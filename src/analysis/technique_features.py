"""Interpretable analysis before training a neural transcription model."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
import json
from pathlib import Path

from src.representation.events import NoteEvent, ScoreData, SourceScore
from src.evaluation.texture_metrics import texture_metrics
from src.evaluation.harmonic_metrics import harmonic_overlap
from src.evaluation.playability_metrics import evaluate_playability
from .melody_alignment import AlignmentResult, align_solo_to_piano


def extract_source_features(source: SourceScore) -> dict:
    """Role-separated texture summaries using quarter-note timing."""
    return {"solo": texture_metrics(source.score.with_notes(source.solo)),
            "accompaniment": texture_metrics(source.score.with_notes(source.accompaniment)),
            "combined": texture_metrics(source.score.with_notes(source.notes)),
            "solo_part_ids": list(source.solo_part_ids),
            "accompaniment_part_ids": list(source.accompaniment_part_ids),
            "time_signatures": [x.value for x in source.score.time_signatures],
            "key_signatures": [x.value for x in source.score.key_signatures]}


def _mapped_notes(notes: tuple[NoteEvent, ...], alignment: AlignmentResult) -> tuple[NoteEvent, ...]:
    mapping = {m.source_measure: m for m in alignment.measure_matches}
    if not mapping:
        return notes
    result = []
    for note in notes:
        measure = mapping.get(note.measure)
        if measure is not None:
            scale = measure.target_duration / measure.source_duration if measure.source_duration else 1.0
            result.append(replace(note, onset=max(0.0, measure.target_onset + (note.onset - measure.source_onset) * scale),
                                  duration=note.duration * scale, measure=measure.target_measure))
    return tuple(result)


def _signature_agreement(a: ScoreData, b: ScoreData, attribute: str,
                         alignment: AlignmentResult) -> float | None:
    if attribute == "time_signatures" and alignment.measure_matches:
        sm, tm = {m.number: m for m in a.measures}, {m.number: m for m in b.measures}
        return sum(sm[m.source_measure].time_signature == tm[m.target_measure].time_signature
                   for m in alignment.measure_matches) / len(alignment.measure_matches)
    left, right = getattr(a, attribute), getattr(b, attribute)
    if not left and not right:
        return None
    # Compare values at aligned measure starts, respecting inherited signatures.
    def current(events, onset):
        return next((e.value for e in reversed(sorted(events, key=lambda e: e.onset))
                     if e.onset <= onset + 1e-8), None)
    anchors = [(m.source_onset, m.target_onset) for m in alignment.measure_matches]
    if not anchors:
        anchors = [(0.0, 0.0)]
    return sum(current(left, s) == current(right, t) for s, t in anchors) / len(anchors)


def _technique_proxies(target: ScoreData, matched_indices: set[int]) -> dict:
    groups: dict[float, list[NoteEvent]] = defaultdict(list)
    for note in target.notes:
        groups[note.onset].append(note)
    chord_groups = [notes for notes in groups.values() if len({n.pitch for n in notes}) > 1]
    bass_octaves = sum(any(a.pitch < 60 and b.pitch - a.pitch == 12 for a in notes for b in notes)
                       for notes in groups.values())
    repeated = transitions = 0
    voices: dict[tuple[str, str], list[NoteEvent]] = defaultdict(list)
    for note in target.notes:
        voices[(note.part_id, note.voice)].append(note)
    for voice in voices.values():
        ordered = sorted(voice, key=lambda n: (n.onset, n.pitch))
        for a, b in zip(ordered, ordered[1:]):
            if b.onset > a.onset + 1e-8:
                transitions += 1
                repeated += a.pitch == b.pitch
    # Sustained/monophonic runs are candidate figuration, not a harmonic classifier.
    ordered_groups = sorted(groups.items())
    broken_runs = sum(len(a) == len(b) == len(c) == 1
                      and len({a[0].pitch % 12, b[0].pitch % 12, c[0].pitch % 12}) == 3
                      and t3 - t1 <= 1.0
                      for (t1, a), (_, b), (t3, c) in zip(ordered_groups, ordered_groups[1:], ordered_groups[2:]))
    added = [n for i, n in enumerate(target.notes) if i not in matched_indices]
    return {"chordal_attack_ratio": len(chord_groups) / len(groups) if groups else 0.0,
            "bass_octave_attack_ratio": bass_octaves / len(groups) if groups else 0.0,
            "repeated_note_attack_ratio": repeated / transitions if transitions else 0.0,
            "broken_chord_candidate_count": broken_runs,
            "inner_voice_notes_per_beat": sum(53 <= n.pitch <= 76 for n in added) / target.duration
                                           if target.duration else 0.0}


def analyze_transcription(source: SourceScore, target: ScoreData,
                          alignment: AlignmentResult | None = None) -> dict:
    """Flat, JSON-safe metrics plus nested source/target diagnostics.

    Coverage uses one-to-one octave-equivalent alignment. Exact onset F1 uses
    epsilon 1e-6 after measure/beat mapping. Target precision includes all piano
    notes, so it measures transcription sparsity as well as melody presence.
    Polyphony is integrated over notated durations including silence. Harmonic
    overlap includes solo and accompaniment and ignores octave/register.
    """
    alignment = alignment or align_solo_to_piano(source.solo, target, source_score=source.score)
    if any(m.source_index >= len(source.solo) or m.target_index >= len(target.notes)
           for m in alignment.matches):
        raise ValueError("Alignment indices do not belong to the provided scores")
    sm = texture_metrics(source.score.with_notes(source.notes))
    tm = texture_metrics(target)
    matches = alignment.matches
    count, target_count = len(source.solo), len(target.notes)
    strict = [m for m in matches if m.source_pitch == m.target_pitch]
    exact_onset = [m for m in strict if abs(m.onset_deviation) <= 1e-6]
    octave_onset = [m for m in matches if abs(m.onset_deviation) <= 1e-6]

    def coverage(n):
        return n / count if count else 1.0

    def f1(n):
        return 2 * n / (count + target_count) if count + target_count else 1.0

    doubling = 0
    chordal = 0
    for match in matches:
        at_attack = [n for n in target.notes if abs(n.onset - match.expected_onset) <= alignment.onset_tolerance + 1e-8]
        pitches = {n.pitch for n in at_attack if n.pitch % 12 == match.source_pitch % 12}
        doubling += any(abs(pitch - match.target_pitch) == 12 for pitch in pitches)
        chordal += any(n.pitch % 12 != match.source_pitch % 12 for n in at_attack)
    measures = {m.source_measure: m for m in alignment.measure_matches}
    altered = 0
    for match in matches:
        s, t = source.solo[match.source_index], target.notes[match.target_index]
        measure = measures.get(s.measure)
        scale = measure.target_duration / measure.source_duration if measure and measure.source_duration else 1.0
        altered += abs(match.onset_deviation) > 1e-6 or abs(t.duration - s.duration * scale) > 1e-6
    # Every original source transition is eligible. Missing endpoints score zero.
    # Direction uses actual intervals, so octave relocation that reverses motion
    # is reported even when pitch-class coverage remains perfect.
    ordered = sorted(range(count), key=lambda i: (source.solo[i].onset, i))
    by_source = {m.source_index: m for m in matches}
    contour = []
    rhythm = []
    for a_index, b_index in zip(ordered, ordered[1:]):
        if source.solo[b_index].onset > source.solo[a_index].onset + 1e-8:
            a, b = by_source.get(a_index), by_source.get(b_index)
            if a is None or b is None:
                contour.append(False)
                rhythm.append(False)
            else:
                src_interval, tgt_interval = b.source_pitch - a.source_pitch, b.target_pitch - a.target_pitch
                def sign(value):
                    return (value > 0) - (value < 0)
                contour.append(sign(src_interval) == sign(tgt_interval))
                rhythm.append(abs((b.target_onset - a.target_onset) - (b.expected_onset - a.expected_onset)) <= 1e-6)
    non_melody_count = target_count - len({m.target_index for m in matches})
    full_source_alignment = align_solo_to_piano(_mapped_notes(source.notes, alignment), target,
                                               onset_tolerance=alignment.onset_tolerance)
    added_count = len(full_source_alignment.unmatched_target_indices)
    hist = Counter(m.octave_shift for m in matches)
    source_low, target_low = sm["lowest_pitch"], tm["lowest_pitch"]
    stats = {
        "melody_coverage_exact": coverage(len(strict)),
        "melody_coverage_octave_tolerant": coverage(len(matches)),
        "melody_exact_pitch_onset_f1": f1(len(exact_onset)),
        "melody_octave_tolerant_onset_f1": f1(len(octave_onset)),
        "melody_exact_onset_recall": coverage(len(exact_onset)),
        "melody_octave_tolerant_onset_recall": coverage(len(octave_onset)),
        "octave_shift_rate": sum(m.octave_shift != 0 for m in matches) / len(matches) if matches else 0.0,
        "octave_doubling_rate": coverage(doubling) if count else 0.0,
        "chordal_melody_rate": coverage(chordal) if count else 0.0,
        "register_relocation_distribution": {str(k): v / len(matches) for k, v in sorted(hist.items())},
        "rhythmic_alteration_ratio": altered / len(matches) if matches else 0.0,
        "melodic_contour_similarity": sum(contour) / len(contour) if contour else None,
        "rhythm_similarity": sum(rhythm) / len(rhythm) if rhythm else None,
        "register_expansion_semitones": tm["pitch_span"] - sm["pitch_span"],
        "bass_expansion_semitones": source_low - target_low if source_low is not None and target_low is not None else 0,
        "note_density_ratio": tm["notes_per_beat"] / sm["notes_per_beat"] if sm["notes_per_beat"] else None,
        "polyphony_delta": tm["average_simultaneous_notes"] - sm["average_simultaneous_notes"],
        "added_note_ratio": added_count / target_count if target_count else 0.0,
        "added_note_count": added_count,
        "non_melody_note_ratio": non_melody_count / target_count if target_count else 0.0,
        "added_non_melody_note_count": non_melody_count,
        "measure_count_ratio": len(target.measures) / len(source.score.measures) if source.score.measures else None,
        "aligned_measure_ratio": len(alignment.measure_matches) / len(source.score.measures) if source.score.measures else None,
        "meter_agreement": _signature_agreement(source.score, target, "time_signatures", alignment),
        "key_signature_agreement": _signature_agreement(source.score, target, "key_signatures", alignment),
        "alignment_quality": alignment.quality,
        "source_texture": sm,
        "target_texture": tm,
        "metric_definitions": {
            "coverage": "Source melody notes matched once each within onset tolerance; exact requires same MIDI pitch.",
            "onset_f1": "2*matched exact-onset notes/(source melody notes + all target piano notes).",
            "octave_shift_rate": "Fraction of matched melody notes moved by a nonzero multiple of 12 semitones.",
            "octave_doubling_rate": "Fraction of all source melody attacks with a target pitch exactly 12 semitones from its aligned melody note.",
            "polyphony": "Time integral of sounding note-event count / score duration, including silence.",
            "density": "Note attacks per quarter note (including grace notes); beat refers to quarter-note units.",
            "harmonic_overlap": "Duration-weighted pitch-class Jaccard over full source and target; silent/silent intervals excluded.",
            "added_note_ratio": "Unmatched target attacks / target attacks after one-to-one octave-equivalent onset matching to full solo+accompaniment.",
            "non_melody_note_ratio": "Target attacks unmatched to solo melody / target attacks, including retained accompaniment.",
            "contour": "Fraction of original consecutive source attack transitions with both notes matched and same actual pitch movement direction; octave relocation may reverse direction.",
            "bass": "Lowest sounding MIDI pitch per quarter-note bin; pitch classes/direction compared after measure/beat mapping.",
            "harmonic_rhythm": "Agreement of pitch-class-set change flags between adjacent quarter-note bins; includes unchanged bins.",
        },
        "analysis_limitations": [
            "Melody presence can match accompaniment with the same pitch class; voice identity is not inferred.",
            "Pitch-class harmonic overlap does not estimate chord roots, tonal function, or reharmonization quality.",
            "Phrase boundaries and cadences are not inferred; structural metrics use measured metadata only.",
            "Added notes is an onset/pitch-class proxy; substitutions, grace decorations, and rewritten durations need human interpretation.",
            "Technique detections are rule-based candidates; no arranger/style classifier is trained.",
            "Contour direction can change under valid octave relocation; chordal solo streams need a primary-voice annotation for melodic interpretation.",
            "Harmonic/bass metrics omit source measures with no measure mapping; review aligned_measure_ratio alongside them.",
        ],
    }
    stats.update(harmonic_overlap(source.score.with_notes(source.notes), target,
                                  _mapped_notes(source.notes, alignment)))
    stats.update(evaluate_playability(target))
    stats.update(_technique_proxies(target, {m.target_index for m in matches}))
    return stats


def write_analysis_report(stats: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return path
