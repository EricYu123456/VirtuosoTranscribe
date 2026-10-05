"""MusicXML/MXL and MIDI import into concert-pitch symbolic events."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import warnings

from music21 import chord, converter, dynamics, harmony, key, meter, note, spanner, stream, tempo

from src.representation.events import (
    DynamicEvent, MeasureInfo, NoteEvent, PartInfo, ScoreData, SignatureEvent,
    TempoEvent, TupletInfo, note_sort_key,
)
from src.representation.timing import beat_position


def _time(value: object) -> float:
    return float(value)


def _unique(events: list) -> tuple:
    return tuple(sorted(set(events), key=lambda event: event.onset))


def parse_musicxml(path: str | Path) -> ScoreData:
    """Parse written MusicXML or sounding-pitch MIDI without quantizing time.

    Chords become individual notes; ties become one sustained note. Rest gaps,
    pickup lengths, changing meter, key signatures, tempo, dynamics, slur
    endpoints, articulations, and tuplet ratios are retained. Repeat signs are
    not expanded. Unsupported microtonal pitches fail explicitly.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() not in {".musicxml", ".xml", ".mxl", ".mid", ".midi"}:
        raise ValueError(f"Unsupported score format: {path.suffix}")
    with warnings.catch_warnings(record=True) as parser_warnings:
        try:
            parsed = converter.parse(str(path), forceSource=True, quantizePost=False)
        except Exception as exc:
            raise ValueError(f"Cannot parse {path.name}: {exc}") from exc
    notices = [str(w.message) for w in parser_warnings]
    if not isinstance(parsed, stream.Score):
        wrapped = stream.Score()
        wrapped.insert(0, parsed if isinstance(parsed, stream.Part)
                       else stream.Part(parsed.elements))
        parsed = wrapped
    if not parsed.parts:
        raise ValueError("The score has no instrumental parts")
    if path.suffix.lower() in {".mid", ".midi"}:
        notices.append("MIDI import cannot recover original notation, slurs, or dynamics.")
    # music21 reads MusicXML transposition into the instrument and pitch flag.
    parsed = parsed.toSoundingPitch(inPlace=False)
    reference_part = parsed.parts[0]
    if not reference_part.getElementsByClass(stream.Measure):
        parsed = parsed.makeMeasures(inPlace=False)
        reference_part = parsed.parts[0]
    original_measures = list(reference_part.getElementsByClass(stream.Measure))
    part_measures = [list(p.getElementsByClass(stream.Measure)) for p in parsed.parts]
    if any(len(measures) != len(original_measures) for measures in part_measures):
        raise ValueError("Instrumental parts have different measure counts")
    measure_infos = []
    start = 0.0
    for index, measure in enumerate(original_measures):
        signature = measure.timeSignature or measure.getContextByClass(meter.TimeSignature)
        signature_text = signature.ratioString if signature is not None else "4/4"
        lengths = [_time(measures[index].duration.quarterLength) for measures in part_measures]
        if index + 1 < len(original_measures):
            lengths.extend(_time(measures[index + 1].offset) - _time(measures[index].offset)
                           for measures in part_measures)
        length = max(lengths)
        if length <= 0:
            length = _time(signature.barDuration.quarterLength) if signature else 4.0
        if signature and length > _time(signature.barDuration.quarterLength) + 1e-6:
            notices.append(f"Overfull measure {index + 1}: duration {length:g} exceeds {signature_text}; actual timing retained.")
        # A short solo does not make the accompaniment's full bar a pickup.
        padding_left = min(_time(measures[index].paddingLeft) for measures in part_measures)
        measure_infos.append(MeasureInfo(index + 1, start, length, signature_text,
                                         str(measure.number), padding_left))
        start += length
    if not measure_infos:
        raise ValueError("The score has no measures")

    all_notes: list[NoteEvent] = []
    parts = []
    time_signatures = []
    key_signatures = []
    tempos = []
    dynamic_events = []
    used_ids = set()
    for part_index, part in enumerate(parsed.parts):
        part_id = str(part.id)
        if part_id in used_ids:
            part_id = f"{part_id}_{part_index + 1}"
        used_ids.add(part_id)
        inst = part.getInstrument(returnDefault=False)
        part_name = part.partName or (inst.instrumentName if inst else None) or f"Part {part_index + 1}"
        parts.append(PartInfo(part_id, part_name, inst.instrumentName or "" if inst else "",
                              inst.midiProgram if inst else None))
        own_measures = list(part.getElementsByClass(stream.Measure))
        if len(own_measures) != len(measure_infos):
            raise ValueError(f"Part {part_name!r} has a different measure count")
        for own, info in zip(own_measures, measure_infos):
            if abs(_time(own.getOffsetInHierarchy(part)) - info.onset) > 1e-6:
                notices.append(f"Incomplete rests in {part_name!r}; measure onsets normalized to the shared timeline.")
                break
        slur_map: dict[int, list[str]] = {}
        for slur_index, slur in enumerate(part.recurse().getElementsByClass(spanner.Slur)):
            for member in slur.getSpannedElements():
                slur_map.setdefault(id(member), []).append(f"{part_id}_slur_{slur_index + 1}")
        # Score-level spanners are common after MusicXML import.
        for slur_index, slur in enumerate(parsed.spannerBundle.getByClass(spanner.Slur)):
            for member in slur.getSpannedElements():
                slur_map.setdefault(id(member), []).append(f"slur_{slur_index + 1}")
        ties: dict[tuple[str, int], int] = {}
        part_notes: list[NoteEvent] = []
        chord_label_warning = False
        for measure_index, measure in enumerate(own_measures):
            info = measure_infos[measure_index]
            for element in measure.recurse():
                onset = info.onset + _time(element.getOffsetInHierarchy(measure))
                if isinstance(element, meter.TimeSignature):
                    time_signatures.append(SignatureEvent(onset, element.ratioString))
                elif isinstance(element, key.KeySignature):
                    mode = element.mode if isinstance(element, key.Key) else ""
                    key_signatures.append(SignatureEvent(onset, f"{element.sharps}:{mode}"))
                elif isinstance(element, tempo.MetronomeMark):
                    bpm = element.getQuarterBPM()
                    if bpm is not None:
                        tempos.append(TempoEvent(onset, float(bpm)))
                elif isinstance(element, dynamics.Dynamic):
                    dynamic_events.append(DynamicEvent(onset, element.value, part_id))
                if isinstance(element, harmony.ChordSymbol) and not element.writeAsChord:
                    if not chord_label_warning:
                        notices.append(f"Chord symbols in {part_name!r} are annotations, not sounding notes; labels are not represented in the canonical note schema.")
                        chord_label_warning = True
                    continue
                if not isinstance(element, (note.Note, chord.Chord)):
                    continue
                voice_context = element.getContextByClass(stream.Voice)
                voice_id = str(voice_context.id) if voice_context is not None else "1"
                atoms = element.notes if isinstance(element, chord.Chord) else [element]
                for atom in atoms:
                    pitch_value = float(atom.pitch.ps)
                    if abs(pitch_value - round(pitch_value)) > 1e-6:
                        raise ValueError("Microtonal pitches are not supported by the MIDI event schema")
                    tuplet_infos = tuple(TupletInfo(t.numberNotesActual, t.numberNotesNormal, t.type)
                                         for t in element.duration.tuplets)
                    event = NoteEvent(
                        pitch=int(round(pitch_value)), onset=onset,
                        duration=_time(element.duration.quarterLength), part_id=part_id,
                        measure=info.number,
                        beat=beat_position(info, onset),
                        voice=voice_id, velocity=atom.volume.velocity or element.volume.velocity or 80,
                        pitch_name=atom.pitch.nameWithOctave,
                        articulations=tuple(type(a).__name__ for a in element.articulations),
                        tuplets=tuplet_infos, slur_ids=tuple(sorted(set(slur_map.get(id(element), [])))),
                        is_grace=bool(element.duration.isGrace),
                    )
                    tie_type = atom.tie.type if atom.tie else None
                    tie_key = (voice_id, event.pitch)
                    if tie_type in {"continue", "stop"} and tie_key in ties:
                        prior_index = ties[tie_key]
                        prior = part_notes[prior_index]
                        if abs(prior.end - event.onset) < 1e-6:
                            part_notes[prior_index] = replace(
                                prior, duration=event.end - prior.onset,
                                slur_ids=tuple(sorted(set(prior.slur_ids + event.slur_ids))))
                            if tie_type == "stop":
                                ties.pop(tie_key)
                            continue
                        notices.append(f"Non-contiguous tie at {part_name}, measure {info.number}")
                    elif tie_type in {"continue", "stop"}:
                        notices.append(f"Unmatched tie at {part_name}, measure {info.number}")
                    part_notes.append(event)
                    if tie_type in {"start", "continue"}:
                        ties[tie_key] = len(part_notes) - 1
                    else:
                        ties.pop(tie_key, None)
        all_notes.extend(part_notes)
    metadata = parsed.metadata
    return ScoreData(
        notes=tuple(sorted(all_notes, key=note_sort_key)), parts=tuple(parts),
        measures=tuple(measure_infos), time_signatures=_unique(time_signatures),
        key_signatures=_unique(key_signatures), tempos=_unique(tempos),
        dynamics=_unique(dynamic_events),
        title=(metadata.title or path.stem) if metadata else path.stem,
        composer=(metadata.composer or "") if metadata else "", warnings=tuple(notices),
    )


parse_score = parse_musicxml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        score = parse_musicxml(args.input)
        score.save(args.output)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(f"Parsed {len(score.notes)} notes, {len(score.parts)} parts, {len(score.measures)} measures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
