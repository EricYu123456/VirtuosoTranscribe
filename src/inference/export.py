"""Validated MusicXML and MIDI export with explicit polyphonic voice lanes."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from music21 import (
    articulations, chord, clef, duration, dynamics, instrument, key, layout,
    metadata, meter, note, pitch, spanner, stream, tempo, tie,
)

from src.data.normalize_score import validate_score
from src.representation.events import NoteEvent, ScoreData


def _make_element(events: list[NoteEvent], length: float, tie_type: str | None):
    pitches = []
    for event in events:
        spelling = pitch.Pitch(event.pitch_name) if event.pitch_name else None
        # A transformed note can carry stale spelling. MIDI pitch is authoritative.
        pitches.append(spelling if spelling is not None and abs(spelling.ps - event.pitch) < 1e-6
                       else event.pitch)
    element = note.Note(pitches[0]) if len(events) == 1 else chord.Chord(pitches)
    element.duration = duration.Duration(length)
    if events[0].is_grace:
        element = element.getGrace()
    elif abs(length - events[0].duration) < 1e-6 and events[0].tuplets:
        # quarterLength derives a compatible ratio; retain bracket endpoints when possible.
        generated = element.duration.tuplets
        for actual, stored in zip(generated, events[0].tuplets):
            if (actual.numberNotesActual, actual.numberNotesNormal) == (stored.actual, stored.normal):
                actual.type = stored.type
    element.volume.velocity = events[0].velocity
    atoms = element.notes if isinstance(element, chord.Chord) else [element]
    for atom, event in zip(atoms, events):
        atom.volume.velocity = event.velocity
        if tie_type:
            atom.tie = tie.Tie(tie_type)
    for name in events[0].articulations:
        articulation_type = getattr(articulations, name, None)
        if articulation_type and issubclass(articulation_type, articulations.Articulation):
            element.articulations.append(articulation_type())
    return element


def to_music21(score: ScoreData) -> stream.Score:
    validate_score(score)
    result = stream.Score()
    result.metadata = metadata.Metadata(title=score.title, composer=score.composer)
    piano_staves = []
    for part_index, part_info in enumerate(score.parts):
        is_piano_staff = part_info.id in {"piano_rh", "piano_lh"}
        part = stream.PartStaff(id=part_info.id) if is_piano_staff else stream.Part(id=part_info.id)
        part.partName = part_info.name
        if part_info.instrument.casefold() == "piano" or part_info.midi_program == 0:
            inst = instrument.Piano()
        elif part_info.midi_program is not None:
            inst = instrument.instrumentFromMidiProgram(part_info.midi_program)
        else:
            inst = instrument.Instrument(instrumentName=part_info.instrument or part_info.name)
        part.insert(0, inst)
        part.atSoundingPitch = True
        if is_piano_staff:
            piano_staves.append(part)
        measure_streams = []
        prior_signature = None
        for info in score.measures:
            measure = stream.Measure(number=info.number)
            measure.paddingLeft = info.padding_left
            if info.time_signature != prior_signature:
                measure.insert(0, meter.TimeSignature(info.time_signature))
            prior_signature = info.time_signature
            if info.number == 1:
                low = part_info.id == "piano_lh"
                if not is_piano_staff:
                    pitches = [n.pitch for n in score.notes if n.part_id == part_info.id]
                    low = bool(pitches) and sum(pitches) / len(pitches) < 60
                measure.insert(0, clef.BassClef() if low else clef.TrebleClef())
            part.insert(info.onset, measure)
            measure_streams.append(measure)

        def place(onset: float, element) -> None:
            for info, measure in zip(score.measures, measure_streams):
                if info.onset - 1e-6 <= onset < info.onset + info.duration - 1e-6:
                    measure.insert(max(0, onset - info.onset), element)
                    return
            raise ValueError(f"Notation event at {onset} lies outside the measure timeline")

        for signature in score.time_signatures:
            # Initial and boundary changes were already added from MeasureInfo.
            if any(abs(signature.onset - m.onset) < 1e-6 for m in score.measures):
                continue
            place(signature.onset, meter.TimeSignature(signature.value))
        for signature in score.key_signatures:
            sharps, _, mode = signature.value.partition(":")
            signature_object = key.KeySignature(int(sharps))
            place(signature.onset, signature_object.asKey(mode) if mode else signature_object)
        if part_index == 0:
            for event in score.tempos:
                place(event.onset, tempo.MetronomeMark(number=event.bpm))
        for event in score.dynamics:
            if event.part_id == part_info.id:
                place(event.onset, dynamics.Dynamic(event.value))

        groups: dict[tuple, list[NoteEvent]] = defaultdict(list)
        for event in score.notes:
            if event.part_id == part_info.id:
                groups[(event.onset, event.duration, event.voice, event.is_grace,
                        event.tuplets, event.articulations)].append(event)
        lane_ends: dict[str, list[float]] = defaultdict(list)
        fragments: dict[tuple[int, str], list[tuple[float, object]]] = defaultdict(list)
        slur_elements: dict[str, list] = defaultdict(list)
        for group_key, events in sorted(groups.items(), key=lambda item: (item[0][0], str(item[0][2]))):
            first = events[0]
            ends = lane_ends[first.voice]
            lane = next((i for i, end in enumerate(ends) if end <= first.onset + 1e-6), len(ends))
            if lane == len(ends):
                ends.append(first.end)
            else:
                ends[lane] = first.end
            voice_name = f"{first.voice}_{lane + 1}"
            spans = [(i, max(first.onset, info.onset), min(first.end, info.onset + info.duration))
                     for i, info in enumerate(score.measures)
                     if first.onset < info.onset + info.duration - 1e-6
                     and first.end > info.onset + 1e-6]
            if first.is_grace:
                spans = [(i, first.onset, first.onset) for i, info in enumerate(score.measures)
                         if info.onset <= first.onset < info.onset + info.duration]
            for span_index, (measure_index, start, end) in enumerate(spans):
                tie_type = None
                if len(spans) > 1:
                    tie_type = "start" if span_index == 0 else "stop" if span_index == len(spans) - 1 else "continue"
                element = _make_element(events, end - start, tie_type)
                offset = start - score.measures[measure_index].onset
                fragments[(measure_index, voice_name)].append((offset, element))
                for slur_id in set(slur for n in events for slur in n.slur_ids):
                    slur_elements[slur_id].append(element)
        all_voice_names = sorted({name for _, name in fragments})
        voice_ids = {name: index + 1 for index, name in enumerate(all_voice_names)}
        for measure_index, (info, measure) in enumerate(zip(score.measures, measure_streams)):
            voice_names = [name for index, name in fragments if index == measure_index]
            if not voice_names:
                voice_names = ["rest"]
            for voice_name in sorted(voice_names):
                voice = stream.Voice(id=voice_ids.get(voice_name, len(voice_ids) + 1))
                cursor = 0.0
                for offset, element in sorted(fragments[(measure_index, voice_name)], key=lambda x: x[0]):
                    if offset > cursor + 1e-6:
                        voice.insert(cursor, note.Rest(quarterLength=offset - cursor))
                    voice.insert(offset, element)
                    cursor = max(cursor, offset + float(element.duration.quarterLength))
                if cursor < info.duration - 1e-6:
                    voice.insert(cursor, note.Rest(quarterLength=info.duration - cursor))
                measure.insert(0, voice)
        for members in slur_elements.values():
            if len(members) >= 2:
                part.insert(0, spanner.Slur(members[0], members[-1]))
        result.insert(0, part)
    if len(piano_staves) == 2:
        result.insert(0, layout.StaffGroup(piano_staves, name="Piano", abbreviation="Pno.",
                                         symbol="brace", barTogether=True))
    result.atSoundingPitch = True
    return result


def export_score(score: ScoreData, path: str | Path) -> Path:
    """Export a score; invalid events fail before a file is written."""
    path = Path(path)
    format_name = {".musicxml": "musicxml", ".xml": "musicxml", ".mxl": "musicxml",
                   ".mid": "midi", ".midi": "midi"}.get(path.suffix.lower())
    if format_name is None:
        raise ValueError("Export requires .musicxml, .xml, .mxl, .mid, or .midi")
    result = to_music21(score)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result.write(format_name, fp=str(path))
    except Exception as exc:
        raise ValueError(f"Cannot export {path.name}: {exc}") from exc
    return path
