from collections import Counter
from dataclasses import replace

from music21 import converter
import pytest

from src.data.parse_musicxml import parse_musicxml
from src.inference.export import export_score
from src.representation.events import MeasureInfo, NoteEvent, PartInfo, ScoreData


def fingerprint(score):
    part_names = {p.id: p.name for p in score.parts}
    return Counter((part_names[n.part_id], n.pitch, round(n.onset, 6), round(n.duration, 6))
                   for n in score.notes)


def test_musicxml_roundtrip_preserves_musical_content(notation_path, tmp_path):
    original = parse_musicxml(notation_path)
    reconstructed = parse_musicxml(export_score(original, tmp_path / "roundtrip.musicxml"))
    assert fingerprint(reconstructed) == fingerprint(original)
    assert [(m.onset, m.duration, m.time_signature) for m in reconstructed.measures] == [
        (m.onset, m.duration, m.time_signature) for m in original.measures]
    assert reconstructed.key_signatures == original.key_signatures
    assert reconstructed.tempos == original.tempos


def test_overlapping_different_durations_export_valid_voices(tmp_path):
    score = ScoreData(
        notes=(NoteEvent(48, 0, 8, "piano_lh"), NoteEvent(55, 1, 2, "piano_lh"),
               NoteEvent(60, 4, 1, "piano_lh", measure=2)),
        parts=(PartInfo("piano_lh", "Piano", "Piano", 0),),
        measures=(MeasureInfo(1, 0, 4), MeasureInfo(2, 4, 4)),
    )
    output = export_score(score, tmp_path / "polyphony.musicxml")
    reconstructed = parse_musicxml(output)
    assert fingerprint(reconstructed) == fingerprint(score)


def test_midi_contains_expected_pitches_and_duration(notation_path, tmp_path):
    original = parse_musicxml(notation_path)
    output = export_score(original, tmp_path / "roundtrip.mid")
    midi_score = converter.parse(output, quantizePost=False)
    notes = midi_score.stripTies().flatten().notes
    pitches = [p.midi for element in notes for p in element.pitches]
    assert sorted(pitches) == sorted(n.pitch for n in original.notes)
    assert float(midi_score.highestTime) >= original.duration


def test_invalid_export_fails_before_writing(tmp_path):
    score = ScoreData((NoteEvent(60, 0, 10, "p"),), (PartInfo("p", "Piano"),),
                      (MeasureInfo(1, 0, 4),))
    output = tmp_path / "invalid.musicxml"
    with pytest.raises(ValueError, match="extends past"):
        export_score(score, output)
    assert not output.exists()
    with pytest.raises(ValueError, match="Export requires"):
        export_score(replace(score, notes=()), tmp_path / "bad.wav")
