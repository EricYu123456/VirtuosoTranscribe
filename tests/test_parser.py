from pathlib import Path

from music21 import instrument, meter, note, stream
import pytest

from src.data.normalize_score import normalize_source
from src.data.parse_musicxml import parse_musicxml
from src.representation.events import ScoreData


def test_ties_tuplets_pickups_and_signatures(notation_path):
    score = parse_musicxml(notation_path)
    source = normalize_source(score, solo_part="Violoncello", accompaniment_parts=["Piano"])
    assert len(source.solo) == 5
    assert source.solo[0].pitch == 65
    assert source.solo[0].onset == 0
    assert source.solo[0].duration == 2
    assert source.solo[1].onset == pytest.approx(2)
    assert source.solo[1].duration == pytest.approx(1 / 3)
    assert source.solo[1].tuplets[0].actual == 3
    assert source.solo[0].articulations == ("Tenuto",)
    assert source.solo[0].slur_ids
    assert [m.number for m in score.measures] == [1, 2, 3]
    assert [m.duration for m in score.measures] == [1, 4, 3]
    assert score.measures[0].padding_left == 3
    assert score.measures[-1].time_signature == "3/4"
    assert score.tempos[0].bpm == 90
    assert score.dynamics[0].value == "p"
    assert [event.value.split(":")[0] for event in score.key_signatures] == ["-1", "1"]
    assert {n.role for n in source.notes} == {"SOLO", "ACC"}


def test_json_roundtrip(notation_path, tmp_path):
    score = parse_musicxml(notation_path)
    assert ScoreData.load(score.save(tmp_path / "score.json")) == score


def test_concert_pitch_transposing_instrument(tmp_path: Path):
    score, part, measure = stream.Score(), stream.Part(), stream.Measure(number=1)
    part.insert(0, instrument.Clarinet())
    part.partName = "Clarinet"
    part.atSoundingPitch = False
    measure.insert(0, meter.TimeSignature("4/4"))
    measure.append(note.Note("C5", quarterLength=4))
    part.append(measure)
    score.insert(0, part)
    score.atSoundingPitch = False
    path = tmp_path / "clarinet.musicxml"
    score.write("musicxml", fp=str(path))
    parsed = parse_musicxml(path)
    assert parsed.notes[0].pitch == 70  # Written C5 -> sounding B-flat4.


def test_ambiguous_solo_requires_selection(notation_path):
    from dataclasses import replace
    from src.representation.events import PartInfo

    score = parse_musicxml(notation_path)
    score = replace(score, parts=score.parts + (PartInfo("extra", "Violin", "Violin", 40),))
    with pytest.raises(ValueError, match="exactly one"):
        normalize_source(score)


def test_missing_and_unsupported_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_musicxml(tmp_path / "missing.musicxml")
    bad = tmp_path / "score.txt"
    bad.write_text("not a score")
    with pytest.raises(ValueError, match="Unsupported"):
        parse_musicxml(bad)
