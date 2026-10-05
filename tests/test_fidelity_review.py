"""Independent musical-fidelity regressions for parser and exporter."""

import struct

import pytest

from src.data.normalize_score import normalize_source, validate_score
from src.data.parse_musicxml import parse_musicxml
from src.inference.export import export_score
from src.representation.events import MeasureInfo, NoteEvent, PartInfo, ScoreData, TupletInfo


def _vlq(value):
    result = [value & 0x7F]
    value >>= 7
    while value:
        result.insert(0, (value & 0x7F) | 0x80)
        value >>= 7
    return bytes(result)


def _score(notes, measures=None, program=0, name="Piano"):
    return ScoreData(tuple(notes), (PartInfo("part", name, name, program),),
                     tuple(measures or [MeasureInfo(1, 0, 4)]))


def _note(pitch, onset, duration, **changes):
    return NoteEvent(pitch, onset, duration, "part", **changes)


def _events(score):
    return sorted((n.pitch, round(n.onset, 7), round(n.duration, 7)) for n in score.notes)


def test_midi_import_preserves_raw_ticks_instead_of_default_music21_quantization(tmp_path):
    track = (b"\x00\xff\x51\x03\x07\xa1\x20" + _vlq(37) + b"\x90\x3c\x5a"
             + _vlq(137) + b"\x80\x3c\x00" + b"\x00\xff\x2f\x00")
    midi = (b"MThd" + struct.pack(">IHHH", 6, 0, 1, 480)
            + b"MTrk" + struct.pack(">I", len(track)) + track)
    path = tmp_path / "unquantized.mid"
    path.write_bytes(midi)
    parsed = parse_musicxml(path)
    assert len(parsed.notes) == 1
    assert parsed.notes[0].onset == pytest.approx(37 / 480)
    assert parsed.notes[0].duration == pytest.approx(137 / 480)


def test_numeric_pitch_is_authoritative_when_notation_spelling_is_stale(tmp_path):
    score = _score([_note(72, 0, 1, pitch_name="C4")])
    restored = parse_musicxml(export_score(score, tmp_path / "stale-spelling.musicxml"))
    assert _events(restored) == _events(score)


def test_polyphonic_same_pitch_cross_bar_ties_keep_separate_sustains(tmp_path):
    score = _score([_note(60, 0, 6, voice="1"), _note(60, 1, 4, voice="1"),
                    _note(67, 2, 3, voice="1"), _note(72, 4, 1, measure=2, voice="melody")],
                   [MeasureInfo(1, 0, 4), MeasureInfo(2, 4, 4)])
    restored = parse_musicxml(export_score(score, tmp_path / "overlap-ties.musicxml"))
    assert _events(restored) == _events(score)


def test_pickup_and_empty_terminal_measure_keep_their_length(tmp_path):
    score = _score([_note(64, 0, 1), _note(67, 1, 3, measure=2)],
                   [MeasureInfo(1, 0, 1, "4/4", "0", 3),
                    MeasureInfo(2, 1, 4, "4/4", "1"),
                    MeasureInfo(3, 5, 4, "4/4", "2")])
    restored = parse_musicxml(export_score(score, tmp_path / "pickup-rest.musicxml"))
    assert [(m.onset, m.duration, m.padding_left) for m in restored.measures] == [(m.onset, m.duration, m.padding_left) for m in score.measures]
    assert _events(restored) == _events(score)


def test_transposing_instrument_export_import_preserves_concert_pitch(tmp_path):
    score = _score([_note(60, 0, 1), _note(72, 1, 2)], program=71, name="Clarinet")
    restored = parse_musicxml(export_score(score, tmp_path / "clarinet.musicxml"))
    assert _events(restored) == _events(score)


def test_tuplets_and_graces_preserve_timing(tmp_path):
    score = _score([_note(59, 0, 0, is_grace=True),
                    _note(60, 0, 1 / 3, tuplets=(TupletInfo(3, 2, "start"),)),
                    _note(62, 1 / 3, 1 / 3, tuplets=(TupletInfo(3, 2),)),
                    _note(64, 2 / 3, 1 / 3, tuplets=(TupletInfo(3, 2, "stop"),))])
    restored = parse_musicxml(export_score(score, tmp_path / "tuplet-grace.musicxml"))
    assert _events(restored) == _events(score)
    assert sum(n.is_grace for n in restored.notes) == 1
    assert all(n.tuplets for n in restored.notes if not n.is_grace)


@pytest.mark.parametrize("event", [
    _note(60, 0, 1, measure=2),
    _note(60, 0, 1, pitch_name="bad-spelling"),
])
def test_invalid_event_measure_or_pitch_spelling_requires_validation_error(event):
    with pytest.raises(ValueError):
        validate_score(_score([event]))


def test_explicit_quantization_updates_measure_and_beat_after_crossing_barline():
    score = ScoreData((_note(60, 0, 4),
                       NoteEvent(72, 3.99, 1, "solo", measure=1, beat=4.99)),
                      (PartInfo("part", "Piano", "Piano", 0),
                       PartInfo("solo", "Violin", "Violin", 40)),
                      (MeasureInfo(1, 0, 4), MeasureInfo(2, 4, 4)))
    normalized = normalize_source(score, quantization=24)
    assert normalized.solo[0].onset == 4
    assert normalized.solo[0].measure == 2
    assert normalized.solo[0].beat == 1


def test_shorter_solo_final_measure_does_not_truncate_longer_accompaniment(tmp_path):
    path = tmp_path / "short-solo.musicxml"
    path.write_text('''<?xml version="1.0" encoding="utf-8"?>
<score-partwise version="4.0"><part-list>
<score-part id="solo"><part-name>Violin</part-name></score-part>
<score-part id="piano"><part-name>Piano</part-name></score-part>
</part-list>
<part id="solo"><measure number="1"><attributes><divisions>1</divisions>
<time><beats>4</beats><beat-type>4</beat-type></time></attributes>
<note><pitch><step>C</step><octave>5</octave></pitch><duration>1</duration><type>quarter</type></note>
</measure></part>
<part id="piano"><measure number="1"><attributes><divisions>1</divisions>
<time><beats>4</beats><beat-type>4</beat-type></time></attributes>
<note><pitch><step>C</step><octave>3</octave></pitch><duration>4</duration><type>whole</type></note>
</measure></part></score-partwise>''', encoding="utf-8")
    parsed = parse_musicxml(path)
    assert parsed.measures[-1].onset + parsed.measures[-1].duration >= max(n.end for n in parsed.notes)
    validate_score(parsed)
