from src.data.segment import segment_pair, segment_score
import pytest
from src.representation.events import (
    MeasureInfo, NoteEvent, PartInfo, ScoreData, SignatureEvent, SourceScore, TempoEvent,
)


def test_sustained_notes_clip_both_boundaries_without_disappearing():
    original = ScoreData((NoteEvent(60, 3.5, 8, "solo", measure=1, beat=4.5),),
                         (PartInfo("solo", "Solo"),),
                         tuple(MeasureInfo(i + 1, i * 4, 4) for i in range(3)),
                         (SignatureEvent(0, "4/4"),), (SignatureEvent(0, "C"),),
                         (TempoEvent(0, 90),))
    windows = segment_score(original, 1)
    assert [s.score.notes[0].duration for s in windows] == [0.5, 4, 3.5]
    assert [s.score.notes[0].onset for s in windows] == [3.5, 0, 0]
    assert [s.score.notes[0].measure for s in windows] == [1, 1, 1]
    assert [(s.note_boundaries[0].clipped_start, s.note_boundaries[0].clipped_end) for s in windows] == [
        (False, True), (True, True), (True, False)]
    assert all(s.note_boundaries[0].original_index == 0 for s in windows)
    assert all(s.note_boundaries[0].original_end == 11.5 for s in windows)
    assert all(s.score.tempos[0].bpm == 90 and s.score.tempos[0].onset == 0 for s in windows)


def test_segments_inherit_changed_meter_and_key():
    score = ScoreData((NoteEvent(60, 5, 1, "solo", measure=2),), (PartInfo("solo", "Solo"),),
                      (MeasureInfo(1, 0, 4), MeasureInfo(2, 4, 3, "3/4")),
                      (SignatureEvent(0, "4/4"), SignatureEvent(4, "3/4")),
                      (SignatureEvent(0, "C"), SignatureEvent(3, "G")))
    second = segment_score(score, 1)[1]
    assert second.score.time_signatures == (SignatureEvent(0, "3/4"),)
    assert second.score.key_signatures == (SignatureEvent(0, "G"),)
    assert second.score.notes[0].onset == 1
    assert second.score.duration == 3


def test_segment_beat_metadata_uses_compound_meter():
    score = ScoreData((NoteEvent(60, 3.5, 0.5, "solo", measure=2),
                       NoteEvent(62, 4.5, 0.5, "solo", measure=2)),
                      (PartInfo("solo", "Solo"),),
                      (MeasureInfo(1, 0, 3, "6/8"), MeasureInfo(2, 3, 3, "6/8")))
    notes = segment_score(score, 1)[1].score.notes
    assert notes[0].onset == 0.5 and notes[0].beat == pytest.approx(4 / 3)
    assert notes[1].onset == 1.5 and notes[1].beat == 2


def test_pair_uses_measure_mapping_and_retains_roles():
    solo = (NoteEvent(60, 1, 1, "solo", role="SOLO"), NoteEvent(62, 5, 1, "solo", measure=2, role="SOLO"))
    source_score = ScoreData(solo, (PartInfo("solo", "Solo"),),
                            (MeasureInfo(1, 0, 4), MeasureInfo(2, 4, 4)))
    original = SourceScore(source_score, solo, (), ("solo",), ())
    target = ScoreData((NoteEvent(67, 1, 1, "piano"), NoteEvent(72, 5, 1, "piano", measure=2),
                        NoteEvent(74, 9, 1, "piano", measure=3)), (PartInfo("piano", "Piano"),),
                       tuple(MeasureInfo(i + 1, i * 4, 4) for i in range(3)))
    windows = segment_pair(original, target, 1)
    assert [(s.target_start_measure, s.target_end_measure) for s in windows] == [(2, 2), (3, 3)]
    assert [s.source.solo[0].pitch for s in windows] == [60, 62]
    assert all(s.target.notes[0].onset == 1 for s in windows)
    assert all(s.source.solo[0].role == "SOLO" for s in windows)
