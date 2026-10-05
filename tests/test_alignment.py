from src.analysis.melody_alignment import align_solo_to_piano
from src.analysis.visualization import write_alignment_visualization
from src.representation.events import MeasureInfo, NoteEvent, PartInfo, ScoreData


def note(pitch, onset, duration=1, measure=None):
    return NoteEvent(pitch, onset, duration, "piano", measure=measure or int(onset // 4) + 1)


def score(notes, measures):
    return ScoreData(tuple(notes), (PartInfo("piano", "Piano"),),
                     tuple(MeasureInfo(i + 1, 4 * i, 4) for i in range(measures)))


def test_octave_shift_and_omitted_notes():
    source = (note(60, 0), note(62, 1), note(64, 2), note(65, 3))
    target = (note(48, 0, 4), note(72, 0), note(76, 2), note(77, 3))
    result = align_solo_to_piano(source, target)
    assert [m.source_index for m in result.matches] == [0, 2, 3]
    assert result.unmatched_source_indices == (1,)
    assert all(m.octave_shift == 12 for m in result.matches)
    assert 0.6 < result.quality < 0.8


def test_large_onset_mismatch_is_not_pitch_coverage():
    result = align_solo_to_piano((note(60, 0),), (note(72, 0.5),))
    assert result.matches == ()
    assert result.quality == 0


def test_octave_doubling_cannot_match_two_repeated_attacks():
    source = (note(60, 0, 0.25), note(60, 0.25, 0.25))
    target = (note(60, 0, 0.5), note(72, 0, 0.5))
    result = align_solo_to_piano(source, target, onset_tolerance=0.5)
    assert len(result.matches) == 1
    assert result.matches[0].source_index == 0
    assert result.unmatched_source_indices == (1,)


def test_unordered_chords_use_one_to_one_optimal_assignment():
    source = (note(64, 0), note(60, 0))
    target = (note(84, 0), note(76, 0), note(72, 0))
    result = align_solo_to_piano(source, target)
    assert [(m.source_index, m.target_index) for m in result.matches] == [(0, 1), (1, 2)]
    assert len({m.target_index for m in result.matches}) == 2


def test_measure_insertion_maps_beats_before_notes():
    source = score((note(60, 1), note(62, 5)), 2)
    target = score((note(67, 1), note(72, 5), note(74, 9)), 3)
    result = align_solo_to_piano(source.notes, target, source_score=source)
    assert [(m.source_measure, m.target_measure) for m in result.measure_matches] == [(1, 2), (2, 3)]
    assert [m.expected_onset for m in result.matches] == [5, 9]
    assert all(m.onset_deviation == 0 for m in result.matches)
    assert all(m.offset == 4 for m in result.matches)


def test_empty_streams_and_invalid_tolerance():
    import pytest
    assert align_solo_to_piano((), ()).quality == 1
    assert align_solo_to_piano((note(60, 0),), ()).unmatched_source_indices == (0,)
    with pytest.raises(ValueError):
        align_solo_to_piano((), (), onset_tolerance=float("nan"))


def test_inspectable_plot_retains_note_indices_and_escapes_title(tmp_path):
    source, target = (note(60, 0),), (note(72, 0),)
    alignment = align_solo_to_piano(source, target)
    path = write_alignment_visualization(source, target, alignment, tmp_path / "alignment.html", title="<test>")
    html = path.read_text(encoding="utf-8")
    assert "<svg" in html and "Source index 0" in html and "Target index 0" in html
    assert "&lt;test&gt;" in html and "<test>" not in html
