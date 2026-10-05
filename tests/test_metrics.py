import pytest

from src.analysis.technique_features import analyze_transcription, extract_source_features, write_analysis_report
from src.evaluation.playability_metrics import evaluate_playability
from src.evaluation.texture_metrics import texture_metrics
from src.representation.events import MeasureInfo, NoteEvent, PartInfo, ScoreData, SourceScore


def note(pitch, onset=0, duration=1, part="solo", role="SOLO"):
    return NoteEvent(pitch, onset, duration, part, role=role)


def score(notes):
    ids = sorted({n.part_id for n in notes})
    return ScoreData(tuple(notes), tuple(PartInfo(x, x) for x in ids), (MeasureInfo(1, 0, 4),))


def source(solo, accompaniment=()):
    return SourceScore(score(solo + accompaniment), solo, accompaniment, ("solo",), ("acc",))


def test_polyphony_is_duration_weighted_including_silence():
    metrics = texture_metrics(score((note(60, 0, 4), note(64, 1, 1))))
    assert metrics["average_simultaneous_notes"] == pytest.approx(1.25)
    assert metrics["max_simultaneous_notes"] == 2
    silence = texture_metrics(score((note(60, 0, 1),)))
    assert silence["average_simultaneous_notes"] == pytest.approx(0.25)
    assert silence["average_chord_size_sounding"] == 1


def test_octave_displacement_coverage_doubling_and_density():
    original = source((note(60), note(62, 1)))
    target = score((note(72, part="piano"), note(60, part="piano"), note(74, 1, part="piano")))
    stats = analyze_transcription(original, target)
    assert stats["melody_coverage_exact"] == 0.5
    assert stats["melody_coverage_octave_tolerant"] == 1
    assert stats["octave_shift_rate"] == 0.5
    assert stats["octave_doubling_rate"] == 0.5
    assert stats["note_density_ratio"] == 1.5
    assert stats["register_expansion_semitones"] == 12
    assert stats["melody_octave_tolerant_onset_f1"] == pytest.approx(0.8)


def test_exact_onset_f1_does_not_count_allowed_small_displacement():
    stats = analyze_transcription(source((note(60),)), score((note(60, 0.1),)))
    assert stats["melody_coverage_exact"] == 1
    assert stats["melody_exact_pitch_onset_f1"] == 0
    assert stats["rhythmic_alteration_ratio"] == 1


def test_two_octaves_distant_bass_is_not_melody_octave_doubling():
    original = source((note(72),))
    target = score((note(72), note(48)))
    assert analyze_transcription(original, target)["octave_doubling_rate"] == 0


def test_full_source_harmony_includes_accompaniment():
    original = source((note(60, 0, 4),), (note(64, 0, 4, "acc", "ACC"), note(67, 0, 4, "acc", "ACC")))
    target = score((note(72, 0, 4),))
    stats = analyze_transcription(original, target)
    assert stats["harmonic_pitch_class_overlap"] == pytest.approx(1 / 3)
    assert stats["beat_pitch_class_overlap"] == pytest.approx(1 / 3)
    assert extract_source_features(original)["accompaniment"]["note_count"] == 2


def test_literal_accompaniment_is_not_reported_as_added_material():
    original = source((note(60, 0, 4),), (note(48, 0, 4, "acc", "ACC"), note(55, 0, 4, "acc", "ACC")))
    stats = analyze_transcription(original, score(original.notes))
    assert stats["added_note_ratio"] == 0
    assert stats["non_melody_note_ratio"] == pytest.approx(2 / 3)
    assert stats["bass_pitch_class_agreement"] == 1
    assert stats["harmonic_rhythm_similarity"] == 1


def test_contour_penalizes_missing_transitions_and_register_reversal():
    original = source((note(60), note(62, 1), note(64, 2)))
    missing = analyze_transcription(original, score((note(72), note(76, 2))))
    assert missing["melodic_contour_similarity"] == 0
    assert missing["rhythm_similarity"] == 0
    reversed_register = analyze_transcription(source((note(60), note(62, 1))), score((note(72), note(62, 1))))
    assert reversed_register["melody_coverage_octave_tolerant"] == 1
    assert reversed_register["melodic_contour_similarity"] == 0


def test_playability_includes_sustained_pitches_after_new_attacks():
    target = score((note(36, 0, 4), note(60, 0, 4), note(84, 1, 1)))
    stats = evaluate_playability(target)
    assert stats["playability_span_violation_ratio"] == pytest.approx(0.25)
    assert stats["peak_estimated_hand_span_semitones"] == 24
    assert stats["playability_score"] < 1
    assert stats["playability_limitations"]


def test_wide_keyboard_chord_can_be_split_between_two_hands():
    target = score((note(36, 0, 4), note(43, 0, 4), note(76, 0, 4), note(84, 0, 4)))
    stats = evaluate_playability(target)
    assert stats["playability_span_violation_ratio"] == 0
    assert stats["peak_estimated_hand_span_semitones"] == 8


def test_ten_keys_can_still_overload_one_reachable_hand():
    target = score(tuple(note(pitch, 0, 4) for pitch in [21, *range(80, 89)]))
    stats = evaluate_playability(target)
    assert stats["peak_simultaneous_keys"] == 10
    assert stats["playability_density_violation_ratio"] == 1
    assert stats["playability_score"] < 1


def test_report_is_json_safe_with_undefined_metrics(tmp_path):
    import json
    stats = analyze_transcription(source(()), score(()))
    report = write_analysis_report(stats, tmp_path / "metrics.json")
    saved = json.loads(report.read_text(encoding="utf-8"))
    assert saved["note_density_ratio"] is None
    assert saved["melody_coverage_octave_tolerant"] == 1
