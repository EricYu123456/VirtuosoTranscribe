from pathlib import Path

import pytest

from scripts.evaluate_all import evaluate_all, reference_metrics
from scripts.make_demo_data import make_demo_data
from src.analysis.melody_alignment import align_solo_to_piano
from src.analysis.technique_features import analyze_transcription
from src.baselines.literal_reduction import reduce_literal
from src.data.normalize_score import normalize_source
from src.data.parse_musicxml import parse_score
from src.inference.export import export_score


def test_original_demo_has_hand_verified_melody_and_harmony(tmp_path):
    make_demo_data(tmp_path)
    source = normalize_source(parse_score(tmp_path / "source.musicxml"))
    target = parse_score(tmp_path / "target.musicxml")
    assert [n.pitch for n in source.solo] == [60, 62, 64, 67, 69, 67, 65, 64, 62, 64, 65, 59, 60]
    assert source.duration == target.duration == 16
    assert len(target.notes) == 29
    alignment = align_solo_to_piano(source.solo, target, source_score=source.score)
    stats = analyze_transcription(source, target, alignment)
    assert stats["melody_coverage_octave_tolerant"] == 1
    assert stats["octave_doubling_rate"] == pytest.approx(4 / 13)


def test_exported_baseline_reference_evaluation(tmp_path):
    make_demo_data(tmp_path / "examples")
    examples = tmp_path / "examples"
    source = normalize_source(parse_score(examples / "source.musicxml"))
    prediction = export_score(reduce_literal(source), tmp_path / "literal.musicxml")
    rows = evaluate_all(prediction, source=examples / "source.musicxml",
                        reference=examples / "target.musicxml")
    assert len(rows) == 1
    assert rows[0]["melody_coverage_octave_tolerant"] == 1
    assert 0 <= rows[0]["reference_exact_note_onset_f1"] <= 1
    assert Path(rows[0]["prediction"]).is_absolute()
    assert reference_metrics(source.score, source.score)["reference_exact_note_onset_f1"] == 1


def test_evaluation_requires_comparison_and_candidates(tmp_path):
    with pytest.raises(ValueError, match="Provide"):
        evaluate_all(tmp_path)
    with pytest.raises(ValueError, match="No MusicXML"):
        evaluate_all(tmp_path, reference=tmp_path / "missing.musicxml")
