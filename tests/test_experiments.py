import csv
import json

import pytest

from scripts.human_evaluation import QUESTIONS, RATING_COLUMNS, aggregate_ratings, prepare_blind_evaluation
from scripts.make_demo_data import make_demo_data
from scripts.run_experiments import run_experiments


@pytest.fixture
def original_manifest(tmp_path):
    make_demo_data(tmp_path / "examples")
    return tmp_path / "examples/pairs.json"


def test_experiment_runner_measures_outputs_and_marks_missing_checkpoints(original_manifest, tmp_path):
    report = run_experiments(original_manifest, tmp_path / "experiments", {
        "methods": [{"name": "literal", "generator": "literal"},
                    {"name": "octave_off", "generator": "rule_based", "controls": {"octave_doubling": 0}},
                    {"name": "octave_on", "generator": "rule_based", "controls": {"octave_doubling": 1}},
                    {"name": "untrained", "generator": "neural", "checkpoint_key": "missing"},
                    {"name": "human", "generator": "human_reference"}]}, split="all")
    rows = {row["method"]: row for row in report["rows"]}
    assert rows["literal"]["status"] == "ok"
    assert rows["untrained"]["status"] == "omitted"
    assert "metrics" not in rows["untrained"]
    assert rows["octave_on"]["metrics"]["octave_doubling_rate"] > rows["octave_off"]["metrics"]["octave_doubling_rate"]
    assert rows["octave_on"]["fingerprint"] != rows["octave_off"]["fingerprint"]
    assert (tmp_path / "experiments/index.html").is_file()
    assert (tmp_path / "experiments/results.csv").is_file()
    assert all(x["bootstrap_95_interval"] is None for x in report["summary"][0]["metrics"].values())


def test_blind_package_uses_anonymous_assets_and_summarizes_only_actual_ratings(original_manifest, tmp_path):
    run_experiments(original_manifest, tmp_path / "experiments", {
        "methods": [{"name": "literal", "generator": "literal"},
                    {"name": "rule_romantic", "generator": "rule_based", "preset": "romantic"}]}, split="all")
    result = prepare_blind_evaluation(tmp_path / "experiments/report.json", tmp_path / "blind")
    assert result["ratings_collected"] == 0
    html = (tmp_path / "blind/participant/index.html").read_text(encoding="utf-8")
    assert "rule_romantic" not in html and "literal" not in html
    assert "key.json" not in html
    blank = aggregate_ratings(result["private_key"], [tmp_path / "blind/participant/ratings_template.csv"], tmp_path / "empty.json")
    assert blank["rating_rows"] == 0 and blank["summary"] == {}
    key = json.loads((tmp_path / "blind/private/key.json").read_text())
    candidate = next(iter(key["candidates"]))
    ratings = tmp_path / "actual.csv"
    with ratings.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RATING_COLUMNS)
        writer.writeheader()
        writer.writerow({"listener_id": "pianist_01", "candidate_id": candidate,
                         **{question: "4" for question in QUESTIONS}})
    actual = aggregate_ratings(result["private_key"], [ratings], tmp_path / "ratings_summary.json")
    assert actual["rating_rows"] == 1 and actual["listeners"] == 1
    method = key["candidates"][candidate]["method"]
    assert actual["summary"][method]["playability"]["mean"] == 4
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_ratings(result["private_key"], [ratings, ratings], tmp_path / "duplicate.json")
    with pytest.raises(ValueError, match="already exists"):
        prepare_blind_evaluation(tmp_path / "experiments/report.json", tmp_path / "blind")
