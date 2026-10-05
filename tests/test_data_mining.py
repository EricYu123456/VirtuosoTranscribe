import json

import pytest

from src.data.mine_pairs import mine_pairs
from src.inference.export import export_score
from src.representation.events import MeasureInfo, NoteEvent, PartInfo, ScoreData


def catalog_row(path, *, source=False, **changes):
    row = {"song_name": "A Little Study (piano arrangement)", "composer_name": "Original Composer",
           "song_length.beats": "4", "song_length.bars": "1", "key": "C major", "meter": "4/4",
           "license": "cc-zero", "license_conflict": "False", "has_paywall": "False",
           "is_official": "False", "subset:all_valid": "True", "tracks": "40-0" if source else "0",
           "mxl": str(path), "metadata": str(path) + ".json"}
    row.update(changes)
    return row


def write_source(path):
    score = ScoreData(tuple(NoteEvent(pitch, index, 1, "solo") for index, pitch in enumerate((72, 74, 76, 77)))
                      + (NoteEvent(48, 0, 4, "acc"),),
                      (PartInfo("solo", "Violin", "Violin", 40), PartInfo("acc", "Piano", "Piano", 0)),
                      (MeasureInfo(1, 0, 4),), title="A Little Study")
    return export_score(score, path)


def write_target(path, pitches):
    score = ScoreData(tuple(NoteEvent(pitch, index, 1, "piano") for index, pitch in enumerate(pitches)),
                      (PartInfo("piano", "Piano", "Piano", 0),), (MeasureInfo(1, 0, 4),), title="A Little Study")
    return export_score(score, path)


def test_metadata_mode_does_not_parse_missing_or_remote_scores(monkeypatch):
    def forbidden_parse(*args, **kwargs):
        raise AssertionError("Metadata mode must never parse or download scores")
    monkeypatch.setattr("src.data.parse_musicxml.parse_score", forbidden_parse)
    result = mine_pairs([catalog_row("missing.mxl", source=True), catalog_row("https://example.com/unfetched.mxl")])
    assert len(result) == 1
    assert result[0].score_review == {}
    assert result[0].review_status == "NEEDS_REVIEW"
    assert result[0].metadata_evidence["meter_agreement"] is True


def test_actual_melody_rejects_title_collision_and_ranks_high_coverage_pair(tmp_path):
    source = write_source(tmp_path / "source.musicxml")
    correct = write_target(tmp_path / "correct.musicxml", (84, 86, 88, 89))
    # No source pitch classes occur in this unrelated study, despite identical metadata.
    unrelated = write_target(tmp_path / "unrelated.musicxml", (73, 75, 78, 80))
    rows = [catalog_row(source, source=True), catalog_row(unrelated), catalog_row(correct)]
    assert len(mine_pairs(rows)) == 2
    filtered = mine_pairs(rows, review_scores=True)
    assert len(filtered) == 1
    assert filtered[0].target_path == str(correct)
    assert filtered[0].score_review["melody_coverage_octave_tolerant"] == 1
    assert filtered[0].score_review["status"] == "PASSED_THRESHOLDS"
    assert filtered[0].review_status == "NEEDS_REVIEW"
    audited = mine_pairs(rows, review_scores=True, include_rejected=True)
    assert audited[0].target_path == str(correct)
    assert audited[1].score_review["status"] == "REJECTED_THRESHOLDS"
    assert audited[1].score_review["melody_coverage_octave_tolerant"] == 0
    assert "melody_coverage_octave_tolerant" in audited[1].score_review["failed_thresholds"]
    assert all(pair.review_status == "NEEDS_REVIEW" for pair in audited)


def test_thresholds_are_explicit_and_partial_coverage_can_be_ranked(tmp_path):
    source = write_source(tmp_path / "source.musicxml")
    perfect = write_target(tmp_path / "perfect.musicxml", (72, 74, 76, 77))
    partial = write_target(tmp_path / "partial.musicxml", (72, 74, 76, 80))
    rows = [catalog_row(source, source=True), catalog_row(partial), catalog_row(perfect)]
    result = mine_pairs(rows, review_scores=True, min_melody_coverage=.5, min_alignment_quality=.3)
    assert len(result) == 2
    assert result[0].target_path == str(perfect)
    assert result[0].confidence > result[1].confidence
    assert result[1].score_review["melody_coverage_octave_tolerant"] == .75
    assert result[1].score_review["thresholds"]["min_melody_coverage"] == .5


def test_local_score_review_resolves_root_and_caches_shared_source(tmp_path, monkeypatch):
    write_source(tmp_path / "source.musicxml")
    write_target(tmp_path / "one.musicxml", (72, 74, 76, 77))
    write_target(tmp_path / "two.musicxml", (84, 86, 88, 89))
    from src.data.parse_musicxml import parse_score
    seen = []
    def counted(path):
        seen.append(path.name)
        return parse_score(path)
    monkeypatch.setattr("src.data.parse_musicxml.parse_score", counted)
    result = mine_pairs([catalog_row("source.musicxml", source=True), catalog_row("one.musicxml"),
                         catalog_row("two.musicxml")], review_scores=True, score_root=tmp_path)
    assert len(result) == 2
    assert seen.count("source.musicxml") == 1
    assert len(seen) == 3


def test_composer_mismatch_and_unknown_duration_are_conservative():
    source = catalog_row("source.mxl", source=True)
    assert mine_pairs([source, catalog_row("target.mxl", composer_name="Different Composer")]) == []
    candidates = mine_pairs([source, catalog_row("target.mxl", **{"song_length.beats": "nan"})])
    assert candidates[0].metadata_evidence["duration_similarity"] is None
    assert candidates[0].confidence < 1


def test_actual_structural_mismatch_is_filtered_even_when_excerpt_melody_matches(tmp_path):
    source = write_source(tmp_path / "source.musicxml")
    pitches = (72, 74, 76, 77, 72, 74, 76, 77)
    score = ScoreData(tuple(NoteEvent(pitch, index, 1, "piano", measure=index // 4 + 1)
                            for index, pitch in enumerate(pitches)),
                      (PartInfo("piano", "Piano", "Piano", 0),),
                      (MeasureInfo(1, 0, 4), MeasureInfo(2, 4, 4)))
    target = export_score(score, tmp_path / "double_length.musicxml")
    candidates = mine_pairs([catalog_row(source, source=True), catalog_row(target)],
                            review_scores=True, include_rejected=True)
    assert candidates[0].score_review["melody_coverage_octave_tolerant"] == 1
    assert "structural_similarity" in candidates[0].score_review["failed_thresholds"]


def test_alignment_resource_budget_fails_explicitly_without_truncation(tmp_path):
    source = write_source(tmp_path / "source.musicxml")
    target = write_target(tmp_path / "target.musicxml", (72, 74, 76, 77))
    candidate = mine_pairs([catalog_row(source, source=True), catalog_row(target)],
                           review_scores=True, include_rejected=True, max_alignment_cells=10)[0]
    assert candidate.score_review["status"] == "REVIEW_LIMIT_EXCEEDED"
    assert candidate.score_review["alignment_cells"] == 25


def test_missing_scores_stay_unapproved_and_are_not_downloaded():
    rows = [catalog_row("https://example.com/source.mxl", source=True), catalog_row("absent.mxl")]
    assert mine_pairs(rows, review_scores=True) == []
    report = mine_pairs(rows, review_scores=True, include_rejected=True)[0]
    assert report.score_review["status"] == "UNAVAILABLE"
    assert "never downloads" in report.score_review["reason"]
    assert report.review_status == "NEEDS_REVIEW"


def test_cli_exports_thresholds_rejections_and_zero_automatic_approvals(tmp_path):
    from scripts.mine_pairs import main
    write_source(tmp_path / "source.musicxml")
    write_target(tmp_path / "unrelated.musicxml", (73, 75, 78, 80))
    rows = [catalog_row("source.musicxml", source=True), catalog_row("unrelated.musicxml")]
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps(rows), encoding="utf-8")
    output, report = tmp_path / "candidates.json", tmp_path / "review.json"
    assert main(["--catalog", str(catalog), "--output", str(output), "--review-scores",
                 "--score-root", str(tmp_path), "--review-report", str(report)]) == 0
    assert json.loads(output.read_text()) == []
    payload = json.loads(report.read_text())
    assert payload["approved_pairs"] == 0
    assert payload["downloads_performed"] is False
    assert payload["score_review_counts"]["REJECTED_THRESHOLDS"] == 1
    assert payload["candidates"][0]["review_status"] == "NEEDS_REVIEW"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -.1, 1.1, True])
def test_review_thresholds_must_be_valid(value):
    with pytest.raises(ValueError, match="finite"):
        mine_pairs([], min_melody_coverage=value)
