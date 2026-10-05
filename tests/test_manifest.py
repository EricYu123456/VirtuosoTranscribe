from copy import deepcopy
import csv
import json
from pathlib import Path

import pytest

from src.data.build_manifest import load_manifest, validate_manifest


def pair(pair_id="pair1", work_id="work1", split="train"):
    return {"pair_id": pair_id, "work_id": work_id,
            "source": {"path": "source.musicxml", "solo_part": "Violin"},
            "target": {"path": "target.musicxml"},
            "license": {"source_status": "CHECKED", "target_status": "CHECKED"},
            "split": split}


def test_relative_paths_resolve_against_manifest_not_cwd(tmp_path, monkeypatch):
    folder = tmp_path / "corpus"
    folder.mkdir()
    (folder / "source.musicxml").write_text("test", encoding="utf-8")
    (folder / "target.musicxml").write_text("test", encoding="utf-8")
    manifest = folder / "manifest.json"
    manifest.write_text(json.dumps({"pairs": [pair()]}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    records = load_manifest(manifest)
    assert records[0]["source"]["path"] == str((folder / "source.musicxml").resolve())


@pytest.mark.parametrize("side,status", [("source", "UNKNOWN"), ("target", "PUBLIC_DOMAIN"), ("source", ""), ("target", None)])
def test_each_side_requires_explicit_legal_check(side, status):
    record = pair()
    record["license"][f"{side}_status"] = status
    with pytest.raises(ValueError, match=f"license.{side}_status must be CHECKED"):
        validate_manifest([record], require_files=False)


def test_work_split_leakage_in_alternate_arrangements_is_rejected():
    with pytest.raises(ValueError, match="split leakage"):
        validate_manifest([pair(), pair("arrangement2", split="test")], require_files=False)
    assert len(validate_manifest([pair(), pair("arrangement2")], require_files=False)) == 2


def test_validation_alias_and_duplicate_pair_ids():
    result = validate_manifest([pair(split="validation")], require_files=False)
    assert result[0]["split"] == "val"
    with pytest.raises(ValueError, match="Duplicate"):
        validate_manifest([pair(), pair()], require_files=False)


def test_csv_manifest_with_part_selectors(tmp_path):
    for name in ("source.musicxml", "target.musicxml"):
        (tmp_path / name).write_text("test", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        fields = ["pair_id", "work_id", "source_path", "target_path", "solo_part", "accompaniment_parts", "source_status", "target_status", "split"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(dict(zip(fields, ["p1", "w1", "source.musicxml", "target.musicxml", "Violin", "Piano RH;Piano LH", "CHECKED", "CHECKED", "train"])))
    result = load_manifest(manifest)
    assert result[0]["source"]["accompaniment_parts"] == ["Piano RH", "Piano LH"]


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(pair_id="../unsafe"),
    lambda p: p.update(work_id=""),
    lambda p: p.update(split="unassigned"),
    lambda p: p["source"].update(path="source.pdf"),
])
def test_invalid_manifest_values(mutate):
    record = pair()
    mutate(record)
    with pytest.raises(ValueError):
        validate_manifest([record], require_files=False)


def test_unchecked_manifest_does_not_mutate_input():
    records = [pair()]
    original = deepcopy(records)
    validate_manifest(records, require_files=False)
    assert records == original


def test_prepare_dataset_rejects_legal_status_before_creating_output(tmp_path):
    from scripts.prepare_dataset import prepare_dataset

    record = pair()
    record["license"]["target_status"] = "UNVERIFIED"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([record]), encoding="utf-8")
    output = tmp_path / "processed"
    with pytest.raises(ValueError, match="target_status must be CHECKED"):
        prepare_dataset(manifest, output)
    assert not output.exists()


def test_prepare_cli_invalid_manifest_exits_with_code_two(tmp_path, capsys):
    from scripts.prepare_dataset import main

    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]", encoding="utf-8")
    with pytest.raises(SystemExit) as result:
        main(["--manifest", str(manifest), "--output", str(tmp_path / "processed")])
    assert result.value.code == 2
    assert "at least one curated pair" in capsys.readouterr().err


def test_prepared_original_fixtures_have_aligned_streams_and_faithful_bundles(tmp_path):
    from scripts.prepare_dataset import prepare_dataset
    from src.representation.detokenizer import detokenize_score
    from src.representation.events import ScoreData, note_sort_key

    manifest = Path(__file__).resolve().parents[1] / "examples" / "pairs.json"
    result = prepare_dataset(manifest, tmp_path, segment_measures=2)
    assert result["pair_count"] == 3
    assert result["segment_count"] >= 3
    for record in result["pairs"]:
        artifact = json.loads((tmp_path / record["processed_path"]).read_text(encoding="utf-8"))
        bundles = artifact["token_bundles"]
        source = ScoreData.from_dict(artifact["source"])
        target = ScoreData.from_dict(artifact["target"])
        reconstructed = detokenize_score(bundles["source_solo"]).notes + detokenize_score(bundles["source_acc"]).notes
        assert tuple(sorted(reconstructed, key=note_sort_key)) == source.notes
        assert detokenize_score(bundles["target_piano"]) == target
    for segment in result["segments"]:
        assert segment["source_solo_tokens"]
        assert segment["source_acc_tokens"]
        assert segment["target_piano_tokens"]
        assert 0 <= segment["alignment"]["quality"] <= 1
        assert segment["split"] in {"train", "val", "test"}
        for bundle in segment["token_bundles"].values():
            assert detokenize_score(json.loads(json.dumps(bundle))).notes
