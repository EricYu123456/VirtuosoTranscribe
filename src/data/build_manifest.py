"""Read and validate curated local source/transcription pairs.

No score downloading is performed. Both legal statuses must be explicitly
CHECKED, and all editions/arrangements of one work must remain in one split.
Relative score paths are always resolved against the manifest's directory.
"""

from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path
import re
from typing import Any, Iterable


_PAIR_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SPLITS = {"train": "train", "val": "val", "validation": "val", "test": "test"}


def _part_list(value: Any) -> list[str] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        value = json.loads(value) if value.lstrip().startswith("[") else value.split(";")
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError("accompaniment_parts must be a list of part names or semicolon-separated names")
    return [item.strip() for item in value]


def _csv_pair(row: dict[str, str]) -> dict[str, Any]:
    def field(*names: str) -> str:
        return next((row[name].strip() for name in names if row.get(name)), "")

    source: dict[str, Any] = {"path": field("source_path", "source.path")}
    target = {"path": field("target_path", "target.path")}
    if field("solo_part", "source.solo_part"):
        source["solo_part"] = field("solo_part", "source.solo_part")
    accompaniment = _part_list(field("accompaniment_parts", "source.accompaniment_parts"))
    if accompaniment is not None:
        source["accompaniment_parts"] = accompaniment
    for name in ("solo_instrument", "key", "meter"):
        if field(name, f"source.{name}"):
            source[name] = field(name, f"source.{name}")
    pair = {"pair_id": field("pair_id"), "work_id": field("work_id"),
            "source": source, "target": target,
            "license": {"source_status": field("source_status", "license.source_status"),
                        "target_status": field("target_status", "license.target_status")},
            "split": field("split")}
    for name in ("composer", "arranger"):
        if field(name):
            pair[name] = field(name)
    return pair


def validate_manifest(pairs: Iterable[dict[str, Any]], *, base_dir: str | Path = ".",
                      require_files: bool = True) -> list[dict[str, Any]]:
    """Validate the entire corpus before any parsing or output writes occur."""
    base_dir = Path(base_dir).resolve()
    normalized = []
    pair_ids = set()
    work_splits: dict[str, str] = {}
    for index, raw in enumerate(pairs):
        if not isinstance(raw, dict):
            raise ValueError(f"Manifest entry {index + 1} must be an object")
        pair = deepcopy(raw)
        pair_id, work_id = pair.get("pair_id"), pair.get("work_id")
        if not isinstance(pair_id, str) or not _PAIR_ID.fullmatch(pair_id):
            raise ValueError(f"Manifest entry {index + 1} requires a safe, nonempty pair_id")
        if pair_id in pair_ids:
            raise ValueError(f"Duplicate pair_id: {pair_id}")
        pair_ids.add(pair_id)
        if not isinstance(work_id, str) or not work_id.strip():
            raise ValueError(f"Pair {pair_id} requires work_id")
        work_id = work_id.strip()
        pair["work_id"] = work_id
        license_info = pair.get("license")
        if not isinstance(license_info, dict):
            raise ValueError(f"Pair {pair_id} requires individually checked source and target legal statuses")
        for side in ("source", "target"):
            status = license_info.get(f"{side}_status")
            if not isinstance(status, str) or status.strip().upper() != "CHECKED":
                raise ValueError(f"Pair {pair_id}: license.{side}_status must be CHECKED")
            license_info[f"{side}_status"] = "CHECKED"
        split = pair.get("split")
        if not isinstance(split, str) or split.strip().lower() not in _SPLITS:
            raise ValueError(f"Pair {pair_id}: split must be train, val, or test")
        split = _SPLITS[split.strip().lower()]
        pair["split"] = split
        if work_id in work_splits and work_splits[work_id] != split:
            raise ValueError(f"Work-level split leakage: {work_id!r} appears in both {work_splits[work_id]} and {split}")
        work_splits[work_id] = split
        for side in ("source", "target"):
            entry = pair.get(side)
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not entry["path"].strip():
                raise ValueError(f"Pair {pair_id} requires {side}.path")
            path = Path(entry["path"])
            path = (path if path.is_absolute() else base_dir / path).resolve()
            if require_files and not path.is_file():
                raise ValueError(f"Pair {pair_id}: {side} score does not exist: {path}")
            if path.suffix.lower() not in {".musicxml", ".xml", ".mxl", ".mid", ".midi"}:
                raise ValueError(f"Pair {pair_id}: unsupported {side} score extension {path.suffix!r}")
            entry["path"] = str(path)
        if "accompaniment_parts" in pair["source"]:
            pair["source"]["accompaniment_parts"] = _part_list(pair["source"]["accompaniment_parts"])
        normalized.append(pair)
    if not normalized:
        raise ValueError("Manifest must contain at least one curated pair")
    return normalized


def load_manifest(path: str | Path, *, require_files: bool = True) -> list[dict[str, Any]]:
    path = Path(path).resolve()
    try:
        if path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8-sig") as handle:
                pairs = [_csv_pair(row) for row in csv.DictReader(handle)]
        elif path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            pairs = payload.get("pairs") if isinstance(payload, dict) else payload
            if not isinstance(pairs, list):
                raise ValueError("JSON manifest must be an array or an object with a pairs array")
        else:
            raise ValueError("Manifest must be CSV or JSON")
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read manifest {path}: {exc}") from exc
    return validate_manifest(pairs, base_dir=path.parent, require_files=require_files)


def write_manifest(pairs: Iterable[dict[str, Any]], path: str | Path) -> Path:
    """Write checked canonical metadata; referenced scores remain local."""
    validated = validate_manifest(pairs, require_files=True)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"schema_version": 1, "pairs": validated}, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
