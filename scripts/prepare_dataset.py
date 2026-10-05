"""Prepare a checked local pair manifest for analysis and future training.

Run ``python -m scripts.prepare_dataset --manifest examples/manifest.json
--output data/processed/demo``. This command does not collect or download data.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from src.data.build_manifest import load_manifest
from src.representation.tokenizer import tokenize_score, tokenize_source


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def prepare_dataset(manifest: str | Path, output: str | Path, *,
                    segment_measures: int = 4, subdivisions: int = 24) -> dict[str, Any]:
    """Parse, align, analyze, and tokenize pairs after validating the full manifest.

    All segments inherit their parent work's split. Archival bundles and clipped
    boundary metadata accompany training token arrays so excerpts are inspectable
    and held notes at boundaries are not mistaken for newly composed attacks.
    """
    if isinstance(segment_measures, bool) or not isinstance(segment_measures, int) or segment_measures < 1:
        raise ValueError("segment_measures must be a positive integer")
    if isinstance(subdivisions, bool) or not isinstance(subdivisions, int) or subdivisions < 1:
        raise ValueError("subdivisions must be a positive integer")
    pairs = load_manifest(manifest)
    # Legal statuses and work-level leakage are checked before the parser is
    # imported or invoked, and before the output directory is created.
    from src.analysis.melody_alignment import align_solo_to_piano
    from src.analysis.technique_features import analyze_transcription, extract_source_features
    from src.data.normalize_score import normalize_source, normalize_target
    from src.data.parse_musicxml import parse_musicxml
    from src.data.segment import segment_pair

    output = Path(output).resolve()
    prepared_pairs = []
    all_segments = []
    artifacts: list[tuple[Path, dict[str, Any]]] = []
    for record in pairs:
        source = normalize_source(parse_musicxml(record["source"]["path"]),
                                  solo_part=record["source"].get("solo_part"),
                                  accompaniment_parts=record["source"].get("accompaniment_parts"))
        target = normalize_target(parse_musicxml(record["target"]["path"]))
        alignment = align_solo_to_piano(source.solo, target, source_score=source.score)
        source_bundles = tokenize_source(source, subdivisions=subdivisions)
        token_bundles = {**source_bundles, "target_piano": tokenize_score(target, role="PIANO", subdivisions=subdivisions)}
        pair_id = record["pair_id"]
        segment_entries = []
        for segment in segment_pair(source, target, measures_per_segment=segment_measures, alignment=alignment):
            segment_id = f"{pair_id}_m{segment.start_measure:03d}_m{segment.end_measure:03d}"
            bundles = {**tokenize_source(segment.source, subdivisions=subdivisions),
                       "target_piano": tokenize_score(segment.target, role="PIANO", subdivisions=subdivisions)}
            local_alignment = align_solo_to_piano(segment.source.solo, segment.target,
                                                  source_score=segment.source.score)
            entry = {"segment_id": segment_id, "pair_id": pair_id, "work_id": record["work_id"],
                     "split": record["split"], "start_measure": segment.start_measure,
                     "end_measure": segment.end_measure,
                     "target_start_measure": segment.target_start_measure,
                     "target_end_measure": segment.target_end_measure,
                     "source_solo_tokens": bundles["source_solo"]["tokens"],
                     "source_acc_tokens": bundles["source_acc"]["tokens"],
                     "target_piano_tokens": bundles["target_piano"]["tokens"],
                     "token_bundles": bundles,
                     "alignment": local_alignment.to_dict(),
                     "technique_features": analyze_transcription(segment.source, segment.target, local_alignment),
                     "source_note_boundaries": [asdict(item) for item in segment.source_note_boundaries],
                     "target_note_boundaries": [asdict(item) for item in segment.target_note_boundaries]}
            if "data_kind" in record:
                entry["data_kind"] = record["data_kind"]
            for field in ("controls", "technique_controls"):
                if field in record:
                    if not isinstance(record[field], dict):
                        raise ValueError(f"Pair {pair_id}: {field} must be a mapping")
                    entry[field] = dict(record[field])
            segment_entries.append(entry)
        if not segment_entries:
            raise ValueError(f"Pair {pair_id} has no contiguous aligned measure segments; inspect its alignment")
        relative_path = f"pairs/{pair_id}.json"
        prepared_record = {**record, "processed_path": relative_path,
                           "alignment": {"method": alignment.method, "quality": alignment.quality},
                           "segment_count": len(segment_entries)}
        prepared_pairs.append(prepared_record)
        all_segments.extend(segment_entries)
        artifacts.append((output / relative_path,
                          {"schema_version": 1, "pair": prepared_record,
                           "source": source.score.to_dict(), "target": target.to_dict(),
                           "roles": {"solo_part_ids": source.solo_part_ids,
                                     "accompaniment_part_ids": source.accompaniment_part_ids},
                           "token_bundles": token_bundles,
                           "source_features": extract_source_features(source),
                           "alignment": alignment.to_dict(),
                           "technique_features": analyze_transcription(source, target, alignment),
                           "segments": segment_entries}))
    dataset = {"schema_version": 1, "manifest_path": str(Path(manifest).resolve()),
               "subdivisions": subdivisions, "segment_measures": segment_measures,
               "pair_count": len(prepared_pairs), "segment_count": len(all_segments),
               "pairs": prepared_pairs, "segments": all_segments}
    # Finish all validation/tokenization before committing any artifacts.
    for path, payload in artifacts:
        _write_json(path, payload)
    _write_json(output / "manifest.json", {"schema_version": 1, "pairs": prepared_pairs})
    _write_json(output / "dataset.json", dataset)
    return dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare checked local source/piano pairs; no data downloading.")
    parser.add_argument("--manifest", required=True, type=Path, help="CSV or JSON curated pair manifest")
    parser.add_argument("--output", required=True, type=Path, help="Processed dataset directory")
    parser.add_argument("--segment-measures", type=int, default=4, help="Measures per aligned segment (default: 4)")
    parser.add_argument("--subdivisions", type=int, default=24, help="Exact grid subdivisions per quarter note (default: 24)")
    args = parser.parse_args(argv)
    try:
        dataset = prepare_dataset(args.manifest, args.output,
                                  segment_measures=args.segment_measures, subdivisions=args.subdivisions)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Prepared {dataset['pair_count']} pairs and {dataset['segment_count']} aligned segments in {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
