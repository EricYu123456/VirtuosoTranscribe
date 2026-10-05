"""Export provenance-preserving historical excerpts using inspected segment mappings.

This does not reassign data splits or create new works. It slices checked source
and target scores at the exact mapped measures recorded by prepare_dataset.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

from src.analysis.melody_alignment import align_solo_to_piano
from src.analysis.technique_features import analyze_transcription
from src.analysis.visualization import write_alignment_visualization
from src.data.build_manifest import load_manifest, validate_manifest
from src.data.normalize_score import normalize_source
from src.data.parse_musicxml import parse_score
from src.data.segment import slice_score_measures
from src.inference.audio import render_audio
from src.inference.export import export_score


def export_real_examples(manifest: str | Path, processed: str | Path, output_dir: str | Path,
                         *, source_measures: list[int], pair_id: str | None = None) -> Path:
    pairs = load_manifest(manifest)
    selected = [p for p in pairs if pair_id is None or p["pair_id"] == pair_id]
    if len(selected) != 1:
        raise ValueError("Select exactly one checked historical pair using --pair-id")
    parent = selected[0]
    if "synthetic" in str(parent.get("data_kind", "")).casefold():
        raise ValueError("Historical listening examples require a real checked pair")
    if not source_measures or any(isinstance(m, bool) or not isinstance(m, int) or m < 1 for m in source_measures):
        raise ValueError("Select positive source measure numbers")
    if len(set(source_measures)) != len(source_measures):
        raise ValueError("Source measures must be unique")
    dataset = json.loads(Path(processed).read_text(encoding="utf-8"))
    records = [s for s in dataset.get("segments", []) if s["pair_id"] == parent["pair_id"]]
    matched = {}
    for measure in source_measures:
        options = [s for s in records if s["start_measure"] == measure and s["end_measure"] == measure]
        if len(options) != 1:
            raise ValueError(f"Source measure {measure} needs one inspected single-bar processed segment")
        matched[measure] = options[0]
    source = normalize_source(parse_score(parent["source"]["path"]),
                              solo_part=parent["source"].get("solo_part"),
                              accompaniment_parts=parent["source"].get("accompaniment_parts"))
    target = parse_score(parent["target"]["path"])
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    excerpts = []
    for measure in source_measures:
        record = matched[measure]
        source_window = slice_score_measures(source.score, measure, measure)
        target_window = slice_score_measures(target, record["target_start_measure"], record["target_end_measure"])
        solo_name = next(p.name for p in source_window.score.parts if p.id in source.solo_part_ids)
        accompaniment_names = list(dict.fromkeys(p.name for p in source_window.score.parts
                                                 if p.id in source.accompaniment_part_ids))
        source_score = replace(source_window.score,
                               title=f"{source.score.title} · source measure {measure}")
        human_target = replace(target_window.score,
                               title=f"{target.title} · source measure {measure}, target {record['target_start_measure']}–{record['target_end_measure']}")
        stem = f"m{measure:03d}"
        source_path = export_score(source_score, root / f"source_{stem}.musicxml")
        target_path = export_score(human_target, root / f"human_target_{stem}.musicxml")
        export_score(source_score, root / f"source_{stem}.mid")
        export_score(human_target, root / f"human_target_{stem}.mid")
        source_score.save(root / f"source_{stem}.json")
        human_target.save(root / f"human_target_{stem}.json")
        source_audio = render_audio(source_score, root / f"source_{stem}.wav")
        target_audio = render_audio(human_target, root / f"human_target_{stem}.wav")
        # Verify exported inputs retain unambiguous source role selection.
        local_source = normalize_source(parse_score(source_path), solo_part=solo_name,
                                        accompaniment_parts=accompaniment_names)
        local_target = parse_score(target_path)
        alignment = align_solo_to_piano(local_source.solo, local_target, source_score=local_source.score)
        metrics = analyze_transcription(local_source, local_target, alignment)
        alignment.save(root / f"alignment_{stem}.json")
        write_alignment_visualization(local_source.solo, local_target, alignment,
                                      root / f"alignment_{stem}.html", title=f"Schubert → Horn · source measure {measure}")
        excerpt = deepcopy(parent)
        excerpt.update(pair_id=f"{parent['pair_id']}_{stem}", data_kind="real_historical_excerpt",
                       source={**deepcopy(parent["source"]), "path": source_path.name, "solo_part": solo_name,
                               "accompaniment_parts": accompaniment_names,
                               "parent_sha256": parent["source"].get("sha256"),
                               "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest()},
                       target={**deepcopy(parent["target"]), "path": target_path.name,
                               "parent_sha256": parent["target"].get("sha256"),
                               "sha256": hashlib.sha256(target_path.read_bytes()).hexdigest()},
                       provenance={"parent_pair_id": parent["pair_id"], "parent_manifest": str(Path(manifest).resolve()),
                                   "processed_dataset": str(Path(processed).resolve()),
                                   "processed_segment_id": record["segment_id"],
                                   "source_measure_start": measure, "source_measure_end": measure,
                                   "target_measure_start": record["target_start_measure"],
                                   "target_measure_end": record["target_end_measure"],
                                   "source_note_boundaries": [asdict(b) for b in source_window.note_boundaries],
                                   "target_note_boundaries": [asdict(b) for b in target_window.note_boundaries],
                                   "transformation": "Inclusive mapped measure slices; boundaries clipped, timing rebased; no arrangement or melody edits.",
                                   "split_note": "Excerpts retain the parent's work ID and training split. They are in-sample listening/overfit diagnostics."},
                       parent_alignment_review=deepcopy(parent.get("alignment_review", {})),
                       alignment_review={"source_attacks": len(local_source.solo),
                                         "melody_matches": len(alignment.matches),
                                         "octave_tolerant_coverage": metrics["melody_coverage_octave_tolerant"],
                                         "quality": alignment.quality,
                                         "scope": "This exported excerpt; inspect the alignment HTML and boundary provenance."},
                       excerpt_analysis=metrics, source_audio=source_audio, target_audio=target_audio)
        excerpts.append(excerpt)
    validate_manifest(excerpts, base_dir=root)
    path = root / "pairs.json"
    path.write_text(json.dumps({"schema_version": 1, "pairs": excerpts}, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--processed", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pair-id")
    parser.add_argument("--source-measure", type=int, action="append", required=True)
    args = parser.parse_args(argv)
    try:
        path = export_real_examples(args.manifest, args.processed, args.output_dir,
                                    source_measures=args.source_measure, pair_id=args.pair_id)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Checked historical excerpts: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
