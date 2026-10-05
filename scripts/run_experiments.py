"""Reproducible baseline, neural, constraint, architecture, and control comparisons.

Missing checkpoints are reported as omissions. Metrics come only from generated
scores; symbolic proxies cannot establish listening quality or generalization.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import replace
from html import escape
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import statistics
import uuid

import yaml

from scripts.evaluate_all import reference_metrics
from src.analysis.technique_features import analyze_transcription
from src.baselines.rule_based import load_preset
from src.data.build_manifest import load_manifest
from src.data.normalize_score import normalize_source
from src.data.parse_musicxml import parse_score
from src.inference.audio import render_audio
from src.inference.export import export_score
from src.inference.generate import _fingerprint, generate_candidates
from src.inference.rerank import rank_candidates


SUMMARY_METRICS = ("melody_coverage_exact", "melody_coverage_octave_tolerant", "octave_doubling_rate",
                   "playability_score", "harmonic_pitch_class_overlap", "note_density_ratio",
                   "register_expansion_semitones", "reference_exact_note_onset_f1")


def _config(value: dict | str | Path) -> dict:
    payload = value if isinstance(value, dict) else yaml.safe_load(Path(value).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("methods"), list) or not payload["methods"]:
        raise ValueError("Experiment config requires a non-empty methods list")
    names = set()
    allowed_generation = {"temperature", "top_p", "beam_width", "max_new_tokens", "device", "soft_strength", "num_threads"}
    global_generation = payload.get("generation", {})
    if not isinstance(global_generation, dict) or set(global_generation) - allowed_generation:
        raise ValueError("Unknown or invalid global generation settings")
    for method in payload["methods"]:
        if not isinstance(method, dict) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", str(method.get("name", ""))):
            raise ValueError("Experiment methods require safe, unique names")
        if method["name"] in names:
            raise ValueError("Experiment method names must be unique")
        names.add(method["name"])
        if method.get("generator") not in {"literal", "rule_based", "neural", "human_reference", "reference_target"}:
            raise ValueError("Unknown experiment generator")
        if method.get("melody_mode", "hard") not in {"none", "soft", "hard"}:
            raise ValueError("Unknown experiment melody mode")
        generation = method.get("generation", {})
        if not isinstance(generation, dict) or set(generation) - allowed_generation:
            raise ValueError("Unknown or invalid method generation settings")
        temperature = method.get("temperature", generation.get("temperature", global_generation.get("temperature", 0.0)))
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature < 0:
            raise ValueError("Experiment temperature must be finite and nonnegative")
        count = method.get("num_candidates", 1)
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 32:
            raise ValueError("Experiment candidate count must be 1..32")
    return payload


def _summary(rows: list[dict], seed: int = 0) -> list[dict]:
    """Macro means by work, with bootstrap intervals over independent work IDs."""
    grouped = defaultdict(list)
    for row in rows:
        if row["status"] == "ok":
            grouped[row["method"]].append(row)
    summaries = []
    rng = random.Random(seed)
    for method, entries in sorted(grouped.items()):
        summary = {"method": method, "pairs": len(entries), "works": len({r["work_id"] for r in entries}), "metrics": {}}
        for key in SUMMARY_METRICS:
            by_work = defaultdict(list)
            for row in entries:
                value = row["metrics"].get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    by_work[row["work_id"]].append(value)
            values = [statistics.fmean(v) for _, v in sorted(by_work.items())]
            if not values:
                continue
            if len(values) >= 2:
                bootstrap = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(1000))
                interval = [bootstrap[24], bootstrap[974]]
            else:
                interval = None
            summary["metrics"][key] = {"work_macro_mean": statistics.fmean(values),
                                        "work_count": len(values), "bootstrap_95_interval": interval}
        summaries.append(summary)
    return summaries


def run_experiments(manifest: str | Path, output_dir: str | Path,
                    config: dict | str | Path = "configs/evaluation/full.yaml", *,
                    checkpoints: dict[str, str | Path] | None = None, split: str = "test",
                    limit: int | None = None, render_listening: bool = True) -> dict:
    config = _config(config)
    checkpoints = checkpoints or {}
    if split not in {"train", "val", "test", "all"}:
        raise ValueError("split must be train, val, test, or all")
    if limit is not None and (isinstance(limit, bool) or limit < 1):
        raise ValueError("limit must be positive")
    pairs = [p for p in load_manifest(manifest) if split == "all" or p["split"] == split]
    if not pairs:
        raise ValueError(f"No pairs in split {split!r}; select an available split explicitly")
    if limit:
        pairs = pairs[:limit]
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Freeze the actual checkpoint bytes once. A training process may replace
    # last.pt while an experiment runs; all ablations must use the same weights.
    snapshots = {}
    for method in config["methods"]:
        if method["generator"] != "neural":
            continue
        path = checkpoints.get(method.get("checkpoint_key", ""), method.get("checkpoint"))
        if not path or not Path(path).is_file():
            continue
        original = Path(path).resolve()
        if original in snapshots:
            continue
        snapshot_dir = root / ".checkpoint_snapshots"
        snapshot_dir.mkdir(exist_ok=True)
        temporary = snapshot_dir / f"{uuid.uuid4().hex}.tmp"
        shutil.copyfile(original, temporary)
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        snapshot = snapshot_dir / f"{digest}.pt"
        temporary.replace(snapshot)
        snapshots[original] = snapshot, digest
    rows = []
    for pair in pairs:
        source = normalize_source(parse_score(pair["source"]["path"]),
                                  solo_part=pair["source"].get("solo_part"),
                                  accompaniment_parts=pair["source"].get("accompaniment_parts"))
        reference = parse_score(pair["target"]["path"])
        for method in config["methods"]:
            generator = "reference_target" if method["generator"] in {"reference_target", "human_reference"} else method["generator"]
            row = {"pair_id": pair["pair_id"], "work_id": pair["work_id"], "split": pair["split"],
                   "method": method["name"], "generator": generator, "status": "ok", "config": method,
                   "data_kind": pair.get("data_kind", "curated_pair")}
            if generator == "reference_target":
                kind = str(pair.get("data_kind", "")).lower()
                synthetic = "synthetic" in kind or "synthetic" in str(pair.get("license", {}).get("origin", "")).lower()
                row["reference_kind"] = "synthetic_reference" if synthetic else "curated_reference"
                row["display_method"] = row["reference_kind"]
            else:
                row["display_method"] = method["name"]
            if generator == "neural":
                checkpoint = checkpoints.get(method.get("checkpoint_key", ""), method.get("checkpoint"))
                if not checkpoint or Path(checkpoint).resolve() not in snapshots:
                    row.update(status="omitted", reason="No existing trained checkpoint supplied for this method")
                    rows.append(row)
                    continue
            controls = load_preset(method.get("preset", "faithful"))
            values = method.get("controls", {})
            directory = root / pair["pair_id"] / method["name"]
            directory.mkdir(parents=True, exist_ok=True)
            try:
                count = method.get("num_candidates", 1)
                seed = int(config.get("seed", 0))
                if generator == "reference_target":
                    candidates = [reference]
                elif generator == "neural":
                    from src.inference.neural_generate import generate_neural_results
                    requested_controls = dict(controls.to_dict(), **values)
                    generation = dict(config.get("generation", {}), **method.get("generation", {}))
                    for name in ("temperature", "top_p", "beam_width", "max_new_tokens", "device"):
                        if name in method:
                            generation[name] = method[name]
                    generation.setdefault("temperature", 0.0)
                    snapshot, digest = snapshots[Path(checkpoint).resolve()]
                    results, metadata = generate_neural_results(source, snapshot,
                                                                requested_controls,
                                                                num_candidates=count, seed=seed,
                                                                melody_mode=method.get("melody_mode", "hard"), **generation)
                    candidates = [result.score for result in results]
                    row["generation_reports"] = [result.report for result in results]
                    row["checkpoint_sha256"] = digest
                    row["checkpoint_metadata"] = {key: metadata[key] for key in
                                                  ("epoch", "global_step", "model_config", "dataset_path", "dataset_sha256",
                                                   "train_work_ids", "validation_work_ids", "test_work_ids",
                                                   "data_kinds", "purpose", "train_segment_count", "parameter_count")
                                                  if key in metadata}
                    row["training_work_overlap"] = pair["work_id"] in metadata.get("train_work_ids", [])
                    row["training_overlap_definition"] = "Exact work_id membership; melodic duplicates and derived works require a separate corpus audit."
                    if metadata.get("purpose") == "overfit sanity check":
                        row["evaluation_role"] = "overfit_checkpoint_diagnostic"
                    elif row["training_work_overlap"]:
                        row["evaluation_role"] = "in_sample_sanity"
                    else:
                        row["evaluation_role"] = "held_out_work_id" if pair["split"] == "test" else "diagnostic_non_test"
                    model_config = metadata.get("model_config", {})
                    row["controls_conditioned"] = bool(model_config.get("conditioned", False))
                    row["architecture"] = model_config.get("architecture", "unknown")
                    if not row["controls_conditioned"]:
                        row["conditioning_note"] = "This checkpoint has no technique conditioning; control values cannot establish learned technique effects."
                else:
                    controls = replace(controls, **values)
                    candidates = generate_candidates(source, controls, num_candidates=count,
                                                     seed=seed, baseline=generator)
                # Soft/flexible controls are neural settings; the heuristic
                # scorer still uses strict-compatible controls for its proxies.
                scoring_controls = dict(controls.to_dict(), **values) if generator == "neural" else controls
                ranking = rank_candidates(source, candidates, scoring_controls)
                selected = ranking[0]["candidate_index"] if method.get("rerank", True) else 0
                score = candidates[selected]
                metrics = analyze_transcription(source, score)
                metrics.update(reference_metrics(score, reference))
                stem = directory / "piano"
                export_score(score, stem.with_suffix(".musicxml"))
                export_score(score, stem.with_suffix(".mid"))
                score.save(stem.with_suffix(".json"))
                row.update(metrics=metrics, selected_candidate=selected,
                           unique_candidates=len({_fingerprint(c) for c in candidates}),
                           fingerprint=_fingerprint(score), score=str(stem.with_suffix(".musicxml")),
                           checkpoint=str(Path(checkpoint).resolve()) if generator == "neural" else None)
                if render_listening:
                    row["audio"] = render_audio(score, stem.with_suffix(".wav"))
                (directory / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
                (directory / "ranking.json").write_text(json.dumps(ranking, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            except (ValueError, OSError, RuntimeError) as exc:
                row.update(status="failed", reason=str(exc))
            rows.append(row)
    report = {"schema_version": 1, "manifest": str(Path(manifest).resolve()), "evaluation_split": split,
              "config": config, "rows": rows, "summary": _summary(rows, int(config.get("seed", 0))),
              "limitations": [
                  "Symbolic heuristics report preservation, texture, harmony, and playability proxies; they do not establish artistic quality.",
                  "Neural methods require actual trained checkpoints; omitted or failed methods are not assigned metric values.",
                  "Bootstrap intervals resample work IDs; fewer than two works yields no interval, and a small corpus cannot support broad conclusions.",
                  "Reference rows describe supplied targets. Programmatically authored targets are labeled synthetic_reference; curated_reference is not a listening judgment.",
                  "Check checkpoint training manifests for work overlap before claiming held-out generalization; train/all splits are diagnostic only.",
                  "Architecture comparisons require matched training data, objectives, and conditioning. The supplied dual_encoder checkpoint is unconditioned; inspect actual checkpoint metadata rather than comparing it as a conditioned dual model.",
                  "Synthetic audio follows score tempo and velocity without pianist expression; listener judgments must be collected separately."]}
    (root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    columns = ["pair_id", "work_id", "split", "method", "generator", "status", "reason", *SUMMARY_METRICS]
    with (root / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(dict(row, **row.get("metrics", {})) for row in rows)
    write_listening_report(report, root / "index.html")
    return report


def write_listening_report(report: dict, path: str | Path) -> Path:
    path = Path(path).resolve()
    rows = []
    for row in report["rows"]:
        metrics = row.get("metrics", {})
        columns = "".join(f"<td>{metrics.get(key, '—'):.3f}</td>" if isinstance(metrics.get(key), (int, float))
                          else "<td>—</td>" for key in SUMMARY_METRICS)
        audio = row.get("audio", {}).get("path")
        listen = f'<audio controls preload="none" src="{escape(Path(audio).relative_to(path.parent).as_posix(), quote=True)}"></audio>' if audio else escape(row.get("reason", "No audio requested"))
        score = row.get("score")
        download = f'<a href="{escape(Path(score).relative_to(path.parent).as_posix(), quote=True)}">MusicXML</a>' if score else ""
        rows.append(f'<tr><td>{escape(row["pair_id"])}</td><td>{escape(row.get("display_method", row["method"]))}</td><td>{escape(row["status"])}</td>{columns}<td>{listen}{download}</td></tr>')
    headings = "".join(f"<th>{escape(key)}</th>" for key in SUMMARY_METRICS)
    limitations = "".join(f"<li>{escape(item)}</li>" for item in report["limitations"])
    document = f'<!doctype html><html><meta charset="utf-8"><title>VirtuosoTranscribe experiments</title><style>body{{font:14px system-ui;margin:28px;color:#172536}}table{{border-collapse:collapse}}th,td{{padding:9px;border:1px solid #d5dfeb}}th{{background:#edf3fa;max-width:180px}}audio{{width:230px;display:block}}.table{{overflow:auto}}li{{margin:7px 0}}</style><h1>Measured transcription comparisons</h1><p>Split: {escape(report["evaluation_split"])}. Values are generated-score measurements. Neural omissions are explicit; no listening ratings have been inferred.</p><div class="table"><table><thead><tr><th>Pair</th><th>Method</th><th>Status</th>{headings}<th>Listening / score</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div><h2>Interpretation limits</h2><ul>{limitations}</ul><p><a href="report.json">Complete report and work-level confidence intervals</a> · <a href="results.csv">CSV</a></p></html>'
    path.write_text(document, encoding="utf-8")
    return path


def checkpoint_arguments(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in result:
            raise ValueError("Use unique --checkpoint NAME=PATH arguments")
        result[name] = Path(path)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation/full.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments"))
    parser.add_argument("--checkpoint", action="append", default=[], help="NAME=PATH; keys are vanilla, conditioned, dual")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-audio", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run_experiments(args.manifest, args.output_dir, args.config,
                                 checkpoints=checkpoint_arguments(args.checkpoint), split=args.split,
                                 limit=args.limit, render_listening=not args.no_audio)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    statuses = defaultdict(int)
    for row in report["rows"]:
        statuses[row["status"]] += 1
    print(f"Measured {statuses['ok']} method/pair rows; omitted {statuses['omitted']}; failed {statuses['failed']}")
    print(f"Report: {(args.output_dir / 'index.html').resolve()}")
    return 1 if statuses["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
