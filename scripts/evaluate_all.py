"""Evaluate exported candidates against the source and/or a reference score."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from src.analysis.technique_features import analyze_transcription
from src.data.normalize_score import normalize_source
from src.data.parse_musicxml import parse_score
from src.evaluation.harmonic_metrics import harmonic_overlap
from src.evaluation.texture_metrics import texture_metrics
from src.representation.events import ScoreData


def reference_metrics(prediction: ScoreData, reference: ScoreData) -> dict:
    """Compare complete target textures; these are not solo-melody metrics."""
    def attacks(score, octave_tolerant=False):
        return Counter((round(n.onset, 6), n.pitch % 12 if octave_tolerant else n.pitch)
                       for n in score.notes if not n.is_grace)
    result = {}
    for name, mode in (("exact", False), ("octave_tolerant", True)):
        pred, ref = attacks(prediction, mode), attacks(reference, mode)
        matches = sum((pred & ref).values())
        total = sum(pred.values()) + sum(ref.values())
        result[f"reference_{name}_note_onset_f1"] = 2 * matches / total if total else 1.0
    result.update({f"reference_{key}": value for key, value in harmonic_overlap(reference, prediction).items()})
    result["reference_measure_count_ratio"] = len(prediction.measures) / len(reference.measures)
    result["reference_metric_definition"] = "All notated attacks, one-to-one multiplicity, absolute quarter-note time."
    return result


def evaluate_all(pred: str | Path, *, source: str | Path | None = None,
                 reference: str | Path | None = None, solo_part: str | None = None,
                 accompaniment_parts: list[str] | None = None) -> list[dict]:
    if source is None and reference is None:
        raise ValueError("Provide --source for melodic preservation or --reference for target comparison")
    pred = Path(pred)
    if pred.is_dir():
        paths = sorted(pred.glob("candidate_*.musicxml"))
        if not paths:
            paths = sorted(p for p in pred.iterdir() if p.suffix.lower() in {".xml", ".musicxml", ".mxl"})
    else:
        paths = [pred]
    if not paths:
        raise ValueError(f"No MusicXML candidates found in {pred}")
    source_score = normalize_source(parse_score(source), solo_part, accompaniment_parts) if source else None
    reference_score = parse_score(reference) if reference else None
    rows = []
    for path in paths:
        prediction = parse_score(path)
        metrics = analyze_transcription(source_score, prediction) if source_score else texture_metrics(prediction)
        if reference_score:
            metrics.update(reference_metrics(prediction, reference_score))
        rows.append({"prediction": str(path.resolve()), **metrics})
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--solo-part")
    parser.add_argument("--accompaniment-part", action="append")
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation/results.json"))
    args = parser.parse_args(argv)
    try:
        rows = evaluate_all(args.pred, source=args.source, reference=args.reference,
                            solo_part=args.solo_part, accompaniment_parts=args.accompaniment_part)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    flat_keys = list(dict.fromkeys(key for row in rows for key, value in row.items()
                                   if not isinstance(value, (dict, list, tuple))))
    with args.output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=flat_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Evaluated {len(rows)} scores; JSON and CSV saved to {args.output.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
