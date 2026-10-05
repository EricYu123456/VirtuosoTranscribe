"""Mine reviewed-license PDMX metadata into unapproved pair proposals."""

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

from src.data.mine_pairs import load_catalog, mine_pairs, save_candidates


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--review-scores", action="store_true",
                        help="Parse existing local MusicXML/MXL files and filter by real alignment; never download")
    parser.add_argument("--score-root", type=Path, help="Base directory for relative catalog score paths")
    parser.add_argument("--min-melody-coverage", type=float, default=.75)
    parser.add_argument("--min-alignment-quality", type=float, default=.6)
    parser.add_argument("--min-structural-similarity", type=float, default=.65)
    parser.add_argument("--onset-tolerance", type=float, default=.125)
    parser.add_argument("--max-score-events", type=int, default=20000)
    parser.add_argument("--max-alignment-cells", type=int, default=2000000)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--include-rejected", action="store_true", help="Keep failed or unavailable score-review proposals for diagnosis")
    parser.add_argument("--review-report", type=Path, help="Detailed thresholds, counts and rejection diagnostics")
    args = parser.parse_args(argv)
    try:
        reviewed = mine_pairs(load_catalog(args.catalog), review_scores=args.review_scores,
                              score_root=args.score_root, min_melody_coverage=args.min_melody_coverage,
                              min_alignment_quality=args.min_alignment_quality,
                              min_structural_similarity=args.min_structural_similarity,
                              onset_tolerance=args.onset_tolerance, include_rejected=True,
                              max_candidates=args.max_candidates, max_score_events=args.max_score_events,
                              max_alignment_cells=args.max_alignment_cells)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    pairs = [pair for pair in reviewed if not args.review_scores or args.include_rejected
             or pair.score_review["status"] == "PASSED_THRESHOLDS"]
    save_candidates(pairs, args.output)
    if args.review_report is not None:
        payload = {"mode": "local_score_alignment" if args.review_scores else "metadata_only",
                   "catalog": str(args.catalog.resolve()), "output": str(args.output.resolve()),
                   "downloads_performed": False, "approved_pairs": 0,
                   "review_status": "NEEDS_REVIEW", "proposals_examined": len(reviewed),
                   "output_candidates": len(pairs),
                   "score_review_counts": dict(Counter(pair.score_review.get("status", "NOT_REQUESTED") for pair in reviewed)),
                   "thresholds": {"min_melody_coverage": args.min_melody_coverage,
                                  "min_alignment_quality": args.min_alignment_quality,
                                  "min_structural_similarity": args.min_structural_similarity,
                                  "onset_tolerance_quarter_notes": args.onset_tolerance,
                                  "max_score_events": args.max_score_events,
                                  "max_alignment_cells": args.max_alignment_cells},
                   "candidates": [asdict(pair) for pair in reviewed]}
        args.review_report.parent.mkdir(parents=True, exist_ok=True)
        args.review_report.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"Proposed {len(pairs)} source/piano pairs; each remains NEEDS_REVIEW")
    if args.review_scores:
        print("Local alignment review: " + str(dict(Counter(pair.score_review["status"] for pair in reviewed))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
