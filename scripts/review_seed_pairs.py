"""Write inspectable diagnostics for acquired Schubert source/transcription candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.analysis.melody_alignment import align_solo_to_piano
from src.analysis.visualization import write_alignment_visualization
from src.data.normalize_score import normalize_source, normalize_target
from src.data.parse_musicxml import parse_score


SEED_LINKS = (("lc5004835", ("6212007", "4945970", "4685175", "1976341")),
              ("lc5133353", ("1976341",)),
              ("lc6389103", ("191296",)),
              ("lc6900961", ("4954098",)),
              ("lc5016466", ("4676133",)))


def review_seed_pairs(source_root: Path, target_root: Path, output: Path) -> list[dict]:
    report = []
    for source_id, target_ids in SEED_LINKS:
        source = normalize_source(parse_score(source_root / f"{source_id}.mxl"))
        print(f"SOURCE {source_id}: {source.score.title}; {len(source.solo)} solo notes; {len(source.score.measures)} bars", flush=True)
        for target_id in target_ids:
            try:
                target = normalize_target(parse_score(target_root / f"pdmx_{target_id}.mxl"))
            except (ValueError, OSError) as exc:
                report.append({"source_id": source_id, "target_id": target_id,
                               "review_status": "REJECTED", "reason": str(exc)})
                print(f"TARGET {target_id}: rejected: {exc}", flush=True)
                continue
            alignment = align_solo_to_piano(source.solo, target, source_score=source.score)
            coverage = len(alignment.matches) / len(source.solo) if source.solo else 0
            print(f"TARGET {target_id}: {len(target.notes)} notes; {len(target.measures)} bars; quality={alignment.quality:.3f}; coverage={coverage:.3f}", flush=True)
            directory = output / f"{source_id}_{target_id}"
            directory.mkdir(parents=True, exist_ok=True)
            alignment.save(directory / "alignment.json")
            source.score.save(directory / "source.json")
            target.save(directory / "target.json")
            write_alignment_visualization(source.solo, target, alignment, directory / "alignment.html",
                                          title=f"{source_id} to {target_id}")
            report.append({"source_id": source_id, "target_id": target_id,
                           "source_title": source.score.title, "target_title": target.title,
                           "source_solo_notes": len(source.solo), "target_notes": len(target.notes),
                           "source_measures": len(source.score.measures), "target_measures": len(target.measures),
                           "quality": alignment.quality, "coverage": coverage,
                           "keys_source": [x.value for x in source.score.key_signatures],
                           "keys_target": [x.value for x in target.key_signatures],
                           "review_status": "NEEDS_PAIR_REVIEW"})
    (output / "review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("data/interim/upstream_candidates"))
    parser.add_argument("--target-root", type=Path, default=Path("data/raw/pdmx_seed"))
    parser.add_argument("--output", type=Path, default=Path("outputs/corpus_review"))
    args = parser.parse_args(argv)
    review_seed_pairs(args.source_root, args.target_root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
