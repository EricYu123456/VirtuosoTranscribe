"""Write source-target metrics and a manually inspectable alignment plot."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.melody_alignment import align_solo_to_piano
from src.analysis.technique_features import analyze_transcription, write_analysis_report
from src.analysis.visualization import write_alignment_visualization
from src.data.normalize_score import normalize_source, normalize_target
from src.data.parse_musicxml import parse_score


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--solo-part")
    parser.add_argument("--accompaniment-part", action="append")
    args = parser.parse_args(argv)
    try:
        source = normalize_source(parse_score(args.source), args.solo_part, args.accompaniment_part)
        target = normalize_target(parse_score(args.target))
        alignment = align_solo_to_piano(source.solo, target, source_score=source.score)
        stats = analyze_transcription(source, target, alignment)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        source.score.save(args.output_dir / "source.json")
        target.save(args.output_dir / "target.json")
        alignment.save(args.output_dir / "alignment.json")
        write_analysis_report(stats, args.output_dir / "metrics.json")
        write_alignment_visualization(source.solo, target, alignment, args.output_dir / "alignment.html")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Melody octave-tolerant coverage: {stats['melody_coverage_octave_tolerant']:.1%}")
    print(f"Alignment and analysis saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
