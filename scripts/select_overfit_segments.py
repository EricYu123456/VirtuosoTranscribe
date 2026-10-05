"""Select exact-alignment, non-grace clips for an explicitly in-sample model check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def select_overfit_segments(dataset: Path, output: Path, count: int = 10,
                            max_tokens: int = 700) -> dict:
    if count < 1 or max_tokens < 8:
        raise ValueError("count and max_tokens must be positive usable bounds")
    payload = json.loads(dataset.read_text(encoding="utf-8"))
    selected = []
    for segment in payload["segments"]:
        fields = [segment[name] for name in
                  ("source_solo_tokens", "source_acc_tokens", "target_piano_tokens")]
        coverage = segment.get("technique_features", {}).get("melody_coverage_octave_tolerant", 0)
        if (segment["split"] == "train" and coverage >= 1 - 1e-8
                and sum(t.startswith("<PITCH_") for t in fields[0]) >= 2
                and all("<DUR_0>" not in tokens for tokens in fields)
                and max(len(fields[2]), len(fields[0]) + len(fields[1]) + 4) < max_tokens):
            selected.append(segment)
    if len(selected) < count:
        raise ValueError(f"Only {len(selected)} eligible clips; requested {count}")
    payload.update(segments=selected[:count], segment_count=count,
                   parent_dataset=str(dataset.resolve()),
                   purpose="real historical transcription overfit sanity check; train only, no held-out quality inference")
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Selected {count}/{len(selected)} exact-alignment train clips", flush=True)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/curated/dataset.json"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/real_overfit"))
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=700)
    args = parser.parse_args(argv)
    select_overfit_segments(args.dataset, args.output, args.count, args.max_tokens)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
