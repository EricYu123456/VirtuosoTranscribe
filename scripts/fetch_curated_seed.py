"""Reproduce the already reviewed seed manifest, checking exact score hashes."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
from urllib.request import urlopen

from src.data.build_manifest import load_manifest
from scripts.fetch_pdmx_subset import fetch_subset


def fetch_curated_seed(manifest: Path, catalog: Path,
                       cache: Path = Path("data/raw/pdmx_seed"), workers: int = 8) -> int:
    pairs = load_manifest(manifest, require_files=False)
    for pair in pairs:
        for side in ("source", "target"):
            record = pair[side]
            destination = Path(record["path"])
            expected = record.get("sha256")
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValueError(f"{pair['pair_id']}: {side} requires reviewed SHA256")
            if destination.exists():
                if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
                    raise ValueError(f"Existing score differs from reviewed hash: {destination}")
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".download")
            if side == "target":
                score_id = str(record["pdmx_score_id"])
                fetch_subset(catalog, [score_id], cache, workers, include_metadata=True)
                shutil.copyfile(cache / f"pdmx_{score_id}.mxl", temporary)
            else:
                url = record["url"]
                if not url.startswith("https://"):
                    raise ValueError("Source URL must use HTTPS")
                with urlopen(url, timeout=90) as response:
                    data = response.read(20 * 1024 * 1024 + 1)
                if len(data) > 20 * 1024 * 1024:
                    raise ValueError("Source score exceeds the acquisition size bound")
                temporary.write_bytes(data)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Downloaded score differs from reviewed hash: {temporary}")
            temporary.replace(destination)
    print(f"Verified {len(pairs)} reviewed real source/transcription pairs", flush=True)
    return len(pairs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/curated_pairs.json"))
    parser.add_argument("--catalog", type=Path, default=Path("data/metadata/pdmx_candidates.json"))
    parser.add_argument("--cache", type=Path, default=Path("data/raw/pdmx_seed"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 16:
        parser.error("workers must be between 1 and 16")
    fetch_curated_seed(args.manifest, args.catalog, args.cache, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
