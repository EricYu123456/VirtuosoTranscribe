"""Inspect upstream PDMX license metadata without treating candidates as checked."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import time
from urllib.request import urlopen


CSV_URL = "https://zenodo.org/api/records/15571083/files/PDMX.csv/content"


def inspect_pdmx(output: Path, csv_path: Path | None = None) -> list[dict]:
    started = time.monotonic()
    candidates = []
    handle = csv_path.open(encoding="utf-8", newline="") if csv_path else io.TextIOWrapper(urlopen(CSV_URL, timeout=120), encoding="utf-8", newline="")
    with handle:
        for count, row in enumerate(csv.DictReader(handle), 1):
            text = " ".join(row.get(k, "") for k in ("title", "song_name", "composer_name", "artist_name")).casefold()
            if (any(x in text for x in ("schubert", "liszt", "standchen", "ständchen", "ave maria", "erlkönig"))
                    or any(x in row.get("metadata", "") for x in ("3252906", "4584986"))):
                row["eligible_license_metadata"] = (
                    row.get("license", "").lower() in {"publicdomain", "cc0", "cc-zero"}
                    and row.get("license_conflict") == "False"
                    and row.get("subset:all_valid") == "True"
                    and row.get("has_paywall") == "False"
                    and row.get("is_official") == "False")
                candidates.append(row)
            if count % 50000 == 0:
                print(f"Inspected {count} rows; {len(candidates)} candidates; {time.monotonic()-started:.1f}s", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"metadata_source": CSV_URL if csv_path is None else str(csv_path),
                                  "record": "https://zenodo.org/records/15571083",
                                  "review_date": "2026-10-05", "rows_inspected": count,
                                  "candidates": candidates}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(candidates)} candidate metadata rows to {output}", flush=True)
    return candidates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/metadata/pdmx_candidates.json"))
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args(argv)
    inspect_pdmx(args.output, args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
