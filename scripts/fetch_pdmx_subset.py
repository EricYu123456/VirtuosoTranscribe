"""Fetch selected, license-eligible PDMX files with verified resumable archives.

PDMX distributes bulk archives. Range requests share one local archive across
workers; only explicitly selected regular members are extracted. Downloading
does not establish a source/target pair or mark a license CHECKED.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import tarfile
import time
from urllib.request import Request, urlopen

from src.data.mine_pairs import load_catalog, metadata_license_eligible


ARCHIVES = {
    "mxl": "49ffd75ecf5489c0be6d41182eb11ff7",
    "metadata": "5bc79445090dd2fe5e96cffa77a3461c",
}
BASE = "https://zenodo.org/api/records/15571083/files"


def fetch_archive(kind: str, directory: Path, workers: int = 8) -> Path:
    url = f"{BASE}/{kind}.tar.gz/content"
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / f"{kind}.tar.gz"
    state_path = archive.with_suffix(".parts.json")
    with urlopen(Request(url, headers={"Range": "bytes=0-0"}), timeout=90) as response:
        content_range = response.headers.get("Content-Range", "")
        if response.status != 206 or not re.fullmatch(r"bytes 0-0/\d+", content_range):
            raise ValueError("Upstream does not support verified HTTP range downloads")
        size = int(content_range.rsplit("/", 1)[1])
    chunk_size = 32 * 1024 * 1024
    chunk_count = (size + chunk_size - 1) // chunk_size
    state = {"url": url, "size": size, "chunk_size": chunk_size, "completed": []}
    if archive.exists() and state_path.exists():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if all(previous.get(k) == state[k] for k in ("url", "size", "chunk_size")) and archive.stat().st_size == size:
            state = previous
    if not archive.exists() or archive.stat().st_size != size:
        with archive.open("wb") as handle:
            handle.truncate(size)
        state["completed"] = []
    started = time.monotonic()

    def fetch_chunk(index: int) -> int:
        begin = index * chunk_size
        end = min(size, begin + chunk_size) - 1
        expected = f"bytes {begin}-{end}/{size}"
        for attempt in range(3):
            try:
                request = Request(url, headers={"Range": f"bytes={begin}-{end}"})
                with urlopen(request, timeout=120) as response, archive.open("r+b") as output:
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError(f"Invalid HTTP range response for chunk {index}")
                    output.seek(begin)
                    remaining = end - begin + 1
                    while remaining:
                        block = response.read(min(1024 * 1024, remaining))
                        if not block:
                            raise OSError(f"Incomplete download of chunk {index}")
                        output.write(block)
                        remaining -= len(block)
                return index
            except (OSError, ValueError):
                if attempt == 2:
                    raise
        raise AssertionError("Unreachable")

    # Each worker owns nonoverlapping byte ranges. Only the parent updates state.
    pending = [i for i in range(chunk_count) if i not in state["completed"]]
    from concurrent.futures import as_completed
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for future in as_completed([executor.submit(fetch_chunk, i) for i in pending]):
            state["completed"].append(future.result())
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            print(f"{kind}: {len(state['completed'])}/{chunk_count} chunks; {time.monotonic()-started:.1f}s", flush=True)
    with archive.open("rb") as archive_handle:
        digest = hashlib.file_digest(archive_handle, "md5").hexdigest()
    if digest != ARCHIVES[kind]:
        raise ValueError(f"Archive checksum mismatch: {archive}")
    print(f"{kind}: upstream MD5 verified", flush=True)
    return archive


def fetch_subset(catalog: Path, score_ids: list[str], output: Path, workers: int = 8,
                 include_metadata: bool = False) -> dict:
    rows = [row for row in load_catalog(catalog)
            if Path(row.get("metadata", "")).stem in score_ids]
    found = {Path(row["metadata"]).stem for row in rows}
    if found != set(score_ids):
        raise ValueError(f"Score IDs absent from catalog: {sorted(set(score_ids)-found)}")
    for row in rows:
        if not metadata_license_eligible(row):
            raise ValueError(f"Score {row['metadata']} is not license eligible")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    downloads = output / "archives"
    extracted = []
    for kind in ("mxl", "metadata") if include_metadata else ("mxl",):
        archive = fetch_archive(kind, downloads, workers)
        names = {row[kind].removeprefix("./"): row for row in rows}
        with tarfile.open(archive, "r:gz") as handle:
            for member in handle:
                name = member.name.removeprefix("./")
                if name not in names:
                    continue
                if not member.isfile() or member.size > 20 * 1024 * 1024:
                    raise ValueError(f"Invalid selected archive member: {name}")
                score_id = Path(names[name]["metadata"]).stem
                destination = output / f"pdmx_{score_id}{Path(name).suffix}"
                with handle.extractfile(member) as source:
                    destination.write_bytes(source.read())
                extracted.append({"score_id": score_id, "kind": kind, "path": str(destination),
                                  "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                                  "archive_member": name})
                print(f"Extracted {kind} score {score_id}", flush=True)
                del names[name]
                if not names:
                    break
        if names:
            raise ValueError(f"Members absent from {archive}: {list(names)}")
    report = {"record": "https://zenodo.org/records/15571083", "review_status": "NEEDS_PAIR_REVIEW",
              "catalog": str(catalog.resolve()), "files": extracted, "license_metadata": rows}
    (output / "acquisition.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--score-ids", nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("data/raw/pdmx_seed"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--include-metadata", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 16:
        parser.error("workers must be between 1 and 16")
    fetch_subset(args.catalog, args.score_ids, args.output, args.workers, args.include_metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
