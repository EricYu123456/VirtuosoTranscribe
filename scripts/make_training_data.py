"""Original, work-disjoint symbolic fixtures for neural pipeline verification.

Targets are auxiliary synthetic arrangements. This corpus checks learnability
and control conditioning; it does not stand in for artistic transcription data.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from scripts.prepare_dataset import prepare_dataset
from src.inference.export import export_score
from src.representation.events import (
    MeasureInfo, NoteEvent, PartInfo, ScoreData, SignatureEvent, TempoEvent,
)


def make_training_data(output: Path, *, works: int = 12) -> dict:
    if works < 12:
        raise ValueError("Use at least12 independent fixture motifs for work-disjoint splits")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    scale = (0, 2, 4, 5, 7, 9, 11)
    for work in range(works):
        solo = []
        accompaniment = []
        measures = tuple(MeasureInfo(i + 1, i * 4, 4, original_number=str(i + 1)) for i in range(4))
        for bar in range(4):
            # Each work has a different melodic interval sequence, rather than
            # representing a transposed copy assigned to a different split.
            degrees = [((work + 1) * (beat + 1) + bar * (work % 3 + 1)
                        + beat * beat + work // 7) % 7 for beat in range(4)]
            for beat, degree in enumerate(degrees):
                solo.append(NoteEvent(72 + scale[degree], bar * 4 + beat, 1, "solo",
                                      measure=bar + 1, beat=beat + 1, role="SOLO"))
            root = (48, 45, 43, 48)[bar]
            for pitch in (root, root + 7):
                accompaniment.append(NoteEvent(pitch, bar * 4, 4, "acc", measure=bar + 1,
                                                role="ACC"))
        source = ScoreData(tuple(solo + accompaniment),
                           (PartInfo("solo", "Violin", "Violin", 40),
                            PartInfo("acc", "Piano", "Piano", 0)), measures,
                           time_signatures=(SignatureEvent(0, "4/4"),),
                           key_signatures=(SignatureEvent(0, "0:"),),
                           tempos=(TempoEvent(0, 90),), title=f"Original neural fixture {work + 1}")
        source_path = output / f"work_{work + 1:03}_source.musicxml"
        export_score(source, source_path)
        split = "train" if work < works - 2 else "val" if work == works - 2 else "test"
        for doubling in (0, 1):
            melody = [replace(n, part_id="piano_rh", role="PIANO") for n in solo]
            if doubling:
                melody += [replace(n, pitch=n.pitch - 12, part_id="piano_rh", role="PIANO") for n in solo]
            bass = [replace(n, part_id="piano_lh", role="PIANO") for n in accompaniment]
            target = source.with_notes(melody + bass,
                                       parts=(PartInfo("piano_rh", "Piano RH", "Piano", 0),
                                              PartInfo("piano_lh", "Piano LH", "Piano", 0)),
                                       title=f"Synthetic target, octave doubling {doubling}")
            target_path = output / f"work_{work + 1:03}_target_{doubling}.musicxml"
            export_score(target, target_path)
            controls = {"melody_preservation": 1.0, "octave_doubling": float(doubling),
                        "texture_density": .5, "virtuosity": .3}
            records.append({"pair_id": f"fixture_{work + 1:03}_d{doubling}",
                            "work_id": f"original_fixture_{work + 1:03}", "split": split,
                            "data_kind": "synthetic_auxiliary", "controls": controls,
                            "source": {"path": source_path.name, "solo_part": "Violin",
                                       "accompaniment_parts": ["Piano"]},
                            "target": {"path": target_path.name, "instrument": "piano"},
                            "license": {"source_status": "CHECKED", "target_status": "CHECKED",
                                        "redistribution": "CC0-1.0",
                                        "origin": "Original algorithmic study, authored for pipeline tests"}})
    manifests = {}
    for name, subset in (("conditioned", records), ("vanilla", records[::2])):
        path = output / f"{name}_pairs.json"
        path.write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8")
        processed = output / f"processed_{name}"
        dataset = prepare_dataset(path, processed)
        by_pair = {row["pair_id"]: row["controls"] for row in subset}
        for entry in dataset["segments"]:
            entry["controls"] = by_pair[entry["pair_id"]]
            entry["data_kind"] = "synthetic_auxiliary"
        (processed / "dataset.json").write_text(json.dumps(dataset, indent=2), encoding="utf-8")
        manifests[name] = str(processed / "dataset.json")
    return manifests


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/interim/training_fixture"))
    parser.add_argument("--works", type=int, default=12)
    args = parser.parse_args(argv)
    try:
        paths = make_training_data(args.output, works=args.works)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(paths, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
