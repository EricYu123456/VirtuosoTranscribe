"""Rebuild original synthetic fixtures; these are not artistic training pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from music21 import chord, dynamics, instrument, key, metadata, meter, note, stream, tempo


def make_pair(solo_instrument, transpose: int = 0):
    source = stream.Score()
    source.metadata = metadata.Metadata(title="VirtuosoTranscribe synthetic study", composer="Synthetic fixture")
    solo = stream.Part(id="solo")
    solo.partName = solo_instrument.instrumentName
    solo.insert(0, solo_instrument)
    accompaniment = stream.Part(id="acc")
    accompaniment.partName = "Piano"
    accompaniment.insert(0, instrument.Piano())
    target = stream.Score()
    target.metadata = metadata.Metadata(title="Synthetic piano target", composer="Synthetic fixture")
    piano = stream.Part(id="piano")
    piano.partName = "Piano"
    piano.insert(0, instrument.Piano())
    melodies = [[60, 62, 64, 67], [69, 67, 65, 64], [62, 64, 65, 59], [60]]
    harmonies = [[48, 52, 55], [45, 48, 52], [43, 47, 50], [48, 52, 55]]
    for index, (melody, harmony) in enumerate(zip(melodies, harmonies), 1):
        solo_measure, acc_measure, target_measure = (stream.Measure(number=index) for _ in range(3))
        for measure in (solo_measure, acc_measure, target_measure):
            if index == 1:
                measure.insert(0, meter.TimeSignature("4/4"))
                measure.insert(0, key.KeySignature(0))
                measure.insert(0, tempo.MetronomeMark(number=84))
                measure.insert(0, dynamics.Dynamic("mp"))
        for pitch in melody:
            event = note.Note(pitch + transpose, quarterLength=4 if len(melody) == 1 else 1)
            solo_measure.append(event)
        acc_measure.insert(0, chord.Chord([p + transpose for p in harmony], quarterLength=4))
        target_melody, target_bass = stream.Voice(id=1), stream.Voice(id=2)
        for onset, pitch in enumerate(melody):
            upper = [pitch + transpose + 12]
            if index == 2:
                upper.append(pitch + transpose)
            target_melody.insert(onset, chord.Chord(upper, quarterLength=4 if len(melody) == 1 else 1))
        target_bass.insert(0, chord.Chord([p + transpose for p in harmony], quarterLength=4))
        target_measure.insert(0, target_melody)
        target_measure.insert(0, target_bass)
        solo.append(solo_measure)
        accompaniment.append(acc_measure)
        piano.append(target_measure)
    source.insert(0, solo)
    source.insert(0, accompaniment)
    target.insert(0, piano)
    return source, target


def make_demo_data(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    pairs = []
    for index, (inst, transposition) in enumerate(((instrument.Violoncello(), 0),
                                                 (instrument.Violin(), 5),
                                                 (instrument.Vocalist(), -2))):
        source, target = make_pair(inst, transposition)
        prefix = "" if index == 0 else f"{inst.instrumentName.lower()}_"
        source_path, target_path = directory / f"{prefix}source.musicxml", directory / f"{prefix}target.musicxml"
        source.write("musicxml", fp=str(source_path))
        target.write("musicxml", fp=str(target_path))
        pairs.append({
            "pair_id": f"synthetic_{index + 1:03}", "work_id": "synthetic_study",
            "composer": "Synthetic fixture", "arranger": "Deterministic fixture",
            "source": {"path": source_path.name, "solo_part": inst.instrumentName,
                       "solo_instrument": inst.instrumentName, "accompaniment_parts": ["Piano"]},
            "target": {"path": target_path.name, "instrument": "piano"},
            "license": {"source_status": "CHECKED", "target_status": "CHECKED",
                        "origin": "Original, programmatically authored synthetic test material",
                        "redistribution": "CC0-1.0"},
            "data_kind": "synthetic_fixture", "split": "train",
        })
    manifest = directory / "pairs.json"
    manifest.write_text(json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("examples"))
    args = parser.parse_args()
    print(make_demo_data(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
