"""Plain solo-plus-accompaniment reduction with octave collision handling."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.baselines._piano import as_piano, bass_indices, can_add, finish, infer_hand, pitch_choices, same_event, solo_notes
from src.representation.events import ScoreData, SourceScore


def reduce_literal(source: SourceScore) -> ScoreData:
    """Retain musical events, merging unisons and moving collisions by octaves.

    This intentionally does not impose the rule-based baseline's hand-span or
    density caps. Its output is a lower bound, with playability reported later.
    """
    if not source.solo:
        raise ValueError("A piano reduction requires a non-empty solo role")
    allocated, merged = solo_notes(source)
    warnings: list[str] = []
    if merged:
        warnings.append(f"Merged {merged} coincident solo unison events into single piano keys.")
    if any(not can_add(note, allocated[:index]) for index, note in enumerate(allocated)):
        warnings.append("Protected solo has overlapping attacks on a single piano key; retained with source timing for review.")
    omitted = 0
    bass = bass_indices(source.accompaniment)
    order = sorted(range(len(source.accompaniment)),
                   key=lambda i: (i not in bass, source.accompaniment[i].onset,
                                  source.accompaniment[i].pitch))
    for index in order:
        original = source.accompaniment[index]
        note = as_piano(original, "lh" if index in bass else infer_hand(original, source), role="ACC")
        if any(same_event(note, previous) for previous in allocated):
            continue
        for pitch in pitch_choices(note, allocated):
            hand = "lh" if index in bass or pitch < 60 else infer_hand(original, source)
            candidate = as_piano(note, hand, pitch)
            if can_add(candidate, allocated):
                allocated.append(candidate)
                break
        else:
            omitted += 1
    if omitted:
        warnings.append(f"Literal reduction omitted {omitted} accompaniment events with no collision-free piano register.")
    if any(not 21 <= original.pitch <= 108 for original in source.solo):
        warnings.append("Solo notes outside the piano range were moved by octaves.")
    return finish(source, allocated, warnings, "literal piano reduction")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--solo-part")
    parser.add_argument("--accompaniment-part", action="append")
    args = parser.parse_args(argv)
    from src.data.parse_musicxml import parse_score
    from src.data.normalize_score import normalize_source
    from src.inference.export import export_score
    try:
        source = normalize_source(parse_score(args.input), solo_part=args.solo_part,
                                  accompaniment_parts=args.accompaniment_part)
        result = reduce_literal(source)
        export_score(result, args.output)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Wrote {args.output} ({len(result.notes)} notes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

