"""Duration-aware note density, polyphony, and register summaries."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Iterator

from src.representation.events import NoteEvent, ScoreData


def sounding_intervals(notes: Iterable[NoteEvent], duration: float | None = None
                       ) -> Iterator[tuple[float, float, tuple[NoteEvent, ...]]]:
    """Half-open sounding intervals, including silence and excluding grace notes."""
    notes = tuple(notes)
    end = duration if duration is not None else max((n.end for n in notes), default=0.0)
    changes: dict[float, list[tuple[bool, int]]] = defaultdict(list)
    changes[0.0]
    changes[end]
    for i, note in enumerate(notes):
        if note.duration > 0 and note.onset < end:
            changes[note.onset].append((True, i))
            changes[min(note.end, end)].append((False, i))
    active: set[int] = set()
    times = sorted(changes)
    for index, start in enumerate(times[:-1]):
        for add, note_index in changes[start]:
            if not add:
                active.discard(note_index)
        for add, note_index in changes[start]:
            if add:
                active.add(note_index)
        stop = times[index + 1]
        if stop > start:
            yield start, stop, tuple(notes[i] for i in sorted(active))


def texture_metrics(score: ScoreData) -> dict[str, float | int | None]:
    notes, duration = score.notes, score.duration
    intervals = tuple(sounding_intervals(notes, duration))
    integral = sum((stop - start) * len(active) for start, stop, active in intervals)
    pitch_integral = sum((stop - start) * len({n.pitch % 12 for n in active})
                         for start, stop, active in intervals)
    sounding_time = sum(stop - start for start, stop, active in intervals if active)
    lowest = min((n.pitch for n in notes), default=None)
    highest = max((n.pitch for n in notes), default=None)
    return {
        "note_count": len(notes),
        "duration_quarter_notes": duration,
        "notes_per_beat": len(notes) / duration if duration else 0.0,
        "notes_per_measure": len(notes) / len(score.measures) if score.measures else 0.0,
        "average_simultaneous_notes": integral / duration if duration else 0.0,
        "average_chord_size_sounding": integral / sounding_time if sounding_time else 0.0,
        "max_simultaneous_notes": max((len(a) for _, _, a in intervals), default=0),
        "average_vertical_pitch_classes": pitch_integral / sounding_time if sounding_time else 0.0,
        "lowest_pitch": lowest,
        "highest_pitch": highest,
        "pitch_span": highest - lowest if lowest is not None else 0,
    }
