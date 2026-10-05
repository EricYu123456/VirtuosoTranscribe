"""Role-aware structured events on an exact quarter-note grid.

The training stream is compact; the JSON bundle also retains the notation and
score metadata necessary for a faithful symbolic round trip. Timing is checked,
never silently rounded. Twenty-four subdivisions support ordinary binary and
triplet rhythms; other tuplets require an explicitly chosen compatible grid.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import math
from typing import Any, Iterable

from .events import MeasureInfo, NoteEvent, ScoreData, SourceScore, note_sort_key
from .timing import beat_position


DEFAULT_SUBDIVISIONS = 24
VELOCITY_BINS = 16
ROLES = frozenset({"SOLO", "ACC", "PIANO", "UNKNOWN"})
BUNDLE_TYPE = "virtuosotranscribe.tokens"


def _ticks(value: float, subdivisions: int, label: str) -> int:
    if isinstance(subdivisions, bool) or not isinstance(subdivisions, int) or subdivisions < 1:
        raise ValueError("subdivisions must be a positive integer")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    scaled = value * subdivisions
    nearest = round(scaled)
    if not math.isclose(scaled, nearest, abs_tol=1e-7, rel_tol=0):
        raise ValueError(f"{label}={value!r} is not representable on the 1/{subdivisions} quarter-note grid")
    return nearest


def _validate_measures(measures: Iterable[MeasureInfo], subdivisions: int) -> dict[int, MeasureInfo]:
    result: dict[int, MeasureInfo] = {}
    previous_end = 0.0
    previous_number = 0
    for measure in measures:
        if isinstance(measure.number, bool) or not isinstance(measure.number, int) or measure.number <= previous_number or measure.number < 1:
            raise ValueError("Measures must have unique, increasing positive numbers")
        _ticks(measure.onset, subdivisions, "measure onset")
        if _ticks(measure.duration, subdivisions, "measure duration") < 1:
            raise ValueError("Measures must have positive durations")
        _ticks(measure.padding_left, subdivisions, "pickup padding")
        if measure.onset < previous_end - 1e-8:
            raise ValueError("Measures must not overlap")
        try:
            beat_position(measure, measure.onset)
        except Exception as exc:
            raise ValueError(f"Invalid meter: {measure.time_signature!r}") from exc
        result[measure.number] = measure
        previous_number = measure.number
        previous_end = measure.onset + measure.duration
    return result


def velocity_bin(velocity: int) -> int:
    if isinstance(velocity, bool) or not isinstance(velocity, int) or not 1 <= velocity <= 127:
        raise ValueError("velocity must be an integer in 1..127")
    return (velocity - 1) // 8


def tokenize_score(score: ScoreData, notes: Iterable[NoteEvent] | None = None,
                   *, role: str | None = None,
                   subdivisions: int = DEFAULT_SUBDIVISIONS) -> dict[str, Any]:
    """Return a JSON-serializable token bundle, optionally for one note stream.

    DUR values count subdivisions of a quarter note, so tied notes may freely
    cross bars and have arbitrarily long durations. VOICE_ID indexes both the
    part and its voice; simultaneous equal pitches in different voices survive.
    """
    _ticks(0, subdivisions, "grid")
    if role is not None and role not in ROLES:
        raise ValueError(f"Unsupported role: {role!r}")
    selected = sorted(score.notes if notes is None else tuple(notes), key=note_sort_key)
    if role is not None:
        selected = [replace(note, role=role) for note in selected]
    measures = _validate_measures(score.measures, subdivisions)
    if selected and not measures:
        raise ValueError("Tokenization requires measure metadata")
    voices = sorted({(note.part_id, note.voice) for note in selected})
    voice_ids = {voice: index for index, voice in enumerate(voices)}
    tokens: list[str] = []
    current_bar = None
    for note in selected:
        if isinstance(note.pitch, bool) or not isinstance(note.pitch, int):
            raise ValueError("MIDI pitch must be an integer")
        if isinstance(note.measure, bool) or not isinstance(note.measure, int):
            raise ValueError("Note measure number must be an integer")
        if note.role not in ROLES:
            raise ValueError(f"Unsupported role: {note.role!r}")
        if note.measure not in measures:
            raise ValueError(f"Note references missing measure {note.measure}")
        measure = measures[note.measure]
        position = _ticks(note.onset - measure.onset, subdivisions, "note position")
        _ticks(note.onset, subdivisions, "note onset")
        if position >= _ticks(measure.duration, subdivisions, "measure duration"):
            raise ValueError(f"Note onset is outside measure {note.measure}")
        duration = _ticks(note.duration, subdivisions, "note duration")
        if duration == 0 and not note.is_grace:
            raise ValueError("Only grace notes may have zero duration")
        if current_bar != note.measure:
            tokens.append(f"<BAR_{note.measure}>")
            current_bar = note.measure
        tokens.extend((f"<POS_{position}>", f"<ROLE_{note.role}>",
                       f"<PITCH_{note.pitch}>", f"<DUR_{duration}>",
                       f"<VELOCITY_BIN_{velocity_bin(note.velocity)}>",
                       f"<VOICE_ID_{voice_ids[(note.part_id, note.voice)]}>"))
    metadata = score.to_dict()
    metadata.pop("notes")
    return {"type": BUNDLE_TYPE, "schema_version": 1,
            "subdivisions": subdivisions, "velocity_bins": VELOCITY_BINS,
            "tokens": tokens, "score_metadata": metadata,
            "voice_table": [{"part_id": part, "voice": voice} for part, voice in voices],
            "note_metadata": [asdict(note) for note in selected]}


def tokenize_source(source: SourceScore, *, subdivisions: int = DEFAULT_SUBDIVISIONS) -> dict[str, Any]:
    """Encode the solo and accompaniment in separate role-labelled streams."""
    return {"source_solo": tokenize_score(source.score, source.solo, role="SOLO", subdivisions=subdivisions),
            "source_acc": tokenize_score(source.score, source.accompaniment, role="ACC", subdivisions=subdivisions)}
