"""Role assignment and optional explicit time-grid normalization."""

from dataclasses import replace
import math

from music21 import pitch

from src.data.identify_roles import identify_roles
from src.representation.events import NoteEvent, ScoreData, SourceScore, note_sort_key
from src.representation.timing import beat_position


def _quantize(note: NoteEvent, grid: int | None, score: ScoreData) -> NoteEvent:
    if grid is None:
        return note
    if isinstance(grid, bool) or not isinstance(grid, int) or grid < 1:
        raise ValueError("quantization must be a positive integer subdivision")
    onset = round(note.onset * grid) / grid
    duration = round(note.duration * grid) / grid
    if duration == 0 and not note.is_grace:
        raise ValueError("Quantization would collapse a note to zero duration")
    owner = next((m for m in score.measures if m.onset <= onset < m.onset + m.duration), None)
    if owner is None:
        raise ValueError("Quantization moved a note outside the score's measure timeline")
    return replace(note, onset=onset, duration=duration, measure=owner.number,
                   beat=beat_position(owner, onset))


def normalize_source(score: ScoreData, solo_part: str | None = None,
                     accompaniment_parts: list[str] | tuple[str, ...] | None = None,
                     *, quantization: int | None = None) -> SourceScore:
    solo_ids, accompaniment_ids = identify_roles(score, solo_part, accompaniment_parts)
    solo = tuple(sorted((_quantize(replace(n, role="SOLO"), quantization, score)
                         for n in score.notes if n.part_id in solo_ids), key=note_sort_key))
    accompaniment = tuple(sorted((_quantize(replace(n, role="ACC"), quantization, score)
                                  for n in score.notes if n.part_id in accompaniment_ids),
                                 key=note_sort_key))
    if not solo:
        raise ValueError("The selected solo part contains no pitched notes")
    selected_ids = set(solo_ids + accompaniment_ids)
    ignored = [p.name for p in score.parts if p.id not in selected_ids]
    notices = score.warnings
    if ignored:
        notices += (f"Unselected parts excluded: {', '.join(ignored)}",)
    selected_score = score.with_notes(solo + accompaniment,
                                      parts=tuple(p for p in score.parts if p.id in selected_ids),
                                      dynamics=tuple(d for d in score.dynamics
                                                     if d.part_id in selected_ids), warnings=notices)
    return SourceScore(selected_score, solo, accompaniment, solo_ids, accompaniment_ids)


def normalize_target(score: ScoreData) -> ScoreData:
    return score.with_notes([replace(n, role="PIANO") for n in score.notes])


def validate_score(score: ScoreData) -> None:
    ids = {p.id for p in score.parts}
    if len(ids) != len(score.parts):
        raise ValueError("Duplicate part identifiers")
    if not score.measures:
        raise ValueError("At least one measure is required")
    previous_end = 0.0
    for index, measure in enumerate(score.measures):
        if measure.number != index + 1:
            raise ValueError("Measures must have consecutive normalized numbers")
        if not math.isfinite(measure.onset) or not math.isfinite(measure.duration):
            raise ValueError("Measure timing must be finite")
        if measure.duration <= 0 or abs(measure.onset - previous_end) > 1e-6:
            raise ValueError("Measure timeline must be contiguous with positive durations")
        previous_end = measure.onset + measure.duration
    for note in score.notes:
        if note.part_id not in ids:
            raise ValueError(f"Unknown note part: {note.part_id}")
        if note.end > previous_end + 1e-6:
            raise ValueError("A note extends past the score's measure timeline")
        if note.measure > len(score.measures):
            raise ValueError("Note measure number is outside the score's measure timeline")
        owner = score.measures[note.measure - 1]
        if note.onset < owner.onset - 1e-6 or note.onset >= owner.onset + owner.duration - 1e-6:
            raise ValueError("Note onset does not belong to its declared measure")
        if note.pitch_name:
            try:
                pitch.Pitch(note.pitch_name)
            except Exception as exc:
                raise ValueError(f"Invalid pitch spelling: {note.pitch_name}") from exc
