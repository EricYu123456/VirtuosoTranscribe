"""Explicit part selection with conservative automatic solo/piano detection."""

from __future__ import annotations

from src.representation.events import PartInfo, ScoreData


def is_piano(part: PartInfo) -> bool:
    text = f"{part.name} {part.instrument}".casefold()
    return "piano" in text or "pianoforte" in text or part.midi_program == 0


def select_parts(score: ScoreData, selector: str) -> tuple[str, ...]:
    """Select exact id first, then exact name/instrument (including two staves)."""
    ids = tuple(p.id for p in score.parts if p.id == selector)
    if ids:
        return ids
    normalized = selector.strip().casefold()
    matched = tuple(p.id for p in score.parts
                    if normalized in {p.name.casefold(), p.instrument.casefold()})
    if not matched:
        available = ", ".join(f"{p.id} ({p.name})" for p in score.parts)
        raise ValueError(f"Unknown part {selector!r}; available: {available}")
    return matched


def identify_roles(score: ScoreData, solo_part: str | None = None,
                   accompaniment_parts: list[str] | tuple[str, ...] | None = None
                   ) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if solo_part is None:
        candidates = [p for p in score.parts if not is_piano(p)]
        if len(candidates) != 1:
            raise ValueError("Automatic solo detection needs exactly one non-piano part; "
                             "specify solo_part / --solo-part explicitly")
        solo_ids = (candidates[0].id,)
    else:
        solo_ids = select_parts(score, solo_part)
    if len(solo_ids) != 1:
        raise ValueError("Select a unique solo part by its id")
    if accompaniment_parts is None:
        accompaniment_ids = tuple(p.id for p in score.parts
                                  if is_piano(p) and p.id not in solo_ids)
    else:
        accompaniment_ids = tuple(dict.fromkeys(
            part_id for selector in accompaniment_parts for part_id in select_parts(score, selector)))
    if not accompaniment_ids:
        raise ValueError("At least one accompaniment part is required")
    if set(solo_ids) & set(accompaniment_ids):
        raise ValueError("Solo and accompaniment parts must be disjoint")
    return solo_ids, accompaniment_ids
