"""Measure-based segmentation with explicit sustained-note clipping provenance."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from src.representation.events import ScoreData, SourceScore, note_sort_key
from src.representation.timing import beat_position
from src.analysis.melody_alignment import AlignmentResult, align_solo_to_piano


@dataclass(frozen=True)
class SegmentNoteBoundary:
    original_index: int
    segment_index: int
    original_onset: float
    original_end: float
    clipped_start: bool
    clipped_end: bool


@dataclass(frozen=True)
class ScoreSegment:
    score: ScoreData
    start_measure: int
    end_measure: int
    start_offset: float
    end_offset: float
    note_boundaries: tuple[SegmentNoteBoundary, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SourceSegment:
    source: SourceScore
    start_measure: int
    end_measure: int
    start_offset: float
    end_offset: float
    note_boundaries: tuple[SegmentNoteBoundary, ...]


@dataclass(frozen=True)
class PairSegment:
    source: SourceScore
    target: ScoreData
    start_measure: int
    end_measure: int
    target_start_measure: int
    target_end_measure: int
    alignment_quality: float
    source_note_boundaries: tuple[SegmentNoteBoundary, ...]
    target_note_boundaries: tuple[SegmentNoteBoundary, ...]


def _slice_events(events, start: float, stop: float, *, by_part: bool = False):
    """Retain the state active at segment start and any subsequent changes."""
    previous = {}
    within = []
    for event in sorted(events, key=lambda e: e.onset):
        key = event.part_id if by_part else "state"
        if event.onset < start:
            previous[key] = event
        elif event.onset < stop:
            within.append(replace(event, onset=event.onset - start))
    keys_at_start = {e.part_id if by_part else "state" for e in within if abs(e.onset) < 1e-8}
    inherited = [replace(e, onset=0.0) for key, e in previous.items() if key not in keys_at_start]
    return tuple(sorted(inherited + within, key=lambda e: e.onset))


def slice_score_measures(score: ScoreData, start_measure: int, end_measure: int) -> ScoreSegment:
    """Inclusive measure slice, rebase onset and measure numbers for round trips.

    Notes crossing either boundary are clipped, never silently discarded. The
    metadata preserves original index/time and marks clipping so later stages
    can avoid treating carried notes as new attacks or reconstruct the original.
    """
    measures = tuple(m for m in score.measures if start_measure <= m.number <= end_measure)
    if not measures:
        raise ValueError("No measures in requested segment")
    if measures[0].number != start_measure or measures[-1].number != end_measure:
        raise ValueError("Segment bounds must identify existing measures")
    start, stop = measures[0].onset, measures[-1].onset + measures[-1].duration
    rebased_measures = tuple(replace(m, number=i + 1, onset=m.onset - start) for i, m in enumerate(measures))
    clipped = []
    for original_index, note in enumerate(score.notes):
        if note.is_grace:
            if not start <= note.onset < stop:
                continue
        elif note.end <= start + 1e-9 or note.onset >= stop - 1e-9:
            continue
        onset = max(note.onset, start)
        end = min(note.end, stop)
        owner = next((m for m in reversed(measures) if m.onset <= onset + 1e-8), measures[0])
        beat = beat_position(owner, onset)
        event = replace(note, onset=onset - start, duration=max(0.0, end - onset),
                        measure=owner.number - start_measure + 1, beat=beat)
        clipped.append((event, original_index, note.onset, note.end,
                        note.onset < start - 1e-9, note.end > stop + 1e-9))
    clipped.sort(key=lambda item: note_sort_key(item[0]))
    boundaries = tuple(SegmentNoteBoundary(index, local, onset, end, left, right)
                       for local, (_, index, onset, end, left, right) in enumerate(clipped))
    result = replace(score, notes=tuple(item[0] for item in clipped), measures=rebased_measures,
                     time_signatures=_slice_events(score.time_signatures, start, stop),
                     key_signatures=_slice_events(score.key_signatures, start, stop),
                     tempos=_slice_events(score.tempos, start, stop),
                     dynamics=_slice_events(score.dynamics, start, stop, by_part=True))
    return ScoreSegment(result, start_measure, end_measure, start, stop, boundaries)


def segment_score(score: ScoreData, measures_per_segment: int = 4, *,
                   stride_measures: int | None = None, min_measures: int = 1) -> tuple[ScoreSegment, ...]:
    if measures_per_segment <= 0 or min_measures <= 0 or min_measures > measures_per_segment:
        raise ValueError("Segment sizes must be positive; min_measures must not exceed size")
    stride = measures_per_segment if stride_measures is None else stride_measures
    if stride <= 0:
        raise ValueError("stride_measures must be positive")
    result = []
    for start in range(0, len(score.measures), stride):
        measures = score.measures[start:start + measures_per_segment]
        if len(measures) >= min_measures:
            result.append(slice_score_measures(score, measures[0].number, measures[-1].number))
    return tuple(result)


def _source_from_segment(source: SourceScore, segment: ScoreSegment) -> SourceScore:
    solo_ids, acc_ids = set(source.solo_part_ids), set(source.accompaniment_part_ids)
    return SourceScore(segment.score,
                       tuple(n for n in segment.score.notes if n.part_id in solo_ids),
                       tuple(n for n in segment.score.notes if n.part_id in acc_ids),
                       source.solo_part_ids, source.accompaniment_part_ids)


def segment_source(source: SourceScore, measures_per_segment: int = 4, *,
                    stride_measures: int | None = None, min_measures: int = 1) -> tuple[SourceSegment, ...]:
    return tuple(SourceSegment(_source_from_segment(source, s), s.start_measure, s.end_measure,
                               s.start_offset, s.end_offset, s.note_boundaries)
                 for s in segment_score(source.score, measures_per_segment,
                                        stride_measures=stride_measures, min_measures=min_measures))


def segment_pair(source: SourceScore, target: ScoreData, measures_per_segment: int = 4,
                  alignment: AlignmentResult | None = None, *,
                  stride_measures: int | None = None, min_measures: int = 1) -> tuple[PairSegment, ...]:
    """Slice monotonic measure-aligned pairs; skip wholly unaligned windows.

    Target measures between the first/last mapped bars are included, including
    inserted measures. This keeps the actual target context rather than deleting
    material silently. Segment alignment quality is recomputed after slicing.
    """
    alignment = alignment or align_solo_to_piano(source.solo, target, source_score=source.score)
    mapping = {m.source_measure: m.target_measure for m in alignment.measure_matches}
    result = []
    for window in segment_score(source.score, measures_per_segment,
                                stride_measures=stride_measures, min_measures=min_measures):
        target_numbers = [mapping[m] for m in range(window.start_measure, window.end_measure + 1) if m in mapping]
        if not target_numbers:
            continue
        target_window = slice_score_measures(target, min(target_numbers), max(target_numbers))
        source_window = _source_from_segment(source, window)
        local_alignment = align_solo_to_piano(source_window.solo, target_window.score, source_score=source_window.score)
        result.append(PairSegment(source_window, target_window.score, window.start_measure, window.end_measure,
                                  target_window.start_measure, target_window.end_measure,
                                  local_alignment.quality, window.note_boundaries, target_window.note_boundaries))
    return tuple(result)
