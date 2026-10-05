"""Octave-equivalent, onset-constrained sequence alignment.

The dynamic program aligns onset *groups*, then solves a one-to-one assignment
inside each chord. An octave doubling can therefore never consume two repeated
melody attacks. Absolute time is used by default; supplying ``source_score``
enables measure alignment and proportional beat mapping before note alignment.
The confidence is a heuristic, not a calibrated probability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Iterable

from src.representation.events import MeasureInfo, NoteEvent, ScoreData


@dataclass(frozen=True)
class NoteMatch:
    source_index: int
    target_index: int
    source_pitch: int
    target_pitch: int
    source_onset: float
    target_onset: float
    expected_onset: float
    onset_deviation: float
    offset: float
    octave_shift: int
    cost: float


@dataclass(frozen=True)
class MeasureMatch:
    source_measure: int
    target_measure: int
    source_onset: float
    target_onset: float
    source_duration: float
    target_duration: float
    cost: float


@dataclass(frozen=True)
class AlignmentResult:
    matches: tuple[NoteMatch, ...]
    unmatched_source_indices: tuple[int, ...]
    unmatched_target_indices: tuple[int, ...]
    measure_matches: tuple[MeasureMatch, ...]
    quality: float
    onset_tolerance: float
    method: str = "onset_group_dynamic_programming"
    limitations: tuple[str, ...] = (
        "Pitch-class matches are evidence of melody presence, not voice identity.",
        "Monotonic alignment does not support formal reordering or many-to-one attacks.",
        "Confidence is a coverage/cost heuristic; manually inspect uncertain pairs.",
    )

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return path


def _groups(notes: tuple[NoteEvent, ...]) -> list[tuple[float, tuple[int, ...]]]:
    result: list[tuple[float, tuple[int, ...]]] = []
    for index in sorted(range(len(notes)), key=lambda i: (notes[i].onset, notes[i].pitch, i)):
        if result and abs(result[-1][0] - notes[index].onset) <= 1e-7:
            onset, indices = result[-1]
            result[-1] = (onset, indices + (index,))
        else:
            result.append((notes[index].onset, (index,)))
    return result


def _assignment(costs: list[list[float]]) -> tuple[float, list[int]]:
    """Rectangular Hungarian assignment; columns >= rows (including gap columns)."""
    n, m = len(costs), len(costs[0])
    u, v, p, way = [0.0] * (n + 1), [0.0] * (m + 1), [0] * (m + 1), [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv, used = [math.inf] * (m + 1), [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], math.inf, 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = costs[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if not p[j0]:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    assigned = [-1] * n
    for j in range(1, m + 1):
        if p[j]:
            assigned[p[j] - 1] = j - 1
    return sum(costs[i][j] for i, j in enumerate(assigned)), assigned


def align_measures(source: ScoreData, target: ScoreData,
                   source_solo: Iterable[NoteEvent] | None = None) -> tuple[MeasureMatch, ...]:
    """Sequence alignment of meter, duration, and melody pitch-class support."""
    sm, tm = source.measures, target.measures
    solo = tuple(source_solo) if source_solo is not None else source.notes
    spcs = {m.number: {n.pitch % 12 for n in solo if n.measure == m.number} for m in sm}
    tpcs = {m.number: {n.pitch % 12 for n in target.notes if n.measure == m.number} for m in tm}
    gap = 0.7
    dp = [[0.0] * (len(tm) + 1) for _ in range(len(sm) + 1)]
    back = [bytearray(len(tm) + 1) for _ in range(len(sm) + 1)]
    for i in range(1, len(sm) + 1):
        dp[i][0], back[i][0] = i * gap, 1
    for j in range(1, len(tm) + 1):
        dp[0][j], back[0][j] = j * gap, 2

    def cost(a: MeasureInfo, b: MeasureInfo) -> float:
        duration_cost = abs(a.duration - b.duration) / max(a.duration, b.duration, 1e-8)
        pcs = spcs[a.number]
        melody_cost = 1 - len(pcs & tpcs[b.number]) / len(pcs) if pcs else 0.0
        return duration_cost * 0.6 + (a.time_signature != b.time_signature) * 0.8 + melody_cost * 0.6

    for i, a in enumerate(sm, 1):
        for j, b in enumerate(tm, 1):
            candidates = (dp[i - 1][j - 1] + cost(a, b), dp[i - 1][j] + gap, dp[i][j - 1] + gap)
            choice = min(range(3), key=lambda k: candidates[k])
            dp[i][j], back[i][j] = candidates[choice], choice
    pairs = []
    i, j = len(sm), len(tm)
    while i or j:
        choice = back[i][j]
        if choice == 0:
            a, b = sm[i - 1], tm[j - 1]
            pairs.append(MeasureMatch(a.number, b.number, a.onset, b.onset,
                                      a.duration, b.duration, cost(a, b)))
            i, j = i - 1, j - 1
        elif choice == 1:
            i -= 1
        else:
            j -= 1
    return tuple(reversed(pairs))


def align_solo_to_piano(source_solo: Iterable[NoteEvent],
                        target_piano: ScoreData | Iterable[NoteEvent], *,
                        source_score: ScoreData | None = None,
                        onset_tolerance: float = 0.125) -> AlignmentResult:
    """Match melody notes once each within ``onset_tolerance`` quarter notes.

    Original input indices are preserved. Simultaneous chords are unordered.
    Pitch substitutions remain unmatched; octave displacements are permitted.
    Accompaniment attacks have zero skip cost, so dense textures do not force
    the solo onto unrelated notes. Gaps in the solo cost one per omitted note.
    """
    if not math.isfinite(onset_tolerance) or onset_tolerance < 0:
        raise ValueError("onset_tolerance must be finite and non-negative")
    source = tuple(source_solo)
    target = target_piano.notes if isinstance(target_piano, ScoreData) else tuple(target_piano)
    measures = (align_measures(source_score, target_piano, source)
                if source_score is not None and isinstance(target_piano, ScoreData) else ())
    measure_map = {m.source_measure: m for m in measures}
    hierarchical = source_score is not None and isinstance(target_piano, ScoreData) and bool(source_score.measures)

    def expected(note: NoteEvent) -> tuple[float | None, float, int | None]:
        if not hierarchical:
            return note.onset, note.duration, None
        m = measure_map.get(note.measure)
        if m is None:
            return None, note.duration, None
        scale = m.target_duration / m.source_duration if m.source_duration else 1.0
        return m.target_onset + (note.onset - m.source_onset) * scale, note.duration * scale, m.target_measure

    source_groups, target_groups = _groups(source), _groups(target)

    def group_match(si: tuple[int, ...], ti: tuple[int, ...]) -> tuple[float, list[tuple[int, int, float, float]]]:
        matrix = []
        for s in si:
            onset, duration, target_measure = expected(source[s])
            row = []
            for t in ti:
                difference = abs(target[t].onset - onset) if onset is not None else math.inf
                valid = (difference <= onset_tolerance + 1e-8 and source[s].pitch % 12 == target[t].pitch % 12
                         and (target_measure is None or target[t].measure == target_measure))
                if valid:
                    time_cost = 0.35 * difference / max(onset_tolerance, 1e-8)
                    dur_cost = 0.15 * abs(target[t].duration - duration) / max(target[t].duration, duration, 1e-8)
                    octave_cost = 0.05 * min(abs(target[t].pitch - source[s].pitch) / 12, 3)
                    row.append(time_cost + dur_cost + octave_cost)
                else:
                    row.append(1e6)
            matrix.append(row + [1.0] * len(si))
        total, assigned = _assignment(matrix)
        matched = [(si[i], ti[j], matrix[i][j], expected(source[si[i]])[0])
                   for i, j in enumerate(assigned) if j < len(ti) and matrix[i][j] < 1.0]
        return total, matched

    n, m = len(source_groups), len(target_groups)
    previous = [0.0] * (m + 1)
    back = [bytearray(m + 1) for _ in range(n + 1)]
    for j in range(1, m + 1):
        back[0][j] = 2
    for i, (_, si) in enumerate(source_groups, 1):
        current = [previous[0] + len(si)] + [0.0] * m
        back[i][0] = 1
        mapped = [expected(source[s])[0] for s in si]
        for j, (tonset, ti) in enumerate(target_groups, 1):
            possible = any(onset is not None and abs(tonset - onset) <= onset_tolerance + 1e-8 for onset in mapped)
            match_cost = group_match(si, ti)[0] if possible else math.inf
            candidates = (previous[j - 1] + match_cost, previous[j] + len(si), current[j - 1])
            choice = min(range(3), key=lambda k: candidates[k])
            current[j], back[i][j] = candidates[choice], choice
        previous = current
    raw = []
    i, j = n, m
    while i or j:
        choice = back[i][j]
        if choice == 0:
            raw.extend(group_match(source_groups[i - 1][1], target_groups[j - 1][1])[1])
            i, j = i - 1, j - 1
        elif choice == 1:
            i -= 1
        else:
            j -= 1
    matches = tuple(NoteMatch(s, t, source[s].pitch, target[t].pitch, source[s].onset,
                               target[t].onset, exp, target[t].onset - exp,
                               target[t].onset - source[s].onset,
                               target[t].pitch - source[s].pitch, cost)
                    for s, t, cost, exp in sorted(raw))
    used_source, used_target = {x.source_index for x in matches}, {x.target_index for x in matches}
    quality = sum(1.0 - x.cost for x in matches) / len(source) if source else 1.0
    return AlignmentResult(matches, tuple(i for i in range(len(source)) if i not in used_source),
                           tuple(i for i in range(len(target)) if i not in used_target),
                           measures, quality, onset_tolerance,
                           "measure_beat_onset_group_dynamic_programming" if hierarchical else
                           "onset_group_dynamic_programming")
