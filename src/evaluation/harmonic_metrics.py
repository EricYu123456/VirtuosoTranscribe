"""Full-source pitch-class overlap; no inferred chord roots or cadences."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from src.representation.events import NoteEvent, ScoreData


def harmonic_overlap(source: ScoreData, target: ScoreData,
                     mapped_source_notes: tuple[NoteEvent, ...] | None = None) -> dict:
    source_notes = source.notes if mapped_source_notes is None else mapped_source_notes
    end = max(max((n.end for n in source_notes), default=0.0), target.duration)
    changes = defaultdict(list)
    changes[0.0]
    changes[end]
    for side, notes in enumerate((source_notes, target.notes)):
        for note in notes:
            if note.duration > 0:
                changes[note.onset].append((side, note.pitch, 1))
                changes[note.end].append((side, note.pitch, -1))
    boundaries = sorted(changes)
    active = [Counter(), Counter()]
    beat_pcs = [[set() for _ in range(math.ceil(end))] for _ in range(2)]
    beat_bass = [[None for _ in range(math.ceil(end))] for _ in range(2)]
    weighted = active_time = 0.0
    for start, stop in zip(boundaries, boundaries[1:]):
        for side, pitch, amount in changes[start]:
            active[side][pitch] += amount
            if not active[side][pitch]:
                del active[side][pitch]
        a, b = ({pitch % 12 for pitch in notes} for notes in active)
        if a or b:
            dt = stop - start
            weighted += dt * len(a & b) / len(a | b)
            active_time += dt
        for side, pcs in enumerate((a, b)):
            for beat in range(int(start), min(math.ceil(stop), len(beat_pcs[side]))):
                beat_pcs[side][beat].update(pcs)
                if active[side]:
                    bass = min(active[side])
                    previous = beat_bass[side][beat]
                    beat_bass[side][beat] = bass if previous is None else min(bass, previous)
    beat_scores = [len(a & b) / len(a | b) for a, b in zip(*beat_pcs) if a or b]
    bass_pairs = [(a, b) for a, b in zip(*beat_bass) if a is not None or b is not None]
    bass_overlap = [a is not None and b is not None and a % 12 == b % 12 for a, b in bass_pairs]
    def sign(value):
        return (value > 0) - (value < 0)
    motion = []
    for (a1, b1), (a2, b2) in zip(bass_pairs, bass_pairs[1:]):
        motion.append(False if None in (a1, b1, a2, b2) else sign(a2 - a1) == sign(b2 - b1))
    harmonic_changes = [(beat_pcs[side][i] != beat_pcs[side][i - 1]
                         for i in range(1, len(beat_pcs[side]))) for side in range(2)]
    rhythm = [a == b for a, b in zip(*harmonic_changes)]
    return {"harmonic_pitch_class_overlap": weighted / active_time if active_time else 1.0,
            "beat_pitch_class_overlap": sum(beat_scores) / len(beat_scores) if beat_scores else 1.0,
            "bass_pitch_class_agreement": sum(bass_overlap) / len(bass_overlap) if bass_overlap else None,
            "bass_motion_direction_similarity": sum(motion) / len(motion) if motion else None,
            "harmonic_rhythm_similarity": sum(rhythm) / len(rhythm) if rhythm else None}
