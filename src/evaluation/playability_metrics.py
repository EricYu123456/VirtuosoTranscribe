"""Sustained-note-aware, deliberately approximate two-hand playability."""

from __future__ import annotations

from collections import defaultdict
import math

from src.representation.events import NoteEvent, ScoreData
from .texture_metrics import sounding_intervals


def _best_split(pitches: list[int], hand_span: int, fingers: int) -> tuple[float, int, int]:
    """Choose two contiguous groups; independently optimistic hand allocation."""
    if not pitches:
        return 0.0, 0, 0
    possibilities = []
    for split in range(len(pitches) + 1):
        hands = (pitches[:split], pitches[split:])
        span = max((hand[-1] - hand[0] if hand else 0 for hand in hands))
        excess = sum(max(0, len(hand) - fingers) for hand in hands)
        penalty = max(0, span - hand_span) / max(hand_span, 1) + excess
        possibilities.append((penalty, span, excess))
    return min(possibilities)


def evaluate_playability(score: ScoreData, *, hand_span_semitones: int = 12,
                         max_notes_per_hand: int = 5,
                         leap_semitones: int = 16,
                         rapid_interval: float = 0.5) -> dict:
    """Proxy score in [0, 1]; all active notes count until their notated release.

    Pianists can release notes early or use pedal; this scorer does neither.
    An optimistic two-hand allocation is recomputed independently per interval.
    Fingering, physical reach, pedal, tempo, and inter-hand movement are unknown.
    Zero score means many proxy violations, not a proof of impossibility.
    """
    if (hand_span_semitones <= 0 or max_notes_per_hand <= 0 or leap_semitones <= 0
            or not math.isfinite(rapid_interval) or rapid_interval <= 0):
        raise ValueError("Playability thresholds must be positive and finite")
    duration = score.duration
    span_time = density_time = collision_time = 0.0
    peak_hand_span = peak_notes = 0
    for start, stop, active in sounding_intervals(score.notes, duration):
        dt = stop - start
        pitches = sorted({n.pitch for n in active})
        _, span, finger_excess = _best_split(pitches, hand_span_semitones, max_notes_per_hand)
        span_time += dt * (span > hand_span_semitones)
        density_time += dt * (finger_excess > 0)
        collision_time += dt * (len(active) > len(pitches))
        peak_hand_span, peak_notes = max(peak_hand_span, span), max(peak_notes, len(pitches))
    # MIDI channels/notation voices offer a reproducible (but imperfect) leap proxy.
    voices: dict[tuple[str, str], dict[float, list[NoteEvent]]] = defaultdict(lambda: defaultdict(list))
    for note in score.notes:
        voices[(note.part_id, note.voice)][note.onset].append(note)
    leap_count = rapid_wide_jump_count = transition_count = 0
    for attacks in voices.values():
        ordered = sorted(attacks.items())
        for (a_time, a), (b_time, b) in zip(ordered, ordered[1:]):
            # Nearest chord distance avoids treating wide, stationary chords as leaps.
            gap = max(min(abs(y.pitch - x.pitch) for y in b) for x in a)
            transition_count += 1
            leap_count += gap > leap_semitones
            rapid_wide_jump_count += gap > leap_semitones and b_time - a_time <= rapid_interval
    span_ratio = span_time / duration if duration else 0.0
    density_ratio = density_time / duration if duration else 0.0
    collision_ratio = collision_time / duration if duration else 0.0
    leap_ratio = leap_count / transition_count if transition_count else 0.0
    score_value = max(0.0, 1 - 0.45 * span_ratio - 0.3 * density_ratio
                      - 0.15 * leap_ratio - 0.1 * collision_ratio)
    return {
        "playability_score": score_value,
        "estimated_difficulty": 1 - score_value,
        "playability_span_violation_ratio": span_ratio,
        "playability_density_violation_ratio": density_ratio,
        "playability_leap_violation_ratio": leap_ratio,
        "playability_collision_ratio": collision_ratio,
        "peak_estimated_hand_span_semitones": peak_hand_span,
        "peak_simultaneous_keys": peak_notes,
        "leap_violation_count": leap_count,
        "rapid_wide_jump_count": rapid_wide_jump_count,
        "playability_thresholds": {"hand_span_semitones": hand_span_semitones,
                                   "max_notes_per_hand": max_notes_per_hand,
                                   "leap_semitones": leap_semitones,
                                   "rapid_interval_quarter_notes": rapid_interval},
        "playability_limitations": [
            "Proxy only: no fingering, hand size, pedal, articulation, or tempo model.",
            "All sustained note durations count; pedal-assisted releases may be playable.",
            "Independent contiguous hand allocation is optimistic and lacks hand continuity.",
            "Leap estimates depend on notation part/voice identities, not inferred hands.",
        ],
    }
