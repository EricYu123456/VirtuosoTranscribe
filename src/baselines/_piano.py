"""Shared allocation rules. All checks consider complete sounding intervals."""

from dataclasses import replace
from typing import Iterable

from src.representation.events import DynamicEvent, NoteEvent, PartInfo, ScoreData, SourceScore

PIANO_LOW = 21
PIANO_HIGH = 108
EPSILON = 1e-7
PIANO_PARTS = (
    PartInfo("piano_rh", "Piano RH", "Piano", 0),
    PartInfo("piano_lh", "Piano LH", "Piano", 0),
)


def overlaps(a: NoteEvent, b: NoteEvent) -> bool:
    """Grace notes occupy their attack instant, without inventing a duration."""
    if a.is_grace or b.is_grace:
        if a.is_grace and b.is_grace:
            return abs(a.onset - b.onset) < EPSILON
        grace, ordinary = (a, b) if a.is_grace else (b, a)
        return ordinary.onset - EPSILON <= grace.onset < ordinary.end - EPSILON
    return a.onset < b.end - EPSILON and b.onset < a.end - EPSILON


def piano_pitch(pitch: int) -> int:
    while pitch < PIANO_LOW:
        pitch += 12
    while pitch > PIANO_HIGH:
        pitch -= 12
    return pitch


def as_piano(note: NoteEvent, hand: str, pitch: int | None = None,
             role: str | None = None) -> NoteEvent:
    pitch = piano_pitch(note.pitch) if pitch is None else pitch
    return replace(note, pitch=pitch, part_id=f"piano_{hand}",
                   voice="1" if role == "SOLO" else "2",
                   role=role or note.role, pitch_name=None)


def bass_indices(notes: tuple[NoteEvent, ...]) -> set[int]:
    """Protect the lowest new attack, including upward overlapping bass motion.

    This is an attack-based salience proxy, not a bass-voice classifier. A held
    lower predecessor must not cause the next moving bass attack to be dropped.
    """
    result = set()
    grouped: dict[float, list[int]] = {}
    for index, note in enumerate(notes):
        grouped.setdefault(note.onset, []).append(index)
    for indices in grouped.values():
        lowest = min(indices, key=lambda i: (notes[i].pitch, -notes[i].duration))
        result.add(lowest)
    return result


def infer_hand(note: NoteEvent, source: SourceScore) -> str:
    part = next((p for p in source.score.parts if p.id == note.part_id), None)
    label = f"{note.part_id} {part.name if part else ''}".lower()
    if any(hint in label for hint in ("left", "bass", "_lh", " lh", "lower")):
        return "lh"
    if any(hint in label for hint in ("right", "treble", "_rh", " rh", "upper")):
        return "rh"
    return "lh" if note.pitch < 60 else "rh"


def same_event(a: NoteEvent, b: NoteEvent) -> bool:
    return (a.pitch == b.pitch and abs(a.onset - b.onset) < EPSILON
            and abs(a.duration - b.duration) < EPSILON
            and a.is_grace == b.is_grace)


def solo_notes(source: SourceScore, shift: int = 0) -> tuple[list[NoteEvent], int]:
    """One piano key represents coincident source-voice unisons once."""
    result: list[NoteEvent] = []
    merged = 0
    for original in source.solo:
        note = as_piano(original, "rh", piano_pitch(original.pitch + shift), "SOLO")
        if any(same_event(note, previous) for previous in result):
            merged += 1
        else:
            result.append(note)
    return result, merged


def can_add(note: NoteEvent, allocated: list[NoteEvent], *, max_span: int | None = None,
            max_per_hand: int | None = None, max_total: int | None = None) -> bool:
    touching = [n for n in allocated if overlaps(note, n)]
    if any(n.pitch == note.pitch for n in touching):
        return False
    melodies = [n.pitch for n in touching if n.role == "SOLO"]
    if note.role not in ("SOLO", "MELODY_DOUBLE") and melodies and note.pitch >= min(melodies):
        return False
    if max_span is None and max_per_hand is None and max_total is None:
        return True
    # An interval can intersect notes that never sound together. Check each
    # actual attack/release boundary, rather than their combined pitch envelope.
    boundaries = {note.onset}
    boundaries.update(n.onset for n in touching if note.onset <= n.onset < note.end)
    for time in sorted(boundaries):
        active = [n for n in touching if n.onset <= time + EPSILON
                  and (n.end > time + EPSILON or (n.is_grace and abs(n.onset - time) < EPSILON))]
        hand = [n for n in active if n.part_id == note.part_id] + [note]
        if max_total is not None and len(active) + 1 > max_total:
            return False
        if max_per_hand is not None and len(hand) > max_per_hand:
            return False
        if max_span is not None and max(n.pitch for n in hand) - min(n.pitch for n in hand) > max_span:
            return False
    return True


def pitch_choices(note: NoteEvent, allocated: list[NoteEvent], *, preferred: int | None = None,
                  below_melody: bool = True) -> list[int]:
    pitches = [p for p in range(PIANO_LOW, PIANO_HIGH + 1) if p % 12 == note.pitch % 12]
    melodies = [n.pitch for n in allocated if n.role == "SOLO" and overlaps(n, note)]
    if below_melody and melodies:
        pitches = [p for p in pitches if p < min(melodies)]
    anchor = piano_pitch(note.pitch) if preferred is None else preferred
    return sorted(pitches, key=lambda p: (abs(p - anchor), p))


def finish(source: SourceScore, notes: Iterable[NoteEvent], warnings: list[str],
           title_suffix: str) -> ScoreData:
    notes = list(notes)
    mapped = []
    for dynamic in source.score.dynamics:
        originals = [n for n in source.notes if n.part_id == dynamic.part_id
                     and n.end >= dynamic.onset - EPSILON]
        if originals:
            nearest_onset = min(originals, key=lambda n: abs(n.onset - dynamic.onset)).onset
            originals = [n for n in originals if abs(n.onset - nearest_onset) < EPSILON]
        hands = {n.part_id for n in notes for original in originals
                 if n.pitch % 12 == original.pitch % 12
                 and abs(n.onset - original.onset) < EPSILON
                 and abs(n.duration - original.duration) < EPSILON}
        if not hands:
            hands = {"piano_rh" if dynamic.part_id in source.solo_part_ids else "piano_lh"}
            warning = "Some dynamics had no retained source-note anchor; assigned by source role for review."
            if warning not in warnings:
                warnings.append(warning)
        for hand in sorted(hands):
            event = DynamicEvent(dynamic.onset, dynamic.value, hand)
            if event not in mapped:
                mapped.append(event)
    return source.score.with_notes(notes, parts=PIANO_PARTS, dynamics=tuple(mapped),
                                   title=f"{source.score.title} — {title_suffix}",
                                   warnings=source.score.warnings + tuple(warnings))

