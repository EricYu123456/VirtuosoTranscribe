"""Incremental event grammar and transparent melody constraints.

Hard decoding protects every annotated solo attack, including simultaneous and
repeated notes. It constrains timing, pitch (exact or one octave) and duration
before sampling; there is no unreported post-generation melody replacement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Mapping, Sequence

from src.representation.events import SourceScore
from src.representation.tokenizer import DEFAULT_SUBDIVISIONS, _ticks, _validate_measures


class GenerationError(ValueError):
    """The learned decoder could not produce a valid bounded piano sequence."""


@dataclass(frozen=True)
class MelodyRequirement:
    bar: int
    position: int
    pitch: int
    duration: int

    @property
    def pitches(self) -> tuple[int, ...]:
        return tuple(p for p in (self.pitch, self.pitch - 12, self.pitch + 12)
                     if 21 <= p <= 108)


@dataclass
class GrammarState:
    phase: str = "start"
    bar: int = 0
    position: int = -1
    pitch: int = 0
    duration: int = 0
    melody_cursor: int = 0
    forced_requirement: int | None = None
    event_count: int = 0
    completed: bool = False
    forced_steps: int = 0
    required_events: int = 0
    force_next_melody: bool = False
    satisfied_requirements: set[int] = field(default_factory=set)
    fields: list[str] = field(default_factory=list)


class PianoTokenGrammar:
    """Accept only complete BAR/POS/ROLE/PITCH/DUR/VELOCITY/VOICE records."""

    def __init__(self, tokens: Mapping[str, int], source: SourceScore, *,
                 melody_mode: str = "hard", subdivisions: int = DEFAULT_SUBDIVISIONS,
                 soft_strength: float = 2.0, melody_preservation: float = 1.0):
        if melody_mode not in {"none", "soft", "hard"}:
            raise ValueError("melody_mode must be none, soft or hard")
        if isinstance(soft_strength, bool) or not isinstance(soft_strength, (int, float)) or not math.isfinite(soft_strength) or soft_strength < 0:
            raise ValueError("soft_strength must be finite and non-negative")
        self.tokens = dict(tokens)
        self.inverse = {value: key for key, value in tokens.items()}
        self.mode = melody_mode
        self.subdivisions = subdivisions
        self.soft_strength = soft_strength * melody_preservation
        self.measures = _validate_measures(source.score.measures, subdivisions)
        if not self.measures:
            raise GenerationError("Generation requires non-empty measure metadata")
        self.end_tick = _ticks(source.duration, subdivisions, "source duration")
        self.requirements = sorted(
            (MelodyRequirement(n.measure,
                               _ticks(n.onset - self.measures[n.measure].onset,
                                      subdivisions, "melody position"),
                               n.pitch, _ticks(n.duration, subdivisions, "melody duration"))
             for n in source.solo),
            key=lambda n: (n.bar, n.position, n.pitch, n.duration))
        self.groups: dict[str, dict[int, int]] = {}
        for token, index in tokens.items():
            match = re.fullmatch(r"<([A-Z_]+)_(\d+)>", token)
            if match:
                self.groups.setdefault(match[1], {})[int(match[2])] = index
        for marker in ("<ROLE_PIANO>", "<EOS>"):
            if marker not in tokens:
                raise GenerationError(f"Checkpoint vocabulary lacks required token {marker}")
        self.eos_id = tokens["<EOS>"]
        if melody_mode == "hard":
            for requirement in self.requirements:
                if not requirement.pitches:
                    raise GenerationError(f"Melody pitch {requirement.pitch} has no piano-range exact/±12 realization")
                if requirement.duration == 0:
                    raise GenerationError("Hard melody generation cannot serialize grace notes without notation sidecars")
                required = [f"<BAR_{requirement.bar}>", f"<POS_{requirement.position}>",
                            f"<DUR_{requirement.duration}>"]
                missing = [token for token in required if token not in tokens]
                if missing or not any(f"<PITCH_{p}>" in tokens for p in requirement.pitches):
                    raise GenerationError(f"Checkpoint vocabulary cannot express a melody requirement: {missing or requirement}")

    def _next(self, state: GrammarState) -> MelodyRequirement | None:
        if self.mode == "hard" and state.melody_cursor < len(self.requirements):
            return self.requirements[state.melody_cursor]
        return None

    def prepare_step(self, state: GrammarState, preferred_id: int) -> None:
        """Force a missing attack only when the model tries to advance past it.

        A learned bass-before-melody chord can otherwise retain its natural
        event order. Requiring melody as every chord's first event would corrupt
        a correctly learned target merely because token ordering differs.
        """
        required = self._next(state)
        if required is None or state.phase not in {"start", "boundary", "position"}:
            return
        token = self.inverse.get(preferred_id, "")
        skipped = preferred_id == self.eos_id
        if token.startswith("<BAR_"):
            skipped = int(token[5:-1]) > required.bar
        elif token.startswith("<POS_"):
            skipped = state.bar == required.bar and int(token[5:-1]) > required.position
        if skipped:
            state.force_next_melody = True

    def allowed_ids(self, state: GrammarState) -> set[int]:
        if state.completed:
            return set()
        required = self._next(state)
        if state.force_next_melody and required is not None and state.phase in {"start", "boundary", "position"}:
            field_name, value = (("BAR", required.bar) if required.bar > state.bar
                                 else ("POS", required.position))
            return {self.groups[field_name][value]}
        if state.phase in {"start", "boundary"}:
            result = {index for bar, index in self.groups.get("BAR", {}).items()
                      if bar in self.measures and bar > state.bar
                      and (required is None or bar <= required.bar)}
            if state.phase == "boundary":
                result.update(self._positions(state, required))
                if required is None:
                    result.add(self.eos_id)
            return result
        if state.phase == "position":
            return self._positions(state, required)
        if state.phase == "role":
            return {self.tokens["<ROLE_PIANO>"]}
        if state.phase == "pitch":
            pitches = self.groups.get("PITCH", {})
            if state.forced_requirement is not None:
                requirement = self.requirements[state.forced_requirement]
                return {pitches[p] for p in requirement.pitches if p in pitches}
            return {index for pitch, index in pitches.items() if 21 <= pitch <= 108}
        if state.phase == "duration":
            durations = self.groups.get("DUR", {})
            if state.forced_requirement is not None and state.force_next_melody:
                return {durations[self.requirements[state.forced_requirement].duration]}
            onset = _ticks(self.measures[state.bar].onset, self.subdivisions, "bar onset") + state.position
            return {index for duration, index in durations.items()
                    if 0 < duration <= self.end_tick - onset}
        if state.phase == "velocity":
            return {index for value, index in self.groups.get("VELOCITY_BIN", {}).items()
                    if 0 <= value < 16}
        if state.phase == "voice":
            return {index for value, index in self.groups.get("VOICE_ID", {}).items()
                    if 0 <= value < 64}
        raise GenerationError(f"Unknown decoder grammar phase {state.phase}")

    def _positions(self, state: GrammarState,
                   required: MelodyRequirement | None) -> set[int]:
        maximum = _ticks(self.measures[state.bar].duration, self.subdivisions, "bar duration") - 1
        if required is not None and required.bar == state.bar:
            maximum = min(maximum, required.position)
        return {index for position, index in self.groups.get("POS", {}).items()
                if max(0, state.position) <= position <= maximum}

    def soft_bias(self, state: GrammarState) -> dict[int, float]:
        """Bias notes at matching attacks; never block or insert melody in soft mode."""
        if self.mode != "soft" or self.soft_strength == 0:
            return {}
        here = [n for n in self.requirements if n.bar == state.bar and n.position == state.position]
        if state.phase == "pitch":
            return {self.tokens[f"<PITCH_{pitch}>"]: self.soft_strength
                    for note in here for pitch in note.pitches
                    if f"<PITCH_{pitch}>" in self.tokens}
        if state.phase == "duration":
            return {self.tokens[f"<DUR_{note.duration}>"]: self.soft_strength
                    for note in here if state.pitch in note.pitches
                    and f"<DUR_{note.duration}>" in self.tokens}
        if state.phase in {"boundary", "position"}:
            return {self.tokens[f"<POS_{note.position}>"]: self.soft_strength / 2
                    for note in self.requirements if note.bar == state.bar
                    and note.position >= state.position
                    and f"<POS_{note.position}>" in self.tokens}
        return {}

    def advance(self, state: GrammarState, token_id: int) -> None:
        if token_id not in self.allowed_ids(state):
            raise GenerationError(f"Invalid token {self.inverse.get(token_id, token_id)!r} in phase {state.phase}")
        token = self.inverse[token_id]
        if token_id == self.eos_id:
            state.completed = True
            return
        if token.startswith("<BAR_"):
            state.bar, state.position = int(token[5:-1]), -1
            state.phase = "position"
        elif token.startswith("<POS_"):
            state.position = int(token[5:-1])
            required = self._next(state)
            state.forced_requirement = (state.melody_cursor if required is not None and state.force_next_melody
                                        and (required.bar, required.position) == (state.bar, state.position)
                                        else None)
            state.phase = "role"
        elif state.phase == "role":
            state.phase = "pitch"
        elif state.phase == "pitch":
            state.pitch = int(token[7:-1])
            state.forced_steps += state.forced_requirement is not None
            if self.mode == "hard" and state.forced_requirement is None:
                state.forced_requirement = next((index for index, note in enumerate(self.requirements)
                                                 if index not in state.satisfied_requirements
                                                 and (note.bar, note.position) == (state.bar, state.position)
                                                 and state.pitch in note.pitches), None)
            state.phase = "duration"
        elif state.phase == "duration":
            state.duration = int(token[5:-1])
            state.forced_steps += state.forced_requirement is not None and state.force_next_melody
            if state.forced_requirement is not None and state.duration != self.requirements[state.forced_requirement].duration:
                state.forced_requirement = None
            state.phase = "velocity"
        elif state.phase == "velocity":
            state.phase = "voice"
        elif state.phase == "voice":
            state.event_count += 1
            if state.forced_requirement is not None:
                state.satisfied_requirements.add(state.forced_requirement)
                while state.melody_cursor in state.satisfied_requirements:
                    state.melody_cursor += 1
                state.required_events += 1
                state.forced_requirement = None
            state.force_next_melody = False
            state.phase = "boundary"
        state.fields.append(token)

    def validate(self, tokens: Sequence[str]) -> GrammarState:
        state = GrammarState()
        for token in tokens:
            if token not in self.tokens:
                raise GenerationError(f"Token absent from checkpoint vocabulary: {token}")
            self.advance(state, self.tokens[token])
        if not state.completed:
            raise GenerationError("Sequence ended without EOS at a complete event boundary")
        return state
