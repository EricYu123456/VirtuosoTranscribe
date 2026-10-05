"""Shared, JSON-serializable intermediate representation.

Times use quarter-note units in concert pitch. Measures are numbered from one;
the original printed number and pickup length are retained separately. Resolved
ties appear as one event, while tuplets retain both rational timing and notation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TupletInfo:
    actual: int
    normal: int
    type: str | None = None


@dataclass(frozen=True)
class NoteEvent:
    pitch: int
    onset: float
    duration: float
    part_id: str
    measure: int = 1
    beat: float = 1.0
    voice: str = "1"
    velocity: int = 80
    role: str = "UNKNOWN"
    pitch_name: str | None = None
    articulations: tuple[str, ...] = ()
    tuplets: tuple[TupletInfo, ...] = ()
    slur_ids: tuple[str, ...] = ()
    is_grace: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.pitch, bool) or not isinstance(self.pitch, int) or not 0 <= self.pitch <= 127:
            raise ValueError(f"MIDI pitch out of range: {self.pitch}")
        if not math.isfinite(self.onset) or self.onset < 0:
            raise ValueError("Note onset must be finite and non-negative")
        if not math.isfinite(self.duration) or self.duration < 0:
            raise ValueError("Note duration must be finite and non-negative")
        if self.duration == 0 and not self.is_grace:
            raise ValueError("Only grace notes may have zero duration")
        if isinstance(self.velocity, bool) or not isinstance(self.velocity, int) or not 1 <= self.velocity <= 127:
            raise ValueError("Velocity must be in 1..127")
        if isinstance(self.measure, bool) or not isinstance(self.measure, int) or self.measure < 1:
            raise ValueError("Normalized measure numbers start at one")

    @property
    def end(self) -> float:
        return self.onset + self.duration


@dataclass(frozen=True)
class MeasureInfo:
    number: int
    onset: float
    duration: float
    time_signature: str = "4/4"
    original_number: str = "1"
    padding_left: float = 0.0


@dataclass(frozen=True)
class SignatureEvent:
    onset: float
    value: str


@dataclass(frozen=True)
class TempoEvent:
    onset: float
    bpm: float


@dataclass(frozen=True)
class DynamicEvent:
    onset: float
    value: str
    part_id: str


@dataclass(frozen=True)
class PartInfo:
    id: str
    name: str
    instrument: str = ""
    midi_program: int | None = None


@dataclass(frozen=True)
class ScoreData:
    notes: tuple[NoteEvent, ...]
    parts: tuple[PartInfo, ...]
    measures: tuple[MeasureInfo, ...]
    time_signatures: tuple[SignatureEvent, ...] = ()
    key_signatures: tuple[SignatureEvent, ...] = ()
    tempos: tuple[TempoEvent, ...] = ()
    dynamics: tuple[DynamicEvent, ...] = ()
    title: str = "Untitled"
    composer: str = ""
    warnings: tuple[str, ...] = ()
    schema_version: int = 1

    @property
    def duration(self) -> float:
        return max([0.0] + [m.onset + m.duration for m in self.measures]
                   + [n.end for n in self.notes])

    def with_notes(self, notes: tuple[NoteEvent, ...] | list[NoteEvent],
                   **changes: Any) -> ScoreData:
        return replace(self, notes=tuple(sorted(notes, key=note_sort_key)), **changes)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ScoreData:
        if payload.get("schema_version", 1) != 1:
            raise ValueError("Unsupported score schema version")
        notes = []
        for raw in payload.get("notes", []):
            raw = dict(raw)
            raw["tuplets"] = tuple(TupletInfo(**x) for x in raw.get("tuplets", []))
            for field_name in ("articulations", "slur_ids"):
                raw[field_name] = tuple(raw.get(field_name, []))
            notes.append(NoteEvent(**raw))
        data = dict(payload, notes=tuple(notes))
        for field_name, item_type in (("parts", PartInfo), ("measures", MeasureInfo),
                                      ("time_signatures", SignatureEvent),
                                      ("key_signatures", SignatureEvent),
                                      ("tempos", TempoEvent), ("dynamics", DynamicEvent)):
            data[field_name] = tuple(item_type(**x) for x in payload.get(field_name, []))
        data["warnings"] = tuple(payload.get("warnings", []))
        return cls(**data)

    @classmethod
    def load(cls, path: str | Path) -> ScoreData:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def note_sort_key(note: NoteEvent) -> tuple:
    return (note.onset, note.part_id, note.voice, note.pitch, note.duration)


@dataclass(frozen=True)
class SourceScore:
    score: ScoreData
    solo: tuple[NoteEvent, ...]
    accompaniment: tuple[NoteEvent, ...]
    solo_part_ids: tuple[str, ...]
    accompaniment_part_ids: tuple[str, ...]

    @property
    def duration(self) -> float:
        return self.score.duration

    @property
    def notes(self) -> tuple[NoteEvent, ...]:
        return tuple(sorted(self.solo + self.accompaniment, key=note_sort_key))
