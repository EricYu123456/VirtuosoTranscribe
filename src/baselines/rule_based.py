"""Technique-controlled piano writing without melody or harmony recomposition."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from collections import defaultdict
import math
from pathlib import Path
import random

from src.baselines._piano import as_piano, bass_indices, can_add, finish, infer_hand, piano_pitch, pitch_choices, same_event, solo_notes
from src.representation.events import NoteEvent, ScoreData, SourceScore
from src.representation.timing import beat_position


@dataclass(frozen=True)
class ArrangementControls:
    melody_preservation: float = 1.0
    octave_doubling: float = 0.0
    register_expansion: float = 0.0
    chord_thickening: float = 0.0
    bass_reinforcement: float = 0.0
    inner_voice_density: float = 0.35
    figuration_density: float = 0.0
    countermelody: float = 0.0
    ornamentation: float = 0.0
    harmonic_freedom: float = 0.0
    rhythmic_freedom: float = 0.0
    virtuosity: float = 0.25
    texture_density: float = 0.35

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{field.name} must be a finite number in 0..1")
        unsupported = [name for name in ("countermelody", "ornamentation",
                                        "harmonic_freedom", "rhythmic_freedom") if getattr(self, name) != 0]
        if unsupported:
            raise ValueError("V1 rule-based generation does not implement nonzero " + ", ".join(unsupported))
        if self.melody_preservation != 1:
            raise ValueError("V1 rule-based generation requires melody_preservation=1; melody rewriting is unsupported")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


PRESETS = {
    "faithful": ArrangementControls(),
    "romantic": ArrangementControls(octave_doubling=.55, register_expansion=.65,
                                    chord_thickening=.4, bass_reinforcement=.5,
                                    inner_voice_density=.5, virtuosity=.5, texture_density=.6),
    "virtuosic": ArrangementControls(octave_doubling=.85, register_expansion=.9,
                                     chord_thickening=.65, bass_reinforcement=.8,
                                     inner_voice_density=.65, virtuosity=.9, texture_density=.9),
    "concert_transcription": ArrangementControls(octave_doubling=.7, register_expansion=.75,
                                                 chord_thickening=.5, bass_reinforcement=.65,
                                                 inner_voice_density=.55, virtuosity=.7, texture_density=.75),
}


def load_preset(preset: str | Path | ArrangementControls = "faithful") -> ArrangementControls:
    if isinstance(preset, ArrangementControls):
        return preset
    name = str(preset).lower()
    if name in PRESETS:
        return PRESETS[name]
    path = Path(preset)
    if not path.is_file():
        raise ValueError(f"Unknown preset {preset!r}; available: {', '.join(PRESETS)}")
    import yaml
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid preset YAML in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("A preset must contain a mapping of control names to values")
    payload = dict(payload)
    payload.pop("preset", None)
    valid = {f.name for f in fields(ArrangementControls)}
    unknown = set(payload) - valid
    if unknown:
        raise ValueError(f"Unknown arrangement controls: {', '.join(sorted(unknown))}")
    return ArrangementControls(**payload)


def _selected(count: int, amount: float, rng: random.Random) -> set[int]:
    if not count or amount == 0:
        return set()
    indices = list(range(count))
    rng.shuffle(indices)
    return set(indices[:max(1, round(count * amount))])


def _figured_source(source: SourceScore, amount: float, rng: random.Random) -> SourceScore:
    """Rewrite selected sustained inner chord tones as eighth-note figures.

    Original bass events and the complete solo remain unchanged. The figure
    uses only source chord pitches, with no harmonic or melodic substitutions.
    """
    if amount == 0:
        return source
    bass = bass_indices(source.accompaniment)
    groups = defaultdict(list)
    for index, event in enumerate(source.accompaniment):
        if index not in bass and not event.is_grace and event.duration >= 1:
            groups[(event.onset, event.duration)].append(index)
    eligible = [(key, indices) for key, indices in sorted(groups.items())
                if len({source.accompaniment[i].pitch for i in indices}) >= 2]
    selected = _selected(len(eligible), amount, rng)
    removed = set()
    additions = []
    for group_index in sorted(selected):
        (onset, length), indices = eligible[group_index]
        notes = sorted((source.accompaniment[i] for i in indices), key=lambda n: n.pitch)
        removed.update(indices)
        position = onset
        index = 0
        while position < onset + length - 1e-7:
            original = notes[index % len(notes)]
            owner = next(m for m in source.score.measures
                         if m.onset <= position < m.onset + m.duration)
            additions.append(replace(original, onset=position,
                                     duration=min(.5, onset + length - position),
                                     measure=owner.number, beat=beat_position(owner, position),
                                     tuplets=(), articulations=(), slur_ids=(), voice="figuration"))
            position += .5
            index += 1
    accompaniment = tuple(n for i, n in enumerate(source.accompaniment) if i not in removed) + tuple(additions)
    score = source.score.with_notes(source.solo + accompaniment)
    return replace(source, score=score, accompaniment=tuple(sorted(accompaniment, key=lambda n: (n.onset, n.pitch))))


def arrange_rule_based(source: SourceScore,
                       controls: ArrangementControls | str | Path = "faithful", *,
                       seed: int = 0) -> ScoreData:
    """Preserve solo attacks/durations and harmony while allocating two hands.

    Added notes are octave-equivalent source tones. Hand constraints are
    checked at all simultaneous sounding boundaries, including sustained notes.
    Protected polyphonic solo material takes precedence over heuristic caps;
    an explicit warning reports any remaining protected-solo difficulty.
    """
    controls = load_preset(controls)
    if not source.solo:
        raise ValueError("A piano arrangement requires a non-empty solo role")
    rng = random.Random(seed)
    source = _figured_source(source, controls.figuration_density, rng)
    max_span = 12 if controls.virtuosity < .8 else 16
    max_hand = 3 + round(2 * controls.texture_density)
    max_total = 4 + round(6 * controls.texture_density)
    limits = dict(max_span=max_span, max_per_hand=max_hand, max_total=max_total)
    warnings: list[str] = []
    median = sorted(n.pitch for n in source.solo)[len(source.solo) // 2]
    shift = 0
    while median + shift < 60:
        shift += 12
    while median + shift > 84:
        shift -= 12
    melodies, merged = solo_notes(source, shift)
    allocated = list(melodies)
    if merged:
        warnings.append(f"Merged {merged} coincident solo unison events into single piano keys.")
    for index, melody in enumerate(allocated):
        if not can_add(melody, allocated[:index], **limits):
            warnings.append("Protected solo contains overlapping unisons or exceeds piano hand limits; retained for melodic fidelity.")
            break
    bass = bass_indices(source.accompaniment)
    bass_added: list[NoteEvent] = []
    omitted = 0

    def allocate(original: NoteEvent, hand: str, preferred: int | None = None) -> NoteEvent | None:
        note = as_piano(original, hand, role="ACC")
        if any(same_event(note, previous) for previous in allocated):
            return note  # Already represented by a sounding piano key.
        for pitch in pitch_choices(note, allocated, preferred=preferred):
            hands = (hand, "lh" if hand == "rh" else "rh")
            for candidate_hand in hands:
                candidate = as_piano(note, candidate_hand, pitch)
                if can_add(candidate, allocated, **limits):
                    allocated.append(candidate)
                    return candidate
        return None

    for index in sorted(bass, key=lambda i: (source.accompaniment[i].onset, source.accompaniment[i].pitch)):
        original = source.accompaniment[index]
        preferred = piano_pitch(original.pitch)
        if controls.register_expansion > .5 and preferred >= 40:
            preferred -= 12
        note = allocate(original, "lh", preferred)
        if note is not None:
            bass_added.append(note)
        else:
            omitted += 1

    # Reserve capacity for audible melody doubling before adding inner voices.
    selected = _selected(len(melodies), controls.octave_doubling, rng)
    doubles = 0
    for index in sorted(selected):
        melody = melodies[index]
        for pitch in (melody.pitch - 12, melody.pitch + 12):
            if not 21 <= pitch <= 108:
                continue
            double = as_piano(melody, "rh", pitch, "MELODY_DOUBLE")
            if can_add(double, allocated, **limits):
                allocated.append(double)
                doubles += 1
                break
    if selected and not doubles:
        warnings.append("Requested melody octave doubling had no playable free hand capacity.")

    for index in sorted(_selected(len(bass_added), controls.bass_reinforcement, rng)):
        bass_note = bass_added[index]
        for pitch in (bass_note.pitch - 12, bass_note.pitch + 12):
            if 21 <= pitch <= 108:
                double = as_piano(bass_note, "lh", pitch, "BASS_DOUBLE")
                if can_add(double, allocated, **limits):
                    allocated.append(double)
                    break

    inner = [i for i in range(len(source.accompaniment)) if i not in bass]
    # Density controls permit 35%-100% of existing inner material; optional
    # added chord voices use only pitches already supplied by the harmony.
    amount = .35 + .65 * controls.inner_voice_density
    retained = _selected(len(inner), amount, rng)
    omitted += len(inner) - len(retained)
    original_added = []
    for position in sorted(retained, key=lambda i: (source.accompaniment[inner[i]].onset,
                                                    source.accompaniment[inner[i]].pitch)):
        original = source.accompaniment[inner[position]]
        note = allocate(original, infer_hand(original, source))
        if note is None:
            omitted += 1
        else:
            original_added.append(note)
    for index in sorted(_selected(len(original_added), controls.chord_thickening, rng)):
        note = original_added[index]
        for pitch in (note.pitch + 12, note.pitch - 12):
            if 21 <= pitch <= 108:
                thick = as_piano(note, infer_hand(note, source), pitch, "CHORD_DOUBLE")
                if can_add(thick, allocated, **limits):
                    allocated.append(thick)
                    break
    if omitted:
        warnings.append(f"Pianistic reduction omitted {omitted} accompaniment events to honor density, collision, or span limits.")
    return finish(source, allocated, warnings, "rule-based piano transcription")

