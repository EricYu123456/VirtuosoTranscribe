"""Validate structured token syntax and reconstruct score events."""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

from .events import MeasureInfo, NoteEvent, ScoreData, TupletInfo
from .timing import beat_position
from .tokenizer import BUNDLE_TYPE, DEFAULT_SUBDIVISIONS, ROLES, VELOCITY_BINS, _ticks, _validate_measures, velocity_bin


_NUMERIC = re.compile(r"^<([A-Z_]+)_(\d+)>$")


def _number(token: str, field: str) -> int:
    match = _NUMERIC.fullmatch(token) if isinstance(token, str) else None
    if match is None or match.group(1) != field:
        raise ValueError(f"Expected <{field}_integer>, received {token!r}")
    return int(match.group(2))


def _records(tokens: Sequence[str]) -> list[dict[str, Any]]:
    if not isinstance(tokens, (list, tuple)):
        raise ValueError("tokens must be a list or tuple of strings")
    records = []
    cursor = 0
    bar = 0
    position = -1
    pending_bar = False
    while cursor < len(tokens):
        token = tokens[cursor]
        if isinstance(token, str) and token.startswith("<BAR_"):
            next_bar = _number(token, "BAR")
            if next_bar <= bar:
                raise ValueError("BAR numbers must be positive and strictly increasing")
            if pending_bar:
                raise ValueError("A BAR must contain at least one complete event")
            bar, position, pending_bar = next_bar, -1, True
            cursor += 1
            continue
        if bar == 0:
            raise ValueError("A note event must be preceded by BAR")
        if cursor + 6 > len(tokens):
            raise ValueError("Incomplete note event")
        next_position = _number(tokens[cursor], "POS")
        if next_position < position:
            raise ValueError("POS values must be nondecreasing within a bar")
        role_token = tokens[cursor + 1]
        role = role_token[6:-1] if isinstance(role_token, str) and role_token.startswith("<ROLE_") and role_token.endswith(">") else None
        if role not in ROLES:
            raise ValueError(f"Invalid ROLE token: {role_token!r}")
        pitch = _number(tokens[cursor + 2], "PITCH")
        duration = _number(tokens[cursor + 3], "DUR")
        velocity = _number(tokens[cursor + 4], "VELOCITY_BIN")
        voice = _number(tokens[cursor + 5], "VOICE_ID")
        if pitch > 127:
            raise ValueError("PITCH must be in 0..127")
        if velocity >= VELOCITY_BINS:
            raise ValueError(f"VELOCITY_BIN must be in 0..{VELOCITY_BINS - 1}")
        records.append({"bar": bar, "position": next_position, "pitch": pitch,
                        "duration": duration, "role": role, "velocity_bin": velocity,
                        "voice_id": voice})
        position, pending_bar = next_position, False
        cursor += 6
    if pending_bar:
        raise ValueError("A BAR must contain at least one complete event")
    return records


def _make_notes(records: list[dict[str, Any]], measures: Iterable[MeasureInfo], subdivisions: int,
                voices: list[dict[str, str]] | None = None,
                sidecars: list[dict[str, Any]] | None = None) -> tuple[NoteEvent, ...]:
    measure_map = _validate_measures(measures, subdivisions)
    if sidecars is not None and len(sidecars) != len(records):
        raise ValueError("Token event count does not match note metadata")
    notes = []
    for index, item in enumerate(records):
        bar = item["bar"]
        if bar not in measure_map:
            raise ValueError(f"BAR {bar} has no measure metadata")
        measure = measure_map[bar]
        position = item["position"]
        if position >= _ticks(measure.duration, subdivisions, "measure duration"):
            raise ValueError(f"POS {position} falls outside BAR {bar}")
        onset = measure.onset + position / subdivisions
        duration = item["duration"] / subdivisions
        if voices is not None:
            if item["voice_id"] >= len(voices):
                raise ValueError("VOICE_ID does not exist in voice_table")
            voice_data = voices[item["voice_id"]]
        else:
            voice_data = {"part_id": "piano", "voice": str(item["voice_id"] + 1)}
        data = {"pitch": item["pitch"], "onset": onset, "duration": duration,
                "measure": bar, "beat": beat_position(measure, onset),
                "part_id": voice_data["part_id"], "voice": voice_data["voice"],
                "role": item["role"], "velocity": min(127, item["velocity_bin"] * 8 + 4)}
        if sidecars is not None:
            metadata = dict(sidecars[index])
            metadata["tuplets"] = tuple(TupletInfo(**x) for x in metadata.get("tuplets", []))
            for name in ("articulations", "slur_ids"):
                metadata[name] = tuple(metadata.get(name, []))
            for name in ("onset", "duration"):
                if _ticks(metadata[name], subdivisions, f"original {name}") == _ticks(data[name], subdivisions, name):
                    data[name] = metadata[name]
            if velocity_bin(metadata["velocity"]) == item["velocity_bin"]:
                data["velocity"] = metadata["velocity"]
            if metadata["onset"] == data["onset"] and metadata["measure"] == bar:
                data["beat"] = metadata["beat"]
            metadata.update(data)
            data = metadata
        if duration == 0 and not data.get("is_grace", False):
            raise ValueError("DUR_0 requires explicit grace-note metadata")
        notes.append(NoteEvent(**data))
    return tuple(notes)


def detokenize_tokens(tokens: Sequence[str], *, measures: Iterable[MeasureInfo] | None = None,
                      subdivisions: int = DEFAULT_SUBDIVISIONS) -> tuple[NoteEvent, ...]:
    """Decode a training stream; omitted measure metadata assumes full 4/4 bars.

    Bare streams cannot reconstruct pickup bars, meter changes, exact velocities,
    instrument identity, or notation. Use detokenize_score for archival bundles.
    """
    _ticks(0, subdivisions, "grid")
    records = _records(tokens)
    if measures is None:
        measures = [MeasureInfo(number, (number - 1) * 4, 4, original_number=str(number))
                    for number in sorted({item["bar"] for item in records})]
    return _make_notes(records, measures, subdivisions)


def detokenize_score(bundle: dict[str, Any]) -> ScoreData:
    """Validate and reconstruct the score represented by an archival bundle."""
    if not isinstance(bundle, dict) or bundle.get("type") != BUNDLE_TYPE or bundle.get("schema_version") != 1:
        raise ValueError("Unsupported token bundle schema")
    if bundle.get("velocity_bins") != VELOCITY_BINS:
        raise ValueError("Unsupported velocity bin count")
    try:
        subdivisions = bundle["subdivisions"]
        _ticks(0, subdivisions, "grid")
        score = ScoreData.from_dict(dict(bundle["score_metadata"], notes=[]))
        records = _records(bundle["tokens"])
        voices = bundle["voice_table"]
        sidecars = bundle["note_metadata"]
        if not isinstance(voices, list) or not isinstance(sidecars, list):
            raise ValueError("voice_table and note_metadata must be lists")
        notes = _make_notes(records, score.measures, subdivisions, voices, sidecars)
        return score.with_notes(notes)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Incomplete or invalid token bundle: {exc}") from exc
