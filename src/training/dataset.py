"""Fixed-work splits and a serializable structured-event vocabulary."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from torch.utils.data import Dataset

from src.models.conditioning import controls_to_vector, features_to_controls


SPECIAL_TOKENS = ("<PAD>", "<BOS>", "<EOS>", "<UNK>",
                  "<SOURCE_SOLO>", "<SOURCE_ACC>", "<TARGET_PIANO>")
TOKEN_FIELDS = ("source_solo_tokens", "source_acc_tokens", "target_piano_tokens")


class Vocabulary:
    """Stable special IDs and complete musical domains, including unseen pitches.

    Grid bounds below cover typical short segments, 32 quarter notes per bar and
    32-quarter-note tied durations. Longer/irregular training tokens are retained
    as observed extensions. Unknown inference events fail explicitly by default.
    """

    def __init__(self, tokens: Iterable[str] = (), *, complete: bool = True,
                 max_bars: int = 256, max_ticks: int = 768, max_voices: int = 64) -> None:
        values = list(SPECIAL_TOKENS)
        if complete:
            values.extend(f"<BAR_{i}>" for i in range(1, max_bars + 1))
            values.extend(f"<POS_{i}>" for i in range(max_ticks))
            values.extend(f"<ROLE_{role}>" for role in ("SOLO", "ACC", "PIANO", "UNKNOWN"))
            values.extend(f"<PITCH_{i}>" for i in range(128))
            values.extend(f"<DUR_{i}>" for i in range(max_ticks + 1))
            values.extend(f"<VELOCITY_BIN_{i}>" for i in range(16))
            values.extend(f"<VOICE_ID_{i}>" for i in range(max_voices))
        values.extend(sorted(set(tokens)))
        self.id_to_token = list(dict.fromkeys(values))
        self.token_to_id = {token: index for index, token in enumerate(self.id_to_token)}

    @property
    def stoi(self) -> dict[str, int]:
        return self.token_to_id

    @property
    def itos(self) -> list[str]:
        return self.id_to_token

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<PAD>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<EOS>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<UNK>"]

    def __len__(self) -> int:
        return len(self.id_to_token)

    def encode(self, tokens: Sequence[str], *, add_bos: bool = False,
               add_eos: bool = False, strict: bool = True) -> list[int]:
        result = []
        for token in tokens:
            if token not in self.token_to_id and strict:
                raise ValueError(f"Token {token!r} is outside this checkpoint's vocabulary")
            result.append(self.token_to_id.get(token, self.unk_id))
        return ([self.bos_id] if add_bos else []) + result + ([self.eos_id] if add_eos else [])

    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = True) -> list[str]:
        result = []
        for index in ids:
            if not 0 <= int(index) < len(self):
                raise ValueError(f"Invalid vocabulary ID {index}")
            token = self.id_to_token[int(index)]
            if not skip_special_tokens or token not in SPECIAL_TOKENS:
                result.append(token)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "tokens": self.id_to_token}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Vocabulary:
        tokens = payload.get("tokens", [])
        if payload.get("schema_version") != 1 or tokens[:len(SPECIAL_TOKENS)] != list(SPECIAL_TOKENS):
            raise ValueError("Invalid vocabulary schema or special token IDs")
        if len(tokens) != len(set(tokens)) or not all(isinstance(t, str) for t in tokens):
            raise ValueError("Vocabulary tokens must be unique strings")
        result = cls(complete=False)
        result.id_to_token = list(tokens)
        result.token_to_id = {token: index for index, token in enumerate(tokens)}
        return result

    @classmethod
    def from_segments(cls, segments: Sequence[dict[str, Any]], **kwargs: Any) -> Vocabulary:
        return cls((token for segment in segments for field in TOKEN_FIELDS
                    for token in segment[field]), **kwargs)


@dataclass
class PreparedDataset:
    segments: list[dict[str, Any]]
    metadata: dict[str, Any]
    path: Path | None = None

    def split(self, name: str) -> list[dict[str, Any]]:
        name = "validation" if name == "val" else name
        return [segment for segment in self.segments if segment["split"] == name]


def load_prepared_dataset(path: str | Path) -> PreparedDataset:
    """Reject work leakage before constructing any training batches."""
    path = Path(path).resolve()
    if path.is_dir():
        path = path / "dataset.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("segments"), list):
        raise ValueError("Expected a schema_version=1 prepared dataset with segments")
    seen_ids: set[str] = set()
    work_splits: dict[str, str] = {}
    segments = []
    for raw in payload["segments"]:
        segment = dict(raw)
        segment_id = segment.get("segment_id")
        work_id = segment.get("work_id")
        split = segment.get("split")
        if split == "val":
            split = "validation"
            segment["split"] = split
        if not isinstance(segment_id, str) or not segment_id or segment_id in seen_ids:
            raise ValueError("Each segment needs a unique non-empty segment_id")
        if not isinstance(work_id, str) or not work_id:
            raise ValueError(f"Segment {segment_id} needs work_id for leakage checks")
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Segment {segment_id} has invalid explicit split {split!r}")
        if work_id in work_splits and work_splits[work_id] != split:
            raise ValueError(f"Work {work_id!r} occurs in multiple splits; split by work, never by segment")
        work_splits[work_id] = split
        seen_ids.add(segment_id)
        for field in TOKEN_FIELDS:
            tokens = segment.get(field)
            if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
                raise ValueError(f"Segment {segment_id}: {field} must be a list of strings")
        if not (segment["source_solo_tokens"] or segment["source_acc_tokens"]) or not segment["target_piano_tokens"]:
            raise ValueError(f"Segment {segment_id} needs at least one non-empty source role and non-empty target musical events")
        segments.append(segment)
    if not segments:
        raise ValueError("Prepared dataset contains no aligned segments")
    return PreparedDataset(segments, {key: value for key, value in payload.items() if key != "segments"}, path)


class TranscriptionDataset(Dataset):
    def __init__(self, segments: Sequence[dict[str, Any]], vocab: Vocabulary,
                 *, max_sequence_length: int = 4096) -> None:
        self.segments = list(segments)
        self.vocab = vocab
        self.max_sequence_length = max_sequence_length
        self.items = [self._encode(segment) for segment in self.segments]

    def _encode(self, segment: dict[str, Any]) -> dict[str, Any]:
        solo = segment["source_solo_tokens"]
        acc = segment["source_acc_tokens"]
        target = segment["target_piano_tokens"]
        source = ["<SOURCE_SOLO>", *solo, "<SOURCE_ACC>", *acc]
        ids = {"source_ids": self.vocab.encode(source, add_bos=True, add_eos=True),
               "solo_ids": self.vocab.encode(solo, add_bos=True, add_eos=True),
               "acc_ids": self.vocab.encode(acc, add_bos=True, add_eos=True),
               "target_ids": self.vocab.encode(target, add_bos=True, add_eos=True)}
        for field, values in ids.items():
            if len(values) > self.max_sequence_length:
                raise ValueError(f"Segment {segment['segment_id']}: {field} length {len(values)} exceeds {self.max_sequence_length}; resegment, do not truncate musical events")
        explicit_controls = segment.get("technique_controls", segment.get("controls"))
        controls = (controls_to_vector(explicit_controls)
                    if explicit_controls is not None
                    else features_to_controls(segment.get("technique_features", {})))
        return {**ids, "segment_id": segment["segment_id"], "work_id": segment["work_id"],
                "controls": controls, "segment": segment}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]

