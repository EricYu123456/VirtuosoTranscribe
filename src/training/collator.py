"""Teacher-forcing shifts, padding masks, and onset-aligned melody supervision."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from .dataset import Vocabulary


def _events(tokens: list[str]) -> list[tuple[int, int, int, int]]:
    """Return (bar, grid position, pitch, prediction index) in token-note order."""
    result = []
    bar = 1
    position = 0
    for index, token in enumerate(tokens):
        if token.startswith("<BAR_"):
            bar = int(token[5:-1])
        elif token.startswith("<POS_"):
            position = int(token[5:-1])
        elif token.startswith("<PITCH_"):
            result.append((bar, position, int(token[7:-1]), index))
    return result


def melody_supervision(segment: dict[str, Any], vocab: Vocabulary) -> list[tuple[list[int], list[int]]]:
    """Pool target pitch slots at the aligned source attack, without hard argmax.

    When no target pitch slot exists at an onset, teacher forcing has no event
    position to supervise. Such attacks remain measurable by generation metrics
    and enforceable by hard decoding. The soft loss does not invent time slots.
    """
    source = _events(segment["source_solo_tokens"])
    target = _events(segment["target_piano_tokens"])
    groups: dict[tuple[int, int], list[int]] = {}
    for bar, position, _, index in target:
        groups.setdefault((bar, position), []).append(index)
    # Segmentation rebases measure indices and aligned onsets to the same local
    # grid. The inspected alignment additionally supplies target note indices
    # for excerpt pairs with a nontrivial beat/time mapping.
    matches = segment.get("alignment", {}).get("matches", [])
    by_source = {m["source_index"]: m for m in matches}
    result = []
    for source_index, (bar, position, pitch, _) in enumerate(source):
        match = by_source.get(source_index)
        if match is not None and 0 <= match["target_index"] < len(target):
            bar, position = target[match["target_index"]][:2]
        positions = groups.get((bar, position), [])
        if positions:
            pitches = [vocab.token_to_id[f"<PITCH_{value}>"]
                       for value in (pitch, pitch - 12, pitch + 12) if 0 <= value <= 127]
            result.append((positions, pitches))
    return result


class TranscriptionCollator:
    def __init__(self, vocab: Vocabulary) -> None:
        self.vocab = vocab

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            raise ValueError("Cannot collate an empty batch")

        def pad(field: str) -> torch.Tensor:
            return pad_sequence([torch.tensor(item[field], dtype=torch.long) for item in items],
                                batch_first=True, padding_value=self.vocab.pad_id)

        target = pad("target_ids")
        batch = {"source_ids": pad("source_ids"), "solo_ids": pad("solo_ids"),
                 "acc_ids": pad("acc_ids"), "target_input_ids": target[:, :-1],
                 "labels": target[:, 1:], "controls": torch.tensor([x["controls"] for x in items], dtype=torch.float32),
                 "segment_ids": [x["segment_id"] for x in items]}
        for field in ("source", "solo", "acc"):
            batch[f"{field}_padding_mask"] = batch[f"{field}_ids"].eq(self.vocab.pad_id)
        batch["target_padding_mask"] = batch["target_input_ids"].eq(self.vocab.pad_id)
        supervision = [melody_supervision(item["segment"], self.vocab) for item in items]
        count = max((len(x) for x in supervision), default=0)
        slots = max((len(positions) for groups in supervision for positions, _ in groups), default=1)
        positions = torch.full((len(items), count, slots), -1, dtype=torch.long)
        pitches = torch.full((len(items), count, 3), -1, dtype=torch.long)
        for row, groups in enumerate(supervision):
            for group, (offsets, allowed) in enumerate(groups):
                positions[row, group, :len(offsets)] = torch.tensor(offsets)
                pitches[row, group, :len(allowed)] = torch.tensor(allowed)
        batch["melody_positions"] = positions
        batch["melody_pitch_ids"] = pitches
        return batch


def model_inputs(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {name: batch[name] for name in ("source_ids", "target_input_ids", "source_padding_mask",
            "target_padding_mask", "solo_ids", "acc_ids", "solo_padding_mask", "acc_padding_mask", "controls")}

