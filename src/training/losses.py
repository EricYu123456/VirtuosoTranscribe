"""Differentiable token and aligned melody-presence objectives."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def melody_presence_loss(logits: torch.Tensor, positions: torch.Tensor,
                         allowed_pitch_ids: torch.Tensor) -> torch.Tensor:
    """Negative log probability that an aligned onset contains the source pitch.

    At each target pitch slot, sum probabilities of exact/source +/- octave.
    Across simultaneous slots use ``1 - product(1 - p)``. This keeps gradients
    connected to the decoder; no sampled tokens, argmax, or decoded metrics enter
    the training loss. Padded slots and attacks are ignored.
    """
    if positions.numel() == 0:
        return logits.sum() * 0.0
    if positions.ndim != 3 or allowed_pitch_ids.shape[:2] != positions.shape[:2]:
        raise ValueError("Melody positions/pitch IDs must have shapes [B,M,K] and [B,M,3]")
    if positions.shape[0] != logits.shape[0]:
        raise ValueError("Melody supervision and logits batch sizes differ")
    valid_slots = positions >= 0
    valid_attacks = valid_slots.any(-1) & (allowed_pitch_ids >= 0).any(-1)
    if not valid_attacks.any():
        return logits.sum() * 0.0
    if (positions[valid_slots] >= logits.shape[1]).any():
        raise ValueError("Melody prediction position exceeds target sequence length")
    probabilities = logits.softmax(-1)
    rows = torch.arange(logits.shape[0], device=logits.device)[:, None, None]
    selected = probabilities[rows, positions.clamp_min(0)]
    ids = allowed_pitch_ids.clamp_min(0).unsqueeze(2).expand(-1, -1, positions.shape[2], -1)
    valid_ids = (allowed_pitch_ids >= 0).unsqueeze(2)
    present = (selected.gather(-1, ids) * valid_ids).sum(-1)
    present = torch.where(valid_slots, present, torch.zeros_like(present)).clamp(0, 1 - 1e-7)
    log_absence = torch.log1p(-present).sum(-1)
    presence = -torch.expm1(log_absence)
    return -presence[valid_attacks].clamp_min(1e-8).log().mean()


def transcription_loss(logits: torch.Tensor, labels: torch.Tensor, *, pad_id: int,
                       melody_positions: torch.Tensor | None = None,
                       melody_pitch_ids: torch.Tensor | None = None,
                       lambda_melody: float = 0.0,
                       label_smoothing: float = 0.0) -> dict[str, torch.Tensor]:
    token = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                            ignore_index=pad_id, label_smoothing=label_smoothing)
    melody = logits.sum() * 0.0
    if lambda_melody:
        if melody_positions is None or melody_pitch_ids is None:
            raise ValueError("Soft melody loss requires aligned onset/pitch supervision")
        melody = melody_presence_loss(logits, melody_positions, melody_pitch_ids)
    return {"loss": token + lambda_melody * melody,
            "token_loss": token, "melody_loss": melody}

