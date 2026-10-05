"""Continuous, interpretable controls shared by training and inference."""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import nn


CONTROL_NAMES = (
    "melody_preservation", "octave_doubling", "register_expansion",
    "chord_thickening", "bass_reinforcement", "inner_voice_density",
    "figuration_density", "countermelody", "ornamentation", "harmonic_freedom",
    "rhythmic_freedom", "virtuosity", "texture_density",
)


def controls_to_vector(controls: Mapping[str, float] | object) -> list[float]:
    """Validate neural controls without imposing the deterministic baseline's limits."""
    if hasattr(controls, "to_dict"):
        controls = controls.to_dict()
    if not isinstance(controls, Mapping):
        raise ValueError("Technique controls must be a mapping")
    unknown = set(controls) - set(CONTROL_NAMES)
    if unknown:
        raise ValueError(f"Unknown technique controls: {sorted(unknown)}")
    result = []
    for name in CONTROL_NAMES:
        value = controls.get(name, 1.0 if name == "melody_preservation" else 0.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be a finite number in 0..1")
        result.append(float(value))
    return result


def features_to_controls(features: Mapping[str, object]) -> list[float]:
    """Map measured techniques to bounded proxy labels, retaining their meaning.

    These labels describe target texture; they are not arranger annotations.
    Explicit ``technique_controls`` in a segment takes precedence in the dataset.
    """
    def numeric(name: str, default: float = 0.0) -> float:
        value = features.get(name, default)
        return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else default

    texture = features.get("target_texture", {})
    if not isinstance(texture, Mapping):
        texture = {}
    polyphony = float(texture.get("average_simultaneous_notes", 0.0) or 0.0)
    density = float(texture.get("notes_per_beat", 0.0) or 0.0)
    def clamp(value: float) -> float:
        return max(0.0, min(1.0, value))
    mapping = {
        "melody_preservation": clamp(numeric("melody_coverage_octave_tolerant", 1.0)),
        "octave_doubling": clamp(numeric("octave_doubling_rate")),
        "register_expansion": clamp(numeric("register_expansion_semitones") / 24),
        "chord_thickening": clamp(numeric("chordal_melody_rate")),
        "bass_reinforcement": clamp(numeric("bass_octave_attack_ratio")),
        "inner_voice_density": clamp(numeric("inner_voice_notes_per_beat") / 4),
        "figuration_density": clamp(numeric("broken_chord_candidate_count") / max(1, density * 4)),
        "countermelody": 0.0,
        "ornamentation": 0.0,
        "harmonic_freedom": clamp(1 - numeric("harmonic_pitch_class_overlap", 1.0)),
        "rhythmic_freedom": clamp(numeric("rhythmic_alteration_ratio")),
        "virtuosity": clamp((polyphony / 10 + density / 8) / 2),
        "texture_density": clamp(density / 8),
    }
    return controls_to_vector(mapping)


class TechniqueConditioning(nn.Module):
    """Encode all thirteen continuous controls as a learned memory prefix."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(len(CONTROL_NAMES), d_model),
                                        nn.GELU(), nn.Linear(d_model, d_model))

    def forward(self, controls: torch.Tensor) -> torch.Tensor:
        if controls.ndim != 2 or controls.shape[-1] != len(CONTROL_NAMES):
            raise ValueError(f"controls must have shape [batch, {len(CONTROL_NAMES)}]")
        if not torch.isfinite(controls).all() or (controls < 0).any() or (controls > 1).any():
            raise ValueError("Technique controls must be finite and within 0..1")
        return self.projection(controls)

