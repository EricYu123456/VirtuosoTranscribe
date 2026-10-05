"""Deterministic symbolic evaluation; these proxies require human validation."""

from .playability_metrics import evaluate_playability

__all__ = ["evaluate_playability"]
