"""Convenience evaluation entry point for dataset and CLI callers."""

from __future__ import annotations

from src.representation.events import ScoreData, SourceScore


def evaluate_pair(source: SourceScore, target: ScoreData, alignment=None) -> dict:
    from src.analysis.technique_features import analyze_transcription
    return analyze_transcription(source, target, alignment)
