"""Dataset-facing alignment entry points."""

from src.analysis.melody_alignment import AlignmentResult, align_measures, align_solo_to_piano

__all__ = ["AlignmentResult", "align_measures", "align_solo_to_piano"]
