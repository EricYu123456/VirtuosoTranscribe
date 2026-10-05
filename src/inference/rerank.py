"""Transparent heuristic reranking of generated piano candidates."""

from __future__ import annotations

from typing import Mapping

from src.baselines.rule_based import ArrangementControls, load_preset
from src.representation.events import ScoreData, SourceScore


def rank_candidates(source: SourceScore, candidates: list[ScoreData],
                    controls: ArrangementControls | str | Mapping[str, float] = "faithful", *,
                    content_scorer=None) -> list[dict]:
    if not candidates:
        raise ValueError("Cannot rank an empty candidate list")
    if isinstance(controls, Mapping):
        from src.inference.neural_generate import load_neural_controls
        requested = load_neural_controls(controls)
    else:
        requested = load_preset(controls).to_dict()
    from src.analysis.technique_features import analyze_transcription
    result = []
    for index, candidate in enumerate(candidates):
        metrics = analyze_transcription(source=source, target=candidate)
        melody = float(metrics["melody_coverage_octave_tolerant"])
        exact = float(metrics["melody_coverage_exact"])
        playable = float(metrics["playability_score"])
        harmony = float(metrics["harmonic_pitch_class_overlap"])
        doubling_match = 1 - abs(float(metrics["octave_doubling_rate"]) - requested["octave_doubling"])
        # Exact pitch is a secondary bonus: it helps distinguish a retained
        # melody from an octave-equivalent bass at the same attack, while
        # deliberate octave relocation still receives the primary coverage.
        observed = _observed_controls(metrics)
        technique_match = sum(1 - abs(observed[name] - requested[name])
                              for name in observed) / len(observed)
        components = {"melody_preservation": melody, "exact_pitch_coverage": exact,
                      "harmonic_consistency": harmony, "playability": playable,
                      "technique_match": technique_match}
        weights = {"melody_preservation": .35, "exact_pitch_coverage": .1,
                   "harmonic_consistency": .2, "playability": .2, "technique_match": .15}
        if content_scorer is not None:
            content = float(content_scorer(source.score.with_notes(source.notes), candidate))
            if not 0 <= content <= 1:
                raise ValueError("Content similarity scorer must return a finite number in 0..1")
            components["musicbert_content_similarity"] = content
            weights.update(melody_preservation=.3, harmonic_consistency=.15,
                           musicbert_content_similarity=.1)
        weighted = sum(weights[name] * value for name, value in components.items())
        result.append({"candidate_index": index, "score": weighted,
                       "score_weights": weights, "score_components": components,
                       "octave_doubling_match": doubling_match,
                       "technique_observations": observed,
                       "technique_observation_limits": "Normalized heuristic proxies; countermelody and ornamentation require annotations and are excluded from technique matching.",
                       "metrics": metrics})
    return sorted(result, key=lambda row: (-row["score"], row["candidate_index"]))


rerank_candidates = rank_candidates


def _observed_controls(metrics: dict) -> dict[str, float]:
    """Bounded, documented proxies; never claim an untrained style classifier."""
    def clip(value):
        return max(0., min(1., float(value)))
    texture = metrics["target_texture"]
    density = texture["notes_per_beat"]
    return {
        "melody_preservation": clip(metrics["melody_coverage_octave_tolerant"]),
        "octave_doubling": clip(metrics["octave_doubling_rate"]),
        "register_expansion": clip(metrics["register_expansion_semitones"] / 24),
        "chord_thickening": clip(metrics["chordal_melody_rate"]),
        "bass_reinforcement": clip(metrics["bass_octave_attack_ratio"]),
        "inner_voice_density": clip(metrics["inner_voice_notes_per_beat"] / 4),
        "figuration_density": clip(metrics["broken_chord_candidate_count"] / max(1, density * 4)),
        "harmonic_freedom": clip(1 - metrics["harmonic_pitch_class_overlap"]),
        "rhythmic_freedom": clip(metrics["rhythmic_alteration_ratio"]),
        "virtuosity": clip((texture["average_simultaneous_notes"] / 10 + density / 8) / 2),
        "texture_density": clip(density / 8),
    }

