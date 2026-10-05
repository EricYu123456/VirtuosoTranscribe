"""Autoregressive Transformer generation with a validated symbolic grammar."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import pickle
from pathlib import Path
from typing import Any, Mapping

from src.baselines.rule_based import ArrangementControls, load_preset
from src.models.constraints import GenerationError, GrammarState, PianoTokenGrammar
from src.representation.detokenizer import detokenize_tokens
from src.representation.events import DynamicEvent, PartInfo, ScoreData, SourceScore
from src.representation.tokenizer import DEFAULT_SUBDIVISIONS, tokenize_source


def load_neural_controls(controls: ArrangementControls | str | Path | Mapping[str, float] = "faithful") -> dict[str, float]:
    """All continuous dimensions are valid neural inputs, including V1 freedoms.

    Baseline operators retain their own narrower validation. A control is a
    conditioning request; its effect depends on the checkpoint's training data.
    """
    from src.models.conditioning import CONTROL_NAMES, controls_to_vector
    if isinstance(controls, Mapping):
        values = dict(controls)
    elif isinstance(controls, ArrangementControls):
        values = controls.to_dict()
    elif Path(str(controls)).is_file():
        import yaml
        values = yaml.safe_load(Path(controls).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("A neural control preset must contain a mapping")
        values = dict(values)
        values.pop("preset", None)
    else:
        values = load_preset(controls).to_dict()
    defaults = {name: 1.0 if name == "melody_preservation" else 0.0 for name in CONTROL_NAMES}
    unknown = set(values) - set(CONTROL_NAMES)
    if unknown:
        raise ValueError("Unknown arrangement controls: " + ", ".join(sorted(unknown)))
    defaults.update(values)
    vector = controls_to_vector(defaults)
    return {name: float(value) for name, value in zip(CONTROL_NAMES, vector)}


@dataclass
class NeuralCandidate:
    score: ScoreData
    tokens: list[str]
    report: dict[str, Any]


@dataclass
class _Beam:
    ids: list[int]
    state: GrammarState
    log_probability: float = 0.0
    model_log_probability: float = 0.0


def tokens_to_piano_score(source: SourceScore, tokens: list[str], *,
                          subdivisions: int = DEFAULT_SUBDIVISIONS) -> ScoreData:
    """Decode generated events against the complete source meter and form."""
    notes = detokenize_tokens(tokens, measures=source.score.measures, subdivisions=subdivisions)
    if not notes:
        raise GenerationError("Learned generation produced no piano events")
    from dataclasses import replace
    allocated = []
    for note in notes:
        if note.role != "PIANO":
            raise GenerationError("Generated output contains a non-piano role")
        # VOICE_ID remains intact; register assigns an explicit notated staff.
        # A checkpoint may contain one or many source target voices.
        allocated.append(replace(note, part_id="piano_rh" if note.pitch >= 60 else "piano_lh"))
    dynamics = tuple(DynamicEvent(event.onset, event.value,
                                 "piano_rh" if event.part_id in source.solo_part_ids else "piano_lh")
                     for event in source.score.dynamics)
    return source.score.with_notes(
        allocated,
        parts=(PartInfo("piano_rh", "Piano RH", "Piano", 0),
               PartInfo("piano_lh", "Piano LH", "Piano", 0)),
        dynamics=dynamics, title=f"{source.score.title} — learned piano transcription")


def _distribution(model, encoded, beam: _Beam, grammar: PianoTokenGrammar, *,
                  device: str, temperature: float):
    import torch
    prefix = torch.tensor([beam.ids], dtype=torch.long, device=device)
    output = model.decode(prefix, encoded)
    if not isinstance(output, torch.Tensor) or output.ndim != 3 or output.shape[:2] != prefix.shape or output.shape[2] != len(grammar.tokens):
        raise GenerationError("Checkpoint decoder must return logits with shape [batch, prefix_length, vocabulary_size]")
    logits = output[0, -1].float()
    grammar.prepare_step(beam.state, int(logits.argmax()))
    allowed = grammar.allowed_ids(beam.state)
    if not allowed:
        raise GenerationError(f"No valid token in grammar phase {beam.state.phase}")
    indices = torch.tensor(sorted(allowed), dtype=torch.long, device=device)
    values = logits[indices]
    if torch.isnan(logits).any() or torch.isposinf(logits).any() or not torch.isfinite(values).all():
        raise GenerationError("Checkpoint returned non-finite logits for valid musical tokens")
    model_log_probs = torch.log_softmax(logits, dim=-1)
    adjusted = values.clone()
    for index, bias in grammar.soft_bias(beam.state).items():
        match = (indices == index).nonzero()
        if match.numel():
            adjusted[match[0, 0]] += bias
    if temperature > 0:
        adjusted /= temperature
    log_probs = torch.log_softmax(adjusted, dim=-1)
    return indices, log_probs, model_log_probs


def _sample_index(log_probs, *, generator, top_p: float):
    import torch
    probabilities = log_probs.exp()
    order = probabilities.argsort(descending=True)
    sorted_probs = probabilities[order]
    excluded = sorted_probs.cumsum(0) - sorted_probs >= top_p
    sorted_probs[excluded] = 0
    sorted_probs /= sorted_probs.sum()
    return int(order[torch.multinomial(sorted_probs, 1, generator=generator)])


def decode_with_model(model, vocabulary, source: SourceScore, *,
                      controls: Mapping[str, float] | None = None,
                      num_candidates: int = 1, seed: int = 0, device: str = "cpu",
                      melody_mode: str = "hard", subdivisions: int = DEFAULT_SUBDIVISIONS,
                      max_new_tokens: int = 2048, temperature: float = .8,
                      top_p: float = .95, beam_width: int = 1,
                      soft_strength: float = 2.0, num_threads: int = 2) -> list[NeuralCandidate]:
    """Decode model logits, with deterministic sampling or optional beam search.

    No targets, memorized segment tables or baseline arrangers are consulted.
    Exhausting the token budget raises a validation error rather than silently
    truncating an event, appending a missing melody, or exporting partial music.
    """
    import torch
    if isinstance(num_threads, bool) or not isinstance(num_threads, int) or num_threads < 1:
        raise ValueError("num_threads must be a positive integer")
    torch.set_num_threads(num_threads)
    if isinstance(num_candidates, bool) or not isinstance(num_candidates, int) or num_candidates < 1:
        raise ValueError("num_candidates must be a positive integer")
    if isinstance(beam_width, bool) or not isinstance(beam_width, int) or beam_width < 1:
        raise ValueError("beam_width must be a positive integer")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 8:
        raise ValueError("max_new_tokens must be an integer of at least 8")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    controls = load_neural_controls(controls or "faithful")
    grammar = PianoTokenGrammar(vocabulary.token_to_id, source,
                                melody_mode=melody_mode, subdivisions=subdivisions,
                                soft_strength=soft_strength,
                                melody_preservation=controls["melody_preservation"])
    bundles = tokenize_source(source, subdivisions=subdivisions)
    solo_tokens = bundles["source_solo"]["tokens"]
    acc_tokens = bundles["source_acc"]["tokens"]
    source_ids = vocabulary.encode(["<SOURCE_SOLO>"] + solo_tokens + ["<SOURCE_ACC>"] + acc_tokens,
                                   add_bos=True, add_eos=True)
    solo_ids = vocabulary.encode(solo_tokens, add_bos=True, add_eos=True)
    acc_ids = vocabulary.encode(acc_tokens, add_bos=True, add_eos=True)
    config = getattr(model, "config", None)
    maximum = getattr(config, "max_sequence_length", 4096)
    if max(len(source_ids), len(solo_ids), len(acc_ids)) > maximum:
        raise GenerationError(f"Source exceeds checkpoint context length {maximum}; segment it before generation")
    max_new_tokens = min(max_new_tokens, maximum - 1)
    from src.models.conditioning import controls_to_vector
    controls_tensor = torch.tensor([controls_to_vector(controls)], dtype=torch.float32, device=device)
    def tensor(ids):
        return torch.tensor([ids], dtype=torch.long, device=device)
    model = model.to(device)
    model.eval()
    results = []
    with torch.no_grad():
        encoded = model.encode(tensor(source_ids), solo_ids=tensor(solo_ids),
                               acc_ids=tensor(acc_ids),
                               controls=controls_tensor if getattr(config, "conditioned", False) else None)
        search_width = max(beam_width, num_candidates) if beam_width > 1 else 1
        for candidate_index in range(num_candidates if beam_width == 1 else 1):
            generator = torch.Generator(device=device).manual_seed(seed + candidate_index)
            beams = [_Beam([vocabulary.bos_id], GrammarState())]
            for _ in range(max_new_tokens):
                expansions: list[_Beam] = []
                for beam in beams:
                    if beam.state.completed:
                        expansions.append(beam)
                        continue
                    indices, log_probs, raw_probs = _distribution(
                        model, encoded, beam, grammar, device=device,
                        temperature=temperature if beam_width == 1 else 1.0)
                    if beam_width > 1:
                        selected = log_probs.topk(min(search_width, len(indices))).indices.tolist()
                    elif temperature == 0:
                        selected = [int(log_probs.argmax())]
                    else:
                        selected = [_sample_index(log_probs, generator=generator, top_p=top_p)]
                    for selected_index in selected:
                        token_id = int(indices[selected_index])
                        state = deepcopy(beam.state)
                        grammar.advance(state, token_id)
                        expansions.append(_Beam(beam.ids + [token_id], state,
                                                beam.log_probability + float(log_probs[selected_index]),
                                                beam.model_log_probability + float(raw_probs[token_id])))
                beams = sorted(expansions, key=lambda item: item.log_probability, reverse=True)[:search_width]
                if all(beam.state.completed for beam in beams):
                    break
            completed = [beam for beam in beams if beam.state.completed]
            if not completed:
                state = beams[0].state
                raise GenerationError(f"Learned decoder exhausted {max_new_tokens} tokens without EOS "
                                      f"(phase={state.phase}, melody={state.required_events}/{len(grammar.requirements)}); "
                                      "increase the budget, segment the input, or train the checkpoint further")
            wanted = num_candidates if beam_width > 1 else 1
            if len(completed) < wanted:
                raise GenerationError(f"Beam decoder completed {len(completed)}/{wanted} requested candidates within {max_new_tokens} tokens")
            for beam_rank, best in enumerate(sorted(completed, key=lambda item: item.log_probability, reverse=True)[:wanted]):
                tokens = vocabulary.decode(best.ids)
                score = tokens_to_piano_score(source, tokens, subdivisions=subdivisions)
                report = {
                    "seed": seed + candidate_index,
                    "melody_mode": melody_mode,
                    "generated_token_count": len(best.ids) - 1,
                    "generated_event_count": best.state.event_count,
                    "autoregressive_log_probability": best.model_log_probability,
                    "constrained_log_probability": best.log_probability,
                    "constraint_forced_token_steps": best.state.forced_steps,
                    "hard_required_melody_events": len(grammar.requirements) if melody_mode == "hard" else 0,
                    "hard_emitted_melody_events": best.state.required_events,
                    "projection_repairs": 0,
                    "post_generation_projection": False,
                    "controls_conditioned": bool(getattr(config, "conditioned", False)),
                    "decoding": "beam" if beam_width > 1 else ("greedy" if temperature == 0 else "nucleus_sampling"),
                    "beam_width": search_width, "beam_rank": beam_rank,
                    "temperature": 1.0 if beam_width > 1 else temperature,
                    "top_p": None if beam_width > 1 else top_p,
                }
                results.append(NeuralCandidate(score, tokens, report))
    return results


def generate_neural_results(source: SourceScore, checkpoint: str | Path,
                            controls: ArrangementControls | str | Path | Mapping[str, float] = "faithful",
                            **kwargs) -> tuple[list[NeuralCandidate], dict[str, Any]]:
    from src.training.trainer import load_checkpoint
    device = kwargs.get("device", "cpu")
    try:
        model, vocabulary, metadata = load_checkpoint(checkpoint, device=device)
    except (KeyError, TypeError, ValueError, pickle.UnpicklingError) as exc:
        reason = str(exc).splitlines()[0]
        raise GenerationError(f"Invalid or unreadable Transformer checkpoint {checkpoint}: {reason}") from exc
    from src.models.conditioning import CONTROL_NAMES
    if "control_names" in metadata and tuple(metadata["control_names"]) != CONTROL_NAMES:
        raise GenerationError("Checkpoint arrangement-control order is incompatible with this model interface")
    subdivisions = metadata.get("subdivisions", DEFAULT_SUBDIVISIONS)
    if isinstance(subdivisions, bool) or not isinstance(subdivisions, int) or subdivisions < 1:
        raise GenerationError("Checkpoint subdivisions must be a positive integer")
    results = decode_with_model(model, vocabulary, source, controls=load_neural_controls(controls),
                                subdivisions=subdivisions, **kwargs)
    return results, metadata


def generate_neural_candidates(source: SourceScore, checkpoint: str | Path,
                               controls: ArrangementControls | str | Path | Mapping[str, float] = "faithful",
                               **kwargs) -> list[ScoreData]:
    results, _ = generate_neural_results(source, checkpoint, controls, **kwargs)
    return [candidate.score for candidate in results]
