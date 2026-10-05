"""Generate, validate, rank and export baseline or learned piano arrangements."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.baselines.literal_reduction import reduce_literal
from src.baselines.rule_based import ArrangementControls, arrange_rule_based, load_preset
from src.inference.rerank import rank_candidates
from src.representation.events import ScoreData, SourceScore


def generate_candidates(source: SourceScore, preset: ArrangementControls | str | Path = "faithful",
                        *, num_candidates: int = 8, seed: int = 0,
                        baseline: str = "rule_based") -> list[ScoreData]:
    if isinstance(num_candidates, bool) or not isinstance(num_candidates, int) or num_candidates < 1:
        raise ValueError("num_candidates must be a positive integer")
    if baseline not in ("rule_based", "literal"):
        raise ValueError("baseline must be rule_based or literal")
    controls = load_preset(preset)
    if baseline == "literal":
        return [reduce_literal(source) for _ in range(num_candidates)]
    result: list[ScoreData] = []
    fingerprints: set[str] = set()
    # Find alternative valid textures when the score permits them. Short or
    # already sparse sources can legitimately have only one available texture.
    attempt = 0
    while len(result) < num_candidates and attempt < max(32, 16 * num_candidates):
        score = arrange_rule_based(source, controls, seed=seed + attempt)
        digest = _fingerprint(score)
        if digest not in fingerprints:
            result.append(score)
            fingerprints.add(digest)
        attempt += 1
    while len(result) < num_candidates:
        result.append(result[len(result) % len(fingerprints)])
    return result


def _fingerprint(score: ScoreData) -> str:
    # Technique-role labels alone do not make a different musical candidate.
    events = sorted((n.pitch, n.onset, n.duration, n.part_id) for n in score.notes)
    return hashlib.sha256(json.dumps(events).encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--preset", default="faithful", help="Preset name or YAML path")
    parser.add_argument("--num_candidates", "--num-candidates", type=int, default=8)
    parser.add_argument("--output_dir", "--output-dir", type=Path, required=True)
    parser.add_argument("--solo-part")
    parser.add_argument("--accompaniment-part", action="append")
    generator_group = parser.add_mutually_exclusive_group()
    generator_group.add_argument("--baseline", choices=("literal", "rule_based"))
    generator_group.add_argument("--checkpoint", type=Path, help="Trained Transformer checkpoint; errors never fall back to a baseline")
    parser.add_argument("--melody-mode", choices=("none", "soft", "hard"), default="hard")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=.8, help="0 selects greedy learned decoding")
    parser.add_argument("--top-p", type=float, default=.95)
    parser.add_argument("--beam-width", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--soft-strength", type=float, default=2.)
    parser.add_argument("--control", action="append", default=[], metavar="NAME=VALUE",
                        help="Override a continuous neural arrangement control")
    parser.add_argument("--musicbert-checkpoint", type=Path)
    parser.add_argument("--musicbert-dictionary", type=Path)
    parser.add_argument("--musicbert-backend", choices=("torchscript", "fairseq"), default="torchscript")
    parser.add_argument("--musicbert-user-dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        return _generate_outputs(args)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.error(str(exc))


def _generate_outputs(args: argparse.Namespace) -> int:
    from src.data.normalize_score import normalize_source
    from src.data.parse_musicxml import parse_score
    from src.inference.export import export_score
    from src.analysis.melody_alignment import align_solo_to_piano
    from src.analysis.visualization import write_alignment_visualization
    source = normalize_source(parse_score(args.input), solo_part=args.solo_part,
                              accompaniment_parts=args.accompaniment_part)
    neural_results = []
    checkpoint_metadata = None
    if args.checkpoint is not None:
        from src.inference.neural_generate import generate_neural_results, load_neural_controls
        controls = load_neural_controls(args.preset)
        for override in args.control:
            try:
                name, value = override.split("=", 1)
                controls[name] = float(value)
            except ValueError as exc:
                raise ValueError("Neural control overrides must use NAME=NUMBER") from exc
        controls = load_neural_controls(controls)
        neural_results, checkpoint_metadata = generate_neural_results(
            source, args.checkpoint, controls, num_candidates=args.num_candidates,
            seed=args.seed, device=args.device, melody_mode=args.melody_mode,
            temperature=args.temperature, top_p=args.top_p, beam_width=args.beam_width,
            max_new_tokens=args.max_new_tokens, soft_strength=args.soft_strength,
            num_threads=args.num_threads)
        candidates = [result.score for result in neural_results]
        generator = "transformer"
    else:
        if args.control:
            raise ValueError("--control overrides require a neural --checkpoint; use a YAML preset for baseline controls")
        controls = load_preset(args.preset)
        generator = args.baseline or "rule_based"
        candidates = generate_candidates(source, controls, num_candidates=args.num_candidates,
                                         seed=args.seed, baseline=generator)
    content_scorer = None
    if args.musicbert_checkpoint is not None:
        if args.musicbert_dictionary is None:
            raise ValueError("--musicbert-checkpoint requires its --musicbert-dictionary")
        from src.inference.musicbert import MusicBERTContentScorer
        content_scorer = MusicBERTContentScorer(
            args.musicbert_checkpoint, args.musicbert_dictionary,
            backend=args.musicbert_backend, user_dir=args.musicbert_user_dir, device=args.device)
    ranking = rank_candidates(source, candidates, controls, content_scorer=content_scorer)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    source.score.save(output_dir / "source.json")
    fingerprints = [_fingerprint(c) for c in candidates]
    for index, score in enumerate(candidates):
        stem = output_dir / f"candidate_{index + 1:02d}"
        export_score(score, stem.with_suffix(".musicxml"))
        export_score(score, stem.with_suffix(".mid"))
        score.save(stem.with_suffix(".json"))
        if neural_results:
            result = neural_results[index]
            (output_dir / f"candidate_{index + 1:02d}.tokens.json").write_text(
                json.dumps({"model_tokens_before_projection": result.tokens,
                            "final_tokens": result.tokens,
                            "projection_repairs": 0,
                            "decoding_report": result.report}, indent=2, allow_nan=False), encoding="utf-8")
            score.save(output_dir / f"candidate_{index + 1:02d}.before_projection.json")
    for row in ranking:
        index = row["candidate_index"]
        analysis_path = output_dir / f"analysis_{index + 1:02d}.json"
        analysis_path.write_text(json.dumps(row["metrics"], ensure_ascii=False, indent=2), encoding="utf-8")
        row["candidate_id"] = f"candidate_{index + 1:02d}"
        row["fingerprint"] = fingerprints[index]
        if neural_results:
            row["generation"] = neural_results[index].report
    best_index = ranking[0]["candidate_index"]
    best = candidates[best_index]
    export_score(best, output_dir / "best.musicxml")
    export_score(best, output_dir / "best.mid")
    best.save(output_dir / "best.json")
    alignment = align_solo_to_piano(source.solo, best, source_score=source.score)
    alignment.save(output_dir / "alignment.json")
    write_alignment_visualization(source.solo, best, alignment, output_dir / "alignment.html",
                                  title=f"{source.score.title}: source solo → best piano arrangement")
    payload = {
        "generator": generator,
        "neural_model": bool(neural_results),
        "seed": args.seed,
        "preset": args.preset,
        "controls": controls if isinstance(controls, dict) else controls.to_dict(),
        "requested_candidates": args.num_candidates,
        "unique_candidates": len(set(fingerprints)),
        "best_candidate": f"candidate_{best_index + 1:02d}",
        "ranking": ranking,
    }
    if neural_results:
        payload.update(checkpoint=str(args.checkpoint.resolve()),
                       checkpoint_metadata=checkpoint_metadata,
                       melody_mode=args.melody_mode,
                       decoder_uses_training_target_lookup=False)
        if not neural_results[0].report["controls_conditioned"]:
            payload["warnings"] = ["This vanilla checkpoint has no control conditioning; requested controls affect reranking only."]
    if content_scorer is not None:
        payload["musicbert"] = {"checkpoint": str(args.musicbert_checkpoint.resolve()),
                                "backend": args.musicbert_backend,
                                "scoring": "normalized cosine similarity of real encoder embeddings"}
    if len(set(fingerprints)) < len(candidates):
        payload.setdefault("warnings", []).append("Fewer distinct textures were generated than requested; duplicate candidates are marked by fingerprint.")
    (output_dir / "ranking.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Wrote {len(candidates)} {generator} candidates ({len(set(fingerprints))} unique) to {output_dir}")
    print(f"Best candidate: {payload['best_candidate']}; " +
          (f"learned {args.melody_mode} decoding" if neural_results else "no neural model used"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

