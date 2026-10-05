"""Free-running generation checks for the mandatory tiny overfit sanity test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from src.models.conditioning import CONTROL_NAMES
from .collator import TranscriptionCollator
from .dataset import TranscriptionDataset, load_prepared_dataset
from .trainer import load_checkpoint


def autoregressive_reconstruction(checkpoint: str | Path, dataset_path: str | Path, *,
                                   split: str = "train", max_segments: int | None = None,
                                   num_threads: int = 2) -> dict[str, Any]:
    """Generate without grammar repair, teacher forcing, or target retrieval.

    Reference event tokens score reproduction after generation. Conditioned
    diagnostics also supply each example's training control annotation, which
    may be derived from target techniques; this is an explicit training-label
    reconstruction test. Production inference supplies requested controls.
    The generation horizon uses model length rather than reference EOS. This
    diagnostic establishes optimization, not generalization or artistic quality.
    """
    torch.set_num_threads(num_threads)
    model, vocab, metadata = load_checkpoint(checkpoint)
    prepared = load_prepared_dataset(dataset_path)
    segments = prepared.split(split)
    if max_segments is not None:
        segments = segments[:max_segments]
    if not segments:
        raise ValueError(f"No segments in split {split!r}")
    data = TranscriptionDataset(segments, vocab, max_sequence_length=model.config.max_sequence_length)
    items = [data[i] for i in range(len(data))]
    batch = TranscriptionCollator(vocab)(items)
    with torch.inference_mode():
        encoded = model.encode(batch["source_ids"], source_padding_mask=batch["source_padding_mask"],
                               solo_ids=batch["solo_ids"], acc_ids=batch["acc_ids"],
                               solo_padding_mask=batch["solo_padding_mask"],
                               acc_padding_mask=batch["acc_padding_mask"], controls=batch["controls"])
        prefix = torch.full((len(data), 1), vocab.bos_id, dtype=torch.long)
        ended = torch.zeros(len(data), dtype=torch.bool)
        sequences: list[list[int]] = [[] for _ in range(len(data))]
        for _ in range(model.config.max_sequence_length - 1):
            logits = model.decode(prefix, encoded)
            predicted = logits[:, -1].argmax(-1)
            predicted = torch.where(ended, torch.full_like(predicted, vocab.pad_id), predicted)
            for row, value in enumerate(predicted.tolist()):
                if not ended[row]:
                    sequences[row].append(value)
            ended |= predicted.eq(vocab.eos_id)
            prefix = torch.cat((prefix, predicted[:, None]), dim=1)
            if ended.all():
                break
    correct = total = exact = pitch_correct = pitch_total = 0
    entries = []
    for item, generated in zip(items, sequences):
        reference = item["target_ids"][1:]
        equal = generated == reference
        exact += equal
        correct += sum(a == b for a, b in zip(generated, reference))
        total += max(len(generated), len(reference))
        generated_tokens = vocab.decode(generated)
        reference_tokens = vocab.decode(reference)
        pitch_correct += sum(a == b for a, b in zip(generated_tokens, reference_tokens)
                             if b.startswith("<PITCH_"))
        pitch_total += sum(token.startswith("<PITCH_") for token in reference_tokens)
        entries.append({"segment_id": item["segment_id"], "work_id": item["work_id"],
                        "controls": dict(zip(CONTROL_NAMES, item["controls"])),
                        "exact_match": equal, "generated_eos": bool(generated and generated[-1] == vocab.eos_id),
                        "generated_tokens": generated_tokens,
                        "reference_tokens": reference_tokens,
                        "generated_length": len(generated), "reference_length": len(reference)})
    return {"checkpoint": str(Path(checkpoint).resolve()), "dataset": str(Path(dataset_path).resolve()),
            "split": split, "segment_count": len(items), "exact_sequence_count": exact,
            "exact_sequence_ratio": exact / len(items), "autoregressive_token_accuracy": correct / total if total else 0.0,
            "positional_pitch_token_accuracy": pitch_correct / pitch_total if pitch_total else None,
            "correct_pitch_tokens": pitch_correct, "reference_pitch_tokens": pitch_total,
            "eos_ratio": float(ended.float().mean()), "checkpoint_epoch": metadata["epoch"],
            "conditioning_annotation_source": "Explicit/feature-derived training control labels" if model.config.conditioned else "Unconditioned model",
            "diagnostic_definition": "Raw greedy decoder; no target-prefix feeding, grammar repair, hard melody constraint, or copied target. Training split reproduction is an overfit check only.",
            "segments": entries}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--max-segments", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-threads", type=int, default=2)
    args = parser.parse_args(argv)
    report = autoregressive_reconstruction(args.checkpoint, args.dataset, split=args.split,
                                            max_segments=args.max_segments, num_threads=args.num_threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Raw greedy generation: {report['exact_sequence_count']}/{report['segment_count']} exact sequences; token accuracy={report['autoregressive_token_accuracy']:.4f}; EOS={report['eos_ratio']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

