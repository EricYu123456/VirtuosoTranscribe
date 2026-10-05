"""Reproducible supervised training and portable resumable checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import time
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.conditioning import CONTROL_NAMES
from src.models.seq2seq import ModelConfig, Seq2SeqTransformer
from .collator import TranscriptionCollator, model_inputs
from .dataset import TranscriptionDataset, Vocabulary, load_prepared_dataset
from .losses import transcription_loss


CHECKPOINT_TYPE = "virtuosotranscribe.transformer"


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 0.0002
    weight_decay: float = 0.01
    gradient_accumulation: int = 1
    gradient_clip: float = 1.0
    warmup_steps: int = 0
    label_smoothing: float = 0.0
    lambda_melody: float = 0.0
    seed: int = 42
    device: str = "cpu"
    num_threads: int = 2
    max_train_segments: int | None = None
    save_every: int = 10
    early_stop_loss: float | None = None

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size", "gradient_accumulation", "num_threads", "save_every"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "gradient_clip"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative and finite")
        if not isinstance(self.warmup_steps, int) or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be a non-negative integer")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing must be within 0..1")
        if not math.isfinite(self.lambda_melody) or self.lambda_melody < 0:
            raise ValueError("lambda_melody must be non-negative and finite")
        if self.max_train_segments is not None and self.max_train_segments < 1:
            raise ValueError("max_train_segments must be positive when supplied")
        if self.early_stop_loss is not None and self.early_stop_loss <= 0:
            raise ValueError("early_stop_loss must be positive when supplied")


def seed_everything(seed: int, num_threads: int = 2) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(num_threads)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def save_checkpoint(path: str | Path, model: Seq2SeqTransformer, vocab: Vocabulary, *,
                    optimizer: torch.optim.Optimizer | None = None,
                    metadata: dict[str, Any] | None = None,
                    training_config: TrainingConfig | None = None,
                    epoch: int = 0, global_step: int = 0,
                    history: list[dict[str, Any]] | None = None,
                    loader_rng_state: torch.Tensor | None = None) -> Path:
    """Save state dictionaries only; weights-only loading needs no Python objects."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": CHECKPOINT_TYPE, "schema_version": 1,
        "model_config": model.config.to_dict(), "vocabulary": vocab.to_dict(),
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "training_config": asdict(training_config) if training_config else None,
        "metadata": dict(metadata or {}), "epoch": epoch, "global_step": global_step,
        "history": list(history or []), "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(), "loader_rng_state": loader_rng_state,
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def _read_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        payload = torch.load(Path(path), map_location=device, weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, EOFError, IndexError, KeyError, TypeError) as exc:
        raise ValueError(f"Unreadable or invalid Transformer checkpoint: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("type") != CHECKPOINT_TYPE or payload.get("schema_version") != 1:
        raise ValueError("Unsupported VirtuosoTranscribe Transformer checkpoint")
    required = ("model_config", "vocabulary", "model_state_dict")
    if any(not isinstance(payload.get(name), dict) for name in required):
        raise ValueError("Incomplete Transformer checkpoint: model configuration, vocabulary, and learned state dictionaries are required")
    if not isinstance(payload.get("metadata", {}), dict):
        raise ValueError("Checkpoint metadata must be a mapping")
    return payload


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[Seq2SeqTransformer, Vocabulary, dict[str, Any]]:
    """Return an evaluated model, its exact training vocabulary, and provenance."""
    payload = _read_checkpoint(path, device)
    vocab = Vocabulary.from_dict(payload["vocabulary"])
    model = Seq2SeqTransformer(len(vocab), ModelConfig.from_dict(payload["model_config"])).to(device)
    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    except RuntimeError as exc:
        raise ValueError(f"Checkpoint weights do not match its model configuration: {exc}") from exc
    model.eval()
    metadata = {**payload.get("metadata", {}), "epoch": payload.get("epoch", 0),
                "global_step": payload.get("global_step", 0),
                "model_config": payload["model_config"],
                "training_config": payload.get("training_config"),
                "history": payload.get("history", [])}
    return model, vocab, metadata


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {name: value.to(device) if isinstance(value, torch.Tensor) else value for name, value in batch.items()}


def evaluate_model(model: Seq2SeqTransformer, loader: DataLoader, *, pad_id: int,
                   config: TrainingConfig, device: torch.device) -> dict[str, float | int]:
    model.eval()
    losses = {"loss": 0.0, "token_loss": 0.0, "melody_loss": 0.0}
    tokens = correct = 0
    with torch.inference_mode():
        for raw in loader:
            batch = _to_device(raw, device)
            logits = model(**model_inputs(batch))
            values = transcription_loss(logits, batch["labels"], pad_id=pad_id,
                                         melody_positions=batch["melody_positions"],
                                         melody_pitch_ids=batch["melody_pitch_ids"],
                                         lambda_melody=config.lambda_melody,
                                         label_smoothing=config.label_smoothing)
            valid = batch["labels"].ne(pad_id)
            count = int(valid.sum())
            for name, value in values.items():
                losses[name] += float(value) * count
            tokens += count
            correct += int(((logits.argmax(-1) == batch["labels"]) & valid).sum())
    return {**{name: value / tokens if tokens else 0.0 for name, value in losses.items()},
            "token_accuracy": correct / tokens if tokens else 0.0, "token_count": tokens}


def train_model(dataset_path: str | Path, output_dir: str | Path, *,
                model_config: ModelConfig | None = None,
                training_config: TrainingConfig | None = None,
                resume: str | Path | None = None,
                progress: bool = True) -> dict[str, Any]:
    """Train explicit train works, inspect validation works, and leave test untouched.

    ``epochs`` is the total desired epoch count, including epochs restored from a
    resume checkpoint. No segment is silently truncated or randomly resplit.
    """
    model_config = model_config or ModelConfig()
    config = training_config or TrainingConfig()
    seed_everything(config.seed, config.num_threads)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; choose device=cpu")
    prepared = load_prepared_dataset(dataset_path)
    train_segments = prepared.split("train")
    validation_segments = prepared.split("validation")
    if not train_segments:
        raise ValueError("Prepared dataset has no explicit train works")
    if config.max_train_segments:
        train_segments = train_segments[:config.max_train_segments]
    vocab = Vocabulary.from_segments(train_segments)
    model = Seq2SeqTransformer(len(vocab), model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history: list[dict[str, Any]] = []
    start_epoch = global_step = 0
    restored = None
    dataset_digest = hashlib.sha256(prepared.path.read_bytes()).hexdigest()
    if resume is not None:
        restored = _read_checkpoint(resume, device)
        if restored["model_config"] != model_config.to_dict():
            raise ValueError("Resume checkpoint model configuration differs from requested model")
        if restored["vocabulary"] != vocab.to_dict():
            raise ValueError("Resume checkpoint vocabulary differs from this training split")
        if restored.get("metadata", {}).get("dataset_sha256") != dataset_digest:
            raise ValueError("Resume dataset changed; use the same prepared dataset for an exact resume")
        if restored.get("optimizer_state_dict") is None:
            raise ValueError("Resume checkpoint has no optimizer state")
        model.load_state_dict(restored["model_state_dict"], strict=True)
        optimizer.load_state_dict(restored["optimizer_state_dict"])
        start_epoch = restored["epoch"]
        global_step = restored["global_step"]
        history = list(restored.get("history", []))
        torch.set_rng_state(restored["torch_rng_state"].cpu())
        random.setstate(restored["python_rng_state"])
        if device.type == "cuda" and restored.get("cuda_rng_states"):
            torch.cuda.set_rng_state_all(restored["cuda_rng_states"])
    collator = TranscriptionCollator(vocab)
    train_data = TranscriptionDataset(train_segments, vocab, max_sequence_length=model_config.max_sequence_length)
    val_data = TranscriptionDataset(validation_segments, vocab, max_sequence_length=model_config.max_sequence_length)
    generator = torch.Generator().manual_seed(config.seed)
    if restored and restored.get("loader_rng_state") is not None:
        generator.set_state(restored["loader_rng_state"].cpu())
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True,
                              collate_fn=collator, generator=generator)
    # Keep evaluation iterator seeds out of the training/global RNG stream so
    # inspecting a resumed checkpoint does not change subsequent dropout masks.
    evaluation_generator = torch.Generator().manual_seed(config.seed + 1)
    eval_loader = DataLoader(train_data, batch_size=config.batch_size, collate_fn=collator,
                             generator=evaluation_generator)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, collate_fn=collator,
                            generator=evaluation_generator)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset_path": str(prepared.path), "dataset_sha256": dataset_digest,
        "subdivisions": prepared.metadata.get("subdivisions", 24),
        "control_names": list(CONTROL_NAMES), "train_segment_count": len(train_segments),
        "validation_segment_count": len(validation_segments),
        "test_segment_count": len(prepared.split("test")),
        "train_work_ids": sorted({x["work_id"] for x in train_segments}),
        "unique_train_source_count": len({tuple(["<SOURCE_SOLO>", *x["source_solo_tokens"],
                                                "<SOURCE_ACC>", *x["source_acc_tokens"]]) for x in train_segments}),
        "unique_train_target_count": len({tuple(x["target_piano_tokens"]) for x in train_segments}),
        "explicit_control_annotation_count": sum("technique_controls" in x or "controls" in x for x in train_segments),
        "inferred_control_annotation_count": sum("technique_controls" not in x and "controls" not in x for x in train_segments),
        "control_annotation_definition": "Explicit requested control labels take precedence; otherwise measured target techniques provide bounded supervision labels. Inference supplies requested controls and never reads target features.",
        "dataset_purpose": prepared.metadata.get("purpose"),
        "origin_manifest": prepared.metadata.get("manifest_path"),
        "parent_dataset": prepared.metadata.get("parent_dataset"),
        "validation_work_ids": sorted({x["work_id"] for x in validation_segments}),
        "test_work_ids": sorted({x["work_id"] for x in prepared.split("test")}),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "torch_version": str(torch.__version__),
        "data_kinds": sorted({str(x.get("data_kind", "unspecified")) for x in train_segments}),
        "purpose": "overfit sanity check" if config.max_train_segments else "supervised research training",
        "soft_melody_definition": "Differentiable exact/+12/-12 presence pooled over teacher-forced target pitch slots at aligned onsets; absent event slots require hard decoding/evaluation.",
    }
    _write_json(output / "vocabulary.json", vocab.to_dict())
    initial = evaluate_model(model, eval_loader, pad_id=vocab.pad_id, config=config, device=device)
    if not history:
        history.append({"epoch": 0, "global_step": 0, "train": initial,
                        "validation": evaluate_model(model, val_loader, pad_id=vocab.pad_id, config=config, device=device) if validation_segments else None})
    if progress:
        print(f"{model_config.architecture} conditioned={model_config.conditioned}: {len(train_data)} train, {len(val_data)} validation segments; initial token loss={initial['token_loss']:.4f}", flush=True)
    started = time.monotonic()
    final_epoch = start_epoch
    for epoch in range(start_epoch + 1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        batches = len(train_loader)
        group_size = min(config.gradient_accumulation, batches)
        for batch_index, raw in enumerate(train_loader):
            if batch_index % config.gradient_accumulation == 0:
                group_size = min(config.gradient_accumulation, batches - batch_index)
            batch = _to_device(raw, device)
            logits = model(**model_inputs(batch))
            values = transcription_loss(logits, batch["labels"], pad_id=vocab.pad_id,
                                         melody_positions=batch["melody_positions"], melody_pitch_ids=batch["melody_pitch_ids"],
                                         lambda_melody=config.lambda_melody, label_smoothing=config.label_smoothing)
            if not torch.isfinite(values["loss"]):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, batch {batch_index}")
            (values["loss"] / group_size).backward()
            if (batch_index + 1) % config.gradient_accumulation == 0 or batch_index + 1 == batches:
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                global_step += 1
                warmup = min(1.0, global_step / config.warmup_steps) if config.warmup_steps else 1.0
                for group in optimizer.param_groups:
                    group["lr"] = config.learning_rate * warmup
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        final_epoch = epoch
        # Epoch-end losses use eval mode so the overfit diagnostic is stable and
        # reported token accuracy includes only non-padding target positions.
        train_metrics = evaluate_model(model, eval_loader, pad_id=vocab.pad_id, config=config, device=device)
        validation_metrics = evaluate_model(model, val_loader, pad_id=vocab.pad_id, config=config, device=device) if validation_segments else None
        record = {"epoch": epoch, "global_step": global_step, "train": train_metrics,
                  "validation": validation_metrics, "elapsed_seconds": time.monotonic() - started}
        history.append(record)
        if progress and (epoch == start_epoch + 1 or epoch % config.save_every == 0):
            val_message = f" val={validation_metrics['token_loss']:.4f}" if validation_metrics else ""
            print(f"epoch={epoch} step={global_step} token_loss={train_metrics['token_loss']:.5f} accuracy={train_metrics['token_accuracy']:.4f}{val_message}", flush=True)
        if epoch % config.save_every == 0:
            save_checkpoint(output / "last.pt", model, vocab, optimizer=optimizer,
                            metadata=metadata, training_config=config, epoch=epoch, global_step=global_step,
                            history=history, loader_rng_state=generator.get_state())
            _write_json(output / "history.json", {"metadata": metadata, "history": history})
        if config.early_stop_loss is not None and train_metrics["token_loss"] <= config.early_stop_loss:
            break
    checkpoint = save_checkpoint(output / "last.pt", model, vocab, optimizer=optimizer,
                                 metadata=metadata, training_config=config, epoch=final_epoch, global_step=global_step,
                                 history=history, loader_rng_state=generator.get_state())
    report = {"checkpoint": str(checkpoint), "metadata": metadata,
              "model_config": model_config.to_dict(), "training_config": asdict(config),
              "history": history, "final_epoch": final_epoch, "global_step": global_step,
              "initial_token_loss": history[0]["train"]["token_loss"],
              "final_token_loss": history[-1]["train"]["token_loss"],
              "final_token_accuracy": history[-1]["train"]["token_accuracy"]}
    _write_json(output / "history.json", report)
    return report

