"""Train a vanilla/dual/conditioned Transformer on inspected prepared pairs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml


def load_training_configuration(path: str | Path) -> tuple[Any, Any, dict[str, Any]]:
    from src.models.seq2seq import ModelConfig
    from src.training.trainer import TrainingConfig

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Training configuration must be a YAML mapping")
    model = ModelConfig.from_dict(payload.get("model", {}))
    training = dict(payload.get("training", {}))
    constraints = payload.get("constraints", {})
    if constraints.get("melody_mode", "none") not in {"none", "soft", "hard"}:
        raise ValueError("constraints.melody_mode must be none/soft/hard")
    if constraints.get("melody_mode") == "soft":
        training.setdefault("lambda_melody", constraints.get("lambda_melody", 0.1))
    else:
        training.setdefault("lambda_melody", 0.0)
    return model, TrainingConfig(**training), payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/model/seq2seq.yaml", type=Path)
    parser.add_argument("--dataset", type=Path, help="Prepared dataset.json or its directory")
    parser.add_argument("--output", type=Path, help="Checkpoint/history directory")
    parser.add_argument("--resume", type=Path, help="Resume complete model + optimizer state")
    parser.add_argument("--epochs", type=int, help="Total epoch count, including restored epochs")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--architecture", choices=("single", "dual"))
    parser.add_argument("--conditioned", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--lambda-melody", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-train-segments", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-threads", type=int)
    parser.add_argument("--early-stop-loss", type=float)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        from src.models.seq2seq import ModelConfig
        from src.training.trainer import TrainingConfig, train_model

        model, training, payload = load_training_configuration(args.config)
        model_values = asdict(model)
        train_values = asdict(training)
        for name in ("architecture", "conditioned"):
            if getattr(args, name) is not None:
                model_values[name] = getattr(args, name)
        for name in ("epochs", "device", "lambda_melody", "learning_rate", "batch_size", "max_train_segments", "seed", "num_threads", "early_stop_loss"):
            if getattr(args, name) is not None:
                train_values[name] = getattr(args, name)
        dataset = args.dataset or payload.get("data", {}).get("dataset")
        output = args.output or payload.get("output_dir", "outputs/checkpoints/seq2seq")
        if dataset is None:
            raise ValueError("Set --dataset or data.dataset in the YAML config after preparing pairs")
        report = train_model(dataset, output, model_config=ModelConfig(**model_values),
                             training_config=TrainingConfig(**train_values), resume=args.resume,
                             progress=not args.quiet)
    except ImportError as exc:
        parser.error(f"Install model dependencies (pip install -e '.[model]'): {exc}")
    except (ValueError, OSError, TypeError) as exc:
        parser.error(str(exc))
    print(f"Saved {report['checkpoint']}; token loss {report['initial_token_loss']:.4f} -> {report['final_token_loss']:.4f}, train accuracy {report['final_token_accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

