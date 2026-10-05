"""Train the unconditional single-encoder neural baseline."""

from __future__ import annotations

import sys

from scripts.train_model import main as train_main


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    return train_main([*arguments, "--architecture", "single", "--no-conditioned", "--lambda-melody", "0"])


if __name__ == "__main__":
    raise SystemExit(main())

