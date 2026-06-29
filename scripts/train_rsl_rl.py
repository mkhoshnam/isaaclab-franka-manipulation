#!/usr/bin/env python3
"""Training entry point for the Franka cabinet manipulation task."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_rsl_rl import main  # noqa: E402


if __name__ == "__main__":
    main()
