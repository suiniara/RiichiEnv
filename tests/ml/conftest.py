"""Expose the optional ML workspace without installing its service dependencies."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "riichienv-ml" / "src"))
