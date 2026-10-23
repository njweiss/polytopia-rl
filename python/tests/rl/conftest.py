"""Milestone 10 tests: CPU only, run by scripts/rl_check.sh. Skipped when torch is missing."""

import pathlib
import sys

import pytest

pytest.importorskip("torch")

# `polytopia_rl` lives in python/ (pure Python, not installed).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
