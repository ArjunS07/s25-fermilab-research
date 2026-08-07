"""Make the project modules importable when running pytest from anywhere.

The project uses absolute imports rooted at ``src`` (e.g. ``from util.geometry.minkowski_utils
import normsq4``), so ``src`` must be on ``sys.path``. This conftest inserts it.
"""
import os
import sys

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
