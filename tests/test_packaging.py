"""Packaging sanity checks."""

from pathlib import Path

import vibetrack


def test_py_typed_marker_shipped():
    """PEP 561 marker must live next to the package so type checkers see our hints."""
    assert (Path(vibetrack.__file__).parent / "py.typed").exists()
