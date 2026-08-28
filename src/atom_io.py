"""I/O facade for .atom binary format — delegates to src.io reader/writer modules."""

from __future__ import annotations

from pathlib import Path

from src.io import (
    ATOM_FORMAT_VERSION,
    ATOM_MAGIC_BYTES,
    load_atom_from_file,
    save_atom_to_file,
)
from src.phasor_atom import BlockAtom

ATOM_MAGIC = ATOM_MAGIC_BYTES
ATOM_VERSION = ATOM_FORMAT_VERSION

save_atom = save_atom_to_file
load_atom = load_atom_from_file

__all__ = [
    "ATOM_MAGIC",
    "ATOM_VERSION",
    "load_atom",
    "save_atom",
]
