"""File I/O domain module.

Separates atom binary saving (writer) and reading (reader) into isolated, single-responsibility modules.
"""

from src.io.atom_reader import (
    AtomHeaderInformation,
    load_atom_from_file,
    read_and_validate_header,
)
from src.io.atom_writer import (
    ATOM_FORMAT_VERSION,
    ATOM_MAGIC_BYTES,
    pack_atom_header,
    save_atom_to_file,
)

__all__ = [
    "ATOM_FORMAT_VERSION",
    "ATOM_MAGIC_BYTES",
    "AtomHeaderInformation",
    "load_atom_from_file",
    "pack_atom_header",
    "read_and_validate_header",
    "save_atom_to_file",
]
