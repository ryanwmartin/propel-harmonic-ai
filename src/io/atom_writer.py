"""File I/O domain — saving BlockAtoms to .atom binary files.

Binary layout (little-endian), version 3:
    Header (32 bytes):
        bytes[4] magic "ATOM"
        u32 version = 3
        u32 harmonic_count (K)
        u32 block_size
        u32 num_blocks
        u32 rows
        u32 cols
        u32 flags (bit 0: shared column order present)
    Harmonic payload:
        For each row and block: f32 anchor, f32[K] A, f32[K] f, f32[K] phase
    Optional ordering payload:
        u32[cols] original column indices in encoded-column order
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import BinaryIO, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from src.phasor_atom import BlockAtom

ATOM_MAGIC_BYTES: bytes = b"ATOM"
ATOM_FORMAT_VERSION: int = 3
COLUMN_ORDER_FLAG: int = 1
HEADER_STRUCT_FORMAT: struct.Struct = struct.Struct("<4s7I")
LEGACY_HEADER_STRUCT_FORMAT: struct.Struct = struct.Struct("<4s6I")


def pack_atom_header(atom: BlockAtom) -> bytes:
    """Pack the version 3 binary header for a BlockAtom."""
    flags = COLUMN_ORDER_FLAG if atom.column_order is not None else 0
    return HEADER_STRUCT_FORMAT.pack(
        ATOM_MAGIC_BYTES,
        ATOM_FORMAT_VERSION,
        atom.K,
        atom.block_size,
        atom.num_blocks,
        atom.rows,
        atom.cols,
        flags,
    )


def write_float_tensor_to_stream(
    output_stream: BinaryIO, tensor_values: torch.Tensor
) -> None:
    """Write float32 tensor contents as little-endian bytes to a stream."""
    float32_array = tensor_values.detach().cpu().numpy().astype("<f4")
    output_stream.write(float32_array.tobytes())


def write_atom_payload(atom: BlockAtom, output_file_stream: BinaryIO) -> None:
    """Write harmonic parameters and optional shared column order."""
    for row_index in range(atom.rows):
        for block_index in range(atom.num_blocks):
            write_float_tensor_to_stream(
                output_file_stream, atom.anchors[row_index, block_index]
            )
            write_float_tensor_to_stream(
                output_file_stream, atom.amplitudes[row_index, block_index]
            )
            write_float_tensor_to_stream(
                output_file_stream, atom.frequencies[row_index, block_index]
            )
            write_float_tensor_to_stream(
                output_file_stream, atom.phases[row_index, block_index]
            )

    if atom.column_order is not None:
        order_array = atom.column_order.detach().cpu().numpy().astype("<u4")
        output_file_stream.write(order_array.tobytes())


def save_atom_to_file(atom: BlockAtom, destination_path: str | Path) -> None:
    """Serialize a BlockAtom parameter container to a .atom binary file."""
    destination_file_path = Path(destination_path)
    with open(destination_file_path, "wb") as output_file_stream:
        output_file_stream.write(pack_atom_header(atom))
        write_atom_payload(atom, output_file_stream)
