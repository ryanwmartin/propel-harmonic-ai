"""Reading and validation for version 2 and version 3 .atom files."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import torch

from src.io.atom_writer import (
    ATOM_FORMAT_VERSION,
    ATOM_MAGIC_BYTES,
    COLUMN_ORDER_FLAG,
    HEADER_STRUCT_FORMAT,
)
from src.phasor_atom import BlockAtom

BYTES_PER_FLOAT32: int = 4
SUPPORTED_FLAGS: int = COLUMN_ORDER_FLAG


@dataclass(frozen=True)
class AtomHeaderInformation:
    harmonic_count: int
    block_size: int
    num_blocks: int
    rows: int
    cols: int
    version: int = ATOM_FORMAT_VERSION
    flags: int = 0


def validate_magic_bytes(magic_bytes: bytes, file_path: Path) -> None:
    if magic_bytes != ATOM_MAGIC_BYTES:
        raise ValueError(
            f"{file_path}: bad magic {magic_bytes!r} (expected {ATOM_MAGIC_BYTES!r})"
        )


def validate_file_version(version_number: int, file_path: Path) -> None:
    if version_number not in (2, ATOM_FORMAT_VERSION):
        raise ValueError(
            f"{file_path}: unsupported version {version_number} "
            f"(expected 2 or {ATOM_FORMAT_VERSION})"
        )


def validate_header_geometry(
    header_info: AtomHeaderInformation, file_path: Path
) -> None:
    if (
        header_info.harmonic_count < 1
        or header_info.block_size < 1
        or header_info.rows < 1
        or header_info.cols < 1
    ):
        raise ValueError(
            f"{file_path}: invalid header dimensions "
            f"(K={header_info.harmonic_count}, block_size={header_info.block_size}, "
            f"rows={header_info.rows}, cols={header_info.cols})"
        )
    expected_num_blocks = (
        header_info.cols + header_info.block_size - 1
    ) // header_info.block_size
    if header_info.num_blocks != expected_num_blocks:
        raise ValueError(
            f"{file_path}: inconsistent geometry — num_blocks={header_info.num_blocks} "
            f"but ceil(cols/block_size)={expected_num_blocks}"
        )
    if header_info.flags & ~SUPPORTED_FLAGS:
        raise ValueError(f"{file_path}: unsupported format flags 0x{header_info.flags:x}")


def read_and_validate_header(
    input_file_stream: BinaryIO, file_path: Path
) -> AtomHeaderInformation:
    prefix = input_file_stream.read(8)
    if len(prefix) != 8:
        raise ValueError(f"{file_path}: truncated header")
    magic_bytes, version_number = struct.unpack("<4sI", prefix)
    validate_magic_bytes(magic_bytes, file_path)
    validate_file_version(version_number, file_path)

    remaining_word_count = 6 if version_number == 3 else 5
    remainder = input_file_stream.read(remaining_word_count * 4)
    if len(remainder) != remaining_word_count * 4:
        raise ValueError(f"{file_path}: truncated header")
    values = struct.unpack(f"<{remaining_word_count}I", remainder)
    harmonic_count, block_size, num_blocks, rows, cols = values[:5]
    flags = values[5] if version_number == 3 else 0

    header_info = AtomHeaderInformation(
        harmonic_count=harmonic_count,
        block_size=block_size,
        num_blocks=num_blocks,
        rows=rows,
        cols=cols,
        version=version_number,
        flags=flags,
    )
    validate_header_geometry(header_info, file_path)
    return header_info


def read_float32_scalar(
    input_file_stream: BinaryIO,
    file_path: Path,
    row_index: int,
    block_index: int,
) -> float:
    raw_bytes = input_file_stream.read(BYTES_PER_FLOAT32)
    if len(raw_bytes) != BYTES_PER_FLOAT32:
        raise ValueError(
            f"{file_path}: truncated payload at row {row_index} block {block_index} anchor"
        )
    return struct.unpack("<f", raw_bytes)[0]


def read_float32_array(
    input_file_stream: BinaryIO,
    file_path: Path,
    element_count: int,
    row_index: int,
    block_index: int,
    field_description: str,
) -> torch.Tensor:
    expected_byte_count = BYTES_PER_FLOAT32 * element_count
    raw_bytes = input_file_stream.read(expected_byte_count)
    if len(raw_bytes) != expected_byte_count:
        raise ValueError(
            f"{file_path}: truncated payload at row {row_index} "
            f"block {block_index} {field_description}"
        )
    return torch.frombuffer(bytearray(raw_bytes), dtype=torch.float32).clone()


def read_atom_payload(
    input_file_stream: BinaryIO,
    file_path: Path,
    header_info: AtomHeaderInformation,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    rows = header_info.rows
    num_blocks = header_info.num_blocks
    harmonic_count = header_info.harmonic_count
    anchors = torch.empty(rows, num_blocks, dtype=torch.float32)
    amplitudes = torch.empty(rows, num_blocks, harmonic_count, dtype=torch.float32)
    frequencies = torch.empty_like(amplitudes)
    phases = torch.empty_like(amplitudes)

    for row_index in range(rows):
        for block_index in range(num_blocks):
            anchors[row_index, block_index] = read_float32_scalar(
                input_file_stream, file_path, row_index, block_index
            )
            for destination, description in (
                (amplitudes, "amplitudes"),
                (frequencies, "frequencies"),
                (phases, "phases"),
            ):
                destination[row_index, block_index] = read_float32_array(
                    input_file_stream,
                    file_path,
                    harmonic_count,
                    row_index,
                    block_index,
                    description,
                )

    column_order = None
    if header_info.flags & COLUMN_ORDER_FLAG:
        expected_bytes = header_info.cols * 4
        raw_order = input_file_stream.read(expected_bytes)
        if len(raw_order) != expected_bytes:
            raise ValueError(f"{file_path}: truncated column-order payload")
        column_order = torch.frombuffer(bytearray(raw_order), dtype=torch.uint32).to(
            torch.int64
        )

    if input_file_stream.read(1):
        raise ValueError(f"{file_path}: unexpected trailing payload bytes")
    return anchors, amplitudes, frequencies, phases, column_order


def load_atom_from_file(source_path: str | Path) -> BlockAtom:
    file_path = Path(source_path)
    with open(file_path, "rb") as input_file_stream:
        header_info = read_and_validate_header(input_file_stream, file_path)
        anchors, amplitudes, frequencies, phases, column_order = read_atom_payload(
            input_file_stream, file_path, header_info
        )

    return BlockAtom(
        anchors=anchors,
        amplitudes=amplitudes,
        frequencies=frequencies,
        phases=phases,
        block_size=header_info.block_size,
        num_blocks=header_info.num_blocks,
        rows=header_info.rows,
        cols=header_info.cols,
        column_order=column_order,
    )
