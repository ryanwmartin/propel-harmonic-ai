"""Block-wise derivative codec — parameter container and decoder interface.

A weight matrix W of shape (rows, cols) is partitioned into blocks of length L.
This module defines the BlockAtom data container which stores per-row, per-block
exact float32 anchor weights and 1-D harmonic parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.decoder.block_decoder import (
    compute_block_length,
    decode_full_tensor,
    decode_row_weights,
    evaluate_block_harmonic_derivative,
)

# Backwards compatibility function aliases
block_length = compute_block_length
eval_block = evaluate_block_harmonic_derivative
decode_row = decode_row_weights
decode_tensor = decode_full_tensor


@dataclass
class BlockAtom:
    """Per-row, per-block harmonic parameters for a weight matrix.

    Attributes:
        anchors: Tensor of shape (rows, num_blocks) holding exact float32 anchor weights.
        amplitudes: Tensor of shape (rows, num_blocks, harmonic_count) holding sinusoid amplitudes.
        frequencies: Tensor of shape (rows, num_blocks, harmonic_count) holding normalized frequencies.
        phases: Tensor of shape (rows, num_blocks, harmonic_count) holding phase offsets in radians.
        block_size: Maximum column length per block.
        num_blocks: Number of blocks per row.
        rows: Total number of rows in the weight matrix.
        cols: Total number of columns in the weight matrix.
        column_order: Optional original column indices in encoded-column order.
    """

    anchors: torch.Tensor
    amplitudes: torch.Tensor
    frequencies: torch.Tensor
    phases: torch.Tensor
    block_size: int
    num_blocks: int
    rows: int
    cols: int
    column_order: torch.Tensor | None = None

    def __post_init__(self) -> None:
        for attribute_name in ("anchors", "amplitudes", "frequencies", "phases"):
            tensor_value = getattr(self, attribute_name)
            if tensor_value.dtype != torch.float32:
                setattr(self, attribute_name, tensor_value.to(torch.float32))

        expected_leading_dimensions = (self.rows, self.num_blocks)
        if self.anchors.shape != expected_leading_dimensions:
            raise ValueError(
                f"anchors shape must be {expected_leading_dimensions}, got {tuple(self.anchors.shape)}"
            )

        for attribute_name in ("amplitudes", "frequencies", "phases"):
            tensor_value = getattr(self, attribute_name)
            if tensor_value.ndim != 3:
                raise ValueError(
                    f"{attribute_name} must be 3-D (rows, num_blocks, harmonic_count), got {tuple(tensor_value.shape)}"
                )

        if self.frequencies.shape != self.amplitudes.shape:
            raise ValueError(
                f"frequencies shape {tuple(self.frequencies.shape)} does not match "
                f"amplitudes shape {tuple(self.amplitudes.shape)}"
            )
        if self.phases.shape != self.amplitudes.shape:
            raise ValueError(
                f"phases shape {tuple(self.phases.shape)} does not match "
                f"amplitudes shape {tuple(self.amplitudes.shape)}"
            )

        leading_dimensions = tuple(self.amplitudes.shape[:2])
        if leading_dimensions != expected_leading_dimensions:
            raise ValueError(
                f"amplitudes leading dims must be {expected_leading_dimensions}, got {leading_dimensions}"
            )

        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("rows and cols must be positive")

        if self.column_order is not None:
            self.column_order = self.column_order.to(dtype=torch.int64)
            if self.column_order.shape != (self.cols,):
                raise ValueError(
                    f"column_order shape must be ({self.cols},), got "
                    f"{tuple(self.column_order.shape)}"
                )
            expected = torch.arange(
                self.cols, dtype=torch.int64, device=self.column_order.device
            )
            if not torch.equal(torch.sort(self.column_order).values, expected):
                raise ValueError("column_order must be a permutation of [0, cols)")

    @property
    def K(self) -> int:
        """Number of harmonics fitted per block (alias for harmonic_count)."""
        return int(self.amplitudes.shape[2])

    @property
    def harmonic_count(self) -> int:
        """Number of harmonics fitted per block."""
        return int(self.amplitudes.shape[2])

    @property
    def device(self) -> torch.device:
        """Torch compute device where atom tensors reside."""
        return self.anchors.device

    def num_params(self) -> int:
        """Total float parameter count across all blocks and rows."""
        return self.rows * self.num_blocks * (1 + 3 * self.harmonic_count)

    def permutation_bytes(self) -> int:
        """Serialized bytes used by the optional shared uint32 column order."""
        return 0 if self.column_order is None else self.cols * 4

    def to(self, target_device: torch.device | str) -> "BlockAtom":
        """Return a copy of this BlockAtom with tensors transferred to target_device."""
        return BlockAtom(
            anchors=self.anchors.to(target_device),
            amplitudes=self.amplitudes.to(target_device),
            frequencies=self.frequencies.to(target_device),
            phases=self.phases.to(target_device),
            block_size=self.block_size,
            num_blocks=self.num_blocks,
            rows=self.rows,
            cols=self.cols,
            column_order=(
                None if self.column_order is None else self.column_order.to(target_device)
            ),
        )

    def detach_clone(self) -> "BlockAtom":
        """Return a detached clone safe to hold after gradient optimization."""
        return BlockAtom(
            anchors=self.anchors.detach().clone(),
            amplitudes=self.amplitudes.detach().clone(),
            frequencies=self.frequencies.detach().clone(),
            phases=self.phases.detach().clone(),
            block_size=self.block_size,
            num_blocks=self.num_blocks,
            rows=self.rows,
            cols=self.cols,
            column_order=(
                None
                if self.column_order is None
                else self.column_order.detach().clone()
            ),
        )
