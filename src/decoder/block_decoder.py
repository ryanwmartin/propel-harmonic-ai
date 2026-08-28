"""Decoder domain — procedural reconstruction of weight tensors from BlockAtoms.

This module implements the exact reference decode contract:
1. Evaluate the 1-D harmonic derivative formula G(x) = Σ_k A_k · sin(2π f_k x + φ_k)
   within each block.
2. Integrate the derivative signal starting from the exact float32 anchor weight via
   a running cumulative sum (prefix-sum).

This decode logic is the single source of truth that the Rust decoder must match.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from src.phasor_atom import BlockAtom

TWO_PI: float = 2.0 * torch.pi


def compute_block_length(
    block_size: int, total_columns: int, block_index: int
) -> int:
    """Calculate the exact length of a block given its index and tensor width.

    Args:
        block_size: Maximum configured block length.
        total_columns: Total number of columns in the weight tensor.
        block_index: Zero-based block index.

    Returns:
        The number of weight elements in this block.
    """
    column_start_index = block_index * block_size
    column_end_index = min(column_start_index + block_size, total_columns)
    return column_end_index - column_start_index


def evaluate_block_harmonic_derivative(
    atom: BlockAtom, row_index: int, block_index: int
) -> torch.Tensor:
    """Evaluate the 1-D harmonic derivative formula for a single block.

    Calculates G(sample_index) = Σ_k A_k · sin(2π f_k sample_index + φ_k) for
    sample_index ∈ [0, block_length).

    Args:
        atom: The BlockAtom parameter container.
        row_index: Zero-based row index in the weight matrix.
        block_index: Zero-based block index within the row.

    Returns:
        1-D float32 tensor of length block_length containing predicted first differences.
    """
    current_block_length = compute_block_length(
        atom.block_size, atom.cols, block_index
    )
    sample_indices = torch.arange(
        current_block_length, dtype=torch.float32, device=atom.device
    )

    frequencies = atom.frequencies[row_index, block_index].unsqueeze(0)
    phases = atom.phases[row_index, block_index].unsqueeze(0)
    amplitudes = atom.amplitudes[row_index, block_index].unsqueeze(0)

    phase_arguments = (
        TWO_PI * frequencies * sample_indices.unsqueeze(1) + phases
    )
    harmonic_components = amplitudes * torch.sin(phase_arguments)
    harmonic_derivative_sum = harmonic_components.sum(dim=1)

    return harmonic_derivative_sum


def decode_row_weights(atom: BlockAtom, row_index: int) -> torch.Tensor:
    """Decode an entire weight matrix row by integrating block derivative estimates.

    Integration semantics (the parity contract): the running prefix-sum
    accumulates in **float32** via ``torch.cumsum`` on a float32 tensor. The
    Rust decoder (Epic 02) must accumulate in f32 to reproduce these outputs
    within the 1e-6 parity tolerance mandated by Epic 03. Do not widen the
    accumulator to f64 on either side.

    Args:
        atom: The BlockAtom parameter container.
        row_index: Zero-based row index to reconstruct.

    Returns:
        1-D float32 tensor of shape (total_columns,) with reconstructed weights.
    """
    reconstructed_row = torch.empty(
        atom.cols, dtype=torch.float32, device=atom.device
    )

    for block_index in range(atom.num_blocks):
        current_block_length = compute_block_length(
            atom.block_size, atom.cols, block_index
        )
        column_start_index = block_index * atom.block_size
        anchor_weight = atom.anchors[row_index, block_index]
        reconstructed_row[column_start_index] = anchor_weight

        if current_block_length > 1:
            harmonic_derivatives = evaluate_block_harmonic_derivative(
                atom, row_index, block_index
            )[: current_block_length - 1]

            # Ŵ[x+1] = anchor + Σ_{i<=x} G(i) — float32 running prefix-sum.
            integrated_weights = anchor_weight + torch.cumsum(
                harmonic_derivatives, dim=0
            )
            reconstructed_row[
                column_start_index + 1 : column_start_index + current_block_length
            ] = integrated_weights

    if atom.column_order is None:
        return reconstructed_row

    original_order_row = torch.empty_like(reconstructed_row)
    original_order_row[atom.column_order] = reconstructed_row
    return original_order_row


def decode_full_tensor(atom: BlockAtom) -> torch.Tensor:
    """Procedurally synthesize the complete weight tensor from block parameters.

    Args:
        atom: The BlockAtom containing per-row, per-block parameters.

    Returns:
        2-D float32 tensor of shape (rows, cols) with reconstructed weights.
    """
    reconstructed_rows = [
        decode_row_weights(atom, row_index) for row_index in range(atom.rows)
    ]
    return torch.stack(reconstructed_rows, dim=0)
