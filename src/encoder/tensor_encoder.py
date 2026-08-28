"""Encoder domain — full tensor encoding orchestration.

Orchestrates block partitioning and per-block sinusoid fitting across all rows
of a 2-D weight matrix to build a self-describing BlockAtom.
"""

from __future__ import annotations

import torch

from src.encoder.block_extractor import count_blocks, extract_blocks
from src.encoder.column_ordering import compute_column_order
from src.encoder.config import EncoderConfig
from src.encoder.harmonic_fitter import fit_harmonics_to_block
from src.phasor_atom import BlockAtom


def extract_difference_slice_for_block(
    all_first_differences: torch.Tensor,
    row_index: int,
    block_index: int,
    actual_block_length: int,
) -> torch.Tensor:
    """Extract unpadded first-difference slice for a specific row and block."""
    if actual_block_length <= 1:
        return torch.empty(0, dtype=torch.float32)

    valid_difference_count = actual_block_length - 1
    return all_first_differences[row_index, block_index, :valid_difference_count]


def encode_tensor(
    weight_tensor: torch.Tensor, configuration: EncoderConfig
) -> BlockAtom:
    """Encode a 2-D weight tensor into a BlockAtom parameter container.

    Args:
        weight_tensor: 2-D tensor of shape (rows, cols).
        configuration: EncoderConfig specifying block_size, harmonic_count, etc.

    Returns:
        BlockAtom parameter container containing exact anchors and fitted sinusoids.
    """
    weight_tensor = weight_tensor.to(torch.float32)
    row_count, total_columns = weight_tensor.shape
    column_order = compute_column_order(weight_tensor, configuration.column_ordering)
    encoded_weight_tensor = (
        weight_tensor if column_order is None else weight_tensor[:, column_order]
    )
    block_count = count_blocks(total_columns, configuration.block_size)

    (
        anchor_weights,
        first_differences,
        block_lengths,
    ) = extract_blocks(encoded_weight_tensor, configuration.block_size)

    all_amplitudes = torch.empty(
        row_count, block_count, configuration.harmonic_count, dtype=torch.float32
    )
    all_frequencies = torch.empty(
        row_count, block_count, configuration.harmonic_count, dtype=torch.float32
    )
    all_phases = torch.empty(
        row_count, block_count, configuration.harmonic_count, dtype=torch.float32
    )

    for row_index in range(row_count):
        for block_index in range(block_count):
            actual_block_length = int(block_lengths[block_index].item())
            difference_slice = extract_difference_slice_for_block(
                first_differences, row_index, block_index, actual_block_length
            )

            (
                amplitudes,
                frequencies,
                phases,
            ) = fit_harmonics_to_block(difference_slice, configuration)

            all_amplitudes[row_index, block_index] = amplitudes
            all_frequencies[row_index, block_index] = frequencies
            all_phases[row_index, block_index] = phases

    return BlockAtom(
        anchors=anchor_weights,
        amplitudes=all_amplitudes,
        frequencies=all_frequencies,
        phases=all_phases,
        block_size=configuration.block_size,
        num_blocks=block_count,
        rows=row_count,
        cols=total_columns,
        column_order=column_order,
    )
