"""Encoder domain — row block geometry and anchor extraction.

Partitions a 2-D weight tensor into row blocks, records exact float32 anchor
weights at block boundaries, and calculates first-difference vectors.
"""

from __future__ import annotations

import torch


def compute_block_length(block_size: int, total_columns: int, block_index: int) -> int:
    """Return the exact element count of block_index given total_columns.

    Every block has length block_size except possibly the terminal block in a row,
    which is truncated if total_columns is not an integer multiple of block_size.

    Args:
        block_size: Configured maximum block length.
        total_columns: Total number of columns in the weight tensor.
        block_index: Zero-based block index.

    Returns:
        Number of weights in this block (integer >= 1).
    """
    column_start_index = block_index * block_size
    column_end_index = min(column_start_index + block_size, total_columns)
    return column_end_index - column_start_index


def count_blocks(total_columns: int, block_size: int) -> int:
    """Return the total number of blocks required to span total_columns."""
    return (total_columns + block_size - 1) // block_size


def extract_blocks(
    weight_tensor: torch.Tensor, block_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Partition weight_tensor into blocks and extract anchor weights and derivatives.

    For each block in each row:
    - The anchor is recorded as weight_tensor[row_index, block_start_column].
    - The derivative is computed as d[x] = W[x+1] - W[x] for x ∈ [0, block_length - 1).

    Args:
        weight_tensor: 2-D float tensor of shape (rows, cols).
        block_size: Maximum block column span.

    Returns:
        Tuple of (anchor_weights, first_differences, block_lengths) where:
        - anchor_weights: Shape (rows, block_count) float32 tensor
        - first_differences: Shape (rows, block_count, block_size - 1) float32 tensor
        - block_lengths: Shape (block_count,) int64 tensor
    """
    if weight_tensor.ndim != 2:
        raise ValueError(
            f"weight_tensor must be 2-D, got shape {tuple(weight_tensor.shape)}"
        )

    row_count, total_columns = weight_tensor.shape
    block_count = count_blocks(total_columns, block_size)
    maximum_difference_length = block_size - 1

    anchor_weights = torch.empty(row_count, block_count, dtype=torch.float32)
    first_differences = torch.zeros(
        row_count, block_count, maximum_difference_length, dtype=torch.float32
    )
    block_lengths = torch.empty(block_count, dtype=torch.int64)

    for block_index in range(block_count):
        current_block_length = compute_block_length(
            block_size, total_columns, block_index
        )
        block_lengths[block_index] = current_block_length
        column_start_index = block_index * block_size

        anchor_weights[:, block_index] = weight_tensor[:, column_start_index]

        if current_block_length > 1:
            block_weight_slice = weight_tensor[
                :, column_start_index : column_start_index + current_block_length
            ]
            calculated_differences = (
                block_weight_slice[:, 1:] - block_weight_slice[:, :-1]
            )
            first_differences[
                :, block_index, : current_block_length - 1
            ] = calculated_differences

    return anchor_weights, first_differences, block_lengths
