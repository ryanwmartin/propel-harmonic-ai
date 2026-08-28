"""Decoder domain module.

Re-exports procedural decoding operations for reconstructing weight matrices
from per-block harmonic parameter containers (BlockAtoms).
"""

from src.decoder.block_decoder import (
    compute_block_length,
    decode_full_tensor,
    decode_row_weights,
    evaluate_block_harmonic_derivative,
)

__all__ = [
    "compute_block_length",
    "decode_full_tensor",
    "decode_row_weights",
    "evaluate_block_harmonic_derivative",
]
