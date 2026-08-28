"""Encoder domain — turns weight tensors into BlockAtoms.

Public API:

- :class:`EncoderConfig` — immutable configuration for a single encode.
- :class:`FitSearchSpace` — bounds for auto-selection.
- :class:`FitResult` — outcome of auto-selection.
- :func:`encode_tensor` — encode one tensor with a fixed configuration.
- :func:`fit_tensor` — auto-select configuration to meet a fidelity target.
- :func:`compute_relative_error` — Frobenius relative error.
- :func:`extract_blocks` — low-level block geometry (re-exported for tests).
- :func:`fit_harmonics_to_block` — low-level per-block fit (re-exported for tests).
"""

from src.encoder.auto_fitter import (
    ConfigEvaluation,
    FitResult,
    FitSearchSpace,
    compute_relative_error,
    fit_tensor,
)
from src.encoder.block_extractor import (
    compute_block_length,
    count_blocks,
    extract_blocks,
)
from src.encoder.column_ordering import compute_column_order, compute_pca_column_order
from src.encoder.config import EncoderConfig
from src.encoder.harmonic_fitter import fit_harmonics_to_block
from src.encoder.tensor_encoder import encode_tensor

__all__ = [
    "ConfigEvaluation",
    "EncoderConfig",
    "FitResult",
    "FitSearchSpace",
    "compute_block_length",
    "compute_column_order",
    "compute_pca_column_order",
    "compute_relative_error",
    "count_blocks",
    "encode_tensor",
    "extract_blocks",
    "fit_harmonics_to_block",
    "fit_tensor",
]
