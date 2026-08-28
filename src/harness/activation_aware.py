"""Experiment 12 stage 1 — closed-form activation-aware fitting math.

Weight-space fitting minimizes ``||W - What||_F``, which treats every input
direction as equally likely. Real inference feeds a layer activations from a
narrow distribution, so the correct stage-1 objective is the **on-distribution
layer output error**:

    E_x ||(W - What) x||^2  =  ||(W - What) M||_F^2,     M = C^{1/2},

where ``C = E[x x^T]`` is the activation second moment estimated from cached
teacher activations (Epic 04, small scale). Because the objective is still a
Frobenius norm after the change of coordinates ``M``, every closed-form fit in
the harness (truncated SVD, shared-basis projection, weighted least squares)
carries over exactly — no gradient training, no distillation loop.

This module holds the shared math:

- :func:`second_moment_square_root` — damped symmetric ``C^{1/2}`` and
  ``C^{-1/2}`` via eigendecomposition.
- :func:`activation_aware_low_rank_factors` — the rank-r minimizer of the
  on-distribution objective, ``What = [W M]_r M^{-1}`` (whitened SVD).
- :func:`column_activation_importance` — per-input-column activation RMS
  ``sqrt(C_jj)`` used to rank sparse-residual entries by their effect on the
  layer output rather than by raw weight error.
- :func:`block_average_second_moment` — the block-position-averaged
  ``(L, L)`` second moment used by the shared-basis family, which shares one
  whitener across all blocks so the fitted basis stays shared.
"""

from __future__ import annotations

import torch

DEFAULT_EIGENVALUE_FLOOR_FRACTION: float = 1e-6
"""Eigenvalues of the second moment are floored at this fraction of the
largest eigenvalue before taking (inverse) square roots, so near-null
activation directions cannot blow up the inverse whitener."""


def second_moment_square_root(
    second_moment: torch.Tensor,
    eigenvalue_floor_fraction: float = DEFAULT_EIGENVALUE_FLOOR_FRACTION,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Damped symmetric square root and inverse square root of ``C``.

    Args:
        second_moment: Symmetric positive semi-definite (n, n) activation
            second moment ``E[x x^T]``.
        eigenvalue_floor_fraction: Eigenvalue floor as a fraction of the
            largest eigenvalue (damping for near-singular directions).

    Returns:
        Tuple ``(M, M_inverse)`` with ``M @ M ≈ C`` (up to damping) and
        ``M @ M_inverse ≈ I``.

    Raises:
        ValueError: If the matrix is not square or the floor fraction is not
            in (0, 1).
    """
    if second_moment.dim() != 2 or second_moment.shape[0] != second_moment.shape[1]:
        raise ValueError(
            f"second_moment must be square, got {tuple(second_moment.shape)}"
        )
    if not 0.0 < eigenvalue_floor_fraction < 1.0:
        raise ValueError(
            "eigenvalue_floor_fraction must be in (0, 1), got "
            f"{eigenvalue_floor_fraction}"
        )
    symmetrized = 0.5 * (second_moment + second_moment.T).to(torch.float64)
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetrized)
    eigenvalue_floor = eigenvalue_floor_fraction * float(
        eigenvalues.max().clamp_min(torch.finfo(torch.float64).tiny)
    )
    clamped_eigenvalues = eigenvalues.clamp_min(eigenvalue_floor)
    sqrt_eigenvalues = clamped_eigenvalues.sqrt()
    square_root = (
        eigenvectors @ torch.diag(sqrt_eigenvalues) @ eigenvectors.T
    ).to(torch.float32)
    inverse_square_root = (
        eigenvectors @ torch.diag(1.0 / sqrt_eigenvalues) @ eigenvectors.T
    ).to(torch.float32)
    return square_root, inverse_square_root


def activation_aware_low_rank_factors(
    weight_matrix: torch.Tensor,
    second_moment: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank-r factors minimizing the on-distribution layer output error.

    Solves ``min_{rank r} ||(W - U V) M||_F`` in closed form: truncate the
    SVD of the whitened matrix ``W M`` and map the column factor back through
    ``M^{-1}``.

    Args:
        weight_matrix: Dense (rows, cols) float32 matrix.
        second_moment: (cols, cols) activation second moment.
        rank: Retained rank, in ``[1, min(rows, cols)]``.

    Returns:
        Tuple ``(row_factor, column_factor)`` of shapes (rows, rank) and
        (rank, cols) whose product is the activation-aware approximation.
    """
    weight_matrix = weight_matrix.to(torch.float32)
    row_count, column_count = weight_matrix.shape
    maximum_rank = min(row_count, column_count)
    if not 1 <= rank <= maximum_rank:
        raise ValueError(
            f"rank must be in [1, {maximum_rank}] for a "
            f"{row_count}x{column_count} matrix, got {rank}"
        )
    if second_moment.shape != (column_count, column_count):
        raise ValueError(
            f"second_moment must be ({column_count}, {column_count}), got "
            f"{tuple(second_moment.shape)}"
        )
    whitener, inverse_whitener = second_moment_square_root(second_moment)
    left_vectors, singular_values, right_vectors_transposed = torch.linalg.svd(
        weight_matrix @ whitener, full_matrices=False
    )
    scale = singular_values[:rank].sqrt()
    row_factor = left_vectors[:, :rank] * scale
    column_factor = (
        scale.unsqueeze(1) * right_vectors_transposed[:rank]
    ) @ inverse_whitener
    return row_factor, column_factor


def column_activation_importance(second_moment: torch.Tensor) -> torch.Tensor:
    """Per-input-column activation RMS ``sqrt(C_jj)``.

    A weight error in column ``j`` perturbs the layer output in proportion to
    the typical magnitude of activation ``x_j``, so sparse-residual entries
    are ranked by ``|error_ij| * sqrt(C_jj)`` instead of raw ``|error_ij|``.

    Args:
        second_moment: (cols, cols) activation second moment.

    Returns:
        Float32 tensor of shape (cols,), clamped to be strictly positive.
    """
    diagonal = torch.diagonal(second_moment).to(torch.float32)
    return diagonal.clamp_min(0.0).sqrt().clamp_min(torch.finfo(torch.float32).tiny)


def block_average_second_moment(
    second_moment: torch.Tensor, block_size: int
) -> torch.Tensor:
    """Average of the per-block diagonal sub-blocks of ``C``.

    The shared-basis family models within-block structure with **one** basis
    per tensor, so its whitener must also be shared across blocks. Averaging
    the (L, L) diagonal sub-blocks of the full (cols, cols) second moment
    yields the single block-position covariance that keeps the fitted basis
    shared while still weighting block positions by how strongly activations
    excite them.

    Args:
        second_moment: (cols, cols) activation second moment.
        block_size: Block length L; must divide cols.

    Returns:
        Float32 (block_size, block_size) averaged second moment.
    """
    column_count = second_moment.shape[0]
    if column_count % block_size != 0:
        raise ValueError(
            f"block_size {block_size} must divide cols {column_count}"
        )
    block_count = column_count // block_size
    blocks = [
        second_moment[
            block_index * block_size : (block_index + 1) * block_size,
            block_index * block_size : (block_index + 1) * block_size,
        ]
        for block_index in range(block_count)
    ]
    return torch.stack(blocks).mean(dim=0).to(torch.float32)
