"""Deterministic shared-column ordering transforms for weight matrices.

A single order is shared by every row so its storage cost is O(cols), and a
procedural matrix multiply can consume reordered weights by gathering the input
activation once with the same order.
"""

from __future__ import annotations

import torch


COLUMN_ORDERING_NONE = "none"
COLUMN_ORDERING_PCA = "pca"
SUPPORTED_COLUMN_ORDERINGS = frozenset(
    {COLUMN_ORDERING_NONE, COLUMN_ORDERING_PCA}
)


def compute_pca_column_order(
    weight_tensor: torch.Tensor, power_iterations: int = 12
) -> torch.Tensor:
    """Return a deterministic order that sorts columns by first-PC projection.

    Columns are samples and rows are features. Power iteration is performed
    without materializing the potentially large row-by-row covariance matrix.
    The result contains original column indices in encoded-column order.
    """
    if weight_tensor.ndim != 2:
        raise ValueError(
            f"Expected a 2-D weight tensor, got shape {tuple(weight_tensor.shape)}"
        )
    if power_iterations < 1:
        raise ValueError("power_iterations must be positive")

    row_count, column_count = weight_tensor.shape
    if column_count <= 1:
        return torch.arange(column_count, dtype=torch.int64, device=weight_tensor.device)

    values = weight_tensor.to(torch.float32)
    centered = values - values.mean(dim=1, keepdim=True)

    # A deterministic non-constant seed avoids introducing RNG state into the
    # binary artifact. Degenerate all-zero matrices fall back to identity.
    direction = torch.linspace(
        -1.0, 1.0, row_count, dtype=torch.float32, device=values.device
    )
    direction_norm = torch.linalg.vector_norm(direction)
    if direction_norm == 0:
        direction = torch.ones(row_count, dtype=torch.float32, device=values.device)
        direction_norm = torch.linalg.vector_norm(direction)
    direction = direction / direction_norm

    for _ in range(power_iterations):
        next_direction = centered @ (centered.transpose(0, 1) @ direction)
        next_norm = torch.linalg.vector_norm(next_direction)
        if not torch.isfinite(next_norm) or float(next_norm) <= torch.finfo(torch.float32).eps:
            return torch.arange(
                column_count, dtype=torch.int64, device=weight_tensor.device
            )
        direction = next_direction / next_norm

    # Fix the arbitrary PCA sign so identical inputs always serialize identically.
    largest_component = int(torch.argmax(direction.abs()).item())
    if float(direction[largest_component]) < 0.0:
        direction = -direction

    projection = centered.transpose(0, 1) @ direction
    return torch.argsort(projection, stable=True).to(torch.int64)


def compute_column_order(
    weight_tensor: torch.Tensor, ordering: str
) -> torch.Tensor | None:
    """Compute the requested shared-column order, or ``None`` for no transform."""
    if ordering == COLUMN_ORDERING_NONE:
        return None
    if ordering == COLUMN_ORDERING_PCA:
        return compute_pca_column_order(weight_tensor)
    raise ValueError(
        f"Unsupported column ordering {ordering!r}; expected one of "
        f"{sorted(SUPPORTED_COLUMN_ORDERINGS)}"
    )
