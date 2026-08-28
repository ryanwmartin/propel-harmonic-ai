"""Experiment 2 — reversible ordering and canonicalization search.

A dense matrix's training-time row/column order is arbitrary. This module
builds cheap, exactly reversible canonicalizations whose state is shared
across many weights:

    transformed[i, j] = weight[row_perm[i], col_perm[j]] * sign[j] / scale[j]

The canonicalization state is O(rows + cols) — permutation indices, one sign
bit per column, one FP16 scale per column — never O(rows * cols). Per the
roadmap, the column transform is moved into the input activations in the hot
path (gather + multiply on ``x``), and the row permutation is applied to the
output vector, so weights are never physically sorted and unsorted per call.

Ordering strategies (all operate on a (dim, n) stack of column vectors; row
orderings pass the transposed matrix):

- ``column_norm_order`` — sort by vector norm (cheap magnitude baseline).
- ``spectral_seriation_order`` — Fiedler-vector ordering of the |cosine|
  similarity graph (joint seriation from the roadmap).
- ``greedy_nearest_neighbor_order`` — TSP-like greedy chain under a
  sign-invariant distance.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

SIGN_BITS_PER_COLUMN: int = 1
"""One stored sign bit per canonicalized column."""

SCALE_BITS_PER_COLUMN: int = 16
"""One FP16 scale per canonicalized column."""

_SCALE_EPSILON: float = 1e-12
"""Floor preventing division by zero for all-zero columns."""


def index_bits(permutation_length: int) -> int:
    """Storage width of one permutation index (16-bit up to 65536 entries)."""
    return 16 if permutation_length <= 1 << 16 else 32


def column_norm_order(column_vectors: torch.Tensor) -> torch.Tensor:
    """Order columns by descending Euclidean norm.

    Args:
        column_vectors: (dim, n) matrix whose columns are ordered.

    Returns:
        int64 permutation tensor of shape (n,).
    """
    norms = torch.linalg.norm(column_vectors, dim=0)
    return torch.argsort(norms, descending=True)


def spectral_seriation_order(column_vectors: torch.Tensor) -> torch.Tensor:
    """Order columns along the Fiedler vector of their similarity graph.

    Builds a sign-invariant |cosine| similarity graph over columns, forms the
    unnormalized graph Laplacian, and sorts columns by the eigenvector of the
    second-smallest eigenvalue. This places strongly correlated columns next
    to each other, the classic seriation heuristic.

    Args:
        column_vectors: (dim, n) matrix whose columns are ordered.

    Returns:
        int64 permutation tensor of shape (n,).
    """
    normalized = column_vectors / torch.linalg.norm(
        column_vectors, dim=0, keepdim=True
    ).clamp_min(_SCALE_EPSILON)
    similarity = (normalized.T @ normalized).abs()
    similarity.fill_diagonal_(0.0)
    degree = similarity.sum(dim=1)
    laplacian = torch.diag(degree) - similarity
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
    del eigenvalues
    fiedler_vector = eigenvectors[:, 1]
    return torch.argsort(fiedler_vector)


def greedy_nearest_neighbor_order(column_vectors: torch.Tensor) -> torch.Tensor:
    """TSP-like greedy chain under a sign-invariant Euclidean distance.

    Starts from the largest-norm column, then repeatedly appends the nearest
    remaining column where distance is ``min(||a - b||, ||a + b||)`` so that
    a later sign-canonicalization pass can absorb the flip.

    Args:
        column_vectors: (dim, n) matrix whose columns are ordered.

    Returns:
        int64 permutation tensor of shape (n,).
    """
    column_count = column_vectors.shape[1]
    squared_norms = (column_vectors * column_vectors).sum(dim=0)
    gram = column_vectors.T @ column_vectors
    # d^2(a, b) with the better of the two signs = |a|^2 + |b|^2 - 2*|<a, b>|.
    sign_invariant_distances = (
        squared_norms.unsqueeze(0) + squared_norms.unsqueeze(1) - 2.0 * gram.abs()
    )

    remaining = torch.ones(column_count, dtype=torch.bool)
    order = torch.empty(column_count, dtype=torch.int64)
    current = int(torch.argmax(squared_norms))
    order[0] = current
    remaining[current] = False
    for position in range(1, column_count):
        distances = sign_invariant_distances[current].clone()
        distances[~remaining] = float("inf")
        current = int(torch.argmin(distances))
        order[position] = current
        remaining[current] = False
    return order


def fit_column_signs(matrix: torch.Tensor) -> torch.Tensor:
    """Flip column signs so each column correlates positively with its left neighbor.

    Args:
        matrix: (rows, cols) matrix in its final column order.

    Returns:
        Float32 tensor of shape (cols,) with entries in {-1.0, +1.0}.
    """
    column_count = matrix.shape[1]
    signs = torch.ones(column_count, dtype=torch.float32)
    previous_column = matrix[:, 0]
    for column_index in range(1, column_count):
        current_column = matrix[:, column_index]
        if torch.dot(previous_column, current_column) < 0.0:
            signs[column_index] = -1.0
            previous_column = -current_column
        else:
            previous_column = current_column
    return signs


def fit_column_scales(matrix: torch.Tensor) -> torch.Tensor:
    """Per-column RMS scales, round-tripped through FP16 storage.

    Args:
        matrix: (rows, cols) matrix in its final column order and signs.

    Returns:
        Float32 tensor of shape (cols,) of strictly positive scales exactly
        representable in FP16.
    """
    root_mean_square = matrix.pow(2).mean(dim=0).sqrt().clamp_min(_SCALE_EPSILON)
    return root_mean_square.to(torch.float16).to(torch.float32).clamp_min(
        _SCALE_EPSILON
    )


@dataclass(frozen=True)
class MatrixCanonicalization:
    """An exactly reversible ordering + sign + scale canonicalization.

    ``None`` components are identity and contribute zero model-state bits.

    Attributes:
        row_permutation: int64 (rows,) or None; ``transformed`` row ``i``
            comes from original row ``row_permutation[i]``.
        column_permutation: int64 (cols,) or None; analogous for columns.
        column_signs: float32 (cols,) of ±1 applied after permutation, or None.
        column_scales: float32 (cols,) positive divisors applied after signs,
            or None.
        rows: Row count of the canonicalized matrix.
        cols: Column count of the canonicalized matrix.
    """

    row_permutation: torch.Tensor | None
    column_permutation: torch.Tensor | None
    column_signs: torch.Tensor | None
    column_scales: torch.Tensor | None
    rows: int
    cols: int

    def apply_to_matrix(self, weight_matrix: torch.Tensor) -> torch.Tensor:
        """Canonicalize a dense matrix (fitting-time only, never hot path)."""
        transformed = weight_matrix
        if self.row_permutation is not None:
            transformed = transformed[self.row_permutation]
        if self.column_permutation is not None:
            transformed = transformed[:, self.column_permutation]
        if self.column_signs is not None:
            transformed = transformed * self.column_signs.unsqueeze(0)
        if self.column_scales is not None:
            transformed = transformed / self.column_scales.unsqueeze(0)
        return transformed

    def restore_matrix(self, transformed_matrix: torch.Tensor) -> torch.Tensor:
        """Exact inverse of :meth:`apply_to_matrix` (diagnostic only)."""
        restored = transformed_matrix
        if self.column_scales is not None:
            restored = restored * self.column_scales.unsqueeze(0)
        if self.column_signs is not None:
            restored = restored * self.column_signs.unsqueeze(0)
        if self.column_permutation is not None:
            unpermuted = torch.empty_like(restored)
            unpermuted[:, self.column_permutation] = restored
            restored = unpermuted
        if self.row_permutation is not None:
            unpermuted = torch.empty_like(restored)
            unpermuted[self.row_permutation] = restored
            restored = unpermuted
        return restored

    def transform_activations(self, activations: torch.Tensor) -> torch.Tensor:
        """Hot-path column transform moved onto the input activations.

        Because ``transformed = W_perm * sign / scale``, computing
        ``W @ x`` from the transformed matrix requires
        ``z[j] = sign[j] * scale[j] * x[col_perm[j]]``.

        Args:
            activations: (cols, batch) input activations.

        Returns:
            (cols, batch) transformed activations ``z``.
        """
        transformed = activations
        if self.column_permutation is not None:
            transformed = transformed[self.column_permutation]
        multiplier = None
        if self.column_signs is not None:
            multiplier = self.column_signs
        if self.column_scales is not None:
            multiplier = (
                self.column_scales
                if multiplier is None
                else multiplier * self.column_scales
            )
        if multiplier is not None:
            transformed = transformed * multiplier.unsqueeze(1)
        return transformed

    def restore_output(self, permuted_output: torch.Tensor) -> torch.Tensor:
        """Hot-path inverse row permutation on the output vector.

        Args:
            permuted_output: (rows, batch) output in canonical row order.

        Returns:
            (rows, batch) output in original row order.
        """
        if self.row_permutation is None:
            return permuted_output
        restored = torch.empty_like(permuted_output)
        restored[self.row_permutation] = permuted_output
        return restored

    def state_bits(self) -> dict[str, int]:
        """Per-field unique model-state bits for all stored components."""
        field_bits: dict[str, int] = {}
        if self.row_permutation is not None:
            field_bits["row_permutation"] = self.rows * index_bits(self.rows)
        if self.column_permutation is not None:
            field_bits["column_permutation"] = self.cols * index_bits(self.cols)
        if self.column_signs is not None:
            field_bits["column_signs"] = self.cols * SIGN_BITS_PER_COLUMN
        if self.column_scales is not None:
            field_bits["column_scales"] = self.cols * SCALE_BITS_PER_COLUMN
        return field_bits

    def extra_transform_operations(self) -> int:
        """Hot-path arithmetic added per layer call (gathers + multiplies)."""
        operation_count = 0
        if self.column_permutation is not None:
            operation_count += self.cols
        if self.column_signs is not None or self.column_scales is not None:
            operation_count += self.cols
        if self.row_permutation is not None:
            operation_count += self.rows
        return operation_count


ORDERING_STRATEGIES = {
    "identity": None,
    "col-norm": column_norm_order,
    "spectral": spectral_seriation_order,
    "greedy-nn": greedy_nearest_neighbor_order,
}
"""Named column-ordering strategies available to the experiment runner."""


def build_canonicalization(
    weight_matrix: torch.Tensor,
    column_strategy: str,
    order_rows: bool = False,
    use_signs: bool = False,
    use_scales: bool = False,
) -> MatrixCanonicalization:
    """Fit a canonicalization for one matrix.

    Args:
        weight_matrix: (rows, cols) float32 matrix.
        column_strategy: Key from :data:`ORDERING_STRATEGIES`.
        order_rows: Also order rows with the same strategy (applied to
            row vectors). Ignored for the identity strategy.
        use_signs: Fit per-column sign flips after ordering.
        use_scales: Fit per-column FP16 RMS scales after signs.

    Returns:
        Fitted :class:`MatrixCanonicalization`.

    Raises:
        ValueError: If ``column_strategy`` is unknown.
    """
    if column_strategy not in ORDERING_STRATEGIES:
        raise ValueError(
            f"column_strategy must be one of {sorted(ORDERING_STRATEGIES)}, "
            f"got {column_strategy!r}"
        )
    weight_matrix = weight_matrix.to(torch.float32)
    row_count, column_count = weight_matrix.shape

    strategy = ORDERING_STRATEGIES[column_strategy]
    column_permutation = None
    row_permutation = None
    if strategy is not None:
        column_permutation = strategy(weight_matrix)
        if order_rows:
            row_permutation = strategy(weight_matrix.T)

    working = weight_matrix
    if row_permutation is not None:
        working = working[row_permutation]
    if column_permutation is not None:
        working = working[:, column_permutation]

    column_signs = None
    if use_signs:
        column_signs = fit_column_signs(working)
        working = working * column_signs.unsqueeze(0)

    column_scales = None
    if use_scales:
        column_scales = fit_column_scales(working)

    return MatrixCanonicalization(
        row_permutation=row_permutation,
        column_permutation=column_permutation,
        column_signs=column_signs,
        column_scales=column_scales,
        rows=row_count,
        cols=column_count,
    )
