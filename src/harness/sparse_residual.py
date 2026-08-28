"""Harness domain — reusable sparse residual component.

Experiment 5 (low-rank + sparse residual) and Experiment 11 (procedural
component + residual) both add a small explicit correction on top of a
structured component:

    weight_matrix = structured_component + sparse_residual

Per the accounting rules, residual values **and** indices are unique model
state and must be fully counted. The residual is applied directly to
activations (``S @ x``) via an index gather + scatter-add, so the dense
residual matrix is never materialized in the hot path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.harness.quantization import quantize_field

RESIDUAL_INDEX_BITS: int = 32
"""Flat (row * cols + col) index storage width. 32 bits covers any tensor in
Pythia-14m or Gemma 2B (< 2^31 elements per projection)."""


@dataclass(frozen=True)
class SparseResidual:
    """Top-magnitude sparse correction for one (rows, cols) matrix.

    Attributes:
        flat_indices: int64 tensor (nnz,) of ``row * cols + col`` positions.
        values: float32 tensor (nnz,) of correction values (already
            round-tripped through the storage precision).
        rows: Row count of the corrected dense matrix.
        cols: Column count of the corrected dense matrix.
        value_bits: Storage precision of each residual value.
    """

    flat_indices: torch.Tensor
    values: torch.Tensor
    rows: int
    cols: int
    value_bits: int

    def __post_init__(self) -> None:
        if self.flat_indices.shape != self.values.shape:
            raise ValueError(
                "flat_indices and values must have identical shapes, got "
                f"{tuple(self.flat_indices.shape)} vs {tuple(self.values.shape)}"
            )
        if self.flat_indices.numel() > 0:
            maximum_index = int(self.flat_indices.max())
            if maximum_index >= self.rows * self.cols:
                raise ValueError(
                    f"flat index {maximum_index} out of range for "
                    f"{self.rows}x{self.cols} matrix"
                )

    @classmethod
    def fit(
        cls,
        error_matrix: torch.Tensor,
        density: float,
        value_bits: int = 16,
        column_importance: torch.Tensor | None = None,
    ) -> "SparseResidual":
        """Keep the top-``density`` fraction of error entries by importance.

        Args:
            error_matrix: Dense (rows, cols) approximation error
                ``W - structured_component`` (research-side only; the hot
                path never sees this matrix).
            density: Fraction of entries to keep, in [0, 1].
            value_bits: Storage precision for residual values.
            column_importance: Optional (cols,) positive weights — typically
                per-column activation RMS ``sqrt(E[x_j^2])`` (Experiment 12).
                When given, entries are ranked by
                ``|error_ij| * importance_j`` (their effect on the layer
                output) instead of raw magnitude. Stored values are always
                the unweighted errors.

        Returns:
            Fitted sparse residual.
        """
        if not 0.0 <= density <= 1.0:
            raise ValueError(f"density must be in [0, 1], got {density}")
        row_count, column_count = error_matrix.shape
        if column_importance is not None and column_importance.shape != (
            column_count,
        ):
            raise ValueError(
                f"column_importance must have shape ({column_count},), got "
                f"{tuple(column_importance.shape)}"
            )
        keep_count = int(round(density * row_count * column_count))
        flat_error = error_matrix.reshape(-1).to(torch.float32)
        if keep_count == 0:
            empty = torch.empty(0)
            return cls(
                flat_indices=empty.to(torch.int64),
                values=empty.to(torch.float32),
                rows=row_count,
                cols=column_count,
                value_bits=value_bits,
            )
        if column_importance is None:
            ranking_scores = flat_error.abs()
        else:
            ranking_scores = (
                error_matrix.to(torch.float32).abs()
                * column_importance.to(torch.float32).unsqueeze(0)
            ).reshape(-1)
        _, top_indices = torch.topk(ranking_scores, keep_count)
        top_indices, _ = torch.sort(top_indices)
        return cls(
            flat_indices=top_indices,
            values=quantize_field(flat_error[top_indices], value_bits),
            rows=row_count,
            cols=column_count,
            value_bits=value_bits,
        )

    @property
    def nnz(self) -> int:
        """Number of stored residual entries."""
        return int(self.flat_indices.numel())

    def state_bits(self) -> dict[str, int]:
        """Per-field unique model-state bits (values + indices)."""
        return {
            "residual_values": self.nnz * self.value_bits,
            "residual_indices": self.nnz * RESIDUAL_INDEX_BITS,
        }

    def total_bytes(self) -> float:
        """Total residual model-state bytes."""
        return sum(self.state_bits().values()) / 8.0

    def apply(self, input_activations: torch.Tensor) -> torch.Tensor:
        """Compute ``S @ x`` without materializing the dense residual.

        Args:
            input_activations: Tensor of shape (cols, batch).

        Returns:
            Tensor of shape (rows, batch).
        """
        batch_size = input_activations.shape[1]
        output = torch.zeros(
            self.rows, batch_size, dtype=torch.float32,
            device=input_activations.device,
        )
        if self.nnz == 0:
            return output
        row_indices = self.flat_indices // self.cols
        column_indices = self.flat_indices % self.cols
        contributions = self.values.unsqueeze(1) * input_activations[
            column_indices
        ]
        output.index_add_(0, row_indices, contributions)
        return output

    def to_dense(self) -> torch.Tensor:
        """Diagnostic dense residual matrix (never used in the hot path)."""
        dense = torch.zeros(self.rows * self.cols, dtype=torch.float32)
        dense[self.flat_indices] = self.values
        return dense.reshape(self.rows, self.cols)
