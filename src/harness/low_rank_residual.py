"""Experiment 5 — low-rank + sparse residual behind the harness interface.

Shared-state mechanism: every entry of the row factor ``U`` (rows, rank) and
the column factor ``V`` (rank, cols) participates in generating an entire row
or column of the represented matrix, so state is shared across ``rows`` or
``cols`` weights rather than stored per weight:

    weight_matrix ~= U @ V + S            (S = sparse top-magnitude residual)

Fitting is **closed-form** — no gradient training, no distillation loop:

- Weight-space mode (Experiment 5): truncated SVD of ``W`` minimizes
  ``||W - U V||_F`` — equal weight on every input direction.
- Activation-aware mode (Experiment 12 stage 1): given the teacher activation
  second moment ``C = E[x x^T]``, truncated SVD of the whitened matrix
  ``W C^{1/2}`` minimizes the **on-distribution layer output error**
  ``E_x ||(W - U V) x||^2``, and the residual is ranked by its effect on the
  output (``|error_ij| * sqrt(C_jj)``) instead of raw magnitude. The second
  moment is fitting-time state only — it is never inference model state.

Hot-path execution never materializes the dense matrix:

    output = U @ (V @ x) + S @ x

The intermediate ``V @ x`` has shape (rank, batch) — far below rows*cols —
and the sparse residual is applied by gather + scatter-add.
"""

from __future__ import annotations

import torch

from src.harness.accounting import StateAccounting
from src.harness.activation_aware import (
    activation_aware_low_rank_factors,
    column_activation_importance,
)
from src.harness.quantization import SUPPORTED_FIELD_BITS, quantize_field
from src.harness.representation import ProceduralRepresentation
from src.harness.sparse_residual import SparseResidual

METADATA_BITS: int = 32 * 8
"""Serialized header metadata (magic, version, geometry, rank, nnz) in bits."""


class LowRankResidualRepresentation(ProceduralRepresentation):
    """Truncated-SVD factors plus sparse top-magnitude residual.

    Args:
        weight_matrix: Dense (rows, cols) float32 matrix to represent.
        rank: Number of retained singular components.
        residual_density: Fraction of entries kept in the sparse residual.
        factor_bits: Storage precision of U and V entries (16 or 32).
        residual_bits: Storage precision of residual values (16 or 32).
        second_moment: Optional (cols, cols) teacher activation second moment
            ``E[x x^T]``. When given, the factors minimize the
            on-distribution layer output error instead of the weight-space
            Frobenius error, and the residual is activation-importance
            ranked (Experiment 12 stage 1). Fitting-time state only.
    """

    def __init__(
        self,
        weight_matrix: torch.Tensor,
        rank: int,
        residual_density: float = 0.0,
        factor_bits: int = 16,
        residual_bits: int = 16,
        second_moment: torch.Tensor | None = None,
    ) -> None:
        if factor_bits not in SUPPORTED_FIELD_BITS:
            raise ValueError(
                f"factor_bits must be one of {sorted(SUPPORTED_FIELD_BITS)}, "
                f"got {factor_bits}"
            )
        weight_matrix = weight_matrix.to(torch.float32)
        row_count, column_count = weight_matrix.shape
        maximum_rank = min(row_count, column_count)
        if not 1 <= rank <= maximum_rank:
            raise ValueError(
                f"rank must be in [1, {maximum_rank}] for a "
                f"{row_count}x{column_count} matrix, got {rank}"
            )
        if second_moment is not None and second_moment.shape != (
            column_count,
            column_count,
        ):
            raise ValueError(
                f"second_moment must be ({column_count}, {column_count}), "
                f"got {tuple(second_moment.shape)}"
            )

        self.rank = rank
        self.factor_bits = factor_bits
        self.activation_aware = second_moment is not None
        self._rows = row_count
        self._cols = column_count

        if second_moment is None:
            left_vectors, singular_values, right_vectors_transposed = (
                torch.linalg.svd(weight_matrix, full_matrices=False)
            )
            scale = singular_values[:rank].sqrt()
            row_factor = left_vectors[:, :rank] * scale
            column_factor = (
                scale.unsqueeze(1) * right_vectors_transposed[:rank]
            )
        else:
            row_factor, column_factor = activation_aware_low_rank_factors(
                weight_matrix, second_moment, rank
            )
        self.row_factor = quantize_field(row_factor, factor_bits)
        self.column_factor = quantize_field(column_factor, factor_bits)

        structured_error = weight_matrix - self.row_factor @ self.column_factor
        residual_importance = (
            column_activation_importance(second_moment)
            if second_moment is not None
            else None
        )
        self.residual = SparseResidual.fit(
            structured_error,
            residual_density,
            residual_bits,
            column_importance=residual_importance,
        )

    # ------------------------------------------------------------------
    # ProceduralRepresentation interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        fitting_mode = "act" if self.activation_aware else "wgt"
        return (
            "low-rank+sparse-residual"
            f"[r={self.rank},density={self.residual.nnz / (self._rows * self._cols):.4f},"
            f"factor={self.factor_bits}b,residual={self.residual.value_bits}b,"
            f"fit={fitting_mode}]"
        )

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def cols(self) -> int:
        return self._cols

    def state_accounting(self) -> StateAccounting:
        field_bits: dict[str, int] = {
            "row_factor": self._rows * self.rank * self.factor_bits,
            "column_factor": self.rank * self._cols * self.factor_bits,
            "metadata": METADATA_BITS,
        }
        field_bits.update(self.residual.state_bits())
        return StateAccounting(
            field_bits=field_bits,
            represented_weight_count=self._rows * self._cols,
        )

    def reconstruct(self) -> torch.Tensor:
        """Diagnostic dense reconstruction ``U @ V + S``."""
        return self.row_factor @ self.column_factor + self.residual.to_dense()

    def transform(self, input_activations: torch.Tensor) -> torch.Tensor:
        """Fused hot path: ``U @ (V @ x) + S @ x`` — dense W never formed."""
        squeeze_output = input_activations.dim() == 1
        activations = (
            input_activations.unsqueeze(1) if squeeze_output else input_activations
        ).to(torch.float32)
        if activations.shape[0] != self._cols:
            raise ValueError(
                f"activation rows {activations.shape[0]} != cols {self._cols}"
            )

        latent = self.column_factor @ activations
        output = self.row_factor @ latent
        output = output + self.residual.apply(activations)
        return output.squeeze(1) if squeeze_output else output

    def estimated_operations_per_weight(self) -> float:
        """MACs per represented dense weight for a batch-1 transform.

        Structured path costs ``rank * (rows + cols)`` MACs; residual adds
        one MAC per stored entry. Normalized by rows*cols.
        """
        structured_operations = 2.0 * self.rank * (self._rows + self._cols)
        residual_operations = 2.0 * self.residual.nnz
        return (structured_operations + residual_operations) / (
            self._rows * self._cols
        )

    def max_transient_scratch_elements(self) -> int:
        """Latent (rank,) plus residual gather buffer (nnz,)."""
        return self.rank + self.residual.nnz

    def max_decoded_weight_elements(self) -> int:
        """No dense-weight values are ever decoded; factors act directly."""
        return 0
