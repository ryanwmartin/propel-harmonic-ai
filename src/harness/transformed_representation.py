"""Experiment 2 — canonicalized wrapper around any procedural family.

Composes a :class:`MatrixCanonicalization` (Experiment 2 state: permutations,
signs, diagonal scales) with any inner :class:`ProceduralRepresentation`
fitted on the canonicalized matrix:

    W @ x  ==  restore_output( inner.transform( transform_activations(x) ) )

Per the roadmap, the column transform lives on the activations and the row
permutation on the output vector — the hot path never physically sorts and
unsorts weight state. All canonicalization state is counted as unique model
state alongside the inner representation's own fields.
"""

from __future__ import annotations

from typing import Callable

import torch

from src.harness.accounting import StateAccounting
from src.harness.orderings import MatrixCanonicalization
from src.harness.representation import ProceduralRepresentation


class TransformedRepresentation(ProceduralRepresentation):
    """A procedural family fitted in a canonicalized coordinate system.

    Args:
        weight_matrix: Original dense (rows, cols) float32 matrix.
        canonicalization: Fitted reversible ordering/sign/scale transform.
        inner_factory: Callable fitting the inner representation on the
            canonicalized matrix, e.g.
            ``lambda m: LowRankResidualRepresentation(m, rank=8)``.
    """

    def __init__(
        self,
        weight_matrix: torch.Tensor,
        canonicalization: MatrixCanonicalization,
        inner_factory: Callable[[torch.Tensor], ProceduralRepresentation],
    ) -> None:
        weight_matrix = weight_matrix.to(torch.float32)
        row_count, column_count = weight_matrix.shape
        if (canonicalization.rows, canonicalization.cols) != (
            row_count,
            column_count,
        ):
            raise ValueError(
                "canonicalization geometry "
                f"({canonicalization.rows}x{canonicalization.cols}) does not "
                f"match weight matrix ({row_count}x{column_count})"
            )
        self._rows = row_count
        self._cols = column_count
        self.canonicalization = canonicalization
        self.inner = inner_factory(
            canonicalization.apply_to_matrix(weight_matrix)
        )
        if (self.inner.rows, self.inner.cols) != (row_count, column_count):
            raise ValueError(
                "inner representation geometry "
                f"({self.inner.rows}x{self.inner.cols}) does not match "
                f"weight matrix ({row_count}x{column_count})"
            )

    # ------------------------------------------------------------------
    # ProceduralRepresentation interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        transform_fields = sorted(self.canonicalization.state_bits())
        transform_label = "+".join(transform_fields) if transform_fields else "identity"
        return f"canon[{transform_label}]({self.inner.name})"

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def cols(self) -> int:
        return self._cols

    def state_accounting(self) -> StateAccounting:
        inner_accounting = self.inner.state_accounting()
        field_bits = dict(inner_accounting.field_bits)
        for field_name, bit_count in self.canonicalization.state_bits().items():
            field_bits[f"canonicalization_{field_name}"] = bit_count
        return StateAccounting(
            field_bits=field_bits,
            represented_weight_count=self._rows * self._cols,
        )

    def reconstruct(self) -> torch.Tensor:
        """Diagnostic dense reconstruction back in the original coordinates."""
        return self.canonicalization.restore_matrix(self.inner.reconstruct())

    def transform(self, input_activations: torch.Tensor) -> torch.Tensor:
        """Fused hot path: activation-space transform, inner fused apply,
        output row un-permutation. No weight sorting per call."""
        squeeze_output = input_activations.dim() == 1
        activations = (
            input_activations.unsqueeze(1) if squeeze_output else input_activations
        ).to(torch.float32)
        if activations.shape[0] != self._cols:
            raise ValueError(
                f"activation rows {activations.shape[0]} != cols {self._cols}"
            )
        transformed_activations = self.canonicalization.transform_activations(
            activations
        )
        permuted_output = self.inner.transform(transformed_activations)
        output = self.canonicalization.restore_output(permuted_output)
        return output.squeeze(1) if squeeze_output else output

    def estimated_operations_per_weight(self) -> float:
        extra_operations = self.canonicalization.extra_transform_operations()
        return self.inner.estimated_operations_per_weight() + (
            extra_operations / (self._rows * self._cols)
        )

    def max_transient_scratch_elements(self) -> int:
        """Transformed activation copy (cols) plus inner scratch."""
        return self._cols + self.inner.max_transient_scratch_elements()

    def max_decoded_weight_elements(self) -> int:
        return self.inner.max_decoded_weight_elements()
