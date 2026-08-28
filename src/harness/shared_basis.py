"""Experiment 3 — shared spectral basis behind the harness interface.

Shared-state mechanism: one basis of K waveforms per **tensor** (not per
block). Every length-``block_size`` block of every row stores only K
coefficients against that common basis:

    weight[row, block, :] ~= coefficients[row, block, :] @ basis    (K x L)

Two basis modes:

- ``"dct"`` — an orthonormal DCT-II cosine bank. The basis is **procedural
  wave synthesis**: each basis row is ``cos(pi * f * (2n + 1) / (2L))`` and
  can be generated on the fly by the same two-term oscillator recurrence the
  harmonic baseline uses. Stored basis state is only the K selected frequency
  indices (16 bits each), amortized across the whole tensor.
- ``"svd"`` — the top-K left singular vectors of the stacked block matrix,
  i.e. the optimal shared basis in the least-squares sense. Its
  ``K * block_size`` float values are honestly counted as unique model state,
  amortized across all blocks.

Given the shared basis, per-block coefficient fitting is **closed-form**
(orthonormal projection / least squares) — no distillation.

Activation-aware mode (Experiment 12 stage 1): given the teacher activation
second moment ``C = E[x x^T]``, the block-position-averaged sub-moment
``C_L`` re-weights the fit so that block positions activations actually
excite dominate the objective. The ``"svd"`` basis becomes the top-K right
singular subspace of the **whitened** stacked blocks (mapped back through
``C_L^{-1/2}``), and ``"dct"`` coefficients become the weighted-least-squares
projection. The second moment is fitting-time state only — never inference
model state, and the hot path is unchanged.

Fused hot-path execution exploits the sharing directly: for each block the
basis is contracted with the activation slice **once** —

    projected[k] = sum_n basis[k, n] * x[block_start + n]      (K values)
    output[row] += sum_k coefficients[row, block, k] * projected[k]

so basis work is amortized over all rows, per-weight cost approaches the
dense MAC cost as rows grow, and no dense weight value is ever decoded.
An optional sparse residual (Experiment 11 guardrail) corrects the largest
errors explicitly.
"""

from __future__ import annotations

import math

import torch

from src.harness.accounting import StateAccounting
from src.harness.activation_aware import (
    block_average_second_moment,
    column_activation_importance,
    second_moment_square_root,
)
from src.harness.quantization import SUPPORTED_FIELD_BITS, quantize_field
from src.harness.representation import ProceduralRepresentation
from src.harness.sparse_residual import SparseResidual

METADATA_BITS: int = 32 * 8
"""Serialized header metadata (magic, version, geometry, mode, K) in bits."""

FREQUENCY_INDEX_BITS: int = 16
"""Storage width of one selected DCT frequency index."""

BASIS_MODES: frozenset[str] = frozenset({"dct", "svd"})


def dct_basis(block_size: int, harmonic_count: int) -> torch.Tensor:
    """Orthonormal DCT-II bank of the ``harmonic_count`` lowest frequencies.

    Returns:
        Float32 tensor of shape (harmonic_count, block_size). Row ``k`` is
        ``scale_k * cos(pi * k * (2n + 1) / (2 * block_size))`` — a pure wave
        that a fused kernel regenerates via an oscillator recurrence instead
        of loading from memory.
    """
    sample_positions = torch.arange(block_size, dtype=torch.float64)
    frequency_indices = torch.arange(harmonic_count, dtype=torch.float64)
    angles = (
        math.pi
        * frequency_indices.unsqueeze(1)
        * (2.0 * sample_positions + 1.0)
        / (2.0 * block_size)
    )
    basis = torch.cos(angles)
    scales = torch.full(
        (harmonic_count, 1), math.sqrt(2.0 / block_size), dtype=torch.float64
    )
    scales[0, 0] = math.sqrt(1.0 / block_size)
    return (scales * basis).to(torch.float32)


class SharedBasisRepresentation(ProceduralRepresentation):
    """One shared K-waveform basis per tensor, per-block coefficients.

    Args:
        weight_matrix: Dense (rows, cols) float32 matrix. ``cols`` must be a
            multiple of ``block_size``.
        block_size: Samples per block (the basis length L).
        harmonic_count: Basis waveforms K shared by every block.
        basis_mode: ``"dct"`` (procedural cosine bank) or ``"svd"``
            (learned optimal basis, stored and counted).
        residual_density: Fraction of entries kept in the sparse residual.
        coefficient_bits: Storage precision of per-block coefficients.
        residual_bits: Storage precision of residual values.
        second_moment: Optional (cols, cols) teacher activation second
            moment. When given, basis and coefficients minimize the
            block-weighted on-distribution output error and the residual is
            activation-importance ranked (Experiment 12 stage 1).
            Fitting-time state only.
    """

    def __init__(
        self,
        weight_matrix: torch.Tensor,
        block_size: int,
        harmonic_count: int,
        basis_mode: str = "dct",
        residual_density: float = 0.0,
        coefficient_bits: int = 16,
        residual_bits: int = 16,
        second_moment: torch.Tensor | None = None,
    ) -> None:
        if basis_mode not in BASIS_MODES:
            raise ValueError(
                f"basis_mode must be one of {sorted(BASIS_MODES)}, got "
                f"{basis_mode!r}"
            )
        if coefficient_bits not in SUPPORTED_FIELD_BITS:
            raise ValueError(
                f"coefficient_bits must be one of {sorted(SUPPORTED_FIELD_BITS)}, "
                f"got {coefficient_bits}"
            )
        weight_matrix = weight_matrix.to(torch.float32)
        row_count, column_count = weight_matrix.shape
        if column_count % block_size != 0:
            raise ValueError(
                f"cols {column_count} must be a multiple of block_size "
                f"{block_size}"
            )
        if not 1 <= harmonic_count <= block_size:
            raise ValueError(
                f"harmonic_count must be in [1, {block_size}], got "
                f"{harmonic_count}"
            )
        if second_moment is not None and second_moment.shape != (
            column_count,
            column_count,
        ):
            raise ValueError(
                f"second_moment must be ({column_count}, {column_count}), "
                f"got {tuple(second_moment.shape)}"
            )

        self.block_size = block_size
        self.harmonic_count = harmonic_count
        self.basis_mode = basis_mode
        self.coefficient_bits = coefficient_bits
        self.activation_aware = second_moment is not None
        self._rows = row_count
        self._cols = column_count
        self.num_blocks = column_count // block_size

        stacked_blocks = weight_matrix.reshape(
            row_count * self.num_blocks, block_size
        )
        block_whitener = None
        if second_moment is not None:
            block_moment = block_average_second_moment(
                second_moment, block_size
            )
            block_whitener, block_inverse_whitener = second_moment_square_root(
                block_moment
            )

        if basis_mode == "dct":
            self.basis = dct_basis(block_size, harmonic_count)
            if block_whitener is None:
                coefficients = stacked_blocks @ self.basis.T
            else:
                coefficients = _weighted_least_squares_coefficients(
                    stacked_blocks, self.basis, block_whitener
                )
        else:
            maximum_svd_harmonics = min(stacked_blocks.shape)
            if harmonic_count > maximum_svd_harmonics:
                raise ValueError(
                    f"svd basis_mode supports at most {maximum_svd_harmonics} "
                    f"harmonics for {stacked_blocks.shape[0]} stacked blocks "
                    f"of size {block_size}, got {harmonic_count}"
                )
            if block_whitener is None:
                _, _, right_vectors_transposed = torch.linalg.svd(
                    stacked_blocks, full_matrices=False
                )
                self.basis = right_vectors_transposed[
                    :harmonic_count
                ].contiguous()
                coefficients = stacked_blocks @ self.basis.T
            else:
                whitened_blocks = stacked_blocks @ block_whitener
                _, _, right_vectors_transposed = torch.linalg.svd(
                    whitened_blocks, full_matrices=False
                )
                whitened_basis = right_vectors_transposed[:harmonic_count]
                # Map back so basis @ x acts on raw activations; coefficients
                # are the orthonormal projection in the whitened metric.
                self.basis = (
                    whitened_basis @ block_inverse_whitener
                ).contiguous()
                coefficients = whitened_blocks @ whitened_basis.T

        self.coefficients = quantize_field(
            coefficients.reshape(row_count, self.num_blocks, harmonic_count),
            coefficient_bits,
        )

        structured_error = weight_matrix - self._structured_reconstruction()
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
            f"shared-basis-{self.basis_mode}"
            f"[L={self.block_size},K={self.harmonic_count},"
            f"density={self.residual.nnz / (self._rows * self._cols):.4f},"
            f"coeff={self.coefficient_bits}b,fit={fitting_mode}]"
        )

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def cols(self) -> int:
        return self._cols

    def state_accounting(self) -> StateAccounting:
        coefficient_count = self._rows * self.num_blocks * self.harmonic_count
        field_bits: dict[str, int] = {
            "coefficients": coefficient_count * self.coefficient_bits,
            "metadata": METADATA_BITS,
        }
        if self.basis_mode == "dct" and not self.activation_aware:
            field_bits["basis_frequency_indices"] = (
                self.harmonic_count * FREQUENCY_INDEX_BITS
            )
        else:
            # SVD bases — and activation-aware DCT variants whose hot path
            # still contracts the stored (possibly re-weighted) basis — are
            # honest unique model state, amortized across all blocks.
            field_bits["shared_basis"] = (
                self.harmonic_count * self.block_size * 32
            )
        field_bits.update(self.residual.state_bits())
        return StateAccounting(
            field_bits=field_bits,
            represented_weight_count=self._rows * self._cols,
        )

    def reconstruct(self) -> torch.Tensor:
        """Diagnostic dense reconstruction (structured + residual)."""
        return self._structured_reconstruction() + self.residual.to_dense()

    def transform(self, input_activations: torch.Tensor) -> torch.Tensor:
        """Fused hot path: basis contracted with activations once per block.

        Never decodes a weight value: the per-block projection
        ``basis @ x_block`` (K, batch) replaces all rows' weight generation,
        then the stored coefficients multiply-accumulate directly into the
        output.
        """
        squeeze_output = input_activations.dim() == 1
        activations = (
            input_activations.unsqueeze(1) if squeeze_output else input_activations
        ).to(torch.float32)
        if activations.shape[0] != self._cols:
            raise ValueError(
                f"activation rows {activations.shape[0]} != cols {self._cols}"
            )

        batch_size = activations.shape[1]
        blocked_activations = activations.reshape(
            self.num_blocks, self.block_size, batch_size
        )
        projected = torch.einsum(
            "kn,bnc->bkc", self.basis, blocked_activations
        )
        output = torch.einsum("rbk,bkc->rc", self.coefficients, projected)
        output = output + self.residual.apply(activations)
        return output.squeeze(1) if squeeze_output else output

    def estimated_operations_per_weight(self) -> float:
        """MACs per represented weight for a batch-1 transform.

        Projection: ``num_blocks * K * L`` MACs shared across all rows;
        accumulation: ``rows * num_blocks * K``; residual: nnz.
        """
        projection_operations = 2.0 * self.num_blocks * (
            self.harmonic_count * self.block_size
        )
        accumulation_operations = 2.0 * (
            self._rows * self.num_blocks * self.harmonic_count
        )
        residual_operations = 2.0 * self.residual.nnz
        return (
            projection_operations + accumulation_operations + residual_operations
        ) / (self._rows * self._cols)

    def max_transient_scratch_elements(self) -> int:
        """Per-block projection (num_blocks * K) plus residual buffer."""
        return self.num_blocks * self.harmonic_count + self.residual.nnz

    def max_decoded_weight_elements(self) -> int:
        """No dense-weight values are ever decoded in transform()."""
        return 0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _structured_reconstruction(self) -> torch.Tensor:
        """Dense decode of the structured component (diagnostic only)."""
        stacked = self.coefficients.reshape(
            self._rows * self.num_blocks, self.harmonic_count
        )
        blocks = stacked @ self.basis
        return blocks.reshape(self._rows, self._cols)


def _weighted_least_squares_coefficients(
    stacked_blocks: torch.Tensor,
    basis: torch.Tensor,
    block_whitener: torch.Tensor,
) -> torch.Tensor:
    """Coefficients minimizing ``||(block - c @ basis) @ M||`` per block.

    Closed form: with ``B~ = basis @ M`` and ``b~ = block @ M``, solve the
    normal equations ``c (B~ B~^T) = b~ B~^T`` once for all blocks.
    """
    whitened_basis = basis @ block_whitener
    gram = whitened_basis @ whitened_basis.T
    projections = (stacked_blocks @ block_whitener) @ whitened_basis.T
    solution = torch.linalg.solve(
        gram.to(torch.float64), projections.to(torch.float64).T
    )
    return solution.T.to(torch.float32)
