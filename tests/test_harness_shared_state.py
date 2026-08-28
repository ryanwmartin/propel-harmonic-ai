"""Contract tests for the shared-state representation families.

Covers Experiment 3 (shared spectral basis) and Experiment 5 (low-rank +
sparse residual): fused-transform correctness against diagnostic
reconstruction, honest state accounting, sparse-residual exactness, and the
zero-dense-materialization gate.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.harness.low_rank_residual import LowRankResidualRepresentation
from src.harness.quantization import quantize_field
from src.harness.shared_basis import SharedBasisRepresentation, dct_basis
from src.harness.sparse_residual import SparseResidual

TRANSFORM_MATCH_TOLERANCE = 1e-4
"""Fused transform vs. dense-matmul-of-reconstruction agreement bound."""


@pytest.fixture()
def weight_matrix() -> torch.Tensor:
    torch.manual_seed(7)
    return torch.randn(48, 128, dtype=torch.float32)


@pytest.fixture()
def activations() -> torch.Tensor:
    torch.manual_seed(11)
    return torch.randn(128, 5, dtype=torch.float32)


class TestSparseResidual:
    def test_zero_density_is_empty_and_free(self) -> None:
        error = torch.randn(8, 16)
        residual = SparseResidual.fit(error, density=0.0)
        assert residual.nnz == 0
        assert residual.total_bytes() == 0.0
        assert torch.equal(
            residual.apply(torch.randn(16, 3)), torch.zeros(8, 3)
        )

    def test_full_density_reproduces_error_exactly_at_fp32(self) -> None:
        error = torch.randn(8, 16)
        residual = SparseResidual.fit(error, density=1.0, value_bits=32)
        assert residual.nnz == 8 * 16
        assert torch.allclose(residual.to_dense(), error)

    def test_apply_matches_dense_residual_matmul(self) -> None:
        torch.manual_seed(3)
        error = torch.randn(12, 24)
        residual = SparseResidual.fit(error, density=0.1, value_bits=32)
        x = torch.randn(24, 4)
        assert torch.allclose(
            residual.apply(x), residual.to_dense() @ x, atol=1e-6
        )

    def test_selects_top_magnitude_entries(self) -> None:
        error = torch.zeros(4, 4)
        error[1, 2] = 100.0
        error[3, 0] = -50.0
        residual = SparseResidual.fit(error, density=2 / 16, value_bits=32)
        dense = residual.to_dense()
        assert dense[1, 2] == 100.0
        assert dense[3, 0] == -50.0

    def test_index_bits_are_counted(self) -> None:
        error = torch.randn(10, 10)
        residual = SparseResidual.fit(error, density=0.2, value_bits=16)
        bits = residual.state_bits()
        assert bits["residual_indices"] == residual.nnz * 32
        assert bits["residual_values"] == residual.nnz * 16

    def test_rejects_invalid_density(self) -> None:
        with pytest.raises(ValueError, match="density"):
            SparseResidual.fit(torch.randn(4, 4), density=1.5)


class TestLowRankResidual:
    def test_full_rank_fp32_reconstructs_exactly(
        self, weight_matrix: torch.Tensor
    ) -> None:
        representation = LowRankResidualRepresentation(
            weight_matrix, rank=48, factor_bits=32
        )
        error = torch.linalg.norm(
            representation.reconstruct() - weight_matrix
        ) / torch.linalg.norm(weight_matrix)
        assert error < 1e-5

    def test_transform_matches_reconstruction_matmul(
        self, weight_matrix: torch.Tensor, activations: torch.Tensor
    ) -> None:
        representation = LowRankResidualRepresentation(
            weight_matrix, rank=12, residual_density=0.02
        )
        expected = representation.reconstruct() @ activations
        actual = representation.transform(activations)
        assert torch.allclose(actual, expected, atol=TRANSFORM_MATCH_TOLERANCE)

    def test_transform_supports_1d_activations(
        self, weight_matrix: torch.Tensor
    ) -> None:
        representation = LowRankResidualRepresentation(weight_matrix, rank=4)
        x = torch.randn(128)
        output = representation.transform(x)
        assert output.shape == (48,)

    def test_no_dense_materialization_gate_passes(
        self, weight_matrix: torch.Tensor
    ) -> None:
        representation = LowRankResidualRepresentation(weight_matrix, rank=8)
        representation.verify_no_dense_materialization()
        assert representation.max_decoded_weight_elements() == 0

    def test_state_accounting_counts_factors_and_residual(
        self, weight_matrix: torch.Tensor
    ) -> None:
        representation = LowRankResidualRepresentation(
            weight_matrix, rank=10, residual_density=0.01,
            factor_bits=16, residual_bits=16,
        )
        accounting = representation.state_accounting()
        assert accounting.field_bits["row_factor"] == 48 * 10 * 16
        assert accounting.field_bits["column_factor"] == 10 * 128 * 16
        assert "residual_values" in accounting.field_bits
        assert "residual_indices" in accounting.field_bits

    def test_rejects_invalid_rank(self, weight_matrix: torch.Tensor) -> None:
        with pytest.raises(ValueError, match="rank"):
            LowRankResidualRepresentation(weight_matrix, rank=0)
        with pytest.raises(ValueError, match="rank"):
            LowRankResidualRepresentation(weight_matrix, rank=49)


class TestSharedBasis:
    def test_dct_basis_is_orthonormal(self) -> None:
        basis = dct_basis(block_size=32, harmonic_count=32)
        gram = basis @ basis.T
        assert torch.allclose(gram, torch.eye(32), atol=1e-5)

    def test_complete_dct_basis_fp32_reconstructs_exactly(
        self, weight_matrix: torch.Tensor
    ) -> None:
        representation = SharedBasisRepresentation(
            weight_matrix, block_size=32, harmonic_count=32,
            basis_mode="dct", coefficient_bits=32,
        )
        error = torch.linalg.norm(
            representation.reconstruct() - weight_matrix
        ) / torch.linalg.norm(weight_matrix)
        assert error < 1e-5

    @pytest.mark.parametrize("basis_mode", ["dct", "svd"])
    def test_transform_matches_reconstruction_matmul(
        self,
        weight_matrix: torch.Tensor,
        activations: torch.Tensor,
        basis_mode: str,
    ) -> None:
        representation = SharedBasisRepresentation(
            weight_matrix, block_size=32, harmonic_count=12,
            basis_mode=basis_mode, residual_density=0.01,
        )
        expected = representation.reconstruct() @ activations
        actual = representation.transform(activations)
        assert torch.allclose(actual, expected, atol=TRANSFORM_MATCH_TOLERANCE)

    def test_svd_basis_beats_dct_at_same_k(
        self, weight_matrix: torch.Tensor
    ) -> None:
        dct = SharedBasisRepresentation(
            weight_matrix, block_size=32, harmonic_count=8,
            basis_mode="dct", coefficient_bits=32,
        )
        svd = SharedBasisRepresentation(
            weight_matrix, block_size=32, harmonic_count=8,
            basis_mode="svd", coefficient_bits=32,
        )
        dct_error = torch.linalg.norm(dct.reconstruct() - weight_matrix)
        svd_error = torch.linalg.norm(svd.reconstruct() - weight_matrix)
        assert svd_error <= dct_error + 1e-6

    def test_dct_basis_state_is_indices_only(
        self, weight_matrix: torch.Tensor
    ) -> None:
        representation = SharedBasisRepresentation(
            weight_matrix, block_size=32, harmonic_count=8, basis_mode="dct"
        )
        accounting = representation.state_accounting()
        assert accounting.field_bits["basis_frequency_indices"] == 8 * 16
        assert "shared_basis" not in accounting.field_bits

    def test_svd_basis_state_is_fully_counted(
        self, weight_matrix: torch.Tensor
    ) -> None:
        representation = SharedBasisRepresentation(
            weight_matrix, block_size=32, harmonic_count=8, basis_mode="svd"
        )
        accounting = representation.state_accounting()
        assert accounting.field_bits["shared_basis"] == 8 * 32 * 32

    def test_no_dense_materialization_gate_passes(
        self, weight_matrix: torch.Tensor
    ) -> None:
        representation = SharedBasisRepresentation(
            weight_matrix, block_size=32, harmonic_count=8
        )
        representation.verify_no_dense_materialization()
        assert representation.max_decoded_weight_elements() == 0

    def test_rejects_indivisible_block_size(
        self, weight_matrix: torch.Tensor
    ) -> None:
        with pytest.raises(ValueError, match="multiple"):
            SharedBasisRepresentation(
                weight_matrix, block_size=48, harmonic_count=8
            )

    def test_rejects_unknown_basis_mode(
        self, weight_matrix: torch.Tensor
    ) -> None:
        with pytest.raises(ValueError, match="basis_mode"):
            SharedBasisRepresentation(
                weight_matrix, block_size=32, harmonic_count=8,
                basis_mode="wavelet",
            )


class TestQuantizationHelper:
    def test_32_bit_is_identity(self) -> None:
        values = torch.randn(16)
        assert torch.equal(quantize_field(values, 32), values)

    def test_16_bit_round_trips_through_fp16(self) -> None:
        values = torch.tensor([math.pi])
        quantized = quantize_field(values, 16)
        assert quantized.dtype == torch.float32
        assert quantized != values
        assert torch.allclose(quantized, values, atol=1e-3)

    def test_rejects_unsupported_width(self) -> None:
        with pytest.raises(ValueError, match="bit width"):
            quantize_field(torch.randn(4), 8)
