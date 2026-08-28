"""Contract tests for Experiment 2 — reversible ordering/transform search.

Covers: exact invertibility of every canonicalization, hot-path equivalence
(activation-space column transform + output row un-permutation equals dense
matmul of the restored matrix), honest permutation/sign/scale state counting,
the zero-dense-materialization gate through the wrapper, and the gate
evaluation logic of the experiment runner.
"""

from __future__ import annotations

import pytest
import torch

from src.harness.experiment_2_orderings import (
    COEFFICIENT_REDUCTION_GATE,
    CanonicalizationSpec,
    GateResult,
    SweepPoint,
    default_canonicalization_grid,
    evaluate_gates,
    run_ordering_experiment,
    verify_exact_inverse,
)
from src.harness.accounting import ExperimentReport, dense_baseline_bits
from src.harness.low_rank_residual import LowRankResidualRepresentation
from src.harness.orderings import (
    ORDERING_STRATEGIES,
    build_canonicalization,
    column_norm_order,
    greedy_nearest_neighbor_order,
    index_bits,
    spectral_seriation_order,
)
from src.harness.shared_basis import SharedBasisRepresentation
from src.harness.transformed_representation import TransformedRepresentation

TRANSFORM_MATCH_TOLERANCE = 1e-4
"""Fused transform vs. dense-matmul-of-reconstruction agreement bound."""


@pytest.fixture()
def weight_matrix() -> torch.Tensor:
    torch.manual_seed(19)
    return torch.randn(48, 128, dtype=torch.float32)


@pytest.fixture()
def activations() -> torch.Tensor:
    torch.manual_seed(23)
    return torch.randn(128, 5, dtype=torch.float32)


class TestOrderingStrategies:
    def test_all_strategies_return_valid_permutations(
        self, weight_matrix: torch.Tensor
    ) -> None:
        for strategy_name, strategy in ORDERING_STRATEGIES.items():
            if strategy is None:
                continue
            permutation = strategy(weight_matrix)
            assert permutation.shape == (weight_matrix.shape[1],), strategy_name
            assert torch.equal(
                torch.sort(permutation).values,
                torch.arange(weight_matrix.shape[1]),
            ), strategy_name

    def test_column_norm_order_is_descending(self) -> None:
        matrix = torch.diag(torch.tensor([1.0, 3.0, 2.0]))
        order = column_norm_order(matrix)
        assert order.tolist() == [1, 2, 0]

    def test_spectral_seriation_groups_correlated_columns(self) -> None:
        torch.manual_seed(5)
        base_a = torch.randn(64)
        base_b = torch.randn(64)
        # Interleave two correlated groups; seriation should separate them.
        columns = [base_a, base_b, base_a * 1.1, base_b * 0.9,
                   base_a * 0.95, base_b * 1.05]
        matrix = torch.stack(
            [column + 0.01 * torch.randn(64) for column in columns], dim=1
        )
        order = spectral_seriation_order(matrix).tolist()
        group_of = {0: "a", 2: "a", 4: "a", 1: "b", 3: "b", 5: "b"}
        labels = [group_of[index] for index in order]
        # Each group must be contiguous after seriation.
        assert labels in (
            ["a", "a", "a", "b", "b", "b"],
            ["b", "b", "b", "a", "a", "a"],
        )

    def test_greedy_nearest_neighbor_chains_similar_columns(self) -> None:
        matrix = torch.tensor(
            [[10.0, 1.0, 9.5, 1.2], [0.0, 5.0, 0.1, 4.8]]
        )
        order = greedy_nearest_neighbor_order(matrix).tolist()
        position = {index: rank for rank, index in enumerate(order)}
        assert abs(position[0] - position[2]) == 1
        assert abs(position[1] - position[3]) == 1

    def test_index_bits_widths(self) -> None:
        assert index_bits(128) == 16
        assert index_bits(1 << 16) == 16
        assert index_bits((1 << 16) + 1) == 32


class TestCanonicalizationInvertibility:
    @pytest.mark.parametrize("spec", default_canonicalization_grid(),
                             ids=lambda spec: spec.label)
    def test_round_trip_is_exact(
        self, spec: CanonicalizationSpec, weight_matrix: torch.Tensor
    ) -> None:
        canonicalization = spec.fit(weight_matrix)
        verify_exact_inverse(canonicalization, weight_matrix)

    def test_permutation_only_round_trip_is_bit_exact(
        self, weight_matrix: torch.Tensor
    ) -> None:
        canonicalization = build_canonicalization(
            weight_matrix, "spectral", order_rows=True
        )
        round_trip = canonicalization.restore_matrix(
            canonicalization.apply_to_matrix(weight_matrix)
        )
        assert torch.equal(round_trip, weight_matrix)

    def test_rejects_unknown_strategy(self, weight_matrix: torch.Tensor) -> None:
        with pytest.raises(ValueError, match="column_strategy"):
            build_canonicalization(weight_matrix, "alphabetical")


class TestCanonicalizationState:
    def test_identity_has_zero_state_bits(self, weight_matrix: torch.Tensor) -> None:
        canonicalization = build_canonicalization(weight_matrix, "identity")
        assert canonicalization.state_bits() == {}

    def test_all_components_are_counted(self, weight_matrix: torch.Tensor) -> None:
        canonicalization = build_canonicalization(
            weight_matrix, "col-norm", order_rows=True,
            use_signs=True, use_scales=True,
        )
        bits = canonicalization.state_bits()
        rows, cols = weight_matrix.shape
        assert bits["row_permutation"] == rows * 16
        assert bits["column_permutation"] == cols * 16
        assert bits["column_signs"] == cols * 1
        assert bits["column_scales"] == cols * 16

    def test_wrapper_accounting_includes_canonicalization_fields(
        self, weight_matrix: torch.Tensor
    ) -> None:
        canonicalization = build_canonicalization(
            weight_matrix, "spectral", use_signs=True, use_scales=True
        )
        wrapped = TransformedRepresentation(
            weight_matrix,
            canonicalization,
            lambda matrix: LowRankResidualRepresentation(matrix, rank=4),
        )
        inner_only = LowRankResidualRepresentation(
            canonicalization.apply_to_matrix(weight_matrix), rank=4
        )
        field_bits = wrapped.state_accounting().field_bits
        assert "canonicalization_column_permutation" in field_bits
        assert "canonicalization_column_signs" in field_bits
        assert "canonicalization_column_scales" in field_bits
        expected_extra = sum(canonicalization.state_bits().values())
        assert (
            wrapped.state_accounting().total_bits
            == inner_only.state_accounting().total_bits + expected_extra
        )


class TestTransformedRepresentationHotPath:
    @pytest.mark.parametrize("spec", default_canonicalization_grid(),
                             ids=lambda spec: spec.label)
    def test_transform_matches_dense_matmul_of_reconstruction(
        self,
        spec: CanonicalizationSpec,
        weight_matrix: torch.Tensor,
        activations: torch.Tensor,
    ) -> None:
        canonicalization = spec.fit(weight_matrix)
        wrapped = TransformedRepresentation(
            weight_matrix,
            canonicalization,
            lambda matrix: LowRankResidualRepresentation(
                matrix, rank=16, residual_density=0.02
            ),
        )
        fused = wrapped.transform(activations)
        dense = wrapped.reconstruct() @ activations
        relative_error = float(
            torch.linalg.norm(fused - dense) / torch.linalg.norm(dense)
        )
        assert relative_error < TRANSFORM_MATCH_TOLERANCE, spec.label

    def test_shared_basis_inner_also_matches(
        self, weight_matrix: torch.Tensor, activations: torch.Tensor
    ) -> None:
        canonicalization = build_canonicalization(
            weight_matrix, "greedy-nn", use_signs=True, use_scales=True
        )
        wrapped = TransformedRepresentation(
            weight_matrix,
            canonicalization,
            lambda matrix: SharedBasisRepresentation(
                matrix, block_size=32, harmonic_count=16, basis_mode="svd"
            ),
        )
        fused = wrapped.transform(activations)
        dense = wrapped.reconstruct() @ activations
        relative_error = float(
            torch.linalg.norm(fused - dense) / torch.linalg.norm(dense)
        )
        assert relative_error < TRANSFORM_MATCH_TOLERANCE

    def test_one_dimensional_activations_round_trip(
        self, weight_matrix: torch.Tensor
    ) -> None:
        canonicalization = build_canonicalization(
            weight_matrix, "col-norm", order_rows=True
        )
        wrapped = TransformedRepresentation(
            weight_matrix,
            canonicalization,
            lambda matrix: LowRankResidualRepresentation(matrix, rank=8),
        )
        vector = torch.randn(weight_matrix.shape[1])
        output = wrapped.transform(vector)
        assert output.shape == (weight_matrix.shape[0],)

    def test_full_rank_inner_recovers_original_layer_output(
        self, weight_matrix: torch.Tensor, activations: torch.Tensor
    ) -> None:
        """With a lossless inner fit, the wrapper must reproduce W @ x."""
        canonicalization = build_canonicalization(
            weight_matrix, "spectral", order_rows=True,
            use_signs=True, use_scales=True,
        )
        full_rank = min(weight_matrix.shape)
        wrapped = TransformedRepresentation(
            weight_matrix,
            canonicalization,
            lambda matrix: LowRankResidualRepresentation(
                matrix, rank=full_rank, factor_bits=32
            ),
        )
        fused = wrapped.transform(activations)
        dense = weight_matrix @ activations
        relative_error = float(
            torch.linalg.norm(fused - dense) / torch.linalg.norm(dense)
        )
        assert relative_error < 1e-4

    def test_no_dense_materialization_through_wrapper(
        self, weight_matrix: torch.Tensor
    ) -> None:
        canonicalization = build_canonicalization(weight_matrix, "spectral")
        wrapped = TransformedRepresentation(
            weight_matrix,
            canonicalization,
            lambda matrix: LowRankResidualRepresentation(matrix, rank=4),
        )
        assert wrapped.max_decoded_weight_elements() == 0
        wrapped.verify_no_dense_materialization()

    def test_geometry_mismatch_is_rejected(self, weight_matrix: torch.Tensor) -> None:
        wrong_geometry = build_canonicalization(
            torch.randn(8, 16), "col-norm"
        )
        with pytest.raises(ValueError, match="geometry"):
            TransformedRepresentation(
                weight_matrix,
                wrong_geometry,
                lambda matrix: LowRankResidualRepresentation(matrix, rank=2),
            )


class TestGateEvaluation:
    @staticmethod
    def _point(
        label: str, family: str, capacity: int, bits: int, error: float,
        canonicalization_bits: int = 0,
    ) -> SweepPoint:
        report = ExperimentReport(
            representation=f"{family}[{capacity}]",
            tensor_description="synthetic",
            shared_executable_bytes=0,
            unique_model_state_bytes=bits / 8.0,
            residual_bytes=0.0,
            transient_scratch_bytes=0.0,
            dense_baseline_bytes=1.0,
            generator_operations_per_weight=1.0,
            dense_weight_materialized=False,
            weight_reconstruction_relative_error=error,
            layer_output_relative_error=error,
            extra_metrics={"state_ratio_to_dense_fp16": 0.0},
        )
        return SweepPoint(
            canonicalization_label=label,
            family=family,
            capacity=capacity,
            state_bits=bits,
            canonicalization_bits=canonicalization_bits,
            layer_output_error=error,
            report=report,
        )

    def test_detects_two_x_coefficient_reduction(self) -> None:
        points = [
            self._point("identity", "low-rank", 8, 8000, 1e-1),
            self._point("identity", "low-rank", 16, 16000, 1e-2),
            self._point("spectral", "low-rank", 4, 4000, 5e-2,
                        canonicalization_bits=100),
            self._point("spectral", "low-rank", 8, 8000, 5e-3,
                        canonicalization_bits=100),
        ]
        results = evaluate_gates(points, dense_bits=1_000_000)
        assert len(results) == 1
        result = results[0]
        assert result.coefficient_reduction == pytest.approx(2.0)
        assert result.passes_reduction_gate
        assert result.passes_permutation_budget

    def test_no_match_when_transformed_never_reaches_identity_error(self) -> None:
        points = [
            self._point("identity", "low-rank", 8, 8000, 1e-3),
            self._point("col-norm", "low-rank", 8, 8000, 1e-1,
                        canonicalization_bits=100),
        ]
        results = evaluate_gates(points, dense_bits=1_000_000)
        assert results[0].coefficient_reduction == 0.0
        assert not results[0].passes_reduction_gate

    def test_permutation_budget_gate(self) -> None:
        points = [
            self._point("identity", "low-rank", 8, 8000, 1e-2),
            self._point("spectral", "low-rank", 4, 4000, 1e-3,
                        canonicalization_bits=400_000),
        ]
        results = evaluate_gates(points, dense_bits=1_000_000)
        assert results[0].coefficient_reduction >= COEFFICIENT_REDUCTION_GATE
        assert not results[0].passes_permutation_budget


class TestExperimentRunner:
    def test_runner_is_deterministic_and_complete(self) -> None:
        torch.manual_seed(2)
        weight_matrix = torch.randn(16, 32)
        grid = [
            CanonicalizationSpec("identity", "identity", False, False, False),
            CanonicalizationSpec("col-norm", "col-norm", False, False, False),
        ]
        first = run_ordering_experiment(
            weight_matrix, "test", seed=1, grid=grid
        )
        second = run_ordering_experiment(
            weight_matrix, "test", seed=1, grid=grid
        )
        assert len(first) == len(second) > 0
        for point_a, point_b in zip(first, second):
            assert point_a.state_bits == point_b.state_bits
            assert point_a.layer_output_error == point_b.layer_output_error

    def test_gate_results_cover_every_family_and_label(self) -> None:
        torch.manual_seed(3)
        weight_matrix = torch.randn(16, 32)
        grid = [
            CanonicalizationSpec("identity", "identity", False, False, False),
            CanonicalizationSpec("spectral", "spectral", False, False, False),
            CanonicalizationSpec("sign+scale", "identity", False, True, True),
        ]
        sweep_points = run_ordering_experiment(
            weight_matrix, "test", seed=1, grid=grid
        )
        gate_results = evaluate_gates(
            sweep_points, dense_baseline_bits(16, 32)
        )
        pairs = {(r.family, r.canonicalization_label) for r in gate_results}
        assert pairs == {
            ("low-rank", "spectral"),
            ("low-rank", "sign+scale"),
            ("shared-svd-basis", "spectral"),
            ("shared-svd-basis", "sign+scale"),
        }
        for result in gate_results:
            assert isinstance(result, GateResult)
            assert result.permutation_ratio >= 0.0
