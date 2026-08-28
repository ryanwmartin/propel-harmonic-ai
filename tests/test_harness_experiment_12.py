"""Contract tests for Experiment 12 stage 1 — activation-aware fitting.

Covers:

- the closed-form activation-aware math (whitening, low-rank optimality on
  the activation distribution, importance ranking, block-averaged moments);
- activation-aware modes of the low-rank and shared-basis families
  (state accounting unchanged where required, hot-path equivalence with the
  diagnostic reconstruction, no dense materialization);
- the activation-cache split logic (deterministic, disjoint, correct moment);
- the experiment runner's gate logic and the synthetic case where
  activation-aware fitting must decisively beat weight-space fitting.
"""

from __future__ import annotations

import pytest
import torch

from src.harness.activation_aware import (
    activation_aware_low_rank_factors,
    block_average_second_moment,
    column_activation_importance,
    second_moment_square_root,
)
from src.harness.activation_cache import split_captured_activations
from src.harness.experiment_12_behavioral import (
    LAYER_OUTPUT_GATE,
    STATE_RATIO_GATE,
    build_synthetic_case,
    evaluate_candidate,
    run_experiment_12,
    summarize_activation_gain,
)
from src.harness.low_rank_residual import LowRankResidualRepresentation
from src.harness.shared_basis import SharedBasisRepresentation
from src.harness.sparse_residual import SparseResidual


def _random_spd_matrix(dimension: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    factor = torch.randn(dimension, dimension, generator=generator)
    return factor @ factor.T / dimension + 0.1 * torch.eye(dimension)


class TestSecondMomentSquareRoot:
    def test_square_root_squares_back_to_moment(self):
        moment = _random_spd_matrix(16)
        square_root, _ = second_moment_square_root(moment)
        assert torch.allclose(square_root @ square_root, moment, atol=1e-4)

    def test_inverse_is_inverse(self):
        moment = _random_spd_matrix(16)
        square_root, inverse_square_root = second_moment_square_root(moment)
        product = square_root @ inverse_square_root
        assert torch.allclose(product, torch.eye(16), atol=1e-4)

    def test_damping_bounds_near_singular_directions(self):
        # Rank-1 moment: without damping the inverse would be infinite.
        direction = torch.randn(8, 1)
        moment = direction @ direction.T
        _, inverse_square_root = second_moment_square_root(moment)
        assert torch.isfinite(inverse_square_root).all()

    def test_rejects_non_square(self):
        with pytest.raises(ValueError, match="square"):
            second_moment_square_root(torch.randn(4, 5))

    def test_rejects_bad_floor_fraction(self):
        with pytest.raises(ValueError, match="eigenvalue_floor_fraction"):
            second_moment_square_root(torch.eye(4), eigenvalue_floor_fraction=1.5)


class TestActivationAwareLowRankFactors:
    def test_beats_weight_space_svd_on_distribution(self):
        """The whitened SVD must have lower on-distribution output error."""
        torch.manual_seed(0)
        weight_matrix = torch.randn(32, 48)
        subspace = torch.linalg.qr(torch.randn(48, 5)).Q
        samples = torch.randn(2048, 5) @ subspace.T + 0.01 * torch.randn(2048, 48)
        moment = (samples.T @ samples / samples.shape[0]).to(torch.float32)
        probes = samples[:256].T.contiguous()

        rank = 5
        row_factor, column_factor = activation_aware_low_rank_factors(
            weight_matrix, moment, rank
        )
        left, singular_values, right_t = torch.linalg.svd(weight_matrix)
        plain_approx = (
            left[:, :rank] * singular_values[:rank]
        ) @ right_t[:rank]

        reference = weight_matrix @ probes
        aware_error = torch.linalg.norm(
            (row_factor @ (column_factor @ probes)) - reference
        )
        plain_error = torch.linalg.norm(plain_approx @ probes - reference)
        assert aware_error < 0.2 * plain_error

    def test_identity_moment_reduces_to_plain_svd(self):
        torch.manual_seed(1)
        weight_matrix = torch.randn(12, 20)
        row_factor, column_factor = activation_aware_low_rank_factors(
            weight_matrix, torch.eye(20), rank=4
        )
        left, singular_values, right_t = torch.linalg.svd(weight_matrix)
        plain_approx = (left[:, :4] * singular_values[:4]) @ right_t[:4]
        assert torch.allclose(
            row_factor @ column_factor, plain_approx, atol=1e-4
        )

    def test_rejects_out_of_range_rank(self):
        with pytest.raises(ValueError, match="rank"):
            activation_aware_low_rank_factors(
                torch.randn(8, 8), torch.eye(8), rank=9
            )

    def test_rejects_mismatched_moment_shape(self):
        with pytest.raises(ValueError, match="second_moment"):
            activation_aware_low_rank_factors(
                torch.randn(8, 10), torch.eye(8), rank=2
            )


class TestColumnActivationImportance:
    def test_matches_diagonal_sqrt(self):
        moment = _random_spd_matrix(10)
        importance = column_activation_importance(moment)
        expected = torch.diagonal(moment).sqrt()
        assert torch.allclose(importance, expected, atol=1e-6)

    def test_strictly_positive_even_for_zero_moment(self):
        importance = column_activation_importance(torch.zeros(6, 6))
        assert (importance > 0).all()


class TestBlockAverageSecondMoment:
    def test_averages_diagonal_blocks(self):
        moment = torch.zeros(8, 8)
        moment[:4, :4] = 2.0 * torch.eye(4)
        moment[4:, 4:] = 4.0 * torch.eye(4)
        averaged = block_average_second_moment(moment, block_size=4)
        assert torch.allclose(averaged, 3.0 * torch.eye(4))

    def test_rejects_non_dividing_block_size(self):
        with pytest.raises(ValueError, match="block_size"):
            block_average_second_moment(torch.eye(10), block_size=4)


class TestImportanceRankedSparseResidual:
    def test_importance_changes_selection(self):
        error = torch.tensor([[1.0, 0.9], [0.0, 0.0]])
        importance = torch.tensor([0.1, 10.0])
        residual = SparseResidual.fit(
            error, density=0.25, column_importance=importance
        )
        # Raw magnitude would pick (0,0)=1.0; importance picks (0,1)=0.9.
        assert residual.flat_indices.tolist() == [1]
        assert torch.allclose(residual.values, torch.tensor([0.9]).half().float())

    def test_stored_values_are_unweighted_errors(self):
        error = torch.tensor([[0.5, -0.25]])
        importance = torch.tensor([1.0, 100.0])
        residual = SparseResidual.fit(
            error, density=0.5, column_importance=importance
        )
        assert torch.allclose(
            residual.values, torch.tensor([-0.25]).half().float()
        )

    def test_rejects_mismatched_importance_shape(self):
        with pytest.raises(ValueError, match="column_importance"):
            SparseResidual.fit(
                torch.randn(4, 6), density=0.1,
                column_importance=torch.ones(5),
            )

    def test_no_importance_preserves_original_behavior(self):
        error = torch.tensor([[1.0, 0.9], [0.0, 0.0]])
        residual = SparseResidual.fit(error, density=0.25)
        assert residual.flat_indices.tolist() == [0]


class TestActivationAwareLowRankRepresentation:
    def test_name_records_fitting_mode(self):
        weight_matrix = torch.randn(16, 24)
        weight_fit = LowRankResidualRepresentation(weight_matrix, rank=4)
        activation_fit = LowRankResidualRepresentation(
            weight_matrix, rank=4, second_moment=_random_spd_matrix(24)
        )
        assert "fit=wgt" in weight_fit.name
        assert "fit=act" in activation_fit.name

    def test_state_accounting_is_identical_across_fit_modes(self):
        """The second moment is fitting-time state, not model state."""
        weight_matrix = torch.randn(16, 24)
        weight_fit = LowRankResidualRepresentation(
            weight_matrix, rank=4, residual_density=0.02
        )
        activation_fit = LowRankResidualRepresentation(
            weight_matrix, rank=4, residual_density=0.02,
            second_moment=_random_spd_matrix(24),
        )
        assert (
            weight_fit.state_accounting().field_bits
            == activation_fit.state_accounting().field_bits
        )

    def test_transform_matches_reconstruction(self):
        weight_matrix = torch.randn(16, 24)
        representation = LowRankResidualRepresentation(
            weight_matrix, rank=4, residual_density=0.05,
            second_moment=_random_spd_matrix(24),
        )
        activations = torch.randn(24, 7)
        assert torch.allclose(
            representation.transform(activations),
            representation.reconstruct() @ activations,
            atol=1e-4,
        )

    def test_no_dense_materialization(self):
        representation = LowRankResidualRepresentation(
            torch.randn(16, 24), rank=4,
            second_moment=_random_spd_matrix(24),
        )
        assert representation.max_decoded_weight_elements() == 0
        representation.verify_no_dense_materialization()

    def test_rejects_mismatched_moment(self):
        with pytest.raises(ValueError, match="second_moment"):
            LowRankResidualRepresentation(
                torch.randn(8, 10), rank=2, second_moment=torch.eye(8)
            )


class TestActivationAwareSharedBasis:
    def test_name_records_fitting_mode(self):
        weight_matrix = torch.randn(16, 32)
        activation_fit = SharedBasisRepresentation(
            weight_matrix, block_size=8, harmonic_count=4, basis_mode="svd",
            second_moment=_random_spd_matrix(32),
        )
        assert "fit=act" in activation_fit.name

    def test_transform_matches_reconstruction_svd(self):
        weight_matrix = torch.randn(16, 32)
        representation = SharedBasisRepresentation(
            weight_matrix, block_size=8, harmonic_count=4, basis_mode="svd",
            residual_density=0.02, second_moment=_random_spd_matrix(32),
        )
        activations = torch.randn(32, 5)
        assert torch.allclose(
            representation.transform(activations),
            representation.reconstruct() @ activations,
            atol=1e-4,
        )

    def test_transform_matches_reconstruction_dct(self):
        weight_matrix = torch.randn(16, 32)
        representation = SharedBasisRepresentation(
            weight_matrix, block_size=8, harmonic_count=4, basis_mode="dct",
            second_moment=_random_spd_matrix(32),
        )
        activations = torch.randn(32, 5)
        assert torch.allclose(
            representation.transform(activations),
            representation.reconstruct() @ activations,
            atol=1e-4,
        )

    def test_activation_aware_dct_counts_stored_basis(self):
        """Re-weighted DCT coefficients pair with a stored basis; the
        frequency-index shortcut only applies to the pure procedural bank."""
        weight_matrix = torch.randn(16, 32)
        plain = SharedBasisRepresentation(
            weight_matrix, block_size=8, harmonic_count=4, basis_mode="dct"
        )
        aware = SharedBasisRepresentation(
            weight_matrix, block_size=8, harmonic_count=4, basis_mode="dct",
            second_moment=_random_spd_matrix(32),
        )
        assert "basis_frequency_indices" in plain.state_accounting().field_bits
        assert "shared_basis" in aware.state_accounting().field_bits

    def test_activation_aware_beats_plain_on_distribution(self):
        """On low-dimensional activations the whitened basis must win."""
        torch.manual_seed(3)
        weight_matrix = torch.randn(24, 32)
        subspace = torch.linalg.qr(torch.randn(32, 4)).Q
        samples = torch.randn(4096, 4) @ subspace.T + 0.01 * torch.randn(4096, 32)
        moment = (samples.T @ samples / samples.shape[0]).to(torch.float32)
        probes = samples[:256].T.contiguous()

        plain = SharedBasisRepresentation(
            weight_matrix, block_size=16, harmonic_count=4, basis_mode="svd"
        )
        aware = SharedBasisRepresentation(
            weight_matrix, block_size=16, harmonic_count=4, basis_mode="svd",
            second_moment=moment,
        )
        reference = weight_matrix @ probes
        plain_error = torch.linalg.norm(plain.transform(probes) - reference)
        aware_error = torch.linalg.norm(aware.transform(probes) - reference)
        assert aware_error < plain_error

    def test_no_dense_materialization(self):
        representation = SharedBasisRepresentation(
            torch.randn(16, 32), block_size=8, harmonic_count=4,
            basis_mode="svd", second_moment=_random_spd_matrix(32),
        )
        assert representation.max_decoded_weight_elements() == 0
        representation.verify_no_dense_materialization()


class TestActivationSplit:
    def test_split_is_deterministic(self):
        samples = torch.randn(100, 8)
        first = split_captured_activations(samples, "p", seed=7)
        second = split_captured_activations(samples, "p", seed=7)
        assert torch.equal(
            first.evaluation_activations, second.evaluation_activations
        )
        assert torch.equal(first.second_moment, second.second_moment)

    def test_fit_and_eval_are_disjoint_and_complete(self):
        samples = torch.arange(40, dtype=torch.float32).reshape(20, 2)
        statistics = split_captured_activations(
            samples, "p", evaluation_fraction=0.25, seed=0
        )
        assert statistics.evaluation_activations.shape == (2, 5)
        assert statistics.fit_sample_count == 15

    def test_second_moment_matches_manual_computation(self):
        samples = torch.randn(64, 6)
        statistics = split_captured_activations(
            samples, "p", evaluation_fraction=0.25, seed=0
        )
        generator = torch.Generator().manual_seed(0)
        permutation = torch.randperm(64, generator=generator)
        fitting_rows = samples[permutation[16:]].to(torch.float64)
        expected = (
            fitting_rows.T @ fitting_rows / fitting_rows.shape[0]
        ).to(torch.float32)
        assert torch.allclose(statistics.second_moment, expected, atol=1e-6)

    def test_rejects_too_few_samples(self):
        with pytest.raises(ValueError, match="at least 4"):
            split_captured_activations(torch.randn(2, 4), "p")

    def test_rejects_bad_fraction(self):
        with pytest.raises(ValueError, match="evaluation_fraction"):
            split_captured_activations(
                torch.randn(10, 4), "p", evaluation_fraction=1.0
            )


@pytest.fixture(scope="module")
def synthetic_case():
    return build_synthetic_case(rows=32, cols=64, activation_rank=6, seed=0)


class TestExperiment12Runner:

    def test_synthetic_activation_aware_passes_both_gates(self, synthetic_case):
        """The designed-to-win case: full-rank W, rank-6 activations."""
        weight_matrix, second_moment, evaluation_activations = synthetic_case
        representation = LowRankResidualRepresentation(
            weight_matrix, rank=8, second_moment=second_moment
        )
        report = evaluate_candidate(
            representation,
            weight_matrix,
            evaluation_activations,
            torch.randn(64, 16),
            "synthetic",
        )
        assert report.layer_output_relative_error <= LAYER_OUTPUT_GATE
        assert (
            report.extra_metrics["state_ratio_to_dense_fp16"]
            <= STATE_RATIO_GATE
        )
        assert report.decision == "advance"

    def test_synthetic_weight_space_fails_at_same_capacity(self, synthetic_case):
        weight_matrix, _, evaluation_activations = synthetic_case
        representation = LowRankResidualRepresentation(weight_matrix, rank=8)
        report = evaluate_candidate(
            representation,
            weight_matrix,
            evaluation_activations,
            torch.randn(64, 16),
            "synthetic",
        )
        assert report.layer_output_relative_error > LAYER_OUTPUT_GATE
        assert report.decision == "retain as baseline"

    def test_random_probe_error_reported_separately(self, synthetic_case):
        """Activation-aware fits trade off-distribution error for
        on-distribution error; both must be visible."""
        weight_matrix, second_moment, evaluation_activations = synthetic_case
        representation = LowRankResidualRepresentation(
            weight_matrix, rank=8, second_moment=second_moment
        )
        report = evaluate_candidate(
            representation,
            weight_matrix,
            evaluation_activations,
            torch.randn(64, 16),
            "synthetic",
        )
        assert (
            report.extra_metrics["random_probe_layer_error"]
            > report.layer_output_relative_error
        )

    def test_runner_produces_paired_reports(self, synthetic_case):
        weight_matrix, second_moment, evaluation_activations = synthetic_case
        reports = run_experiment_12(
            weight_matrix, second_moment, evaluation_activations, "synthetic"
        )
        activation_count = sum(
            1 for r in reports if ",fit=act]" in r.representation
        )
        weight_count = sum(
            1 for r in reports if ",fit=wgt]" in r.representation
        )
        assert activation_count == weight_count > 0

    def test_gain_summary_pairs_configurations(self, synthetic_case):
        weight_matrix, second_moment, evaluation_activations = synthetic_case
        reports = run_experiment_12(
            weight_matrix, second_moment, evaluation_activations, "synthetic"
        )
        summary = summarize_activation_gain(reports)
        assert "low-rank+sparse-residual[r=" in summary
        # Every paired line must show a finite gain column.
        assert "x  " in summary
