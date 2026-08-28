"""Contract tests for the Experiment 1 harness — harmonic derivative baseline.

Gates verified here:

1. Oscillator recurrence decode matches the reference per-sample-sin decode
   within the cross-algorithm drift tolerance (5e-6). The strict 1e-6 parity
   contract applies Python-vs-Rust on the same recurrence in Epic 02.
2. Fused transform (generate + MAC, no dense weights) matches dense matmul of
   the diagnostic reconstruction to float32 accumulation tolerance.
3. State accounting counts every field and matches the analytical formula.
4. The no-dense-materialization contract rejects dense-scale scratch.
5. Precision sweep points quantize honestly (smaller state, bounded error).
"""

from __future__ import annotations

import pytest
import torch

from src.codec import generate_synthetic_target_matrix
from src.encoder import EncoderConfig, encode_tensor
from src.harness import (
    DENSE_BASELINE_BITS_PER_WEIGHT,
    HarmonicDerivativeRepresentation,
    StateAccounting,
)
from src.harness.experiment_1_harmonic import (
    CROSS_ALGORITHM_DRIFT_TOLERANCE,
    SweepPoint,
    evaluate_sweep_point,
    measure_oscillator_parity,
)
from src.phasor_atom import decode_tensor

ROWS, COLS = 16, 64
BLOCK_SIZE, HARMONIC_COUNT = 16, 4


@pytest.fixture(scope="module")
def fitted_atom():
    torch.manual_seed(0)
    weight_matrix = generate_synthetic_target_matrix(ROWS, COLS)
    configuration = EncoderConfig(
        block_size=BLOCK_SIZE, harmonic_count=HARMONIC_COUNT, refinement_steps=50
    )
    return weight_matrix, encode_tensor(weight_matrix, configuration)


class TestOscillatorParity:
    def test_recurrence_matches_reference_decode(self, fitted_atom) -> None:
        _, atom = fitted_atom
        representation = HarmonicDerivativeRepresentation(atom)
        parity_delta = measure_oscillator_parity(representation)
        assert parity_delta <= CROSS_ALGORITHM_DRIFT_TOLERANCE

    def test_reconstruct_shape(self, fitted_atom) -> None:
        _, atom = fitted_atom
        representation = HarmonicDerivativeRepresentation(atom)
        assert representation.reconstruct().shape == (ROWS, COLS)


class TestFusedTransform:
    def test_transform_matches_dense_matmul_of_reconstruction(
        self, fitted_atom
    ) -> None:
        _, atom = fitted_atom
        representation = HarmonicDerivativeRepresentation(atom)
        reconstruction = representation.reconstruct()

        torch.manual_seed(1)
        activations = torch.randn(COLS, 4, dtype=torch.float32)
        fused_output = representation.transform(activations)
        dense_output = reconstruction @ activations

        assert fused_output.shape == (ROWS, 4)
        assert torch.allclose(fused_output, dense_output, atol=1e-4, rtol=1e-4)

    def test_transform_accepts_1d_activations(self, fitted_atom) -> None:
        _, atom = fitted_atom
        representation = HarmonicDerivativeRepresentation(atom)
        activations = torch.randn(COLS, dtype=torch.float32)
        output = representation.transform(activations)
        assert output.shape == (ROWS,)

    def test_transform_rejects_wrong_activation_length(self, fitted_atom) -> None:
        _, atom = fitted_atom
        representation = HarmonicDerivativeRepresentation(atom)
        with pytest.raises(ValueError, match="input activations"):
            representation.transform(torch.randn(COLS + 1, dtype=torch.float32))


class TestStateAccounting:
    def test_field_bits_match_analytical_formula(self, fitted_atom) -> None:
        _, atom = fitted_atom
        representation = HarmonicDerivativeRepresentation(atom)
        accounting = representation.state_accounting()

        block_count = atom.num_blocks
        anchor_count = ROWS * block_count
        coefficient_count = anchor_count * HARMONIC_COUNT

        assert accounting.field_bits["anchors"] == anchor_count * 32
        assert accounting.field_bits["amplitudes"] == coefficient_count * 32
        assert accounting.field_bits["frequencies"] == coefficient_count * 32
        assert accounting.field_bits["phases"] == coefficient_count * 32
        assert accounting.represented_weight_count == ROWS * COLS

    def test_quantized_fields_shrink_state(self, fitted_atom) -> None:
        _, atom = fitted_atom
        full_precision = HarmonicDerivativeRepresentation(atom).state_accounting()
        half_precision = HarmonicDerivativeRepresentation(
            atom, anchor_bits=16, coefficient_bits=16
        ).state_accounting()
        assert half_precision.total_bits < full_precision.total_bits

    def test_ratio_to_dense_fp16(self, fitted_atom) -> None:
        _, atom = fitted_atom
        accounting = HarmonicDerivativeRepresentation(atom).state_accounting()
        expected_ratio = accounting.total_bits / (
            ROWS * COLS * DENSE_BASELINE_BITS_PER_WEIGHT
        )
        assert accounting.ratio_to_dense_fp16() == pytest.approx(expected_ratio)

    def test_negative_field_bits_rejected(self) -> None:
        with pytest.raises(ValueError, match="negative bit count"):
            StateAccounting(field_bits={"bad": -1}, represented_weight_count=1)


class TestMaterializationContract:
    def test_scratch_is_much_smaller_than_dense(self, fitted_atom) -> None:
        _, atom = fitted_atom
        representation = HarmonicDerivativeRepresentation(atom)
        representation.verify_no_dense_materialization()
        assert representation.max_transient_scratch_elements() < ROWS * COLS

    def test_invalid_precision_rejected(self, fitted_atom) -> None:
        _, atom = fitted_atom
        with pytest.raises(ValueError, match="anchor_bits"):
            HarmonicDerivativeRepresentation(atom, anchor_bits=8)
        with pytest.raises(ValueError, match="coefficient_bits"):
            HarmonicDerivativeRepresentation(atom, coefficient_bits=64)


class TestSweepEvaluation:
    def test_lossless_ceiling_sweep_point(self) -> None:
        """K = block_size/2 must remain the exact-reconstruction ceiling."""
        torch.manual_seed(0)
        weight_matrix = generate_synthetic_target_matrix(8, 32)
        sweep_point = SweepPoint(
            block_size=16, harmonic_count=8, anchor_bits=32, coefficient_bits=32
        )
        calibration = torch.randn(32, 4, dtype=torch.float32)
        report = evaluate_sweep_point(
            weight_matrix, sweep_point, refinement_steps=0, calibration_activations=calibration
        )
        assert report.weight_reconstruction_relative_error < 1e-4
        assert report.layer_output_relative_error < 1e-4
        assert not report.dense_weight_materialized

    def test_report_contains_state_metrics(self) -> None:
        torch.manual_seed(0)
        weight_matrix = generate_synthetic_target_matrix(8, 32)
        sweep_point = SweepPoint(
            block_size=16, harmonic_count=4, anchor_bits=16, coefficient_bits=16
        )
        calibration = torch.randn(32, 2, dtype=torch.float32)
        report = evaluate_sweep_point(
            weight_matrix, sweep_point, refinement_steps=0, calibration_activations=calibration
        )
        assert "state_bits_per_weight" in report.extra_metrics
        assert report.extra_metrics["state_ratio_to_dense_fp16"] > 0
        rendered = report.format_report()
        assert "Dense weight materialized: no" in rendered


class TestReferenceDecodeUnchanged:
    def test_facade_decode_still_matches(self, fitted_atom) -> None:
        """The Epic 01 reference decode contract is untouched by the harness."""
        _, atom = fitted_atom
        reference = decode_tensor(atom)
        assert reference.shape == (ROWS, COLS)
        assert reference.dtype == torch.float32
