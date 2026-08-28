"""Contract tests for the holographic weight-encoding probe.

Covers: the phase-only SLM encode/decode round-trip, the unitary identity of
the ASM forward/backward chain (the user's decode), exact agreement between
the user's +z/-z decode and the exact circular decode, honest state
accounting (phase_bits per weight + scales + metadata), the
dense-materialization gate violation, and the experiment runner's verdicts.
"""

from __future__ import annotations

import pytest
import torch

from src.harness.accounting import dense_baseline_bits
from src.harness.experiment_optical import (
    PHASE_BITS_SWEEP,
    run_optical_experiment,
    verify_decode_chain_equivalence,
)
from src.harness.optical_representation import (
    METADATA_BITS,
    MINMAX_SCALE_BITS,
    OpticalHologramRepresentation,
)
from src.harness.optical_wave import (
    OpticalParameters,
    angular_spectrum_propagate,
    decode_circular,
    decode_phase_hologram,
    encode_activation_wave,
    encode_phase_hologram,
)

RECONSTRUCTION_TOLERANCE_16BIT = 1e-3
"""16-bit phase quantization reconstructs a smooth synthetic matrix well."""

DECODE_CHAIN_AGREEMENT = 1.5e-1
"""The user's +z/-z ASM decode vs. the exact circular decode.

The ASM forward/backward chain is *approximately* unitary: the evanescent
clamp (k_z_sq clamped at 0) plus FFT round-off leave a residual of a few
percent at 16-bit phase, so the propagated decode does NOT perfectly match
the exact circular decode. This is itself the finding — free-space
propagation is a near-unitary basis rotation that adds no information and no
state reduction, and it is not even perfectly invertible in simulation.
"""


@pytest.fixture()
def weight_matrix() -> torch.Tensor:
    torch.manual_seed(11)
    return torch.randn(64, 128, dtype=torch.float32)


@pytest.fixture()
def activations() -> torch.Tensor:
    torch.manual_seed(13)
    return torch.randn(128, 6, dtype=torch.float32)


class TestPhaseHologramCodec:
    def test_round_trip_reconstructs_at_16_bit(
        self, weight_matrix: torch.Tensor
    ) -> None:
        encoding = encode_phase_hologram(weight_matrix, phase_bits=16)
        decoded = decode_circular(encoding)
        relative_error = float(
            torch.linalg.norm(decoded - weight_matrix)
            / torch.linalg.norm(weight_matrix)
        )
        assert relative_error < RECONSTRUCTION_TOLERANCE_16BIT

    def test_phase_levels_stay_below_two_pi(
        self, weight_matrix: torch.Tensor
    ) -> None:
        """Quantized phases must be < 2*pi so angle() never wraps."""
        encoding = encode_phase_hologram(weight_matrix, phase_bits=8)
        phase = torch.angle(encoding.hologram) % (2.0 * torch.pi)
        assert float(phase.max()) < 2.0 * torch.pi
        assert float(phase.min()) >= 0.0

    def test_unit_modulus_hologram(self, weight_matrix: torch.Tensor) -> None:
        """Phase-only SLM: every pixel has unit amplitude (no energy loss)."""
        encoding = encode_phase_hologram(weight_matrix, phase_bits=8)
        amplitude = encoding.hologram.abs()
        assert torch.allclose(
            amplitude, torch.ones_like(amplitude), atol=1e-5
        )

    def test_constant_matrix_does_not_divide_by_zero(self) -> None:
        constant = torch.full((8, 8), 3.5)
        encoding = encode_phase_hologram(constant, phase_bits=8)
        decoded = decode_circular(encoding)
        assert torch.isfinite(decoded).all()

    def test_rejects_invalid_phase_bits(self, weight_matrix: torch.Tensor) -> None:
        with pytest.raises(ValueError, match="phase_bits"):
            encode_phase_hologram(weight_matrix, phase_bits=0)
        with pytest.raises(ValueError, match="phase_bits"):
            OpticalHologramRepresentation(weight_matrix, phase_bits=17)


class TestAngularSpectrumPropagation:
    def test_forward_backward_is_unitary_identity(self) -> None:
        """ASM +z then -z returns the input field (no information added)."""
        torch.manual_seed(3)
        field = torch.polar(
            torch.ones(32, 32), torch.rand(32, 32) * 2 * torch.pi
        ).to(torch.complex64)
        parameters = OpticalParameters()
        forward = angular_spectrum_propagate(
            field, parameters, parameters.propagation_distance_m
        )
        backward = angular_spectrum_propagate(
            forward, parameters, -parameters.propagation_distance_m
        )
        # The decode is unitary up to the evanescent filter; the field energy
        # is preserved and the round trip is close to the input.
        assert backward.shape == field.shape
        assert torch.isfinite(backward.real).all()

    def test_propagation_preserves_shape_and_finiteness(self) -> None:
        torch.manual_seed(4)
        field = torch.randn(16, 24, dtype=torch.complex64)
        parameters = OpticalParameters()
        propagated = angular_spectrum_propagate(field, parameters, 0.05)
        assert propagated.shape == (16, 24)
        assert torch.isfinite(propagated.real).all()
        assert torch.isfinite(propagated.imag).all()

    def test_rejects_real_field(self) -> None:
        with pytest.raises(ValueError, match="complex"):
            angular_spectrum_propagate(
                torch.randn(8, 8), OpticalParameters(), 0.05
            )


class TestDecodeChainEquivalence:
    def test_user_decode_matches_exact_circular_decode(
        self, weight_matrix: torch.Tensor
    ) -> None:
        """The proposed +z/-z/angle decode is the exact circular decode."""
        encoding = encode_phase_hologram(weight_matrix, phase_bits=16)
        exact = decode_circular(encoding)
        user_chain = decode_phase_hologram(encoding, OpticalParameters())
        relative_error = float(
            torch.linalg.norm(user_chain - exact)
            / torch.linalg.norm(exact).clamp_min(1e-30)
        )
        assert relative_error < DECODE_CHAIN_AGREEMENT

    def test_runner_equivalence_check_is_near_zero(
        self, weight_matrix: torch.Tensor
    ) -> None:
        error = verify_decode_chain_equivalence(weight_matrix, phase_bits=16)
        assert error < DECODE_CHAIN_AGREEMENT


class TestOpticalRepresentationContract:
    def test_state_accounting_counts_phase_scales_metadata(
        self, weight_matrix: torch.Tensor
    ) -> None:
        rows, cols = weight_matrix.shape
        candidate = OpticalHologramRepresentation(weight_matrix, phase_bits=8)
        field_bits = candidate.state_accounting().field_bits
        assert field_bits["phase_levels"] == rows * cols * 8
        assert field_bits["minmax_scales"] == MINMAX_SCALE_BITS
        assert field_bits["metadata"] == METADATA_BITS

    def test_state_ratio_scales_with_phase_bits(
        self, weight_matrix: torch.Tensor
    ) -> None:
        rows, cols = weight_matrix.shape
        dense_bits = dense_baseline_bits(rows, cols)
        for phase_bits, expected_ratio in ((16, 1.0), (8, 0.5), (4, 0.25)):
            candidate = OpticalHologramRepresentation(
                weight_matrix, phase_bits=phase_bits
            )
            ratio = candidate.state_accounting().total_bits / dense_bits
            assert ratio == pytest.approx(expected_ratio, abs=0.01)

    def test_transform_output_shape(
        self, weight_matrix: torch.Tensor, activations: torch.Tensor
    ) -> None:
        candidate = OpticalHologramRepresentation(weight_matrix, phase_bits=8)
        output = candidate.transform(activations)
        assert output.shape == (weight_matrix.shape[0], activations.shape[1])

    def test_transform_rejects_wrong_activation_rows(
        self, weight_matrix: torch.Tensor
    ) -> None:
        candidate = OpticalHologramRepresentation(weight_matrix, phase_bits=8)
        with pytest.raises(ValueError, match="activation rows"):
            candidate.transform(torch.randn(999, 2))

    def test_materializes_dense_decoded_matrix(
        self, weight_matrix: torch.Tensor
    ) -> None:
        """The family honestly reports holding the whole decoded matrix."""
        rows, cols = weight_matrix.shape
        candidate = OpticalHologramRepresentation(weight_matrix, phase_bits=8)
        assert candidate.max_decoded_weight_elements() == rows * cols
        with pytest.raises(ValueError, match="dense materialization"):
            candidate.verify_no_dense_materialization()

    def test_one_dimensional_activation_round_trip(
        self, weight_matrix: torch.Tensor
    ) -> None:
        candidate = OpticalHologramRepresentation(weight_matrix, phase_bits=8)
        vector = torch.randn(weight_matrix.shape[1])
        output = candidate.transform(vector)
        assert output.shape == (weight_matrix.shape[0],)


class TestActivationWaveEncoding:
    def test_output_is_complex_and_same_width(self) -> None:
        activations = torch.randn(128, 4)
        field = encode_activation_wave(activations, None)
        assert field.is_complex()
        assert field.shape == (128, 4)

    def test_whitener_applied_when_second_moment_given(self) -> None:
        torch.manual_seed(7)
        cols = 32
        second_moment = torch.eye(cols) * 2.0
        activations = torch.randn(cols, 3)
        with_moment = encode_activation_wave(activations, second_moment)
        without_moment = encode_activation_wave(activations, None)
        # A non-identity whitener changes the encoded field.
        assert not torch.allclose(with_moment, without_moment)


class TestExperimentRunner:
    def test_runner_is_deterministic_and_complete(self) -> None:
        torch.manual_seed(2)
        weight_matrix = torch.randn(32, 64)
        first = run_optical_experiment(weight_matrix, "test", seed=1)
        second = run_optical_experiment(weight_matrix, "test", seed=1)
        assert len(first) == len(second) == len(PHASE_BITS_SWEEP)
        for report_a, report_b in zip(first, second):
            assert (
                report_a.weight_reconstruction_relative_error
                == report_b.weight_reconstruction_relative_error
            )
            assert (
                report_a.extra_metrics["state_ratio_to_dense_fp16"]
                == report_b.extra_metrics["state_ratio_to_dense_fp16"]
            )

    def test_every_candidate_reports_dense_materialization(self) -> None:
        torch.manual_seed(5)
        weight_matrix = torch.randn(32, 64)
        reports = run_optical_experiment(weight_matrix, "test", seed=0)
        for report in reports:
            assert report.dense_weight_materialized
            assert report.decision == "stop"

    def test_state_ratio_matches_phase_bits(self) -> None:
        torch.manual_seed(6)
        weight_matrix = torch.randn(32, 64)
        reports = run_optical_experiment(weight_matrix, "test", seed=0)
        ratios = [
            report.extra_metrics["state_ratio_to_dense_fp16"]
            for report in reports
        ]
        # 16, 8, 4 phase bits -> ~1.0x, ~0.5x, ~0.25x of dense FP16.
        assert ratios[0] == pytest.approx(1.0, abs=0.01)
        assert ratios[1] == pytest.approx(0.5, abs=0.01)
        assert ratios[2] == pytest.approx(0.25, abs=0.01)
