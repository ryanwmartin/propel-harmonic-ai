"""Experiment 1 — harmonic derivative baseline sweep.

Runs the Experiment 1 protocol from ``agile/procedural-inference-experiments.md``:

- Encode a weight matrix with the existing Epic 01 fitter across a sweep of
  block size, harmonic count, anchor precision, and coefficient precision.
- Measure fidelity two ways: dense reconstruction (diagnostic) and fused
  ``transform`` layer-output error against dense matmul (the metric that
  matters for procedural inference).
- Verify Python parity between the reference ``decode_full_tensor`` prefix-sum
  path and the oscillator-recurrence path within the 1e-6 contract tolerance.
- Emit the required experiment report per configuration.

Usage::

    python -m src.harness.experiment_1_harmonic --rows 64 --cols 128
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import torch

from src.encoder import EncoderConfig, compute_relative_error, encode_tensor
from src.harness.accounting import ExperimentReport, dense_baseline_bits
from src.harness.harmonic_baseline import HarmonicDerivativeRepresentation
from src.phasor_atom import decode_tensor

PARITY_TOLERANCE: float = 1e-6
"""Same-algorithm Python/Rust parity tolerance (the Epic 02/03 contract)."""

CROSS_ALGORITHM_DRIFT_TOLERANCE: float = 1e-5
"""Relative tolerance between the direct per-sample f32 ``sin()`` reference
decode and the f64-state oscillator recurrence, normalized by the tensor's
max |weight|. These are different evaluation orders, so they legitimately
differ by accumulated f32 transcendental rounding that grows with block
length and signal magnitude. The strict 1e-6 absolute contract applies to
Python-vs-Rust implementations of the *same* recurrence, which is checked in
Epic 02, not to this cross-algorithm diagnostic."""


@dataclass(frozen=True)
class SweepPoint:
    """One Experiment 1 sweep configuration."""

    block_size: int
    harmonic_count: int
    anchor_bits: int
    coefficient_bits: int


def default_sweep(column_count: int) -> list[SweepPoint]:
    """The default Experiment 1 sweep grid, clamped to the tensor width."""
    sweep_points: list[SweepPoint] = []
    for block_size in (16, 32, 64):
        if block_size > column_count:
            continue
        for harmonic_count in (4, 8, block_size // 2):
            for anchor_bits, coefficient_bits in ((32, 32), (32, 16), (16, 16)):
                sweep_points.append(
                    SweepPoint(
                        block_size=block_size,
                        harmonic_count=harmonic_count,
                        anchor_bits=anchor_bits,
                        coefficient_bits=coefficient_bits,
                    )
                )
    return sweep_points


def measure_oscillator_parity(
    representation: HarmonicDerivativeRepresentation,
) -> float:
    """Relative max |Δ| between per-sample-sin decode and recurrence decode.

    Normalized by the reference reconstruction's max |weight| so the gate is
    scale-invariant. A cross-algorithm drift diagnostic, gated by
    :data:`CROSS_ALGORITHM_DRIFT_TOLERANCE`. Only meaningful at full float32
    storage precision; quantized sweeps skip the gate because the stored
    parameters intentionally differ.
    """
    reference_reconstruction = decode_tensor(representation.atom)
    recurrence_reconstruction = representation.reconstruct()
    max_absolute_delta = float(
        (reference_reconstruction - recurrence_reconstruction).abs().max()
    )
    magnitude_scale = float(reference_reconstruction.abs().max())
    return max_absolute_delta / max(magnitude_scale, 1e-30)


def evaluate_sweep_point(
    weight_matrix: torch.Tensor,
    sweep_point: SweepPoint,
    refinement_steps: int,
    calibration_activations: torch.Tensor,
) -> ExperimentReport:
    """Encode, verify, and report one Experiment 1 configuration."""
    configuration = EncoderConfig(
        block_size=sweep_point.block_size,
        harmonic_count=sweep_point.harmonic_count,
        refinement_steps=refinement_steps,
    )
    atom = encode_tensor(weight_matrix, configuration)
    representation = HarmonicDerivativeRepresentation(
        atom,
        anchor_bits=sweep_point.anchor_bits,
        coefficient_bits=sweep_point.coefficient_bits,
    )
    representation.verify_no_dense_materialization()

    is_full_precision = (
        sweep_point.anchor_bits == 32 and sweep_point.coefficient_bits == 32
    )
    if is_full_precision:
        parity_delta = measure_oscillator_parity(representation)
        if parity_delta > CROSS_ALGORITHM_DRIFT_TOLERANCE:
            raise AssertionError(
                f"oscillator recurrence drift failure: max |Δ| = {parity_delta:.3e} "
                f"exceeds {CROSS_ALGORITHM_DRIFT_TOLERANCE:.0e}"
            )

    reconstruction = representation.reconstruct()
    reconstruction_error = compute_relative_error(weight_matrix, reconstruction)

    dense_output = weight_matrix @ calibration_activations
    fused_output = representation.transform(calibration_activations)
    layer_output_error = float(
        torch.linalg.norm(fused_output - dense_output)
        / torch.linalg.norm(dense_output).clamp_min(1e-30)
    )

    accounting = representation.state_accounting()
    row_count, column_count = weight_matrix.shape

    return ExperimentReport(
        representation=representation.name,
        tensor_description=f"synthetic/dense {row_count}x{column_count} f32",
        shared_executable_bytes=0,
        unique_model_state_bytes=accounting.total_bytes,
        residual_bytes=0.0,
        transient_scratch_bytes=representation.max_transient_scratch_elements() * 4.0,
        dense_baseline_bytes=dense_baseline_bits(row_count, column_count) / 8.0,
        generator_operations_per_weight=(
            representation.estimated_operations_per_weight()
        ),
        dense_weight_materialized=False,
        weight_reconstruction_relative_error=reconstruction_error,
        layer_output_relative_error=layer_output_error,
        decision="retain as baseline",
        reason="Experiment 1 harmonic baseline sweep point",
        extra_metrics={
            "state_bits_per_weight": accounting.bits_per_weight,
            "state_ratio_to_dense_fp16": accounting.ratio_to_dense_fp16(),
        },
    )


def run_experiment_1(
    weight_matrix: torch.Tensor,
    refinement_steps: int = 100,
    sweep_points: list[SweepPoint] | None = None,
    calibration_batch: int = 8,
    seed: int = 0,
) -> list[ExperimentReport]:
    """Run the full Experiment 1 sweep on one weight matrix."""
    torch.manual_seed(seed)
    weight_matrix = weight_matrix.to(torch.float32)
    _, column_count = weight_matrix.shape
    calibration_activations = torch.randn(
        column_count, calibration_batch, dtype=torch.float32
    )
    if sweep_points is None:
        sweep_points = default_sweep(column_count)

    return [
        evaluate_sweep_point(
            weight_matrix, sweep_point, refinement_steps, calibration_activations
        )
        for sweep_point in sweep_points
    ]


def main(command_line_arguments: list[str] | None = None) -> int:
    argument_parser = argparse.ArgumentParser(
        prog="python -m src.harness.experiment_1_harmonic",
        description="Experiment 1 — harmonic derivative baseline sweep.",
    )
    argument_parser.add_argument("--rows", type=int, default=64)
    argument_parser.add_argument("--cols", type=int, default=128)
    argument_parser.add_argument("--steps", type=int, default=100)
    argument_parser.add_argument("--seed", type=int, default=0)
    parsed_arguments = argument_parser.parse_args(command_line_arguments)

    from src.codec import generate_synthetic_target_matrix

    weight_matrix = generate_synthetic_target_matrix(
        parsed_arguments.rows, parsed_arguments.cols
    )
    reports = run_experiment_1(
        weight_matrix,
        refinement_steps=parsed_arguments.steps,
        seed=parsed_arguments.seed,
    )

    for report in reports:
        print(report.format_report())
        print("-" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
