"""Holographic weight-encoding experiment runner.

Sweeps the SLM phase resolution (``phase_bits``) and scores the
:class:`~src.harness.optical_representation.OpticalHologramRepresentation`
against the program-wide gates, reusing the shared
:func:`~src.harness.experiment_shared_state.evaluate_candidate` measurement
so every number is directly comparable to Experiments 1/2/3/5/12.

What this runner measures
-------------------------
1. **State reduction claim** — ``state_ratio_to_dense_fp16`` per
   ``phase_bits`` (16/8/4 -> ~1.0x / ~0.5x / ~0.25x of dense FP16). The only
   reduction is phase quantization; the ASM propagation carries no state.
2. **Fidelity** — dense-reconstruction relative error (diagnostic) and the
   detected-power layer-output error on spectrally flat (Gaussian) probes.
3. **The user's decode chain** — ``decode_phase_hologram`` (forward +z,
   backward -z, read angle) is verified to be *identical* to the exact
   circular decode, confirming ASM propagation is unitary and carries no
   information.

Usage::

    python -m src.harness.experiment_optical --source synthetic
    python -m src.harness.experiment_optical --source pythia
"""

from __future__ import annotations

import argparse
import sys

import torch

from src.encoder import compute_relative_error
from src.harness.accounting import ExperimentReport, dense_baseline_bits
from src.harness.experiment_shared_state import (
    LAYER_OUTPUT_GATE,
    STATE_RATIO_GATE,
    load_pythia_matrices,
)
from src.harness.optical_representation import OpticalHologramRepresentation
from src.harness.optical_wave import (
    OpticalParameters,
    decode_circular,
    decode_phase_hologram,
)

PHASE_BITS_SWEEP: tuple[int, ...] = (16, 8, 4)
"""SLM phase resolutions: FP16-parity, SLM-8-bit, and 4-bit-incumbent-parity."""


def evaluate_optical_candidate(
    candidate: OpticalHologramRepresentation,
    weight_matrix: torch.Tensor,
    calibration_activations: torch.Tensor,
    tensor_description: str,
) -> ExperimentReport:
    """Measure the hologram under the required report template.

    Unlike the shared evaluator, this reports the dense-materialization gate
    honestly: the optical transform holds the entire decoded real matrix in
    the propagation domain, so ``dense_weight_materialized`` is True and the
    candidate can never pass the program's no-materialization requirement.
    """
    reconstruction = candidate.reconstruct()
    reconstruction_error = compute_relative_error(weight_matrix, reconstruction)

    dense_output = weight_matrix @ calibration_activations
    optical_output = candidate.transform(calibration_activations)
    layer_output_error = float(
        torch.linalg.norm(optical_output - dense_output)
        / torch.linalg.norm(dense_output).clamp_min(1e-30)
    )

    accounting = candidate.state_accounting()
    state_ratio = accounting.ratio_to_dense_fp16()
    dense_element_count = candidate.rows * candidate.cols
    materializes_dense = (
        candidate.max_decoded_weight_elements() >= dense_element_count
    )
    passes_gates = (
        state_ratio <= STATE_RATIO_GATE
        and layer_output_error <= LAYER_OUTPUT_GATE
        and not materializes_dense
    )
    row_count, column_count = weight_matrix.shape

    return ExperimentReport(
        representation=candidate.name,
        tensor_description=tensor_description,
        shared_executable_bytes=0,
        unique_model_state_bytes=accounting.total_bytes,
        residual_bytes=0.0,
        transient_scratch_bytes=(
            candidate.max_transient_scratch_elements() * 4.0
        ),
        dense_baseline_bytes=dense_baseline_bits(row_count, column_count) / 8.0,
        generator_operations_per_weight=(
            candidate.estimated_operations_per_weight()
        ),
        dense_weight_materialized=materializes_dense,
        weight_reconstruction_relative_error=reconstruction_error,
        layer_output_relative_error=layer_output_error,
        decision="advance" if passes_gates else "stop",
        reason=(
            "meets <=50% dense state at <=1e-2 layer output error with no "
            "dense materialization"
            if passes_gates
            else (
                "fails: propagation is a unitary identity (no state reduction "
                "beyond phase quantization), the hot path materializes the "
                "dense decoded matrix, and/or layer-output error exceeds 1e-2"
            )
        ),
        extra_metrics={
            "state_bits_per_weight": accounting.bits_per_weight,
            "state_ratio_to_dense_fp16": state_ratio,
        },
    )


def run_optical_experiment(
    weight_matrix: torch.Tensor,
    tensor_description: str,
    second_moment: torch.Tensor | None = None,
    calibration_batch: int = 8,
    seed: int = 0,
) -> list[ExperimentReport]:
    """Fit and measure the hologram at every phase resolution."""
    torch.manual_seed(seed)
    weight_matrix = weight_matrix.to(torch.float32)
    _, column_count = weight_matrix.shape
    calibration_activations = torch.randn(
        column_count, calibration_batch, dtype=torch.float32
    )
    reports: list[ExperimentReport] = []
    for phase_bits in PHASE_BITS_SWEEP:
        candidate = OpticalHologramRepresentation(
            weight_matrix,
            phase_bits=phase_bits,
            second_moment=second_moment,
        )
        report = evaluate_optical_candidate(
            candidate,
            weight_matrix,
            calibration_activations,
            tensor_description,
        )
        reports.append(report)
    return reports


def verify_decode_chain_equivalence(
    weight_matrix: torch.Tensor, phase_bits: int = 16
) -> float:
    """Confirm the user's +z/-z decode equals the exact circular decode.

    Returns the relative error between the two decode paths; ~0 proves the
    ASM propagation chain is an exact identity (unitary) and therefore
    carries no information and no state reduction.
    """
    candidate = OpticalHologramRepresentation(
        weight_matrix, phase_bits=phase_bits
    )
    exact = decode_circular(candidate.encoding)
    user_chain = decode_phase_hologram(
        candidate.encoding, OpticalParameters()
    )
    return float(
        torch.linalg.norm(user_chain - exact)
        / torch.linalg.norm(exact).clamp_min(1e-30)
    )


def summarize_optical(reports: list[ExperimentReport]) -> str:
    """One-line-per-resolution summary with the gate verdict."""
    lines = [
        f"{'phase bits':>10} {'state/dense':>12} {'recon err':>12} "
        f"{'layer err':>12} {'ops/w':>8} {'dense mat':>10}  verdict",
    ]
    for report in reports:
        state_ratio = report.extra_metrics["state_ratio_to_dense_fp16"]
        layer_error = report.layer_output_relative_error
        materializes = report.dense_weight_materialized
        if (
            state_ratio <= STATE_RATIO_GATE
            and layer_error <= LAYER_OUTPUT_GATE
            and not materializes
        ):
            verdict = "passes all gates"
        elif materializes:
            verdict = "materializes dense matrix"
        elif state_ratio <= STATE_RATIO_GATE:
            verdict = "state OK, error too high"
        else:
            verdict = "no state reduction (quantization only)"
        phase_bits = int(round(state_ratio * 16))
        lines.append(
            f"{phase_bits:>10} {state_ratio:>12.4f} "
            f"{report.weight_reconstruction_relative_error:>12.4e} "
            f"{layer_error:>12.4e} "
            f"{report.generator_operations_per_weight:>8.2f} "
            f"{'yes' if materializes else 'no':>10}  {verdict}"
        )
    return "\n".join(lines)


def main(command_line_arguments: list[str] | None = None) -> int:
    argument_parser = argparse.ArgumentParser(
        prog="python -m src.harness.experiment_optical",
        description="Holographic phase-only SLM weight encoding probe.",
    )
    argument_parser.add_argument(
        "--source", choices=("synthetic", "pythia"), default="synthetic"
    )
    argument_parser.add_argument("--rows", type=int, default=64)
    argument_parser.add_argument("--cols", type=int, default=128)
    argument_parser.add_argument("--seed", type=int, default=0)
    argument_parser.add_argument(
        "--full-reports", action="store_true",
        help="Print the full required report per configuration.",
    )
    parsed_arguments = argument_parser.parse_args(command_line_arguments)

    if parsed_arguments.source == "pythia":
        matrices = load_pythia_matrices()
    else:
        from src.codec import generate_synthetic_target_matrix

        matrices = [
            (
                f"synthetic/dense {parsed_arguments.rows}x{parsed_arguments.cols} f32",
                generate_synthetic_target_matrix(
                    parsed_arguments.rows, parsed_arguments.cols
                ),
            )
        ]

    for tensor_name, weight_matrix in matrices:
        reports = run_optical_experiment(
            weight_matrix, tensor_name, seed=parsed_arguments.seed
        )
        chain_error = verify_decode_chain_equivalence(weight_matrix)
        print(f"=== {tensor_name} {tuple(weight_matrix.shape)} ===")
        print(
            f"decode-chain (ASM +z/-z) vs exact circular decode rel err: "
            f"{chain_error:.3e}  (0 => propagation is a unitary identity)"
        )
        print(summarize_optical(reports))
        if parsed_arguments.full_reports:
            for report in reports:
                print("-" * 72)
                print(report.format_report())
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
