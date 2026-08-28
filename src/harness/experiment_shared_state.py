"""Experiments 3 + 5 — shared-state families vs. the harmonic baseline.

Compares three representation families under identical accounting on the same
weight matrix:

- Experiment 1 baseline: independent per-block harmonic derivatives.
- Experiment 3: shared DCT / SVD basis with per-block coefficients.
- Experiment 5: low-rank factors plus sparse residual.

All fits are closed-form (projection / SVD / top-K) — no distillation. Each
candidate is measured on:

- unique model-state bits per weight and ratio to dense FP16;
- dense-reconstruction relative error (diagnostic);
- fused layer-output relative error against dense matmul (decision metric);
- generator/transform operations per weight;
- the zero-dense-materialization gate.

Usage::

    python -m src.harness.experiment_shared_state --source synthetic
    python -m src.harness.experiment_shared_state --source pythia
"""

from __future__ import annotations

import argparse
import sys

import torch

from src.encoder import compute_relative_error
from src.harness.accounting import ExperimentReport, dense_baseline_bits
from src.harness.low_rank_residual import LowRankResidualRepresentation
from src.harness.representation import ProceduralRepresentation
from src.harness.shared_basis import SharedBasisRepresentation

LAYER_OUTPUT_GATE: float = 1e-2
"""Program-wide pre-distillation layer-output relative error gate."""

STATE_RATIO_GATE: float = 0.5
"""Advancement gate: state must be <= 50% of dense FP16."""


def evaluate_candidate(
    representation: ProceduralRepresentation,
    weight_matrix: torch.Tensor,
    calibration_activations: torch.Tensor,
    tensor_description: str,
) -> ExperimentReport:
    """Measure one fitted candidate under the required report template."""
    representation.verify_no_dense_materialization()

    reconstruction = representation.reconstruct()
    reconstruction_error = compute_relative_error(weight_matrix, reconstruction)

    dense_output = weight_matrix @ calibration_activations
    fused_output = representation.transform(calibration_activations)
    layer_output_error = float(
        torch.linalg.norm(fused_output - dense_output)
        / torch.linalg.norm(dense_output).clamp_min(1e-30)
    )

    accounting = representation.state_accounting()
    state_ratio = accounting.ratio_to_dense_fp16()
    passes_gates = (
        state_ratio <= STATE_RATIO_GATE
        and layer_output_error <= LAYER_OUTPUT_GATE
    )
    row_count, column_count = weight_matrix.shape
    residual_bits = sum(
        bits
        for field_name, bits in accounting.field_bits.items()
        if field_name.startswith("residual")
    )

    return ExperimentReport(
        representation=representation.name,
        tensor_description=tensor_description,
        shared_executable_bytes=0,
        unique_model_state_bytes=accounting.total_bytes,
        residual_bytes=residual_bits / 8.0,
        transient_scratch_bytes=(
            representation.max_transient_scratch_elements() * 4.0
        ),
        dense_baseline_bytes=dense_baseline_bits(row_count, column_count) / 8.0,
        generator_operations_per_weight=(
            representation.estimated_operations_per_weight()
        ),
        dense_weight_materialized=False,
        weight_reconstruction_relative_error=reconstruction_error,
        layer_output_relative_error=layer_output_error,
        decision="advance" if passes_gates else "retain as baseline",
        reason=(
            "meets <=50% dense state at <=1e-2 layer output error"
            if passes_gates
            else "fails state<=50% and/or layer-output<=1e-2 gate"
        ),
        extra_metrics={
            "state_bits_per_weight": accounting.bits_per_weight,
            "state_ratio_to_dense_fp16": state_ratio,
        },
    )


def build_candidates(
    weight_matrix: torch.Tensor,
    block_size: int = 64,
) -> list[ProceduralRepresentation]:
    """Fit the shared-state candidate grid for one weight matrix."""
    row_count, column_count = weight_matrix.shape
    maximum_rank = min(row_count, column_count)
    candidates: list[ProceduralRepresentation] = []

    for rank_fraction in (0.05, 0.10, 0.25, 0.50):
        rank = max(1, int(round(rank_fraction * maximum_rank)))
        for residual_density in (0.0, 0.01, 0.05):
            candidates.append(
                LowRankResidualRepresentation(
                    weight_matrix,
                    rank=rank,
                    residual_density=residual_density,
                    factor_bits=16,
                    residual_bits=16,
                )
            )

    usable_block_size = (
        block_size if column_count % block_size == 0 else maximum_usable_block(
            column_count
        )
    )
    for basis_mode in ("dct", "svd"):
        for harmonic_fraction in (0.25, 0.50, 0.75):
            harmonic_count = max(
                1, int(round(harmonic_fraction * usable_block_size))
            )
            for residual_density in (0.0, 0.01):
                candidates.append(
                    SharedBasisRepresentation(
                        weight_matrix,
                        block_size=usable_block_size,
                        harmonic_count=harmonic_count,
                        basis_mode=basis_mode,
                        residual_density=residual_density,
                        coefficient_bits=16,
                        residual_bits=16,
                    )
                )
    return candidates


def maximum_usable_block(column_count: int) -> int:
    """Largest power-of-two block size <= 64 that divides column_count."""
    for candidate_size in (64, 32, 16, 8, 4, 2):
        if column_count % candidate_size == 0:
            return candidate_size
    return 1


def run_shared_state_experiment(
    weight_matrix: torch.Tensor,
    tensor_description: str,
    calibration_batch: int = 8,
    seed: int = 0,
) -> list[ExperimentReport]:
    """Fit and measure all shared-state candidates on one weight matrix."""
    torch.manual_seed(seed)
    weight_matrix = weight_matrix.to(torch.float32)
    _, column_count = weight_matrix.shape
    calibration_activations = torch.randn(
        column_count, calibration_batch, dtype=torch.float32
    )
    return [
        evaluate_candidate(
            candidate, weight_matrix, calibration_activations, tensor_description
        )
        for candidate in build_candidates(weight_matrix)
    ]


def summarize_pareto(reports: list[ExperimentReport]) -> str:
    """One-line-per-candidate summary sorted by state ratio."""
    lines = [
        f"{'state/dense':>12} {'layer err':>12} {'ops/w':>8} {'decision':>10}  representation",
    ]
    for report in sorted(
        reports, key=lambda r: r.extra_metrics["state_ratio_to_dense_fp16"]
    ):
        lines.append(
            f"{report.extra_metrics['state_ratio_to_dense_fp16']:>12.4f} "
            f"{report.layer_output_relative_error:>12.4e} "
            f"{report.generator_operations_per_weight:>8.2f} "
            f"{report.decision:>10}  {report.representation}"
        )
    return "\n".join(lines)


def load_pythia_matrices() -> list[tuple[str, torch.Tensor]]:
    """Real trained projection matrices from EleutherAI/pythia-14m."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    checkpoint_path = hf_hub_download(
        "EleutherAI/pythia-14m", "model.safetensors"
    )
    state = load_file(checkpoint_path)
    tensor_names = [
        "gpt_neox.layers.0.attention.query_key_value.weight",
        "gpt_neox.layers.0.mlp.dense_4h_to_h.weight",
    ]
    return [
        (name, state[name].to(torch.float32)) for name in tensor_names
    ]


def main(command_line_arguments: list[str] | None = None) -> int:
    argument_parser = argparse.ArgumentParser(
        prog="python -m src.harness.experiment_shared_state",
        description="Experiments 3+5 — shared-state families vs. baseline.",
    )
    argument_parser.add_argument(
        "--source", choices=("synthetic", "pythia"), default="synthetic"
    )
    argument_parser.add_argument("--rows", type=int, default=64)
    argument_parser.add_argument("--cols", type=int, default=128)
    argument_parser.add_argument("--seed", type=int, default=0)
    argument_parser.add_argument(
        "--full-reports", action="store_true",
        help="Print the full required report per candidate.",
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
        reports = run_shared_state_experiment(
            weight_matrix, tensor_name, seed=parsed_arguments.seed
        )
        print(f"=== {tensor_name} {tuple(weight_matrix.shape)} ===")
        print(summarize_pareto(reports))
        if parsed_arguments.full_reports:
            for report in reports:
                print("-" * 72)
                print(report.format_report())
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
