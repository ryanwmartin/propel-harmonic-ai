"""Experiment 12 stage 1 — activation-aware closed-form fits vs. weight-space.

The Exp 3/5 measurement showed closed-form **weight-space** fitting cannot
reach the pre-distillation gate on real trained weights: requiring
``What x ~= W x`` for random probes is equivalent to requiring
``What ~= W``. Stage 1 of Experiment 12 changes the fitting target, not the
family: the same low-rank and shared-basis representations are re-fitted
against the teacher's **activation second moment** (cached from real
Pythia-14m forward passes, Epic 04 small scale), so the closed-form solve
minimizes output error where activations actually live.

Decision metric: **on-distribution** layer output relative error, measured on
held-out activation columns the fit never saw. The random-probe error is also
reported to show how much of the error mass moved off-distribution.

Gates (program-wide, pre-distillation stage-1 flavor):

- unique model state <= 50% of dense FP16;
- held-out on-distribution layer output relative error <= 1e-2;
- per-layer activation MSE < 5e-4 (the roadmap's stage-1 distillation gate)
  reported alongside;
- zero dense-weight materialization (interface-enforced).

Usage::

    python -m src.harness.experiment_12_behavioral --source synthetic
    python -m src.harness.experiment_12_behavioral --source pythia
"""

from __future__ import annotations

import argparse
import sys

import torch

from src.harness.accounting import ExperimentReport, dense_baseline_bits
from src.harness.low_rank_residual import LowRankResidualRepresentation
from src.harness.representation import ProceduralRepresentation
from src.harness.shared_basis import SharedBasisRepresentation

LAYER_OUTPUT_GATE: float = 1e-2
"""Held-out on-distribution layer-output relative error gate."""

STATE_RATIO_GATE: float = 0.5
"""Advancement gate: unique state must be <= 50% of dense FP16."""

ACTIVATION_MSE_GATE: float = 5e-4
"""Roadmap stage-1 distillation target, reported for context."""

RANK_FRACTIONS: tuple[float, ...] = (0.05, 0.10, 0.25, 0.40)
RESIDUAL_DENSITIES: tuple[float, ...] = (0.0, 0.01, 0.05)
HARMONIC_FRACTIONS: tuple[float, ...] = (0.25, 0.50, 0.75)


def evaluate_candidate(
    representation: ProceduralRepresentation,
    weight_matrix: torch.Tensor,
    evaluation_activations: torch.Tensor,
    random_probes: torch.Tensor,
    tensor_description: str,
) -> ExperimentReport:
    """Measure one fitted candidate on held-out teacher activations."""
    representation.verify_no_dense_materialization()

    dense_output = weight_matrix @ evaluation_activations
    fused_output = representation.transform(evaluation_activations)
    on_distribution_error = _relative_error(fused_output, dense_output)
    activation_mse = float(
        torch.mean((fused_output - dense_output) ** 2)
    )

    random_dense_output = weight_matrix @ random_probes
    random_fused_output = representation.transform(random_probes)
    random_probe_error = _relative_error(
        random_fused_output, random_dense_output
    )

    accounting = representation.state_accounting()
    state_ratio = accounting.ratio_to_dense_fp16()
    passes_gates = (
        state_ratio <= STATE_RATIO_GATE
        and on_distribution_error <= LAYER_OUTPUT_GATE
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
        weight_reconstruction_relative_error=_relative_error(
            representation.reconstruct(), weight_matrix
        ),
        layer_output_relative_error=on_distribution_error,
        decision="advance" if passes_gates else "retain as baseline",
        reason=(
            "meets <=50% dense state at <=1e-2 held-out on-distribution error"
            if passes_gates
            else "fails state<=50% and/or on-distribution<=1e-2 gate"
        ),
        extra_metrics={
            "state_ratio_to_dense_fp16": state_ratio,
            "state_bits_per_weight": accounting.bits_per_weight,
            "activation_mse": activation_mse,
            "random_probe_layer_error": random_probe_error,
        },
    )


def build_candidate_pairs(
    weight_matrix: torch.Tensor,
    second_moment: torch.Tensor,
    block_size: int = 64,
) -> list[ProceduralRepresentation]:
    """Fit each capacity point twice: weight-space and activation-aware."""
    row_count, column_count = weight_matrix.shape
    maximum_rank = min(row_count, column_count)
    candidates: list[ProceduralRepresentation] = []

    for rank_fraction in RANK_FRACTIONS:
        rank = max(1, int(round(rank_fraction * maximum_rank)))
        for residual_density in RESIDUAL_DENSITIES:
            for moment in (None, second_moment):
                candidates.append(
                    LowRankResidualRepresentation(
                        weight_matrix,
                        rank=rank,
                        residual_density=residual_density,
                        factor_bits=16,
                        residual_bits=16,
                        second_moment=moment,
                    )
                )

    usable_block_size = (
        block_size
        if column_count % block_size == 0
        else _maximum_usable_block(column_count)
    )
    stacked_block_count = row_count * (column_count // usable_block_size)
    maximum_svd_harmonics = min(usable_block_size, stacked_block_count)
    seen_harmonic_counts: set[int] = set()
    for harmonic_fraction in HARMONIC_FRACTIONS:
        harmonic_count = min(
            max(1, int(round(harmonic_fraction * usable_block_size))),
            maximum_svd_harmonics,
        )
        if harmonic_count in seen_harmonic_counts:
            continue
        seen_harmonic_counts.add(harmonic_count)
        for residual_density in (0.0, 0.01):
            for moment in (None, second_moment):
                candidates.append(
                    SharedBasisRepresentation(
                        weight_matrix,
                        block_size=usable_block_size,
                        harmonic_count=harmonic_count,
                        basis_mode="svd",
                        residual_density=residual_density,
                        coefficient_bits=16,
                        residual_bits=16,
                        second_moment=moment,
                    )
                )
    return candidates


def run_experiment_12(
    weight_matrix: torch.Tensor,
    second_moment: torch.Tensor,
    evaluation_activations: torch.Tensor,
    tensor_description: str,
    seed: int = 0,
) -> list[ExperimentReport]:
    """Fit and measure all candidate pairs for one projection."""
    torch.manual_seed(seed)
    weight_matrix = weight_matrix.to(torch.float32)
    _, column_count = weight_matrix.shape
    random_probes = torch.randn(column_count, 64, dtype=torch.float32)
    return [
        evaluate_candidate(
            candidate,
            weight_matrix,
            evaluation_activations,
            random_probes,
            tensor_description,
        )
        for candidate in build_candidate_pairs(weight_matrix, second_moment)
    ]


def summarize(reports: list[ExperimentReport]) -> str:
    """Per-candidate summary sorted by state ratio, then error."""
    lines = [
        f"{'state/dense':>12} {'on-dist err':>12} {'rand err':>10} "
        f"{'act MSE':>10} {'decision':>10}  representation",
    ]
    for report in sorted(
        reports,
        key=lambda r: (
            r.extra_metrics["state_ratio_to_dense_fp16"],
            r.layer_output_relative_error,
        ),
    ):
        lines.append(
            f"{report.extra_metrics['state_ratio_to_dense_fp16']:>12.4f} "
            f"{report.layer_output_relative_error:>12.4e} "
            f"{report.extra_metrics['random_probe_layer_error']:>10.3e} "
            f"{report.extra_metrics['activation_mse']:>10.3e} "
            f"{report.decision:>10}  {report.representation}"
        )
    return "\n".join(lines)


def summarize_activation_gain(reports: list[ExperimentReport]) -> str:
    """Pair up fit=wgt / fit=act variants and report the error ratio."""
    weight_fits: dict[str, ExperimentReport] = {}
    activation_fits: dict[str, ExperimentReport] = {}
    for report in reports:
        if ",fit=act]" in report.representation:
            key = report.representation.replace(",fit=act]", "]")
            activation_fits[key] = report
        elif ",fit=wgt]" in report.representation:
            key = report.representation.replace(",fit=wgt]", "]")
            weight_fits[key] = report

    lines = [
        f"{'wgt on-dist':>12} {'act on-dist':>12} {'gain':>8}  configuration",
    ]
    for key in sorted(weight_fits.keys() & activation_fits.keys()):
        weight_error = weight_fits[key].layer_output_relative_error
        activation_error = activation_fits[key].layer_output_relative_error
        gain = weight_error / max(activation_error, 1e-30)
        lines.append(
            f"{weight_error:>12.4e} {activation_error:>12.4e} "
            f"{gain:>7.2f}x  {key}"
        )
    return "\n".join(lines)


def _relative_error(computed: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        torch.linalg.norm(computed - reference)
        / torch.linalg.norm(reference).clamp_min(1e-30)
    )


def _maximum_usable_block(column_count: int) -> int:
    for candidate_size in (64, 32, 16, 8, 4, 2):
        if column_count % candidate_size == 0:
            return candidate_size
    return 1


def build_synthetic_case(
    rows: int = 64,
    cols: int = 128,
    activation_rank: int = 12,
    sample_count: int = 4096,
    noise_scale: float = 0.002,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Synthetic sanity case: full-rank weights, low-dimensional activations.

    The weight matrix is near full rank (weight-space truncation must fail)
    but activations live on an ``activation_rank``-dimensional subspace plus
    small isotropic noise — exactly the regime where activation-aware
    fitting should win decisively. ``noise_scale`` sets the irreducible
    off-subspace output error floor; the default keeps the floor below the
    1e-2 gate so the designed-to-win case can pass it.

    Returns:
        Tuple of (weight_matrix, fitting second moment, held-out evaluation
        activations).
    """
    generator = torch.Generator().manual_seed(seed)
    weight_matrix = torch.randn(rows, cols, generator=generator)

    subspace = torch.linalg.qr(
        torch.randn(cols, activation_rank, generator=generator)
    ).Q
    latent = torch.randn(sample_count, activation_rank, generator=generator)
    noise = noise_scale * torch.randn(sample_count, cols, generator=generator)
    activation_rows = latent @ subspace.T + noise

    split_point = sample_count // 2
    fitting_rows = activation_rows[:split_point].to(torch.float64)
    second_moment = (
        fitting_rows.T @ fitting_rows / fitting_rows.shape[0]
    ).to(torch.float32)
    evaluation_activations = (
        activation_rows[split_point:].T.contiguous().to(torch.float32)
    )
    return weight_matrix, second_moment, evaluation_activations


def main(command_line_arguments: list[str] | None = None) -> int:
    argument_parser = argparse.ArgumentParser(
        prog="python -m src.harness.experiment_12_behavioral",
        description=(
            "Experiment 12 stage 1 — activation-aware closed-form fitting."
        ),
    )
    argument_parser.add_argument(
        "--source", choices=("synthetic", "pythia"), default="synthetic"
    )
    argument_parser.add_argument("--seed", type=int, default=0)
    argument_parser.add_argument(
        "--full-reports", action="store_true",
        help="Print the full required report per candidate.",
    )
    parsed_arguments = argument_parser.parse_args(command_line_arguments)

    if parsed_arguments.source == "pythia":
        from src.harness.activation_cache import (
            capture_pythia_activation_statistics,
            load_projection_weight,
        )

        statistics = capture_pythia_activation_statistics(
            seed=parsed_arguments.seed
        )
        cases = [
            (
                f"pythia-14m/{projection_name} "
                f"(fit N={projection_statistics.fit_sample_count})",
                load_projection_weight(projection_name),
                projection_statistics.second_moment,
                projection_statistics.evaluation_activations,
            )
            for projection_name, projection_statistics in statistics.items()
        ]
    else:
        weight_matrix, second_moment, evaluation_activations = (
            build_synthetic_case(seed=parsed_arguments.seed)
        )
        cases = [
            (
                "synthetic/full-rank W, rank-12 activation subspace",
                weight_matrix,
                second_moment,
                evaluation_activations,
            )
        ]

    for description, weight_matrix, second_moment, evaluation_activations in cases:
        reports = run_experiment_12(
            weight_matrix,
            second_moment,
            evaluation_activations,
            description,
            seed=parsed_arguments.seed,
        )
        print(f"=== {description} {tuple(weight_matrix.shape)} ===")
        print(summarize(reports))
        print("--- activation-aware gain (same capacity, wgt vs act fit) ---")
        print(summarize_activation_gain(reports))
        if parsed_arguments.full_reports:
            for report in reports:
                print("-" * 72)
                print(report.format_report())
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
