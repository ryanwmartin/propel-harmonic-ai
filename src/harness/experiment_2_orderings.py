"""Experiment 2 — reversible ordering and transform search runner.

For each canonicalization (column ordering strategy x optional row ordering,
sign flips, diagonal scales) this runner refits the shared-state families of
Experiments 3/5 on the canonicalized matrix and sweeps their capacity
(low-rank ``rank``; shared-basis ``K``). The decision metric is the roadmap's
Experiment 2 gate, evaluated per family:

- **Coefficient reduction:** the minimum state bits a transformed candidate
  needs to match (or beat) the *best* layer-output error the identity
  candidate achieves at each capacity point; advancement needs >= 2x.
- **Permutation budget:** all canonicalization bytes must stay under 25% of
  dense FP16 bytes.
- **Exact invertibility:** verified for every fitted canonicalization.

Usage::

    python -m src.harness.experiment_2_orderings --source synthetic
    python -m src.harness.experiment_2_orderings --source pythia
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Callable

import torch

from src.harness.accounting import ExperimentReport, dense_baseline_bits
from src.harness.experiment_shared_state import evaluate_candidate
from src.harness.low_rank_residual import LowRankResidualRepresentation
from src.harness.orderings import (
    ORDERING_STRATEGIES,
    MatrixCanonicalization,
    build_canonicalization,
)
from src.harness.representation import ProceduralRepresentation
from src.harness.shared_basis import SharedBasisRepresentation
from src.harness.transformed_representation import TransformedRepresentation

COEFFICIENT_REDUCTION_GATE: float = 2.0
"""Roadmap gate: >= 2x fewer state bits at matched layer-output error."""

PERMUTATION_BUDGET_RATIO: float = 0.25
"""Roadmap gate: canonicalization bytes must be < 25% of dense FP16 bytes."""

INVERTIBILITY_TOLERANCE: float = 0.0
"""Permutations and sign flips must invert exactly; scales are exact divides
of FP16-representable values, so restore_matrix must reproduce the source
bit-for-bit up to f32 multiply/divide round-trip (checked with allclose)."""


@dataclass(frozen=True)
class CanonicalizationSpec:
    """One named canonicalization configuration to evaluate."""

    label: str
    column_strategy: str
    order_rows: bool
    use_signs: bool
    use_scales: bool

    def fit(self, weight_matrix: torch.Tensor) -> MatrixCanonicalization:
        return build_canonicalization(
            weight_matrix,
            column_strategy=self.column_strategy,
            order_rows=self.order_rows,
            use_signs=self.use_signs,
            use_scales=self.use_scales,
        )


def default_canonicalization_grid() -> list[CanonicalizationSpec]:
    """The Experiment 2 sweep grid: orderings x signs x scales."""
    grid = [CanonicalizationSpec("identity", "identity", False, False, False)]
    for strategy in ("col-norm", "spectral", "greedy-nn"):
        grid.append(
            CanonicalizationSpec(strategy, strategy, False, False, False)
        )
        grid.append(
            CanonicalizationSpec(
                f"{strategy}+rows", strategy, True, False, False
            )
        )
        grid.append(
            CanonicalizationSpec(
                f"{strategy}+sign+scale", strategy, False, True, True
            )
        )
        grid.append(
            CanonicalizationSpec(
                f"{strategy}+rows+sign+scale", strategy, True, True, True
            )
        )
    grid.append(
        CanonicalizationSpec("sign+scale", "identity", False, True, True)
    )
    return grid


def verify_exact_inverse(
    canonicalization: MatrixCanonicalization, weight_matrix: torch.Tensor
) -> None:
    """Raise if canonicalize -> restore does not reproduce the source."""
    round_trip = canonicalization.restore_matrix(
        canonicalization.apply_to_matrix(weight_matrix)
    )
    if not torch.allclose(round_trip, weight_matrix, rtol=1e-6, atol=1e-7):
        maximum_error = float((round_trip - weight_matrix).abs().max())
        raise ValueError(
            "canonicalization round trip is not exact: max abs error "
            f"{maximum_error:.3e}"
        )


def inner_capacity_sweep(
    weight_matrix: torch.Tensor,
) -> dict[str, list[tuple[int, Callable[[torch.Tensor], ProceduralRepresentation]]]]:
    """Capacity ladders per inner family: (capacity, factory) pairs.

    Capacity is the family's swept knob (rank or harmonic count K); factories
    fit the family on whatever (canonicalized) matrix they are given.
    """
    row_count, column_count = weight_matrix.shape
    maximum_rank = min(row_count, column_count)

    def low_rank_factory(
        rank: int,
    ) -> Callable[[torch.Tensor], ProceduralRepresentation]:
        return lambda matrix: LowRankResidualRepresentation(
            matrix, rank=rank, residual_density=0.0, factor_bits=16
        )

    block_size = usable_block_size(column_count)

    def shared_basis_factory(
        harmonic_count: int,
    ) -> Callable[[torch.Tensor], ProceduralRepresentation]:
        return lambda matrix: SharedBasisRepresentation(
            matrix,
            block_size=block_size,
            harmonic_count=harmonic_count,
            basis_mode="svd",
            residual_density=0.0,
            coefficient_bits=16,
        )

    rank_ladder = sorted(
        {
            max(1, int(round(fraction * maximum_rank)))
            for fraction in (0.05, 0.10, 0.20, 0.35, 0.50, 0.75)
        }
    )
    # An SVD basis of the stacked (rows * num_blocks, block_size) matrix has
    # at most min(rows * num_blocks, block_size) singular vectors.
    maximum_harmonics = min(
        block_size, row_count * (column_count // block_size)
    )
    harmonic_ladder = sorted(
        {
            min(
                maximum_harmonics,
                max(1, int(round(fraction * block_size))),
            )
            for fraction in (0.125, 0.25, 0.375, 0.50, 0.75)
        }
    )
    return {
        "low-rank": [(rank, low_rank_factory(rank)) for rank in rank_ladder],
        "shared-svd-basis": [
            (harmonic_count, shared_basis_factory(harmonic_count))
            for harmonic_count in harmonic_ladder
        ],
    }


def usable_block_size(column_count: int) -> int:
    """Largest power-of-two block size <= 64 dividing column_count."""
    for candidate_size in (64, 32, 16, 8, 4, 2):
        if column_count % candidate_size == 0:
            return candidate_size
    return 1


@dataclass(frozen=True)
class SweepPoint:
    """One measured (canonicalization, family, capacity) configuration."""

    canonicalization_label: str
    family: str
    capacity: int
    state_bits: int
    canonicalization_bits: int
    layer_output_error: float
    report: ExperimentReport


def run_ordering_experiment(
    weight_matrix: torch.Tensor,
    tensor_description: str,
    calibration_batch: int = 8,
    seed: int = 0,
    grid: list[CanonicalizationSpec] | None = None,
) -> list[SweepPoint]:
    """Measure every (canonicalization x family x capacity) configuration."""
    torch.manual_seed(seed)
    weight_matrix = weight_matrix.to(torch.float32)
    _, column_count = weight_matrix.shape
    calibration_activations = torch.randn(
        column_count, calibration_batch, dtype=torch.float32
    )

    sweep_points: list[SweepPoint] = []
    families = inner_capacity_sweep(weight_matrix)
    for spec in grid if grid is not None else default_canonicalization_grid():
        canonicalization = spec.fit(weight_matrix)
        verify_exact_inverse(canonicalization, weight_matrix)
        canonicalization_bits = sum(canonicalization.state_bits().values())
        for family_name, ladder in families.items():
            for capacity, factory in ladder:
                candidate = TransformedRepresentation(
                    weight_matrix, canonicalization, factory
                )
                report = evaluate_candidate(
                    candidate,
                    weight_matrix,
                    calibration_activations,
                    tensor_description,
                )
                sweep_points.append(
                    SweepPoint(
                        canonicalization_label=spec.label,
                        family=family_name,
                        capacity=capacity,
                        state_bits=candidate.state_accounting().total_bits,
                        canonicalization_bits=canonicalization_bits,
                        layer_output_error=report.layer_output_relative_error,
                        report=report,
                    )
                )
    return sweep_points


@dataclass(frozen=True)
class GateResult:
    """Experiment 2 gate evaluation for one (canonicalization, family) pair."""

    canonicalization_label: str
    family: str
    coefficient_reduction: float
    permutation_ratio: float
    passes_reduction_gate: bool
    passes_permutation_budget: bool
    matched_error: float
    identity_bits: int
    transformed_bits: int


def evaluate_gates(
    sweep_points: list[SweepPoint], dense_bits: int
) -> list[GateResult]:
    """Score every non-identity canonicalization against the identity ladder.

    For each family, the identity ladder defines (state_bits, error) targets.
    A transformed configuration matches a target when its layer-output error
    is <= the identity error; the coefficient reduction is the best ratio
    ``identity_bits / transformed_bits`` over all matched targets.
    """
    results: list[GateResult] = []
    families = sorted({point.family for point in sweep_points})
    labels = sorted(
        {
            point.canonicalization_label
            for point in sweep_points
            if point.canonicalization_label != "identity"
        }
    )
    for family in families:
        identity_ladder = sorted(
            (
                point
                for point in sweep_points
                if point.family == family
                and point.canonicalization_label == "identity"
            ),
            key=lambda point: point.state_bits,
        )
        for label in labels:
            transformed_ladder = [
                point
                for point in sweep_points
                if point.family == family
                and point.canonicalization_label == label
            ]
            if not identity_ladder or not transformed_ladder:
                continue
            best_reduction = 0.0
            best_error = float("inf")
            best_identity_bits = 0
            best_transformed_bits = 0
            for identity_point in identity_ladder:
                matching = [
                    point
                    for point in transformed_ladder
                    if point.layer_output_error <= identity_point.layer_output_error
                ]
                if not matching:
                    continue
                cheapest = min(matching, key=lambda point: point.state_bits)
                reduction = identity_point.state_bits / cheapest.state_bits
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_error = identity_point.layer_output_error
                    best_identity_bits = identity_point.state_bits
                    best_transformed_bits = cheapest.state_bits
            permutation_ratio = (
                transformed_ladder[0].canonicalization_bits / dense_bits
            )
            results.append(
                GateResult(
                    canonicalization_label=label,
                    family=family,
                    coefficient_reduction=best_reduction,
                    permutation_ratio=permutation_ratio,
                    passes_reduction_gate=(
                        best_reduction >= COEFFICIENT_REDUCTION_GATE
                    ),
                    passes_permutation_budget=(
                        permutation_ratio <= PERMUTATION_BUDGET_RATIO
                    ),
                    matched_error=best_error,
                    identity_bits=best_identity_bits,
                    transformed_bits=best_transformed_bits,
                )
            )
    return results


def summarize_sweep(sweep_points: list[SweepPoint]) -> str:
    """Per-configuration table sorted by family, canonicalization, capacity."""
    lines = [
        f"{'family':>17} {'canonicalization':>22} {'cap':>4} "
        f"{'state/dense':>12} {'layer err':>12}"
    ]
    for point in sorted(
        sweep_points,
        key=lambda p: (p.family, p.canonicalization_label, p.capacity),
    ):
        state_ratio = point.report.extra_metrics["state_ratio_to_dense_fp16"]
        lines.append(
            f"{point.family:>17} {point.canonicalization_label:>22} "
            f"{point.capacity:>4} {state_ratio:>12.4f} "
            f"{point.layer_output_error:>12.4e}"
        )
    return "\n".join(lines)


def summarize_gates(gate_results: list[GateResult]) -> str:
    """Gate table: coefficient reduction and permutation budget per pair."""
    lines = [
        f"{'family':>17} {'canonicalization':>22} {'coef. reduction':>16} "
        f"{'perm/dense':>11} {'gate':>6}"
    ]
    for result in sorted(
        gate_results,
        key=lambda r: (r.family, -r.coefficient_reduction),
    ):
        verdict = (
            "PASS"
            if result.passes_reduction_gate and result.passes_permutation_budget
            else "fail"
        )
        reduction_text = (
            f"{result.coefficient_reduction:.2f}x"
            if result.coefficient_reduction > 0.0
            else "no match"
        )
        lines.append(
            f"{result.family:>17} {result.canonicalization_label:>22} "
            f"{reduction_text:>16} {result.permutation_ratio:>11.4f} "
            f"{verdict:>6}"
        )
    return "\n".join(lines)


def main(command_line_arguments: list[str] | None = None) -> int:
    argument_parser = argparse.ArgumentParser(
        prog="python -m src.harness.experiment_2_orderings",
        description="Experiment 2 — reversible ordering/transform search.",
    )
    argument_parser.add_argument(
        "--source", choices=("synthetic", "pythia"), default="synthetic"
    )
    argument_parser.add_argument("--rows", type=int, default=64)
    argument_parser.add_argument("--cols", type=int, default=128)
    argument_parser.add_argument("--seed", type=int, default=0)
    argument_parser.add_argument(
        "--full-sweep", action="store_true",
        help="Print every measured configuration, not only the gate table.",
    )
    parsed_arguments = argument_parser.parse_args(command_line_arguments)

    if parsed_arguments.source == "pythia":
        from src.harness.experiment_shared_state import load_pythia_matrices

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
        row_count, column_count = weight_matrix.shape
        dense_bits = dense_baseline_bits(row_count, column_count)
        sweep_points = run_ordering_experiment(
            weight_matrix, tensor_name, seed=parsed_arguments.seed
        )
        gate_results = evaluate_gates(sweep_points, dense_bits)
        print(f"=== {tensor_name} {tuple(weight_matrix.shape)} ===")
        if parsed_arguments.full_sweep:
            print(summarize_sweep(sweep_points))
            print()
        print(summarize_gates(gate_results))
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
