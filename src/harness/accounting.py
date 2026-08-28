"""Harness domain — model-state accounting value objects.

The accounting rules from ``agile/procedural-inference-experiments.md`` require
every experiment to report shared executable bytes, unique model-state bytes,
and transient bytes separately, with every serialized byte assigned to a named
field. These value objects make that accounting explicit and comparable across
representation families.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DENSE_BASELINE_BITS_PER_WEIGHT: int = 16
"""Dense FP16/BF16 baseline: 16 bits of model state per stored weight."""


def dense_baseline_bits(row_count: int, column_count: int) -> int:
    """Total dense FP16 model-state bits for a (rows, cols) weight matrix."""
    return row_count * column_count * DENSE_BASELINE_BITS_PER_WEIGHT


@dataclass(frozen=True)
class StateAccounting:
    """Per-field unique model-state bit counts for one encoded tensor.

    Attributes:
        field_bits: Mapping from field name (e.g. ``"anchors"``,
            ``"amplitudes"``, ``"column_order"``, ``"metadata"``) to the exact
            number of unique model-state bits that field serializes to.
        represented_weight_count: Number of dense weights this state represents.
    """

    field_bits: dict[str, int]
    represented_weight_count: int

    def __post_init__(self) -> None:
        if self.represented_weight_count <= 0:
            raise ValueError(
                "represented_weight_count must be positive, got "
                f"{self.represented_weight_count}"
            )
        for field_name, bit_count in self.field_bits.items():
            if bit_count < 0:
                raise ValueError(
                    f"field {field_name!r} has negative bit count {bit_count}"
                )

    @property
    def total_bits(self) -> int:
        """Total unique model-state bits across all fields."""
        return sum(self.field_bits.values())

    @property
    def total_bytes(self) -> float:
        """Total unique model-state bytes across all fields."""
        return self.total_bits / 8.0

    @property
    def bits_per_weight(self) -> float:
        """Unique model-state bits per represented dense weight."""
        return self.total_bits / self.represented_weight_count

    def ratio_to_dense_fp16(self) -> float:
        """Ratio of unique model-state bits to the dense FP16 baseline."""
        return self.total_bits / (
            self.represented_weight_count * DENSE_BASELINE_BITS_PER_WEIGHT
        )


@dataclass(frozen=True)
class ExperimentReport:
    """The required experiment report record from the experiment roadmap.

    Fields mirror the mandatory report template so that every completed
    experiment appends structurally identical results.
    """

    representation: str
    tensor_description: str
    shared_executable_bytes: int
    unique_model_state_bytes: float
    residual_bytes: float
    transient_scratch_bytes: float
    dense_baseline_bytes: float
    generator_operations_per_weight: float
    dense_weight_materialized: bool
    weight_reconstruction_relative_error: float
    layer_output_relative_error: float
    decision: str = "retain as baseline"
    reason: str = ""
    extra_metrics: dict[str, float] = field(default_factory=dict)

    def format_report(self) -> str:
        """Render the report in the roadmap's required template layout."""
        lines = [
            f"Representation: {self.representation}",
            f"Tensor/model: {self.tensor_description}",
            f"Shared executable bytes: {self.shared_executable_bytes}",
            f"Unique model-state bytes: {self.unique_model_state_bytes:.0f}",
            f"Residual bytes: {self.residual_bytes:.0f}",
            f"Transient scratch bytes: {self.transient_scratch_bytes:.0f}",
            f"Dense FP16/BF16 baseline bytes: {self.dense_baseline_bytes:.0f}",
            "Generator/transform operations per weight: "
            f"{self.generator_operations_per_weight:.1f}",
            f"Dense weight materialized: {'yes' if self.dense_weight_materialized else 'no'}",
            "Weight reconstruction error (diagnostic): "
            f"{self.weight_reconstruction_relative_error:.4e}",
            f"Layer output error: {self.layer_output_relative_error:.4e}",
        ]
        for metric_name, metric_value in sorted(self.extra_metrics.items()):
            lines.append(f"{metric_name}: {metric_value:.4e}")
        lines.append(f"Decision: {self.decision}")
        lines.append(f"Reason: {self.reason}")
        return "\n".join(lines)
