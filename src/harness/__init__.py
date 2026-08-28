"""Common experiment harness — shared interface for procedural representations.

Every candidate representation in ``agile/procedural-inference-experiments.md``
implements :class:`ProceduralRepresentation` so that state accounting, fidelity,
and operation counts are measured identically across families.

Public API:

- :class:`ProceduralRepresentation` — the candidate interface.
- :class:`StateAccounting` — per-field model-state bit accounting.
- :class:`ExperimentReport` — the required experiment report record.
- :func:`dense_baseline_bits` — FP16 dense layer state baseline.
"""

from src.harness.accounting import (
    DENSE_BASELINE_BITS_PER_WEIGHT,
    ExperimentReport,
    StateAccounting,
    dense_baseline_bits,
)
from src.harness.harmonic_baseline import HarmonicDerivativeRepresentation
from src.harness.representation import ProceduralRepresentation

__all__ = [
    "DENSE_BASELINE_BITS_PER_WEIGHT",
    "ExperimentReport",
    "HarmonicDerivativeRepresentation",
    "ProceduralRepresentation",
    "StateAccounting",
    "dense_baseline_bits",
]
