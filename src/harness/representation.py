"""Harness domain — the shared candidate-representation interface.

Every procedural family (harmonic baseline, shared bases, low-rank + residual,
butterfly layers, coordinate generators, ...) implements this interface so the
benchmark harness can compare them under identical accounting and fidelity
measurement, per ``agile/procedural-inference-experiments.md``.

The two execution paths are deliberately separate:

- :meth:`reconstruct` is a **diagnostic** that may materialize a dense tensor
  for research comparison. It never counts as procedural inference.
- :meth:`transform` is the **hot path** contract: apply the represented matrix
  to activations without allocating a dense (rows, cols) weight buffer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from src.harness.accounting import StateAccounting


class ProceduralRepresentation(ABC):
    """A fitted procedural representation of one 2-D weight matrix."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short unique identifier of the representation family."""

    @property
    @abstractmethod
    def rows(self) -> int:
        """Row count of the represented dense matrix."""

    @property
    @abstractmethod
    def cols(self) -> int:
        """Column count of the represented dense matrix."""

    @abstractmethod
    def state_accounting(self) -> StateAccounting:
        """Exact per-field unique model-state bit accounting."""

    @abstractmethod
    def reconstruct(self) -> torch.Tensor:
        """Diagnostic dense reconstruction Ŵ of shape (rows, cols).

        Allowed to allocate a dense tensor; used only for fidelity diagnostics,
        never in the measured hot path.
        """

    @abstractmethod
    def transform(self, input_activations: torch.Tensor) -> torch.Tensor:
        """Fused hot path: compute ``Ŵ @ input_activations`` procedurally.

        Args:
            input_activations: 1-D tensor of shape (cols,) or 2-D tensor of
                shape (cols, batch).

        Returns:
            Tensor of shape (rows,) or (rows, batch).

        Implementations must not allocate a dense (rows, cols) weight buffer.
        Per-block or per-tile scratch bounded well below rows*cols is allowed
        and must be reported by :meth:`max_transient_scratch_elements`.
        """

    @abstractmethod
    def estimated_operations_per_weight(self) -> float:
        """Estimated generator + MAC arithmetic operations per generated weight."""

    @abstractmethod
    def max_transient_scratch_elements(self) -> int:
        """Upper bound on all transient scratch elements allocated by transform().

        Includes generator state (oscillators, recurrent state, basis values)
        plus decoded-weight accumulators. Reported as transient bytes in the
        experiment accounting.
        """

    @abstractmethod
    def max_decoded_weight_elements(self) -> int:
        """Upper bound on decoded-weight values in flight during transform().

        This counts only buffer elements holding reconstructed dense-weight
        values (not generator state such as oscillator registers). The
        no-materialization gate compares this bound to the dense element
        count, because a decoded-weight buffer approaching rows*cols is dense
        materialization in disguise.
        """

    def verify_no_dense_materialization(self) -> None:
        """Raise if in-flight decoded weights amount to a dense matrix."""
        dense_element_count = self.rows * self.cols
        decoded_element_count = self.max_decoded_weight_elements()
        if decoded_element_count >= dense_element_count:
            raise ValueError(
                f"{self.name}: decoded-weight buffer ({decoded_element_count} "
                f"elements) is not smaller than the dense matrix "
                f"({dense_element_count} elements); this is dense "
                "materialization in disguise"
            )
