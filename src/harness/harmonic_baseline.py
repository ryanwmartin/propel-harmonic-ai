"""Experiment 1 — harmonic derivative baseline behind the harness interface.

This wraps the Epic 01 BlockAtom in the :class:`ProceduralRepresentation`
contract with the two hot-path upgrades Experiment 1 requires:

1. **Oscillator recurrence.** The per-sample transcendental
   ``sin(2π f n + φ)`` is replaced by a two-term rotation recurrence::

       sin_state[n+1] = sin_state[n] * cos_step + cos_state[n] * sin_step
       cos_state[n+1] = cos_state[n] * cos_step - sin_state[n] * sin_step

   requiring only four multiplies and two adds per harmonic per sample, with
   transcendentals evaluated once per block to initialize the states.

   Oscillator state is carried in float64 so the rotation does not drift over
   a block; the derivative sum is cast to float32 before entering the pinned
   float32 prefix-sum accumulator, preserving the Epic 02 parity contract.
   This recurrence output is the single source of truth the Rust kernel must
   match within 1e-6; the direct per-sample f32 ``sin()`` reference decode is
   kept as a drift diagnostic only.

2. **Fused generate + MAC.** :meth:`transform` integrates each block's
   derivative estimate and multiplies against the input activation in the
   same loop, so decoded weights live only in per-block accumulator registers
   (one float32 accumulator per row) and are never written to a dense
   (rows, cols) buffer.

The representation also supports the Experiment 1 precision sweep: anchors and
sinusoid coefficients can be independently quantized to float16 storage while
computation stays in float32, and the state accounting reports the reduced bit
widths honestly.
"""

from __future__ import annotations

import torch

from src.harness.accounting import StateAccounting
from src.harness.quantization import SUPPORTED_FIELD_BITS, quantize_field
from src.harness.representation import ProceduralRepresentation
from src.phasor_atom import BlockAtom

TWO_PI: float = 2.0 * torch.pi

METADATA_BITS: int = 32 * 8
"""Serialized .atom header metadata (magic, version, geometry) in bits."""


class HarmonicDerivativeRepresentation(ProceduralRepresentation):
    """Anchor + harmonic first-difference representation of one weight matrix.

    Args:
        atom: Fitted BlockAtom parameter container.
        anchor_bits: Storage precision of block anchors (16 or 32).
        coefficient_bits: Storage precision of amplitudes/frequencies/phases
            (16 or 32).
    """

    def __init__(
        self,
        atom: BlockAtom,
        anchor_bits: int = 32,
        coefficient_bits: int = 32,
    ) -> None:
        if anchor_bits not in SUPPORTED_FIELD_BITS:
            raise ValueError(
                f"anchor_bits must be one of {sorted(SUPPORTED_FIELD_BITS)}, "
                f"got {anchor_bits}"
            )
        if coefficient_bits not in SUPPORTED_FIELD_BITS:
            raise ValueError(
                f"coefficient_bits must be one of {sorted(SUPPORTED_FIELD_BITS)}, "
                f"got {coefficient_bits}"
            )

        self.anchor_bits = anchor_bits
        self.coefficient_bits = coefficient_bits
        self.atom = BlockAtom(
            anchors=quantize_field(atom.anchors, anchor_bits),
            amplitudes=quantize_field(atom.amplitudes, coefficient_bits),
            frequencies=quantize_field(atom.frequencies, coefficient_bits),
            phases=quantize_field(atom.phases, coefficient_bits),
            block_size=atom.block_size,
            num_blocks=atom.num_blocks,
            rows=atom.rows,
            cols=atom.cols,
            column_order=atom.column_order,
        )

    # ------------------------------------------------------------------
    # ProceduralRepresentation interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return (
            "harmonic-derivative"
            f"[L={self.atom.block_size},K={self.atom.K},"
            f"anchor={self.anchor_bits}b,coeff={self.coefficient_bits}b]"
        )

    @property
    def rows(self) -> int:
        return self.atom.rows

    @property
    def cols(self) -> int:
        return self.atom.cols

    def state_accounting(self) -> StateAccounting:
        anchor_count = self.atom.rows * self.atom.num_blocks
        coefficient_count = anchor_count * self.atom.K
        field_bits: dict[str, int] = {
            "anchors": anchor_count * self.anchor_bits,
            "amplitudes": coefficient_count * self.coefficient_bits,
            "frequencies": coefficient_count * self.coefficient_bits,
            "phases": coefficient_count * self.coefficient_bits,
            "metadata": METADATA_BITS,
        }
        if self.atom.column_order is not None:
            field_bits["column_order"] = self.atom.cols * 32
        return StateAccounting(
            field_bits=field_bits,
            represented_weight_count=self.atom.rows * self.atom.cols,
        )

    def reconstruct(self) -> torch.Tensor:
        """Diagnostic dense decode via the oscillator recurrence.

        Matches the reference ``decode_full_tensor`` contract (float32
        prefix-sum accumulator) while exercising the same recurrence the fused
        hot path uses, so parity checks cover the production decode math.
        """
        reconstructed = torch.empty(
            self.atom.rows, self.atom.cols, dtype=torch.float32
        )
        for block_index in range(self.atom.num_blocks):
            block_weights = self._generate_block_weights(block_index)
            column_start = block_index * self.atom.block_size
            reconstructed[
                :, column_start : column_start + block_weights.shape[1]
            ] = block_weights

        if self.atom.column_order is None:
            return reconstructed
        original_order = torch.empty_like(reconstructed)
        original_order[:, self.atom.column_order] = reconstructed
        return original_order

    def transform(self, input_activations: torch.Tensor) -> torch.Tensor:
        """Fused hot path: ``Ŵ @ x`` without materializing Ŵ.

        Per block: initialize oscillator states from block parameters, then in
        a single pass integrate the derivative into a per-row float32 weight
        accumulator and immediately multiply-accumulate against the matching
        input activation. Decoded weights never leave registers/scratch of
        shape (rows,).
        """
        squeeze_output = input_activations.ndim == 1
        activations = (
            input_activations.unsqueeze(1) if squeeze_output else input_activations
        ).to(torch.float32)
        if activations.shape[0] != self.atom.cols:
            raise ValueError(
                f"input activations must have {self.atom.cols} rows, got "
                f"{activations.shape[0]}"
            )

        if self.atom.column_order is not None:
            activations = activations[self.atom.column_order]

        batch_size = activations.shape[1]
        output = torch.zeros(self.atom.rows, batch_size, dtype=torch.float32)

        for block_index in range(self.atom.num_blocks):
            block_length = self._block_length(block_index)
            column_start = block_index * self.atom.block_size

            # Weight accumulator: the only place decoded weights ever exist.
            weight_accumulator = self.atom.anchors[:, block_index].clone()
            output += torch.outer(
                weight_accumulator, activations[column_start]
            )

            if block_length <= 1:
                continue

            sin_state, cos_state, sin_step, cos_step = self._init_oscillators(
                block_index
            )
            amplitudes = self.atom.amplitudes[:, block_index].to(torch.float64)

            for sample_offset in range(block_length - 1):
                derivative = (amplitudes * sin_state).sum(dim=1).to(torch.float32)
                weight_accumulator = weight_accumulator + derivative
                output += torch.outer(
                    weight_accumulator, activations[column_start + 1 + sample_offset]
                )
                sin_state, cos_state = (
                    sin_state * cos_step + cos_state * sin_step,
                    cos_state * cos_step - sin_state * sin_step,
                )

        return output.squeeze(1) if squeeze_output else output

    def estimated_operations_per_weight(self) -> float:
        """Recurrence advance (6K) + derivative MAC (2K) + integrate + MAC (2)."""
        return 8.0 * self.atom.K + 2.0

    def max_transient_scratch_elements(self) -> int:
        """Oscillator states + steps (4·rows·K) + weight accumulator (rows)."""
        return self.atom.rows * (4 * self.atom.K + 1)

    def max_decoded_weight_elements(self) -> int:
        """One in-flight decoded weight per row (the block accumulator)."""
        return self.atom.rows

    # ------------------------------------------------------------------
    # Oscillator recurrence internals
    # ------------------------------------------------------------------

    def _block_length(self, block_index: int) -> int:
        column_start = block_index * self.atom.block_size
        column_end = min(column_start + self.atom.block_size, self.atom.cols)
        return column_end - column_start

    def _init_oscillators(
        self, block_index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate transcendentals once per block for all rows and harmonics.

        Returns (sin_state, cos_state, sin_step, cos_step), each of shape
        (rows, K) in float64. States start at sample_index = 0, i.e.
        sin(φ), cos(φ). Float64 state keeps the rotation recurrence
        drift-free within a block; only the summed derivative is cast to
        float32 for the pinned prefix-sum accumulator.
        """
        phases = self.atom.phases[:, block_index].to(torch.float64)
        step_angles = TWO_PI * self.atom.frequencies[:, block_index].to(
            torch.float64
        )
        return (
            torch.sin(phases),
            torch.cos(phases),
            torch.sin(step_angles),
            torch.cos(step_angles),
        )

    def _generate_block_weights(self, block_index: int) -> torch.Tensor:
        """Decode one block for all rows via the oscillator recurrence.

        Diagnostic-only helper used by :meth:`reconstruct`; the fused hot path
        never calls this because it would materialize a (rows, block_length)
        slab per block.
        """
        block_length = self._block_length(block_index)
        block_weights = torch.empty(
            self.atom.rows, block_length, dtype=torch.float32
        )
        block_weights[:, 0] = self.atom.anchors[:, block_index]

        if block_length <= 1:
            return block_weights

        sin_state, cos_state, sin_step, cos_step = self._init_oscillators(
            block_index
        )
        amplitudes = self.atom.amplitudes[:, block_index].to(torch.float64)
        weight_accumulator = block_weights[:, 0].clone()

        for sample_offset in range(block_length - 1):
            derivative = (amplitudes * sin_state).sum(dim=1).to(torch.float32)
            weight_accumulator = weight_accumulator + derivative
            block_weights[:, 1 + sample_offset] = weight_accumulator
            sin_state, cos_state = (
                sin_state * cos_step + cos_state * sin_step,
                cos_state * cos_step - sin_state * sin_step,
            )

        return block_weights
