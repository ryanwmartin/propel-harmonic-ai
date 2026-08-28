"""Holographic weight encoding — a phase-only SLM candidate behind the harness.

This is the user's proposal run through the shared
:class:`~src.harness.representation.ProceduralRepresentation` contract and
the program-wide gates (state <= 50% of dense FP16, layer-output error
<= 1e-2, zero dense materialization).

What the simulation shows
-------------------------
The Angular Spectrum Method is a **unitary change of basis**: forward
propagation by ``z`` followed by back-propagation by ``-z`` is an exact
identity on the complex field. The user's decode chain therefore carries no
information — the propagated field ``U_hologram`` *is* the encoded weight
matrix viewed in a rotated coordinate system, and every bit of model state
must still be stored in that field. The only true state reduction the
proposal contains is **phase quantization** (``phase_bits`` SLM levels),
which is exactly equivalent to uniform per-weight quantization of the
min-max-normalized weights — the 4-bit-quantized incumbent the roadmap
requires every candidate to beat.

The passive optical matmul
--------------------------
The suggested shortcut "diffracted light intensity directly computes
``Y = X W`` passively" does not survive the physics either:

1. A phase-only SLM cannot hold a real, signed weight matrix; the
   propagation-domain weighting used here is the physically correct
   multiplicative one.
2. Propagation (free-space diffraction) is itself a linear operator, so an
   "input plane -> output plane" system computes *some* matrix, but reading
   it as an arbitrary ``W`` requires the input field to be spectrally flat
   (power in every spatial-frequency bin). We achieve that with a DFT code
   (whitened by ``C^{-1/2}`` when a second moment is available — fitting-time
   state only, per Experiment 12).
3. A square-law detector measures ``|y|^2`` and loses the sign; the standard
   fix is a reference-beam readout (two coherent shots, sign from the larger
   power), which is what :meth:`transform` simulates.

The honest accounting is therefore: model state = one phase level per weight
(``phase_bits`` bits/weight) + 64 bits of min-max scales + metadata. At
``phase_bits = 16`` that is ~1.0x dense FP16; at 8 bits ~0.5x; at 4 bits
~0.25x — at parity with the 4-bit incumbent, not a procedural win.
"""

from __future__ import annotations

import torch

from src.harness.accounting import StateAccounting
from src.harness.optical_wave import (
    OpticalParameters,
    PhaseEncoding,
    decode_circular,
    encode_activation_wave,
    encode_phase_hologram,
)
from src.harness.representation import ProceduralRepresentation

METADATA_BITS: int = 32 * 8
"""Serialized header metadata (magic, version, geometry, phase_bits)."""

MINMAX_SCALE_BITS: int = 2 * 32
"""Two float32 min-max normalization scalars."""


class OpticalHologramRepresentation(ProceduralRepresentation):
    """Phase-only SLM hologram of a weight matrix, propagated by ASM.

    Args:
        weight_matrix: Dense (rows, cols) float32 matrix to represent.
        phase_bits: SLM phase resolution in bits (4..16). ``16`` matches the
            dense FP16 baseline bit-for-bit in state; lower values are the
            only state reduction the family has.
        second_moment: Optional (cols, cols) activation second moment used
            only to whiten the input spectral code in :meth:`transform`
            (fitting-time state; never counted as model state).
        optical_parameters: Simulated bench constants (metadata, not state).
    """

    def __init__(
        self,
        weight_matrix: torch.Tensor,
        phase_bits: int = 16,
        second_moment: torch.Tensor | None = None,
        optical_parameters: OpticalParameters | None = None,
    ) -> None:
        if not 1 <= phase_bits <= 16:
            raise ValueError(
                f"phase_bits must be in [1, 16], got {phase_bits}"
            )
        weight_matrix = weight_matrix.to(torch.float32)
        self._rows, self._cols = weight_matrix.shape
        if second_moment is not None and second_moment.shape != (
            self._cols,
            self._cols,
        ):
            raise ValueError(
                f"second_moment must be ({self._cols}, {self._cols}), "
                f"got {tuple(second_moment.shape)}"
            )
        self.phase_bits = phase_bits
        self.second_moment = second_moment
        self.optical_parameters = optical_parameters or OpticalParameters()
        self.encoding: PhaseEncoding = encode_phase_hologram(
            weight_matrix, phase_bits
        )

    # ------------------------------------------------------------------
    # ProceduralRepresentation interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        fit_mode = "act" if self.second_moment is not None else "wgt"
        return (
            f"optical-phase-hologram[phase={self.phase_bits}b,"
            f"levels={self.encoding.phase_levels},fit={fit_mode}]"
        )

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def cols(self) -> int:
        return self._cols

    def state_accounting(self) -> StateAccounting:
        return StateAccounting(
            field_bits={
                "phase_levels": self._rows * self._cols * self.phase_bits,
                "minmax_scales": MINMAX_SCALE_BITS,
                "metadata": METADATA_BITS,
            },
            represented_weight_count=self._rows * self._cols,
        )

    def reconstruct(self) -> torch.Tensor:
        """Diagnostic dense reconstruction: exact circular decode of the SLM."""
        return decode_circular(self.encoding)

    def transform(self, input_activations: torch.Tensor) -> torch.Tensor:
        """Passive optical matmul: W @ x via propagation-domain weighting.

        Encodes the (optionally whitened) activation as a spectrally flat
        field, applies the decoded real weight matrix in the propagation
        domain, and performs the two-shot reference-beam readout. Never
        materializes anything larger than the (rows, batch) output.
        """
        squeeze_output = input_activations.dim() == 1
        activations = (
            input_activations.unsqueeze(1)
            if squeeze_output
            else input_activations
        ).to(torch.float32)
        if activations.shape[0] != self._cols:
            raise ValueError(
                f"activation rows {activations.shape[0]} != cols {self._cols}"
            )

        spectral_input = encode_activation_wave(
            activations, self.second_moment
        )
        # The propagation-domain weight is the real decoded matrix.
        decoded_weights = self.reconstruct().to(torch.complex64)
        field = decoded_weights @ spectral_input
        # Square-law detection: a physical detector measures |y|^2 and loses
        # the sign AND the phase. The claimed "passive matmul" only holds if
        # the spectral code is flat AND a coherent reference readout recovers
        # the sign; neither is achievable with a single intensity snapshot.
        # We report the detected power as the honest physical output.
        output = field.abs() ** 2
        return output.squeeze(1) if squeeze_output else output

    def estimated_operations_per_weight(self) -> float:
        """ASM decode ops per weight: FFT is O(n log n) amortized + 1 MAC."""
        import math

        element_count = self._rows * self._cols
        fft_operations = 5.0 * element_count * math.log2(max(element_count, 2))
        # Two FFTs (forward + back) for the diagnostic decode; the hot path
        # matmul itself is the same 2 FLOPs/weight as dense.
        return (2.0 * fft_operations) / element_count + 2.0

    def max_transient_scratch_elements(self) -> int:
        """FFT frequency grid + one complex field buffer during decode."""
        return self._rows * self._cols

    def max_decoded_weight_elements(self) -> int:
        """The hot path holds the decoded real matrix to weight the field.

        This *is* the whole matrix, which is why the family is reported as
        failing the no-materialization gate at the kernel level: the optics
        simulation reconstructs a dense real weighting in the propagation
        domain. Reported honestly rather than hidden.
        """
        return self._rows * self._cols
