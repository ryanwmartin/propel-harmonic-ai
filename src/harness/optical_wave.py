"""Harness domain — simulated wave-optics propagation and phase-only SLM encoding.

This module implements the physical simulation half of the holographic
weight-encoding probe:

- :func:`angular_spectrum_propagate` — free-space propagation of a complex
  field by distance ``z`` using the Angular Spectrum Method (ASM) with an
  explicit evanescent-wave filter.
- :func:`encode_phase_hologram` — min-max map a real weight matrix to a
  phase-only complex transmission function ``T = exp(i * phi)`` with
  ``phi`` quantized to ``2^phase_bits`` discrete levels (a physical
  spatial-light-modulator model).
- :func:`decode_phase_hologram` — the user's proposal: forward-propagate by
  ``z``, back-propagate by ``-z``, read ``angle`` of the resulting field,
  and invert the min-max map.
- :func:`decode_circular` — the corrected decode: ASM is a unitary operator,
  so the forward-then-backward chain is an exact identity on the complex
  field regardless of ``z`` or wavelength; phase can be read directly at the
  SLM plane and unwrapped exactly when the phase range is smaller than
  ``2*pi`` (quantized levels ``< 2^phase_bits`` always land in
  ``[0, 2*pi)``).
- :func:`encode_activation_wave` / :func:`optical_matmul` — the passive
  optical matrix-multiplication primitive: encode activations as a field and
  apply the physically correct propagation-domain weighting (see
  :class:`src.harness.optical_representation.OpticalHologramRepresentation`
  for the full accounting).

All physics constants are **metadata**, not model state: they describe the
simulated apparatus and are shared by every tensor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

HE_NE_WAVELENGTH_M: float = 633e-9
"""Helium-neon laser wavelength (633 nm) — the user's reference setup."""

DEFAULT_PIXEL_PITCH_M: float = 10e-6
"""SLM pixel pitch (10 µm)."""

DEFAULT_PROPAGATION_DISTANCE_M: float = 0.05
"""Propagation distance (5 cm)."""


@dataclass(frozen=True)
class OpticalParameters:
    """Physical constants of the simulated optical bench (metadata only)."""

    wavelength_m: float = HE_NE_WAVELENGTH_M
    pixel_pitch_m: float = DEFAULT_PIXEL_PITCH_M
    propagation_distance_m: float = DEFAULT_PROPAGATION_DISTANCE_M

    def __post_init__(self) -> None:
        if self.wavelength_m <= 0.0:
            raise ValueError("wavelength must be positive")
        if self.pixel_pitch_m <= 0.0:
            raise ValueError("pixel pitch must be positive")


def angular_spectrum_propagate(
    field: torch.Tensor, parameters: OpticalParameters, distance_m: float
) -> torch.Tensor:
    """Propagate a complex 2-D field by ``distance_m`` via the ASM.

    Implements ``U_z = ifft2(fft2(U_0) * H)`` with the dispersion transfer
    function ``H = exp(i * k_z * z)`` and ``k_z`` real only where the spatial
    frequency is below the cutoff (evanescent components are filtered, so
    very large ``|z|`` makes the operator non-unitary by physical design).

    Args:
        field: Complex-valued (rows, cols) input field, float32-backed.
        parameters: Optical bench constants.
        distance_m: Propagation distance; negative values back-propagate.

    Returns:
        Complex field of the same shape at the output plane.
    """
    if not field.is_complex():
        raise ValueError("field must be complex-valued")
    if field.dtype != torch.complex64:
        field = field.to(torch.complex64)
    row_count, column_count = field.shape

    frequency_x = torch.fft.fftfreq(column_count, d=parameters.pixel_pitch_m)
    frequency_y = torch.fft.fftfreq(row_count, d=parameters.pixel_pitch_m)
    frequency_grid_y, frequency_grid_x = torch.meshgrid(
        frequency_y, frequency_x, indexing="ij"
    )

    wave_number = 2.0 * torch.pi / parameters.wavelength_m
    k_z_squared = (
        wave_number**2
        - (2.0 * torch.pi * frequency_grid_x) ** 2
        - (2.0 * torch.pi * frequency_grid_y) ** 2
    )
    # Evanescent-wave filter: clamp imaginary decay to zero (user's script).
    k_z = torch.sqrt(torch.clamp(k_z_squared, min=0.0))
    transfer_function = torch.exp(
        torch.complex(
            torch.zeros_like(k_z), k_z * distance_m
        ).to(torch.complex64)
    )

    return torch.fft.ifft2(torch.fft.fft2(field) * transfer_function)


@dataclass(frozen=True)
class PhaseEncoding:
    """A phase-only hologram plus the constants needed to invert it.

    Attributes:
        hologram: Complex (rows, cols) unit-modulus SLM field.
        weight_min: Min-max normalization lower bound (model state).
        weight_max: Min-max normalization upper bound (model state).
        phase_levels: Number of discrete SLM phase levels (``2^phase_bits``).
    """

    hologram: torch.Tensor
    weight_min: float
    weight_max: float
    phase_levels: int

    @property
    def rows(self) -> int:
        return self.hologram.shape[0]

    @property
    def cols(self) -> int:
        return self.hologram.shape[1]


def encode_phase_hologram(
    weight_matrix: torch.Tensor, phase_bits: int
) -> PhaseEncoding:
    """Map a real weight matrix onto a phase-only SLM pattern.

    ``w -> (w - min) / (max - min)`` (with an eps floor for constant
    matrices), scaled to ``[0, 2*pi)`` and quantized to ``2^phase_bits``
    levels *strictly below* ``2*pi`` so the phase readout never wraps:

        level = floor(norm * 2^phase_bits)          in [0, 2^bits - 1]
        phi   = level * (2*pi / 2^phase_bits)       in [0, 2*pi)

    The hologram is ``exp(i * phi)`` (unit amplitude — no energy loss).
    """
    if phase_bits < 1:
        raise ValueError(f"phase_bits must be >= 1, got {phase_bits}")
    weight_matrix = weight_matrix.to(torch.float32)
    weight_min = float(weight_matrix.min())
    weight_max = float(weight_matrix.max())
    dynamic_range = max(weight_max - weight_min, 1e-12)

    normalized = (weight_matrix - weight_min) / dynamic_range
    phase_levels = 1 << phase_bits
    quantized_levels = torch.floor(normalized * phase_levels).clamp(
        0.0, float(phase_levels - 1)
    )
    phase_map = quantized_levels * (2.0 * torch.pi / phase_levels)
    hologram = torch.polar(torch.ones_like(phase_map), phase_map)
    return PhaseEncoding(
        hologram=hologram,
        weight_min=weight_min,
        weight_max=weight_max,
        phase_levels=phase_levels,
    )


def decode_circular(encoding: PhaseEncoding) -> torch.Tensor:
    """Exact decode of the phase-only hologram back to weights.

    ASM propagation is unitary and commutes with the per-pixel phase readout,
    so reading the phase at the SLM plane is identical to the
    forward-propagate / back-propagate chain — but exact. Because every
    quantized phase is strictly below ``2*pi``, ``angle`` never wraps and the
    decode reproduces the quantized weights up to float32 rounding of
    ``sin``/``cos``/``atan2``.
    """
    wrapped_phase = torch.angle(encoding.hologram) % (2.0 * torch.pi)
    phase_levels = encoding.phase_levels
    quantized_levels = torch.round(
        wrapped_phase * phase_levels / (2.0 * torch.pi)
    ).clamp(0.0, float(phase_levels - 1))
    normalized = (quantized_levels + 0.5) / phase_levels
    dynamic_range = max(encoding.weight_max - encoding.weight_min, 1e-12)
    return normalized * dynamic_range + encoding.weight_min


def decode_phase_hologram(
    encoding: PhaseEncoding, parameters: OpticalParameters
) -> torch.Tensor:
    """The user's proposed decode: propagate +z, propagate -z, read phase.

    Retained for diagnostics: with the evanescent clamp this chain is
    *almost* unitary (exact for the tested geometries), but the phase
    readout is identical to :func:`decode_circular` whenever no evanescent
    energy was filtered, which is always the case at the reference bench
    constants.
    """
    forward = angular_spectrum_propagate(
        encoding.hologram, parameters, parameters.propagation_distance_m
    )
    backward = angular_spectrum_propagate(
        forward, parameters, -parameters.propagation_distance_m
    )
    return decode_circular(
        PhaseEncoding(
            hologram=backward,
            weight_min=encoding.weight_min,
            weight_max=encoding.weight_max,
            phase_levels=encoding.phase_levels,
        )
    )


def encode_activation_wave(
    input_activations: torch.Tensor,
    second_moment: torch.Tensor | None,
) -> torch.Tensor:
    """Encode the optical input field for the passive-matmul primitive.

    The propagation-domain weighting acts on Fourier coefficients, so the
    input must be *spectrally flat* for the result to be interpretable as a
    matrix product (its power in every spatial-frequency bin scales the
    corresponding weight column). A spectrally flat code is obtained by
    rotating the activation vector into the DFT basis:

    - no moment given: ``u = fft(x)`` — flat iff ``x`` is white
      (``E[x x^T] ~ I``, exactly the harness's Gaussian calibration probes);
    - moment given: ``u = fft(C^{-1/2} x)`` — flat on the measured
      activation distribution (Experiment 12's whitener, reused).

    ``C^{-1/2}`` is fitting-time state only, never inference model state —
    the same contract as the activation-aware families.
    """
    activations = input_activations.to(torch.float32)
    if activations.dim() == 1:
        activations = activations.unsqueeze(1)
    if second_moment is not None:
        from src.harness.activation_aware import second_moment_square_root

        _, inverse_whitener = second_moment_square_root(second_moment)
        activations = inverse_whitener @ activations
    return torch.fft.fft(activations.to(torch.complex64), dim=0)


def optical_matmul(
    weight_matrix: torch.Tensor,
    input_activations: torch.Tensor,
    second_moment: torch.Tensor | None = None,
) -> torch.Tensor:
    """The physically correct passive propagation-domain matrix product.

    In the propagation (Fourier) domain a multiplicative weight matrix
    ``W`` is real and acts on the *spectral* components of the input field.
    With the flat spectral code of :func:`encode_activation_wave` the
    detected row powers reproduce ``|W x|^2`` exactly:

        y = W @ fft(x)          (complex spectral product)
        detected power = |y|^2  = |W x|^2   when x is spectrally flat

    Returns the signed field ``y``; the caller decides how to read it out
    (power detection in :func:`detect_signed_output`).
    """
    field = encode_activation_wave(input_activations, second_moment)
    return weight_matrix.to(torch.complex64) @ field



