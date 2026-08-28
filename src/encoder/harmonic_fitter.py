"""Encoder domain — 1-D per-block harmonic fitting.

Fits K sinusoids G(x) = Σ_k A_k · sin(2π f_k x + φ_k) to the first-difference
signal of an individual row block using:
1. FFT warm-start spectral peak detection.
2. Optional Adam gradient refinement on the 1-D block parameters.
"""

from __future__ import annotations

import math

import torch

from src.encoder.config import EncoderConfig


def fit_harmonics_to_block(
    difference_signal: torch.Tensor, configuration: EncoderConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit sinusoids to a 1-D block first-difference signal.

    Args:
        difference_signal: 1-D float tensor containing differences for one block.
        configuration: EncoderConfig holding harmonic_count and refinement_steps.

    Returns:
        Tuple of (amplitudes, frequencies, phases), each a 1-D float32 tensor of shape (K,).
    """
    difference_signal = difference_signal.to(torch.float32).flatten()

    if difference_signal.numel() == 0:
        return _generate_zero_harmonics(configuration.harmonic_count)

    initial_amplitudes, initial_frequencies, initial_phases = _fft_warm_start(
        difference_signal, configuration.harmonic_count
    )

    if configuration.refinement_steps > 0:
        return _refine_parameters_with_adam(
            difference_signal,
            initial_amplitudes,
            initial_frequencies,
            initial_phases,
            configuration,
        )

    return initial_amplitudes, initial_frequencies, initial_phases


def _generate_zero_harmonics(
    harmonic_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return zero-initialized parameter tensors for zero-length or degenerate signals."""
    return (
        torch.zeros(harmonic_count, dtype=torch.float32),
        torch.zeros(harmonic_count, dtype=torch.float32),
        torch.zeros(harmonic_count, dtype=torch.float32),
    )


def _compute_bin_amplitude_scales(
    signal_length: int, half_spectrum_length: int
) -> torch.Tensor:
    """Return per-bin FFT-magnitude-to-sinusoid-amplitude scale factors.

    Interior bins map a real sinusoid of amplitude A to magnitude A·N/2, so the
    scale is 2/N. The DC bin (and the Nyquist bin for even N) carry the full
    magnitude A·N, so their scale is 1/N.
    """
    amplitude_scales = torch.full(
        (half_spectrum_length,), 2.0 / signal_length, dtype=torch.float32
    )
    amplitude_scales[0] = 1.0 / signal_length
    if signal_length % 2 == 0 and half_spectrum_length == signal_length // 2 + 1:
        amplitude_scales[-1] = 1.0 / signal_length
    return amplitude_scales


def _fft_warm_start(
    difference_signal: torch.Tensor, harmonic_count: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate initial sinusoid parameters from FFT spectral peaks.

    The DC bin participates in peak selection: a constant offset in the
    first-difference signal is a *linear trend* in the weights, which the anchor
    alone cannot absorb. DC is exactly representable in the harmonic format as
    A·sin(2π·0·x + π/2) = A, so it competes with the AC bins on equal footing.
    """
    signal_length = difference_signal.shape[0]
    frequency_spectrum = torch.fft.fft(difference_signal)

    half_spectrum_length = signal_length // 2 + 1
    magnitude_spectrum = frequency_spectrum[:half_spectrum_length].abs()
    amplitude_scales = _compute_bin_amplitude_scales(
        signal_length, half_spectrum_length
    )
    sinusoid_amplitudes_per_bin = magnitude_spectrum * amplitude_scales

    peak_selection_count = min(harmonic_count, half_spectrum_length)
    top_spectral_peaks = torch.topk(sinusoid_amplitudes_per_bin, peak_selection_count)

    initial_amplitudes = torch.zeros(harmonic_count, dtype=torch.float32)
    initial_frequencies = torch.zeros(harmonic_count, dtype=torch.float32)
    initial_phases = torch.zeros(harmonic_count, dtype=torch.float32)

    for harmonic_index, (peak_amplitude, bin_index_tensor) in enumerate(
        zip(top_spectral_peaks.values, top_spectral_peaks.indices)
    ):
        bin_index = int(bin_index_tensor.item())

        # Frequency in cycles per sample within the block
        initial_frequencies[harmonic_index] = bin_index / signal_length

        # Sinusoid amplitude with per-bin DC/Nyquist-aware scaling
        initial_amplitudes[harmonic_index] = float(peak_amplitude.item())

        # Phase relationship: φ = angle(X[f]) + π/2.
        # For DC (angle 0 or π) this yields sin(φ) = ±1, reproducing the sign.
        bin_complex_value = frequency_spectrum[bin_index]
        initial_phases[harmonic_index] = (
            torch.angle(bin_complex_value) + math.pi / 2.0
        )

    return initial_amplitudes, initial_frequencies, initial_phases


def _refine_parameters_with_adam(
    difference_signal: torch.Tensor,
    initial_amplitudes: torch.Tensor,
    initial_frequencies: torch.Tensor,
    initial_phases: torch.Tensor,
    configuration: EncoderConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Refine initial harmonic estimates via Adam optimization on this 1-D block.

    Tracks the lowest-loss parameters observed across all steps — including the
    FFT warm start itself — so refinement can never return a worse fit than the
    initialization. This matters when K covers all spectral bins: the warm start
    is then an exact DFT reconstruction and gradient steps would only perturb it.
    """
    signal_length = difference_signal.shape[0]
    sample_indices = torch.arange(signal_length, dtype=torch.float32)

    def _evaluate_loss(
        amplitude_values: torch.Tensor,
        frequency_values: torch.Tensor,
        phase_values: torch.Tensor,
    ) -> torch.Tensor:
        phase_arguments = (
            2.0 * math.pi * frequency_values.unsqueeze(0) * sample_indices.unsqueeze(1)
            + phase_values.unsqueeze(0)
        )
        predicted_signal = (
            amplitude_values.unsqueeze(0) * torch.sin(phase_arguments)
        ).sum(dim=1)
        return torch.nn.functional.mse_loss(predicted_signal, difference_signal)

    amplitudes = initial_amplitudes.clone().requires_grad_(True)
    frequencies = initial_frequencies.clone().requires_grad_(True)
    phases = initial_phases.clone().requires_grad_(True)

    optimizer = torch.optim.Adam(
        [amplitudes, frequencies, phases], lr=configuration.learning_rate
    )

    with torch.no_grad():
        best_loss = float(
            _evaluate_loss(initial_amplitudes, initial_frequencies, initial_phases)
        )
    best_parameters = (
        initial_amplitudes.clone(),
        initial_frequencies.clone(),
        initial_phases.clone(),
    )

    for _step in range(configuration.refinement_steps):
        optimizer.zero_grad()
        loss_value = _evaluate_loss(amplitudes, frequencies, phases)
        loss_value.backward()
        optimizer.step()

        with torch.no_grad():
            current_loss = float(_evaluate_loss(amplitudes, frequencies, phases))
        if current_loss < best_loss:
            best_loss = current_loss
            best_parameters = (
                amplitudes.detach().clone(),
                frequencies.detach().clone(),
                phases.detach().clone(),
            )

    return best_parameters
