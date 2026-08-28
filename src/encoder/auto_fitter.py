"""Encoder domain — auto-selection state machine.

Auto-selects (block_size, harmonic_count) to meet a target fidelity requirement.
Uses pattern matching (`match ... case`) to evaluate escalation strategies
and adjust encoding parameters iteratively.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import torch

from src.encoder.config import EncoderConfig
from src.encoder.tensor_encoder import encode_tensor
from src.phasor_atom import BlockAtom, decode_tensor


@dataclass(frozen=True)
class ConfigEvaluation:
    """Fidelity measurement for one (block_size, harmonic_count) configuration.

    Attributes:
        config: The EncoderConfig that was evaluated.
        mean_squared_error: MSE between original and reconstructed tensor.
        relative_error: Frobenius relative error ||W - Ŵ||_F / ||W||_F.
        parameter_count: Total scalar parameter count of the encoded atom.
    """

    config: EncoderConfig
    mean_squared_error: float
    relative_error: float
    parameter_count: int


@dataclass(frozen=True)
class FitResult:
    """Outcome of an auto-selection parameter search.

    Attributes:
        atom: The best fitted BlockAtom produced.
        relative_error: Final Frobenius relative error ||W - Ŵ||_F / ||W||_F.
        config: The chosen EncoderConfig instance.
        converged: Whether relative_error <= search_space.target_relative_error.
        evaluations: Per-configuration fidelity breakdown for every encode/decode
            cycle the search performed, in evaluation order.
    """

    atom: BlockAtom
    relative_error: float
    config: EncoderConfig
    converged: bool
    evaluations: tuple[ConfigEvaluation, ...] = ()


@dataclass(frozen=True)
class FitSearchSpace:
    """Bounds and convergence criteria for the auto-selection search.

    Attributes:
        target_relative_error: Target upper bound for relative error.
        max_harmonic_count: Maximum harmonic count K allowed during escalation.
        min_block_size: Minimum block size allowed when halving block size.
        max_iterations: Maximum search iterations allowed.
    """

    target_relative_error: float
    max_harmonic_count: int = 64
    min_block_size: int = 8
    max_iterations: int = 10


class StrategyEscalation(enum.Enum):
    """Available parameter adjustment strategies when fidelity target is not met."""

    DOUBLE_HARMONIC_COUNT = "double_harmonic_count"
    HALVE_BLOCK_SIZE = "halve_block_size"
    SEARCH_EXHAUSTED = "search_exhausted"


def select_next_escalation_strategy(
    current_configuration: EncoderConfig, search_space: FitSearchSpace
) -> StrategyEscalation:
    """Determine the next parameter adjustment strategy using structural pattern matching."""
    harmonic_capacity_available = (
        current_configuration.harmonic_count < search_space.max_harmonic_count
    )
    block_size_reducible = (
        current_configuration.block_size > search_space.min_block_size
    )

    match (harmonic_capacity_available, block_size_reducible):
        case (True, _):
            return StrategyEscalation.DOUBLE_HARMONIC_COUNT
        case (False, True):
            return StrategyEscalation.HALVE_BLOCK_SIZE
        case (False, False):
            return StrategyEscalation.SEARCH_EXHAUSTED


def double_harmonic_count_strategy(
    current_configuration: EncoderConfig, search_space: FitSearchSpace
) -> EncoderConfig:
    """Double harmonic count K up to max_harmonic_count."""
    increased_harmonic_count = min(
        current_configuration.harmonic_count * 2,
        search_space.max_harmonic_count,
    )
    return current_configuration.with_harmonic_count(increased_harmonic_count)


def halve_block_size_strategy(
    current_configuration: EncoderConfig,
    initial_harmonic_count: int,
    search_space: FitSearchSpace,
) -> EncoderConfig:
    """Halve block_size down to min_block_size and reset harmonic_count to initial value."""
    reduced_block_size = max(
        current_configuration.block_size // 2, search_space.min_block_size
    )
    return current_configuration.with_block_size(
        reduced_block_size
    ).with_harmonic_count(initial_harmonic_count)


def apply_escalation_strategy(
    strategy: StrategyEscalation,
    current_configuration: EncoderConfig,
    initial_harmonic_count: int,
    search_space: FitSearchSpace,
) -> EncoderConfig:
    """Execute strategy escalation using pattern matching."""
    match strategy:
        case StrategyEscalation.DOUBLE_HARMONIC_COUNT:
            return double_harmonic_count_strategy(
                current_configuration, search_space
            )
        case StrategyEscalation.HALVE_BLOCK_SIZE:
            return halve_block_size_strategy(
                current_configuration, initial_harmonic_count, search_space
            )
        case StrategyEscalation.SEARCH_EXHAUSTED:
            return current_configuration


def compute_relative_error(
    original_tensor: torch.Tensor, reconstructed_tensor: torch.Tensor
) -> float:
    """Calculate Frobenius norm relative error ||original - reconstructed|| / ||original||."""
    difference_norm = (original_tensor - reconstructed_tensor).norm()
    original_norm = original_tensor.norm().clamp_min(1e-12)
    return float(difference_norm / original_norm)


def fit_tensor(
    weight_tensor: torch.Tensor,
    initial_configuration: EncoderConfig,
    search_space: FitSearchSpace,
) -> FitResult:
    """Auto-select (block_size, harmonic_count) to satisfy target_relative_error.

    Args:
        weight_tensor: 2-D weight matrix to encode.
        initial_configuration: Initial EncoderConfig configuration.
        search_space: FitSearchSpace bounds and target error.

    Returns:
        FitResult containing best BlockAtom, final relative error, and chosen EncoderConfig.
    """
    weight_tensor = weight_tensor.to(torch.float32)
    current_configuration = initial_configuration
    initial_harmonic_count = initial_configuration.harmonic_count
    evaluation_history: list[ConfigEvaluation] = []

    def _evaluate_configuration(
        configuration: EncoderConfig,
    ) -> tuple[BlockAtom, float]:
        encoded_atom = encode_tensor(weight_tensor, configuration)
        reconstructed_tensor = decode_tensor(encoded_atom)
        calculated_relative_error = compute_relative_error(
            weight_tensor, reconstructed_tensor
        )
        calculated_mse = float(
            torch.nn.functional.mse_loss(reconstructed_tensor, weight_tensor)
        )
        evaluation_history.append(
            ConfigEvaluation(
                config=configuration,
                mean_squared_error=calculated_mse,
                relative_error=calculated_relative_error,
                parameter_count=encoded_atom.num_params(),
            )
        )
        return encoded_atom, calculated_relative_error

    for _iteration in range(search_space.max_iterations):
        encoded_atom, calculated_relative_error = _evaluate_configuration(
            current_configuration
        )

        if calculated_relative_error <= search_space.target_relative_error:
            return FitResult(
                atom=encoded_atom,
                relative_error=calculated_relative_error,
                config=current_configuration,
                converged=True,
                evaluations=tuple(evaluation_history),
            )

        escalation_strategy = select_next_escalation_strategy(
            current_configuration, search_space
        )

        match escalation_strategy:
            case StrategyEscalation.SEARCH_EXHAUSTED:
                break
            case _:
                current_configuration = apply_escalation_strategy(
                    escalation_strategy,
                    current_configuration,
                    initial_harmonic_count,
                    search_space,
                )

    # Final evaluation with last tried configuration
    final_atom, final_relative_error = _evaluate_configuration(
        current_configuration
    )

    return FitResult(
        atom=final_atom,
        relative_error=final_relative_error,
        config=current_configuration,
        converged=final_relative_error <= search_space.target_relative_error,
        evaluations=tuple(evaluation_history),
    )
