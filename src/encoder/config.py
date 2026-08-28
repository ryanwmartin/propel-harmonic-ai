"""Encoder domain — configuration value object.

Bundles all parameters controlling block partition geometry, sinusoid fitting
counts, and Adam refinement iteration budgets into an immutable container.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.encoder.column_ordering import (
    COLUMN_ORDERING_NONE,
    SUPPORTED_COLUMN_ORDERINGS,
)


@dataclass(frozen=True)
class EncoderConfig:
    """Immutable configuration container for tensor block encoding.

    Attributes:
        block_size: Maximum weight count per block.
        harmonic_count: Number of sinusoids (K) fitted per block.
        refinement_steps: Adam gradient optimization steps per block (0 = pure FFT).
        learning_rate: Adam learning rate for per-block parameter polish.
        column_ordering: Shared-column transform (``"none"`` or ``"pca"``).
    """

    block_size: int
    harmonic_count: int
    refinement_steps: int = 200
    learning_rate: float = 1e-2
    column_ordering: str = COLUMN_ORDERING_NONE

    def __post_init__(self) -> None:
        if self.block_size < 2:
            raise ValueError(f"block_size must be >= 2, got {self.block_size}")
        if self.harmonic_count < 1:
            raise ValueError(
                f"harmonic_count must be >= 1, got {self.harmonic_count}"
            )
        if self.refinement_steps < 0:
            raise ValueError(
                f"refinement_steps must be >= 0, got {self.refinement_steps}"
            )
        if self.learning_rate <= 0.0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if self.column_ordering not in SUPPORTED_COLUMN_ORDERINGS:
            raise ValueError(
                f"column_ordering must be one of {sorted(SUPPORTED_COLUMN_ORDERINGS)}, "
                f"got {self.column_ordering!r}"
            )

    def with_harmonic_count(self, new_harmonic_count: int) -> "EncoderConfig":
        """Return a copy of this configuration with a new harmonic count."""
        return EncoderConfig(
            block_size=self.block_size,
            harmonic_count=new_harmonic_count,
            refinement_steps=self.refinement_steps,
            learning_rate=self.learning_rate,
            column_ordering=self.column_ordering,
        )

    def with_block_size(self, new_block_size: int) -> "EncoderConfig":
        """Return a copy of this configuration with a new block size."""
        return EncoderConfig(
            block_size=new_block_size,
            harmonic_count=self.harmonic_count,
            refinement_steps=self.refinement_steps,
            learning_rate=self.learning_rate,
            column_ordering=self.column_ordering,
        )


# Semantic class alias
EncoderConfiguration = EncoderConfig
