"""Harness domain — shared storage-precision helpers.

Every representation family supports the same precision sweep: parameters may
be stored at reduced precision (float16) while computation stays in float32.
Quantization is modeled as a round-trip through the storage dtype so fidelity
measurements reflect exactly what a serialized artifact would decode to.
"""

from __future__ import annotations

import torch

SUPPORTED_FIELD_BITS: frozenset[int] = frozenset({16, 32})
"""Storage bit widths supported by the precision sweep."""


def quantize_field(field_tensor: torch.Tensor, bit_width: int) -> torch.Tensor:
    """Round-trip a float32 field through the requested storage precision.

    Args:
        field_tensor: Field values in float32 (or convertible).
        bit_width: Storage precision; 32 keeps float32, 16 round-trips
            through float16.

    Returns:
        Float32 tensor whose values are exactly representable at the
        requested storage precision.

    Raises:
        ValueError: If ``bit_width`` is not supported.
    """
    if bit_width not in SUPPORTED_FIELD_BITS:
        raise ValueError(
            f"bit width must be one of {sorted(SUPPORTED_FIELD_BITS)}, "
            f"got {bit_width}"
        )
    if bit_width == 32:
        return field_tensor.to(torch.float32)
    return field_tensor.to(torch.float16).to(torch.float32)
