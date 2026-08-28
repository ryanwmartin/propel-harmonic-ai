"""Batch domain — processing and encoding multiple weight matrices in parallel or sequence.

Separates batch tensor conversion from single tensor encoding.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from src.encoder.config import EncoderConfig
from src.encoder.tensor_encoder import encode_tensor
from src.phasor_atom import BlockAtom


def encode_tensor_batch(
    named_tensors: Mapping[str, torch.Tensor],
    shared_configuration: EncoderConfig,
) -> dict[str, BlockAtom]:
    """Encode multiple named weight tensors using a uniform shared configuration.

    Args:
        named_tensors: Mapping of tensor names (e.g. layer names) to weight tensors.
        shared_configuration: Single shared EncoderConfig applied to all tensors.

    Returns:
        Dictionary mapping layer names to their corresponding BlockAtom outputs.
    """
    return {
        layer_name: encode_tensor(weight_tensor, shared_configuration)
        for layer_name, weight_tensor in named_tensors.items()
    }


def encode_tensor_batch_individual(
    named_tensors: Mapping[str, torch.Tensor],
    layer_configurations: Mapping[str, EncoderConfig],
) -> dict[str, BlockAtom]:
    """Encode multiple weight tensors using distinct per-layer configurations.

    Args:
        named_tensors: Mapping of layer names to weight tensors.
        layer_configurations: Mapping of layer names to individual EncoderConfig instances.

    Returns:
        Dictionary mapping layer names to their corresponding BlockAtom outputs.

    Raises:
        ValueError: If a layer name in named_tensors is missing from layer_configurations.
    """
    encoded_atoms: dict[str, BlockAtom] = {}
    for layer_name, weight_tensor in named_tensors.items():
        if layer_name not in layer_configurations:
            raise ValueError(
                f"No EncoderConfig provided for tensor {layer_name!r}"
            )
        specific_configuration = layer_configurations[layer_name]
        encoded_atoms[layer_name] = encode_tensor(
            weight_tensor, specific_configuration
        )
    return encoded_atoms
