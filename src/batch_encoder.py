"""Batch tensor conversion facade — delegates to src.batch domain."""

from src.batch.batch_processor import (
    encode_tensor_batch,
    encode_tensor_batch_individual,
)

__all__ = [
    "encode_tensor_batch",
    "encode_tensor_batch_individual",
]
