"""Phasor Inference — HWave reference codec.

Encodes a weight tensor ``W`` into per-row, per-block harmonic parameters
``Θ`` (anchors + 1D sinusoids fitting the first difference) and procedurally
decodes ``Θ`` back into ``Ŵ`` via oscillator + prefix-sum integration. This
package is the source of truth the Rust decoder must match.

Domain layout:

- ``src.encoder`` — encoding pipeline (block extraction, harmonic fitting,
  auto-selection of block_size / K).
- ``src.phasor_atom`` — BlockAtom container + decode contract.
- ``src.atom_io`` — ``.atom`` binary file save / load.
- ``src.batch_encoder`` — encode multiple named tensors in one call.
- ``src.codec`` — HWaveCodec facade composing all of the above.
"""

from src.phasor_atom import BlockAtom, block_length, decode_row, decode_tensor, eval_block

__all__ = [
    "BlockAtom",
    "block_length",
    "decode_row",
    "decode_tensor",
    "eval_block",
    "HWaveCodec",
]


def __getattr__(name: str):
    # Lazy import so `python -m src.codec` does not import src.codec twice.
    if name == "HWaveCodec":
        from src.codec import HWaveCodec

        return HWaveCodec
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
