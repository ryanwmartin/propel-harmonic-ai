"""Stress test: encode/decode a real TinyLM safetensor and measure fidelity.

Downloads the actual ``model.safetensors`` of a real tiny language model
(`EleutherAI/pythia-14m <https://huggingface.co/EleutherAI/pythia-14m>`_, a
14M-parameter GPT-NeoX) from the Hugging Face Hub, runs the HWave block-wise
derivative codec on its trained weight matrices, and reports:

* on-disk size of the original ``.safetensors`` tensor vs. the encoded ``.atom``
* parameter-count ratio (encoded ``Θ`` vs. dense weights)
* reconstruction fidelity (relative Frobenius error, MSE, max abs error)
* baseline vs. reversible shared PCA column ordering, including permutation bytes

Scientific finding pinned by these tests
----------------------------------------
A *trained* weight row's first difference is **spectrally broadband** (close to
white noise). Unlike the smooth synthetic tensors in ``tests/test_codec.py``,
its energy is spread across *all* FFT bins. Consequently the direct harmonic
codec only reconstructs real weights exactly when ``K = block_size / 2`` — i.e.
when the FFT warm start forms a **complete real DFT** of the block. At that
setting the format is lossless (to float32 rounding) at ``~1.5×`` dense
parameter cost. Below that budget the truncated spectrum cannot capture the
broadband derivative and the running prefix-sum diverges.

This is precisely the derivative-compactness diagnostic the Epic 03 gate asks
for, measured on real weights: the direct codec is **not** a compressor for
trained LLM weights, which motivates the distillation path (Epics 04–06).

The safetensors file is cached in the standard Hugging Face hub cache, so it is
downloaded only once across runs.
"""

from __future__ import annotations

import os

import pytest
import torch

pytest.importorskip("safetensors", reason="safetensors is required to read .safetensors files")
pytest.importorskip("huggingface_hub", reason="huggingface_hub is required to download the model")

from safetensors import safe_open  # noqa: E402

from src.codec import HWaveCodec  # noqa: E402
from src.encoder import EncoderConfig, compute_relative_error  # noqa: E402
from src.entropy_coder import (  # noqa: E402
    decode_bytes as entropy_decode_bytes,
    encode_bytes as entropy_encode_bytes,
    measure_encoding,
)
from src.phasor_atom import decode_tensor  # noqa: E402
from src.search_compressor import compress_bytes, decode_bytes  # noqa: E402

# A real, tiny, open-weights LM. 14M parameters, GPT-NeoX architecture, F16.
TINY_LM_REPO_ID = "EleutherAI/pythia-14m"
TINY_LM_WEIGHTS_FILE = "model.safetensors"

# Representative trained 2-D weight matrices spanning the two main layer kinds
# (attention QKV projection and MLP down-projection) and both tensor widths.
STRESS_TENSORS = [
    "gpt_neox.layers.0.attention.query_key_value.weight",  # (384, 128)
    "gpt_neox.layers.0.mlp.dense_4h_to_h.weight",          # (128, 512)
]

# `block_size / harmonic_count` configurations to stress. "full" resolves to
# K = block_size // 2 (a complete real DFT of the block -> lossless).
STRESS_CONFIGURATIONS = [
    ("baseline K=8", 128, 8, "none"),
    ("PCA ordered K=8", 128, 8, "pca"),
    ("baseline K=32", 128, 32, "none"),
    ("PCA ordered K=32", 128, 32, "pca"),
    ("baseline full spectrum", 128, "full", "none"),
    ("PCA ordered full spectrum", 128, "full", "pca"),
    ("baseline full, small blocks", 64, "full", "none"),
    ("PCA ordered full, small blocks", 64, "full", "pca"),
]

# Reconstruction is exact to float32 rounding at the full-spectrum budget.
LOSSLESS_RELATIVE_ERROR_TOLERANCE = 1e-4
LOSSLESS_MAX_ABS_TOLERANCE = 1e-4


@pytest.fixture(scope="module")
def safetensors_path() -> str:
    """Download (once) and return the local path of the tiny LM's safetensors."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(TINY_LM_REPO_ID, TINY_LM_WEIGHTS_FILE)


def _load_weight_tensor(safetensors_path: str, tensor_name: str) -> torch.Tensor:
    """Load one tensor from the safetensors file as a float32 CPU tensor."""
    with safe_open(safetensors_path, framework="pt") as tensor_file:
        return tensor_file.get_tensor(tensor_name).to(torch.float32)


def _format_bytes(byte_count: int) -> str:
    """Human-readable byte size."""
    value = float(byte_count)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} GB"


def _resolve_harmonic_count(block_size: int, harmonic_count) -> int:
    """Resolve the symbolic 'full' harmonic count to block_size // 2."""
    if harmonic_count == "full":
        return block_size // 2
    return int(harmonic_count)


class TestRealSafetensorStress:
    """Encode/decode stress test over real trained weight matrices."""

    @pytest.mark.parametrize("tensor_name", STRESS_TENSORS)
    def test_real_tensor_encode_decode_fidelity(
        self, safetensors_path, tensor_name, tmp_path, capsys
    ):
        weight_tensor = _load_weight_tensor(safetensors_path, tensor_name)
        assert weight_tensor.ndim == 2

        codec = HWaveCodec(compute_device="cpu")
        row_count, column_count = weight_tensor.shape
        dense_parameter_count = weight_tensor.numel()
        dense_fp32_byte_size = dense_parameter_count * 4
        dense_fp16_byte_size = dense_parameter_count * 2

        print(f"\n{'=' * 88}")
        print(f"REAL SAFETENSOR STRESS TEST :: {TINY_LM_REPO_ID}")
        print(f"  tensor            = {tensor_name}")
        print(f"  shape             = ({row_count}, {column_count})")
        print(f"  dense parameters  = {dense_parameter_count:,} "
              f"({_format_bytes(dense_fp32_byte_size)} as fp32)")
        print(f"{'=' * 88}")
        print(
            f"  {'configuration':>34}  {'rel_err':>11}  {'MSE':>11}  "
            f"{'max_abs':>11}  {'FP32':>7}  {'FP16':>7}  {'.atom size':>11}"
        )
        print(f"  {'-' * 103}")

        for label, block_size, harmonic_count_spec, column_ordering in STRESS_CONFIGURATIONS:
            harmonic_count = _resolve_harmonic_count(block_size, harmonic_count_spec)
            configuration = EncoderConfig(
                block_size=block_size,
                harmonic_count=harmonic_count,
                refinement_steps=0,  # pure FFT warm start — deterministic, closed form
                column_ordering=column_ordering,
            )

            # --- Encode ---
            atom = codec.encode(weight_tensor, configuration)

            # --- Measure serialized size on disk ---
            atom_path = tmp_path / (
                f"{tensor_name.replace('.', '_')}_bs{block_size}_K{harmonic_count}_"
                f"{column_ordering}.atom"
            )
            codec.save(atom, atom_path)
            atom_byte_size = os.path.getsize(atom_path)

            # --- Decode from the reloaded file (full serialization round-trip) ---
            reloaded_atom = codec.load(atom_path)
            reconstructed = codec.decode(reloaded_atom)

            # --- Fidelity metrics ---
            relative_error = compute_relative_error(weight_tensor, reconstructed)
            mean_squared_error = float(
                torch.nn.functional.mse_loss(reconstructed, weight_tensor)
            )
            max_abs_error = float((reconstructed - weight_tensor).abs().max())
            fp32_byte_ratio = atom_byte_size / dense_fp32_byte_size
            fp16_byte_ratio = atom_byte_size / dense_fp16_byte_size

            print(
                f"  {label:>34}  {relative_error:>11.4e}  "
                f"{mean_squared_error:>11.4e}  {max_abs_error:>11.4e}  "
                f"{fp32_byte_ratio:>6.2f}x  {fp16_byte_ratio:>6.2f}x  "
                f"{_format_bytes(atom_byte_size):>11}"
            )

            # Structural invariants that must hold for every configuration.
            assert reconstructed.shape == weight_tensor.shape
            assert atom.block_size == block_size
            assert atom.K == harmonic_count
            assert (atom.column_order is not None) == (column_ordering == "pca")
            assert atom_byte_size == 32 + atom.num_params() * 4 + atom.permutation_bytes()

            if harmonic_count_spec == "full":
                # K = block_size/2 is a complete real DFT of each block: the
                # format reconstructs the real weights to float32 rounding.
                assert relative_error < LOSSLESS_RELATIVE_ERROR_TOLERANCE, (
                    f"full-spectrum encode should be lossless, got rel_err={relative_error:.3e}"
                )
                assert max_abs_error < LOSSLESS_MAX_ABS_TOLERANCE, (
                    f"full-spectrum encode should be lossless, got max_abs={max_abs_error:.3e}"
                )
            else:
                # Below the full-spectrum budget the derivative is broadband and
                # the truncated fit diverges — this is the Epic 03 gate signal.
                assert relative_error > 0.5, (
                    f"expected sub-spectral K={harmonic_count} to diverge on "
                    f"broadband real weights, got rel_err={relative_error:.3e}"
                )

    def test_full_spectrum_roundtrip_is_bit_exact_through_file(
        self, safetensors_path, tmp_path
    ):
        """At the lossless budget, decode is deterministic through a file round-trip."""
        weight_tensor = _load_weight_tensor(
            safetensors_path, "gpt_neox.layers.0.mlp.dense_4h_to_h.weight"
        )
        codec = HWaveCodec(compute_device="cpu")
        configuration = EncoderConfig(
            block_size=128, harmonic_count=64, refinement_steps=0
        )

        atom = codec.encode(weight_tensor, configuration)
        atom_path = tmp_path / "lossless.atom"
        codec.save(atom, atom_path)

        decode_direct = decode_tensor(atom)
        decode_reloaded = decode_tensor(codec.load(atom_path))

        # Serialization round-trip is bit-identical.
        assert torch.equal(decode_direct, decode_reloaded)
        # And both reconstruct the real weights to f32 rounding.
        assert (
            compute_relative_error(weight_tensor, decode_reloaded)
            < LOSSLESS_RELATIVE_ERROR_TOLERANCE
        )


def test_real_safetensor_summary(safetensors_path, tmp_path, capsys):
    """Standalone summary: original file size vs. aggregate .atom size + fidelity."""
    codec = HWaveCodec(compute_device="cpu")
    configuration = EncoderConfig(block_size=128, harmonic_count=64, refinement_steps=0)

    original_file_size = os.path.getsize(safetensors_path)
    total_dense_parameters = 0
    total_atom_bytes = 0
    worst_relative_error = 0.0

    with safe_open(safetensors_path, framework="pt") as tensor_file:
        # All 2-D transformer weight matrices. The two (50304, 128) embedding
        # tables are excluded: they dominate the per-block Python encode loop and
        # are not representative of the layer weights the codec targets.
        tensor_names = [
            name
            for name in tensor_file.keys()
            if len(tensor_file.get_slice(name).get_shape()) == 2
            and "embed" not in name
        ]
        for tensor_name in tensor_names:
            weight_tensor = tensor_file.get_tensor(tensor_name).to(torch.float32)
            atom = codec.encode(weight_tensor, configuration)
            atom_path = tmp_path / f"{tensor_name.replace('.', '_')}.atom"
            codec.save(atom, atom_path)
            total_atom_bytes += os.path.getsize(atom_path)
            total_dense_parameters += weight_tensor.numel()
            reconstructed = codec.decode(atom)
            worst_relative_error = max(
                worst_relative_error, compute_relative_error(weight_tensor, reconstructed)
            )

    print(f"\n{'=' * 88}")
    print(f"FULL-MODEL TRANSFORMER-WEIGHT SUMMARY :: {TINY_LM_REPO_ID}")
    print(f"  2-D weight tensors encoded   = {len(tensor_names)} (embeddings excluded)")
    print(f"  dense parameters (2-D)       = {total_dense_parameters:,}")
    print(f"  original .safetensors (all)  = {_format_bytes(original_file_size)} (F16, incl. embeddings + 1-D)")
    print(f"  encoded .atom total (2-D)    = {_format_bytes(total_atom_bytes)}")
    print(f"  worst relative error         = {worst_relative_error:.4e}")
    print(f"{'=' * 88}")

    assert len(tensor_names) > 0
    assert worst_relative_error < LOSSLESS_RELATIVE_ERROR_TOLERANCE


def test_lossless_search_compressor_on_real_fp16_rows(safetensors_path, capsys):
    """Search exact byte programs independently for every real matrix row.

    Safetensor weights are binary FP16, not decimal digits. Compressing their
    original little-endian row bytes preserves every IEEE-754 bit and compares
    against the correct two-byte-per-weight baseline. Each row is one fixed
    search block; framing and generator parameters are included in the ratio.
    """
    total_raw_bytes = 0
    total_encoded_bytes = 0
    total_row_count = 0
    candidate_counts: dict[str, int] = {}

    for tensor_name in STRESS_TENSORS:
        with safe_open(safetensors_path, framework="pt") as tensor_file:
            weight_tensor = tensor_file.get_tensor(tensor_name).contiguous()
        assert weight_tensor.dtype == torch.float16

        raw_tensor_bytes = weight_tensor.view(torch.uint8).numpy().tobytes()
        row_byte_count = weight_tensor.shape[1] * weight_tensor.element_size()
        reconstructed_rows = bytearray()
        tensor_encoded_bytes = 0

        for row_start in range(0, len(raw_tensor_bytes), row_byte_count):
            source_row = raw_tensor_bytes[row_start : row_start + row_byte_count]
            encoding = compress_bytes(source_row)
            serialized = encoding.serialize()
            reconstructed_rows.extend(decode_bytes(serialized))
            tensor_encoded_bytes += len(serialized)
            candidate_counts[encoding.kind.name] = (
                candidate_counts.get(encoding.kind.name, 0) + 1
            )

        assert bytes(reconstructed_rows) == raw_tensor_bytes
        total_raw_bytes += len(raw_tensor_bytes)
        total_encoded_bytes += tensor_encoded_bytes
        total_row_count += weight_tensor.shape[0]
        print(
            f"\nLOSSLESS ROW SEARCH :: {tensor_name}\n"
            f"  rows / row bytes = {weight_tensor.shape[0]} / {row_byte_count:,}\n"
            f"  raw FP16 bytes   = {_format_bytes(len(raw_tensor_bytes))}\n"
            f"  encoded bytes    = {_format_bytes(tensor_encoded_bytes)}\n"
            f"  encoded/raw      = {tensor_encoded_bytes / len(raw_tensor_bytes):.4f}x"
        )

    print(f"  candidate counts = {candidate_counts}")
    print(f"  aggregate ratio  = {total_encoded_bytes / total_raw_bytes:.4f}x")

    assert total_raw_bytes > 0
    assert total_encoded_bytes > 0
    # Exactness, not a presumed win, is the contract: random-looking trained
    # weights are permitted to choose the bounded-overhead RAW fallback.
    assert total_encoded_bytes <= total_raw_bytes + total_row_count * 10


def test_adaptive_entropy_coder_on_real_fp16_weights(safetensors_path, capsys):
    """Arithmetic-code real FP16 bytes and require bit-exact reconstruction.

    Whole-tensor coding measures the best context reuse available to this
    adaptive byte model. Row-blocked coding measures a more parallel inference
    layout where every row can be decoded independently. All stream headers and
    checksums are included in reported storage.
    """
    total_raw_bytes = 0
    total_tensor_encoded_bytes = 0
    total_row_encoded_bytes = 0

    print(f"\n{'=' * 88}")
    print(f"ADAPTIVE ARITHMETIC CODER :: {TINY_LM_REPO_ID}")
    print(
        f"  {'tensor':>55}  {'raw':>10}  {'tensor':>10}  "
        f"{'ratio':>8}  {'bits/w':>8}  {'row ratio':>10}"
    )
    print(f"  {'-' * 111}")

    for tensor_name in STRESS_TENSORS:
        with safe_open(safetensors_path, framework="pt") as tensor_file:
            weight_tensor = tensor_file.get_tensor(tensor_name).contiguous()
        assert weight_tensor.dtype == torch.float16

        raw_tensor_bytes = weight_tensor.view(torch.uint8).numpy().tobytes()
        tensor_encoding = entropy_encode_bytes(raw_tensor_bytes)
        tensor_metrics = measure_encoding(raw_tensor_bytes, tensor_encoding)
        reconstructed_tensor_bytes = entropy_decode_bytes(tensor_encoding)

        # Byte equality is stronger than numerical closeness: every FP16 sign,
        # exponent, and mantissa bit must be identical after decoding.
        assert reconstructed_tensor_bytes == raw_tensor_bytes
        reconstructed_tensor = torch.frombuffer(
            bytearray(reconstructed_tensor_bytes), dtype=torch.float16
        ).reshape(weight_tensor.shape)
        assert torch.equal(reconstructed_tensor, weight_tensor)

        row_byte_count = weight_tensor.shape[1] * weight_tensor.element_size()
        reconstructed_rows = bytearray()
        row_encoded_bytes = 0
        for row_start in range(0, len(raw_tensor_bytes), row_byte_count):
            source_row = raw_tensor_bytes[row_start : row_start + row_byte_count]
            row_encoding = entropy_encode_bytes(source_row)
            reconstructed_rows.extend(entropy_decode_bytes(row_encoding))
            row_encoded_bytes += len(row_encoding)
        assert bytes(reconstructed_rows) == raw_tensor_bytes

        total_raw_bytes += len(raw_tensor_bytes)
        total_tensor_encoded_bytes += tensor_metrics.encoded_bytes
        total_row_encoded_bytes += row_encoded_bytes
        print(
            f"  {tensor_name:>55}  {_format_bytes(len(raw_tensor_bytes)):>10}  "
            f"{_format_bytes(tensor_metrics.encoded_bytes):>10}  "
            f"{tensor_metrics.encoded_to_source_ratio:>7.4f}x  "
            f"{tensor_metrics.bits_per_fp16_weight:>8.3f}  "
            f"{row_encoded_bytes / len(raw_tensor_bytes):>9.4f}x"
        )

    tensor_ratio = total_tensor_encoded_bytes / total_raw_bytes
    row_ratio = total_row_encoded_bytes / total_raw_bytes
    print(f"  aggregate tensor-stream ratio = {tensor_ratio:.4f}x")
    print(f"  aggregate row-stream ratio    = {row_ratio:.4f}x")
    print(f"{'=' * 88}")

    assert total_raw_bytes > 0
    assert total_tensor_encoded_bytes > 0
    assert total_row_encoded_bytes > 0
