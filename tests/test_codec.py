"""Contract tests for the HWave block-wise derivative codec.

These tests pin the exact behaviors the Rust decoder (Epic 02) and the parity
test (Epic 03) rely on: the decode formula, the .atom binary layout, and
serialization round-trip fidelity.
"""

import struct

import pytest
import torch

from src.atom_io import ATOM_MAGIC, ATOM_VERSION, load_atom, save_atom
from src.codec import HWaveCodec
from src.encoder import (
    EncoderConfig,
    FitSearchSpace,
    compute_relative_error,
    encode_tensor,
    extract_blocks,
    fit_harmonics_to_block,
    fit_tensor,
)
from src.encoder.column_ordering import compute_pca_column_order
from src.phasor_atom import (
    BlockAtom,
    block_length,
    decode_row,
    decode_tensor,
    eval_block,
)


def _structured_target(row_count: int, column_count: int) -> torch.Tensor:
    row_coords = torch.linspace(-1.0, 1.0, row_count).unsqueeze(1)
    col_coords = torch.linspace(-1.0, 1.0, column_count).unsqueeze(0)
    synthetic = (
        0.8
        * torch.sin(2 * torch.pi * 3.0 * (0.9 * row_coords + 0.4 * col_coords) + 0.3)
        + 0.5
        * torch.sin(2 * torch.pi * 6.0 * (-0.2 * row_coords + 1.0 * col_coords) + 1.1)
        + 0.3
        * torch.sin(2 * torch.pi * 1.5 * (1.0 * row_coords - 0.7 * col_coords) + 2.0)
    ) * torch.exp(-0.4 * (row_coords.abs() + col_coords.abs()))
    return synthetic.to(torch.float32)


def _make_atom(
    row_count: int, column_count: int, block_size: int, harmonic_count: int
) -> BlockAtom:
    block_count = (column_count + block_size - 1) // block_size
    return BlockAtom(
        anchors=torch.randn(row_count, block_count),
        amplitudes=torch.randn(row_count, block_count, harmonic_count),
        frequencies=torch.randn(row_count, block_count, harmonic_count),
        phases=torch.randn(row_count, block_count, harmonic_count),
        block_size=block_size,
        num_blocks=block_count,
        rows=row_count,
        cols=column_count,
    )


def _default_config(block_size: int, harmonic_count: int, steps: int = 200) -> EncoderConfig:
    return EncoderConfig(
        block_size=block_size,
        harmonic_count=harmonic_count,
        refinement_steps=steps,
    )




class TestColumnOrdering:
    def test_pca_order_is_deterministic_permutation(self):
        weights = _structured_target(12, 31)
        first = compute_pca_column_order(weights)
        second = compute_pca_column_order(weights)
        assert torch.equal(first, second)
        assert torch.equal(torch.sort(first).values, torch.arange(weights.shape[1]))

    def test_ordered_full_spectrum_roundtrip_restores_original_columns(self):
        weights = torch.randn(7, 32)
        config = EncoderConfig(
            block_size=16,
            harmonic_count=8,
            refinement_steps=0,
            column_ordering="pca",
        )
        atom = encode_tensor(weights, config)
        reconstructed = decode_tensor(atom)
        assert atom.column_order is not None
        assert compute_relative_error(weights, reconstructed) < 1e-4

    def test_ordered_file_roundtrip_preserves_order_and_size(self, tmp_path):
        weights = torch.randn(5, 32)
        codec = HWaveCodec()
        config = EncoderConfig(
            block_size=16,
            harmonic_count=4,
            refinement_steps=0,
            column_ordering="pca",
        )
        atom = codec.encode(weights, config)
        path = tmp_path / "ordered.atom"
        codec.save(atom, path)
        loaded = codec.load(path)

        expected_size = 32 + atom.num_params() * 4 + weights.shape[1] * 4
        assert path.stat().st_size == expected_size
        assert torch.equal(loaded.column_order, atom.column_order.cpu())
        assert torch.equal(codec.decode(loaded), codec.decode(atom))

class TestBlockAtom:
    def test_shapes_and_params(self):
        row_count, column_count, block_size, harmonic_count = 4, 64, 16, 8
        atom = _make_atom(row_count, column_count, block_size, harmonic_count)
        assert atom.K == harmonic_count
        assert atom.num_blocks == 4
        assert atom.num_params() == row_count * 4 * (1 + 3 * harmonic_count)

    def test_rejects_bad_anchor_shape(self):
        with pytest.raises(ValueError, match="anchors"):
            BlockAtom(
                anchors=torch.randn(4, 3),
                amplitudes=torch.randn(4, 4, 8),
                frequencies=torch.randn(4, 4, 8),
                phases=torch.randn(4, 4, 8),
                block_size=16,
                num_blocks=4,
                rows=4,
                cols=64,
            )

    def test_rejects_bad_harmonic_shape(self):
        with pytest.raises(ValueError, match="amplitudes"):
            BlockAtom(
                anchors=torch.randn(4, 4),
                amplitudes=torch.randn(4, 4, 7),
                frequencies=torch.randn(4, 4, 8),
                phases=torch.randn(4, 4, 8),
                block_size=16,
                num_blocks=4,
                rows=4,
                cols=64,
            )

    def test_rejects_wrong_leading_dims(self):
        with pytest.raises(ValueError, match="leading dims"):
            BlockAtom(
                anchors=torch.randn(4, 4),
                amplitudes=torch.randn(4, 3, 8),
                frequencies=torch.randn(4, 3, 8),
                phases=torch.randn(4, 3, 8),
                block_size=16,
                num_blocks=4,
                rows=4,
                cols=64,
            )


class TestBlockMath:
    def test_block_length(self):
        assert block_length(16, 64, 0) == 16
        assert block_length(16, 64, 3) == 16
        assert block_length(16, 60, 3) == 12  # 48..60
        assert block_length(16, 17, 1) == 1  # 16..17

    def test_extract_blocks_exact(self):
        weight_tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        anchors, differences, block_lengths = extract_blocks(weight_tensor, block_size=3)
        assert anchors.shape == (3, 2)
        assert differences.shape == (3, 2, 2)
        assert block_lengths.tolist() == [3, 1]
        # Block 0: [0,1,2] -> anchor=0, d=[1,1]
        assert anchors[0, 0].item() == 0.0
        assert differences[0, 0].tolist() == [1.0, 1.0]
        # Block 1: [3] -> anchor=3, d zero-padded
        assert anchors[0, 1].item() == 3.0
        assert differences[0, 1].tolist() == [0.0, 0.0]

    def test_decode_accumulates_in_float32(self):
        # Parity contract with the Rust decoder (Epic 02/03): the prefix-sum
        # accumulates in f32 via torch.cumsum. Pin against a manual f32 cumsum.
        torch.manual_seed(7)
        atom = _make_atom(1, 128, 128, 8)
        decoded_row = decode_row(atom, 0)
        derivatives = eval_block(atom, 0, 0)[:127]
        expected = atom.anchors[0, 0] + torch.cumsum(derivatives, dim=0)
        assert torch.equal(decoded_row[1:], expected)

    def test_decode_row_integrates(self):
        atom = BlockAtom(
            anchors=torch.tensor([[5.0]]),
            amplitudes=torch.tensor([[[1.0]]]),
            frequencies=torch.tensor([[[0.0]]]),
            phases=torch.tensor([[[torch.pi / 2]]]),
            block_size=8,
            num_blocks=1,
            rows=1,
            cols=8,
        )
        decoded_row = decode_row(atom, 0)
        expected = torch.tensor([5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        assert torch.allclose(decoded_row, expected, atol=1e-6)


class TestHarmonicFitter:
    def test_fits_constant_derivative_linear_trend(self):
        # A linear trend in the weights = constant first-difference (pure DC).
        # The anchor cannot absorb slope, so the fitter must capture DC exactly
        # via A·sin(2π·0·x + π/2) = A.
        constant_difference = torch.full((31,), 0.125, dtype=torch.float32)
        config = EncoderConfig(block_size=32, harmonic_count=1, refinement_steps=0)
        amplitudes, frequencies, phases = fit_harmonics_to_block(
            constant_difference, config
        )
        sample_indices = torch.arange(31, dtype=torch.float32)
        arguments = 2 * torch.pi * frequencies[0] * sample_indices + phases[0]
        prediction = amplitudes[0] * torch.sin(arguments)
        assert torch.allclose(prediction, constant_difference, atol=1e-5)

    def test_linear_trend_roundtrip(self):
        # End-to-end: a perfectly linear row must reconstruct with no drift.
        row = torch.linspace(0.0, 1.0, 64).unsqueeze(0)
        config = EncoderConfig(block_size=32, harmonic_count=2, refinement_steps=0)
        atom = encode_tensor(row, config)
        reconstructed = decode_tensor(atom)
        assert torch.allclose(reconstructed, row, atol=1e-5)

    def test_fits_pure_sinusoid(self):
        signal_length = 32
        sample_indices = torch.arange(signal_length, dtype=torch.float32)
        difference_signal = 2.5 * torch.sin(
            2 * torch.pi * 4.0 * sample_indices / signal_length + 0.7
        )
        config = EncoderConfig(
            block_size=signal_length, harmonic_count=1, refinement_steps=500
        )
        amplitudes, frequencies, phases = fit_harmonics_to_block(
            difference_signal, config
        )
        arguments = 2 * torch.pi * frequencies[0] * sample_indices + phases[0]
        prediction = amplitudes[0] * torch.sin(arguments)
        mse = torch.nn.functional.mse_loss(prediction, difference_signal).item()
        assert mse < 1e-4


class TestCodec:
    def test_roundtrip_structured_tensor(self):
        row_count, column_count, block_size, harmonic_count = 32, 64, 16, 8
        weight_tensor = _structured_target(row_count, column_count)
        codec = HWaveCodec()
        config = _default_config(block_size, harmonic_count)
        atom = codec.encode(weight_tensor, config)
        reconstructed = codec.decode(atom)

        assert reconstructed.shape == weight_tensor.shape
        mse = torch.nn.functional.mse_loss(reconstructed, weight_tensor).item()
        rel_error = (reconstructed - weight_tensor).norm().item() / weight_tensor.norm().item()
        assert mse < 1e-3
        assert rel_error < 5e-2

    def test_decode_deterministic(self):
        row_count, column_count, block_size, harmonic_count = 16, 32, 16, 4
        weight_tensor = _structured_target(row_count, column_count)
        codec = HWaveCodec()
        config = _default_config(block_size, harmonic_count, steps=50)
        atom = codec.encode(weight_tensor, config)
        first_decode = codec.decode(atom)
        second_decode = codec.decode(atom)
        assert torch.equal(first_decode, second_decode)

    def test_encode_rejects_1d(self):
        codec = HWaveCodec()
        config = _default_config(block_size=8, harmonic_count=4)
        with pytest.raises(ValueError):
            codec.encode(torch.randn(16), config)

    def test_edge_block_shorter_than_block_size(self):
        row_count, column_count, block_size, harmonic_count = 4, 20, 16, 4
        weight_tensor = _structured_target(row_count, column_count)
        codec = HWaveCodec()
        config = _default_config(block_size, harmonic_count, steps=50)
        atom = codec.encode(weight_tensor, config)
        reconstructed = codec.decode(atom)
        assert reconstructed.shape == (row_count, column_count)
        # Last block has length 4; its anchor is exact.
        assert reconstructed[0, 16].item() == weight_tensor[0, 16].item()


class TestAutoFitter:
    def test_auto_selects_parameters(self):
        row_count, column_count = 16, 32
        weight_tensor = _structured_target(row_count, column_count)
        codec = HWaveCodec()
        initial_config = EncoderConfig(
            block_size=32, harmonic_count=2, refinement_steps=100
        )
        search_space = FitSearchSpace(target_relative_error=0.05)
        result = codec.fit(weight_tensor, initial_config, search_space)
        assert result.relative_error <= 0.05
        assert result.atom.block_size == result.config.block_size
        assert result.atom.K == result.config.harmonic_count

    def test_fit_records_per_config_breakdown(self):
        row_count, column_count = 8, 32
        weight_tensor = _structured_target(row_count, column_count)
        codec = HWaveCodec()
        initial_config = EncoderConfig(
            block_size=32, harmonic_count=2, refinement_steps=50
        )
        search_space = FitSearchSpace(target_relative_error=0.05, max_iterations=4)
        result = codec.fit(weight_tensor, initial_config, search_space)

        assert len(result.evaluations) >= 1
        for evaluation in result.evaluations:
            assert evaluation.mean_squared_error >= 0.0
            assert evaluation.relative_error >= 0.0
            assert evaluation.parameter_count > 0
        # The final evaluation corresponds to the returned config and error.
        final_evaluation = result.evaluations[-1]
        assert final_evaluation.config == result.config
        assert final_evaluation.relative_error == pytest.approx(
            result.relative_error
        )


class TestAtomFileFormat:
    def test_binary_layout(self, tmp_path):
        row_count, column_count, block_size, harmonic_count = 4, 32, 16, 8
        codec = HWaveCodec()
        config = _default_config(block_size, harmonic_count, steps=50)
        atom = codec.encode(
            _structured_target(row_count, column_count), config
        )
        file_path = tmp_path / "test.atom"
        codec.save(atom, file_path)

        raw_bytes = file_path.read_bytes()
        block_count = 2
        header_size = 4 + 7 * 4  # magic + 7 u32s (including flags)
        payload_size = row_count * block_count * (1 + 3 * harmonic_count) * 4
        assert len(raw_bytes) == header_size + payload_size

        magic, version, file_k, file_bs, file_nb, file_rows, file_cols, flags = struct.unpack(
            "<4s7I", raw_bytes[:header_size]
        )
        assert magic == ATOM_MAGIC
        assert version == ATOM_VERSION
        assert file_k == harmonic_count
        assert file_bs == block_size
        assert file_nb == block_count
        assert file_rows == row_count
        assert file_cols == column_count
        assert flags == 0

    def test_save_load_roundtrip(self, tmp_path):
        row_count, column_count, block_size, harmonic_count = 8, 32, 16, 8
        codec = HWaveCodec()
        config = _default_config(block_size, harmonic_count, steps=50)
        atom = codec.encode(
            _structured_target(row_count, column_count), config
        )
        file_path = tmp_path / "roundtrip.atom"
        codec.save(atom, file_path)

        reloaded_atom = codec.load(file_path)
        assert reloaded_atom.block_size == block_size
        assert reloaded_atom.K == harmonic_count
        assert reloaded_atom.rows == row_count
        assert reloaded_atom.cols == column_count
        assert torch.equal(atom.anchors, reloaded_atom.anchors)
        assert torch.equal(atom.amplitudes, reloaded_atom.amplitudes)
        assert torch.equal(atom.frequencies, reloaded_atom.frequencies)
        assert torch.equal(atom.phases, reloaded_atom.phases)
        # Decoded outputs are bit-identical after round-trip.
        assert torch.equal(codec.decode(atom), codec.decode(reloaded_atom))

    def test_load_accepts_legacy_v2_file(self, tmp_path):
        atom = _make_atom(2, 8, 8, 2)
        file_path = tmp_path / "legacy-v2.atom"
        header = struct.pack("<4s6I", ATOM_MAGIC, 2, 2, 8, 1, 2, 8)
        payload = bytearray()
        for row_index in range(atom.rows):
            payload.extend(atom.anchors[row_index, 0].numpy().astype("<f4").tobytes())
            payload.extend(atom.amplitudes[row_index, 0].numpy().astype("<f4").tobytes())
            payload.extend(atom.frequencies[row_index, 0].numpy().astype("<f4").tobytes())
            payload.extend(atom.phases[row_index, 0].numpy().astype("<f4").tobytes())
        file_path.write_bytes(header + payload)

        loaded = load_atom(file_path)
        assert loaded.column_order is None
        assert torch.equal(loaded.anchors, atom.anchors)
        assert torch.equal(decode_tensor(loaded), decode_tensor(atom))

    def test_load_rejects_bad_magic(self, tmp_path):
        file_path = tmp_path / "bad.atom"
        file_path.write_bytes(b"XXXX" + b"\x00" * 64)
        with pytest.raises(ValueError, match="bad magic"):
            load_atom(file_path)

    def test_load_rejects_wrong_version(self, tmp_path):
        file_path = tmp_path / "v1.atom"
        file_path.write_bytes(ATOM_MAGIC + struct.pack("<I", 1) + b"\x00" * 64)
        with pytest.raises(ValueError, match="unsupported version"):
            load_atom(file_path)

    def test_load_rejects_truncated(self, tmp_path):
        file_path = tmp_path / "trunc.atom"
        file_path.write_bytes(ATOM_MAGIC + struct.pack("<I", ATOM_VERSION))
        with pytest.raises(ValueError, match="truncated"):
            load_atom(file_path)

    def test_load_rejects_inconsistent_geometry(self, tmp_path):
        # Header claims num_blocks=3 but ceil(cols/block_size) = ceil(32/16) = 2.
        file_path = tmp_path / "badgeom.atom"
        header = ATOM_MAGIC + struct.pack("<6I", ATOM_VERSION, 8, 16, 3, 4, 32)
        file_path.write_bytes(header + b"\x00" * 1024)
        with pytest.raises(ValueError, match="inconsistent geometry"):
            load_atom(file_path)

    def test_load_rejects_zero_dimensions(self, tmp_path):
        file_path = tmp_path / "zerodim.atom"
        header = ATOM_MAGIC + struct.pack("<6I", ATOM_VERSION, 8, 16, 2, 0, 32)
        file_path.write_bytes(header + b"\x00" * 1024)
        with pytest.raises(ValueError, match="invalid header dimensions"):
            load_atom(file_path)


class TestBatchEncoder:
    def test_encode_tensor_batch(self):
        from src.batch_encoder import encode_tensor_batch

        tensors = {
            "layer_0": _structured_target(8, 32),
            "layer_1": _structured_target(8, 32),
        }
        config = _default_config(block_size=16, harmonic_count=4, steps=50)
        results = encode_tensor_batch(tensors, config)

        assert set(results.keys()) == {"layer_0", "layer_1"}
        for name, atom in results.items():
            assert atom.rows == 8
            assert atom.cols == 32

    def test_encode_tensor_batch_individual(self):
        from src.batch_encoder import encode_tensor_batch_individual

        tensors = {
            "layer_0": _structured_target(8, 32),
            "layer_1": _structured_target(8, 32),
        }
        configs = {
            "layer_0": _default_config(block_size=16, harmonic_count=4, steps=50),
            "layer_1": _default_config(block_size=8, harmonic_count=2, steps=50),
        }
        results = encode_tensor_batch_individual(tensors, configs)

        assert results["layer_0"].block_size == 16
        assert results["layer_0"].K == 4
        assert results["layer_1"].block_size == 8
        assert results["layer_1"].K == 2

    def test_encode_tensor_batch_individual_missing_config(self):
        from src.batch_encoder import encode_tensor_batch_individual

        tensors = {"layer_0": _structured_target(8, 32)}
        configs = {}
        with pytest.raises(ValueError, match="No EncoderConfig"):
            encode_tensor_batch_individual(tensors, configs)
