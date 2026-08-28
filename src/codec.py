"""HWaveCodec facade — composes encoder, decoder, and I/O domains.

This module acts as a high-level facade. All core domain logic is delegated to:
- src.encoder: Encoding and parameter auto-selection
- src.decoder: Procedural oscillator and running prefix-sum decoding
- src.io: Reading and writing .atom binary files
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from src.encoder import (
    EncoderConfig,
    FitResult,
    FitSearchSpace,
    compute_relative_error,
    encode_tensor,
    fit_tensor,
)
from src.io import load_atom_from_file, save_atom_to_file
from src.phasor_atom import BlockAtom, decode_tensor


class HWaveCodec:
    """Convenience facade for encoding, decoding, and saving/loading weight matrices."""

    def __init__(self, compute_device: torch.device | str = "cpu") -> None:
        self.device = torch.device(compute_device)

    def encode(
        self, weight_tensor: torch.Tensor, configuration: EncoderConfig
    ) -> BlockAtom:
        """Fit per-block harmonic parameters to weight_tensor."""
        self._validate_two_dimensional_tensor(weight_tensor)
        return encode_tensor(weight_tensor.to(torch.float32), configuration)

    def fit(
        self,
        weight_tensor: torch.Tensor,
        initial_configuration: EncoderConfig,
        search_space: FitSearchSpace,
    ) -> FitResult:
        """Auto-select block_size and harmonic_count to satisfy target error."""
        self._validate_two_dimensional_tensor(weight_tensor)
        return fit_tensor(
            weight_tensor.to(torch.float32), initial_configuration, search_space
        )

    def decode(self, atom: BlockAtom) -> torch.Tensor:
        """Procedurally synthesize reconstructed weight matrix Ŵ from BlockAtom Θ."""
        return decode_tensor(atom.to(self.device)).detach().cpu()

    @staticmethod
    def save(atom: BlockAtom, destination_file_path: str | Path) -> None:
        """Save BlockAtom to a .atom binary file (delegates to src.io.atom_writer)."""
        save_atom_to_file(atom, destination_file_path)

    @staticmethod
    def load(source_file_path: str | Path) -> BlockAtom:
        """Load BlockAtom from a .atom binary file (delegates to src.io.atom_reader)."""
        return load_atom_from_file(source_file_path)

    @staticmethod
    def _validate_two_dimensional_tensor(weight_tensor: torch.Tensor) -> None:
        if weight_tensor.ndim != 2:
            raise ValueError(
                f"Expected a 2-D weight tensor, got shape {tuple(weight_tensor.shape)}"
            )


def generate_synthetic_target_matrix(row_count: int, column_count: int) -> torch.Tensor:
    """Generate a smooth synthetic target tensor with spectral structure for CLI demos."""
    row_coordinates = torch.linspace(-1.0, 1.0, row_count).unsqueeze(1)
    column_coordinates = torch.linspace(-1.0, 1.0, column_count).unsqueeze(0)
    synthetic_tensor = (
        0.8
        * torch.sin(
            2 * torch.pi * 3.0 * (0.9 * row_coordinates + 0.4 * column_coordinates)
            + 0.3
        )
        + 0.5
        * torch.sin(
            2 * torch.pi * 6.0 * (-0.2 * row_coordinates + 1.0 * column_coordinates)
            + 1.1
        )
        + 0.3
        * torch.sin(
            2 * torch.pi * 1.5 * (1.0 * row_coordinates - 0.7 * column_coordinates)
            + 2.0
        )
    ) * torch.exp(-0.4 * (row_coordinates.abs() + column_coordinates.abs()))
    return synthetic_tensor.to(torch.float32)


def main(command_line_arguments: list[str] | None = None) -> int:
    argument_parser = argparse.ArgumentParser(
        prog="python -m src.codec",
        description="HWave reference codec CLI.",
    )
    argument_parser.add_argument("--rows", type=int, default=128, help="number of rows")
    argument_parser.add_argument("--cols", type=int, default=128, help="number of columns")
    argument_parser.add_argument("--block-size", type=int, default=32, help="block length")
    argument_parser.add_argument("--K", type=int, default=8, help="harmonics per block")
    argument_parser.add_argument("--steps", type=int, default=200, help="Adam steps per block")
    argument_parser.add_argument("--lr", type=float, default=1e-2, help="Adam learning rate")
    argument_parser.add_argument("--seed", type=int, default=0, help="random seed")
    argument_parser.add_argument(
        "--target-rel-error",
        type=float,
        default=None,
        help="auto-select (block_size, K) to meet target relative error",
    )
    argument_parser.add_argument(
        "--out", type=str, default=None, help="optional .atom output file path"
    )
    parsed_arguments = argument_parser.parse_args(command_line_arguments)

    torch.manual_seed(parsed_arguments.seed)
    weight_matrix = generate_synthetic_target_matrix(
        parsed_arguments.rows, parsed_arguments.cols
    )

    codec_instance = HWaveCodec(compute_device="cpu")
    print(
        f"Encoding {parsed_arguments.rows}x{parsed_arguments.cols} tensor: "
        f"block_size={parsed_arguments.block_size}, K={parsed_arguments.K}, steps={parsed_arguments.steps}"
    )

    base_configuration = EncoderConfig(
        block_size=parsed_arguments.block_size,
        harmonic_count=parsed_arguments.K,
        refinement_steps=parsed_arguments.steps,
        learning_rate=parsed_arguments.lr,
    )

    if parsed_arguments.target_rel_error is not None:
        search_space = FitSearchSpace(
            target_relative_error=parsed_arguments.target_rel_error,
        )
        fit_result = codec_instance.fit(
            weight_matrix, base_configuration, search_space
        )
        atom = fit_result.atom

        print("\nPer-configuration breakdown:")
        print(f"  {'block_size':>10}  {'K':>4}  {'MSE':>12}  {'rel_error':>12}  {'params':>10}")
        for evaluation in fit_result.evaluations:
            print(
                f"  {evaluation.config.block_size:>10}"
                f"  {evaluation.config.harmonic_count:>4}"
                f"  {evaluation.mean_squared_error:>12.4e}"
                f"  {evaluation.relative_error:>12.4e}"
                f"  {evaluation.parameter_count:>10}"
            )

        print(
            f"\nAuto-selected: block_size={fit_result.config.block_size}, "
            f"K={fit_result.config.harmonic_count} "
            f"(converged={fit_result.converged})"
        )
    else:
        atom = codec_instance.encode(weight_matrix, base_configuration)

    reconstructed_matrix = codec_instance.decode(atom)

    mean_squared_error = float(
        torch.nn.functional.mse_loss(reconstructed_matrix, weight_matrix)
    )
    relative_error = compute_relative_error(weight_matrix, reconstructed_matrix)

    dense_parameter_count = parsed_arguments.rows * parsed_arguments.cols
    parameter_ratio = atom.num_params() / dense_parameter_count

    print("\nResult:")
    print(f"  MSE             = {mean_squared_error:.6e}")
    print(f"  Relative error  = {relative_error:.6e}  (||W - Ŵ|| / ||W||)")
    print(
        f"  Parameters      = {atom.num_params()} floats "
        f"({parameter_ratio:.2f}x dense)"
    )
    print(
        f"  Config          = block_size={atom.block_size}, "
        f"K={atom.K}, num_blocks={atom.num_blocks}"
    )

    if parsed_arguments.out:
        codec_instance.save(atom, parsed_arguments.out)
        print(f"  Saved atom      -> {parsed_arguments.out}")

        reloaded_atom = codec_instance.load(parsed_arguments.out)
        reloaded_reconstruction = codec_instance.decode(reloaded_atom)
        roundtrip_max_delta = float(
            (reconstructed_matrix - reloaded_reconstruction).abs().max()
        )
        print(
            f"  Reload max |Δ|  = {roundtrip_max_delta:.3e} (serialization round-trip)"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
