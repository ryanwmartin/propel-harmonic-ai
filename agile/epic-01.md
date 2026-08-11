# Epic 01: Python Reference Codec (Encoder + Decoder)

## Status
Ready

## Goal
A runnable `HWaveCodec` that encodes a 128×128 weight tensor `W` into a Phasor Atom
`Θ = {A_k, f_k, θ_k, φ_k, γ}` via 2D-FFT warm-start + Adam refinement, and decodes `Θ`
back into `Ŵ` procedurally. This is the **source of truth** the Rust decoder must match.

## User Stories
1. As a researcher, I want to encode any 128×128 tensor into ~`4K+1` floats so I can quantify the fidelity vs. K trade-off.
2. As a systems engineer, I want a deterministic decode function so the Rust port has an exact reference to test against.

## Acceptance Criteria
- [ ] `HWaveCodec.encode()` + `HWaveCodec.decode()` round-trip a 128×128 synthetic tensor
- [ ] MSE and relative error printed for every encode/decode cycle
- [ ] CLI runs: `python -m src.codec --dim 128 --K 16 --steps 500`
- [ ] `Θ` serialized to disk as `.atom` binary file

## Tasks
- [ ] Create Python project skeleton
  - Files: `requirements.txt`, `src/__init__.py`
  - Details: `numpy`, `scipy`, `torch`, `matplotlib` in requirements
- [ ] Implement `PhasorAtom` parameter container
  - Files: `src/phasor_atom.py`
  - Details: `@dataclass` holding `amplitudes`, `frequencies`, `angles`, `phases`, `decay` as `torch.Tensor`
- [ ] Implement `G(Θ, r, c)` evaluation function
  - Files: `src/phasor_atom.py`
  - Details: Given coordinate grids `r, c ∈ [-1,1]`, compute `Σ_k A_k · sin(2π·f_k·(r·cos θ_k + c·sin θ_k) + φ_k) · exp(-γ·(|r|+|c|))`
- [ ] Implement FFT warm-start encoder
  - Files: `src/encoder.py`
  - Details: 2D-FFT of `W` → extract top-K peaks → map to `(A_k, f_k, θ_k, φ_k)` initial guess
- [ ] Implement Adam refinement loop
  - Files: `src/encoder.py`
  - Details: `loss = MSE(G(Θ, r, c), W)`; optimize all Θ parameters jointly
- [ ] Implement `HWaveCodec` facade
  - Files: `src/codec.py`
  - Details: `.encode(W, K, steps) → Θ`; `.decode(Θ, dim) → Ŵ`; `.save(Θ, path)` / `.load(path) → Θ`
- [ ] Add serialization to `.atom` binary format
  - Files: `src/codec.py`
  - Details: Flat `f32` array: header (`tile_dim`, `K`) then `4K+1` values
- [ ] CLI entry point
  - Files: `src/__main__.py`
  - Details: `--dim 128 --K 16 --steps 500` → encode synthetic tensor, decode, print MSE + relative error
