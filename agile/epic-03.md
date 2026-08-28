# Epic 03: End-to-End Fidelity Gate & Parity Test

## Status
Ready

## Goal
Prove the prototype actually works and decide go/no-go. Three proofs: (1) Rust decode ==
Python decode, (2) matched filter == dense matmul, (3) zero allocations — plus the one
experiment that matters: **reconstruction relative error vs. (`block_size`, `K`)** on a
real weight tensor, which tells us whether weight *derivatives* are spectrally compact.

## User Stories
1. As a researcher, I want a fidelity-vs-(`block_size`,`K`) surface so I know whether the block-wise derivative representation can express a real weight matrix before investing in optimization.
2. As an engineer, I want automated parity and zero-alloc tests so the core claims are continuously verifiable.
3. As a researcher, I want a direct measurement of how many harmonics a real weight row's first-difference needs per block — the single number that decides if this architecture is viable.

## Acceptance Criteria
- [ ] **Parity:** for a shared `Θ`, Rust `decode_weight_scalar(r,c)` == Python `decode()[r,c]` within **1e-6** for a sample of coordinates.
- [ ] **Correctness:** `MatchedFilterLayer` with an identity generator reproduces input (dot-product sanity).
- [ ] **Zero-alloc:** `process()` runs under a tracking allocator with **0** allocations.
- [ ] **Fidelity gate:** a table of reconstruction relative error over a grid of `block_size ∈ {16, 32, 64, 128}` × `K ∈ {4, 8, 16, 32}` on a **real** `nn.Linear` weight (not just `randn`), with a stated go/no-go threshold.
- [ ] **Derivative compactness diagnostic:** a measurement of how many harmonics capture 99% of the energy of a real weight row's first-difference within a block.

## Tasks
- [ ] Rust↔Python parity test
  - Files: `python/hwave_codec.py`, `crates/phasor-synthesis/tests/parity.rs`, a shared fixture `.atom` + golden `Ŵ.npy`
  - Details: Python encodes a fixed seed tensor, writes `Θ` and the decoded `Ŵ`. Rust loads `Θ`, decodes the same coords, asserts `|rust − python| < 1e-6`. **Must use tolerance, not bit-equality** (libm vs. platform libm differ in last ULP). Note: cumsum is order-dependent, so both sides must integrate in the same order.
- [ ] Dot-product correctness test
  - Files: `crates/phasor-dsp/tests/bit_exactness.rs`
  - Details: identity-generator `MatchedFilterLayer` returns input unchanged.
- [ ] Zero-allocation audit
  - Files: `crates/phasor-dsp/tests/zero_alloc.rs`
  - Details: custom `#[global_allocator]` counting allocs; assert `process` performs 0 allocations after setup.
- [ ] Derivative compactness diagnostic
  - Files: `python/experiments/derivative_spectrum.py`
  - Details: pull one real weight matrix (e.g., a small pretrained `nn.Linear`, or a Llama/Gemma FFN slice via `safetensors`). For several rows and a given `block_size`: take the first-difference of each block, FFT it, and report (a) the number of harmonics capturing 99% of block energy, (b) the DC/near-DC energy fraction (the drift driver). **This is the viability signal.** If derivatives are broad-band (no compact harmonic fit), the gate should fail.
- [ ] Fidelity-gate experiment
  - Files: `python/experiments/fidelity_vs_k.py`
  - Details: encode the real weight at each (`block_size`, `K`) grid point, record MSE + relative error `‖W−Ŵ‖/‖W‖` + parameter count. Output a table showing the fidelity-vs-size trade-off. **This is the decision point.**
- [ ] Write the go/no-go verdict
  - Files: `agile/verdict.md`
  - Details: state the best achievable relative error and the (`block_size`, `K`) it required, plus the parameter count vs. dense. If relative error is too high at every tractable config, the next phase is Path B (distillation) — **not** Rust optimization.

## Shared-column ordering experiment

A reversible PCA-based shared-column order is implemented in
`src/encoder/column_ordering.py` and persisted once per tensor in `.atom` v3. Decode
restores original column positions automatically. The stress test counts the permutation
payload (`4 * cols` bytes) in all size ratios.

On two real Pythia-14m matrices with `block_size=128`, ordering did **not** make the
low-budget derivative spectrum easier to encode:

| tensor | K | baseline rel. error | PCA-ordered rel. error |
|---|---:|---:|---:|
| attention QKV `(384,128)` | 8 | 1.2707 | 1.3566 |
| attention QKV `(384,128)` | 32 | 1.0770 | 1.1951 |
| MLP down `(128,512)` | 8 | 1.2031 | 1.2622 |
| MLP down `(128,512)` | 32 | 0.9699 | 0.9969 |

Full-spectrum reconstruction remains intact (`2.75e-5`–`2.89e-5` relative error for
ordered tensors), proving sort/unsort and serialization correctness. The negative low-K
result means shared PCA ordering is available as an experimental transform but does not
change the direct-codec go/no-go signal; behavioral distillation remains the next path.

## Gate Outcome
- **Pass (relative error ≤ agreed threshold at tractable `block_size`/`K` with real parameter savings):** proceed to a Phase-2 epic — multi-layer demo, block-parallel decode, SIMD.
- **Fail (derivatives broad-band, no compact fit):** pivot the *encoding* (Path B distillation, which fits `Θ` to behavior rather than values) before any further Rust perf work. The block integrator decode path is retained regardless — only the `Θ`-fitting objective changes.
