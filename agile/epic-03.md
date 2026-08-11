# Epic 03: End-to-End Fidelity Gate & Parity Test

## Status
Ready

## Goal
Prove the prototype actually works and decide go/no-go. Three proofs: (1) Rust decode ==
Python decode, (2) matched filter == dense matmul, (3) zero allocations — plus the one
experiment that matters: **reconstruction MSE vs. K** on a real weight tensor.

## User Stories
1. As a researcher, I want a fidelity-vs-K curve so I know whether the HWave representation can express a real weight matrix before investing in optimization.
2. As an engineer, I want automated parity and zero-alloc tests so the core claims are continuously verifiable.

## Acceptance Criteria
- [ ] **Parity:** for a shared `Θ`, Rust `decode_weight_scalar(r,c)` == Python `decode()[r,c]` within **1e-6** for a sample of coordinates.
- [ ] **Correctness:** `MatchedFilterLayer` with an identity generator reproduces input (dot-product sanity).
- [ ] **Zero-alloc:** `process()` runs under a tracking allocator with **0** allocations.
- [ ] **Fidelity gate:** a plot/table of reconstruction MSE vs. K ∈ {8,16,32,64,128} on a **real** `nn.Linear` weight (not just `randn`), with a stated go/no-go threshold.

## Tasks
- [ ] Rust↔Python parity test
  - Files: `python/hwave_codec.py`, `crates/phasor-synthesis/tests/parity.rs`, a shared fixture `.atom` + golden `Ŵ.npy`
  - Details: Python encodes a fixed seed tensor, writes `Θ` and the decoded `Ŵ`. Rust loads `Θ`, decodes the same coords, asserts `|rust − python| < 1e-6`. **Must use tolerance, not bit-equality** (libm vs. platform libm differ in last ULP).
- [ ] Dot-product correctness test
  - Files: `crates/phasor-dsp/tests/bit_exactness.rs`
  - Details: identity-generator `MatchedFilterLayer` returns input unchanged (the existing v2 test is fine).
- [ ] Zero-allocation audit
  - Files: `crates/phasor-dsp/tests/zero_alloc.rs`
  - Details: custom `#[global_allocator]` counting allocs; assert `PhasorMlp::forward` (or `process`) performs 0 allocations after setup.
- [ ] Fidelity-gate experiment
  - Files: `python/experiments/fidelity_vs_k.py`
  - Details: pull one real weight matrix (e.g., a small pretrained `nn.Linear`, or a Llama/Gemma FFN slice via `safetensors`), tile to 128×128, encode at each K, record MSE + relative error `‖W−Ŵ‖/‖W‖`. Output a table. **This is the decision point.**
- [ ] Write the go/no-go verdict
  - Files: `agile/verdict.md`
  - Details: state the best achievable relative error and the K it required. If relative error is too high at every tractable K, the next phase is Path B (distillation) or a basis change (independent `f_r, f_c`), **not** Rust optimization.

## Gate Outcome
- **Pass (relative error ≤ agreed threshold at tractable K):** proceed to a Phase-2 epic — oscillator-bank recurrence, SIMD, multi-layer demo.
- **Fail:** pivot the *encoding* (Path B distillation, or replace the directional basis with independent `(f_r, f_c)` per harmonic) before any further Rust perf work.
