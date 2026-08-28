# Epic 02: Rust Hot-Path Decoder + Zero-Alloc Block Integrator

## Status
Ready

## Goal
A Rust workspace with a `BlockAtomIter` that procedurally synthesizes weight elements from
a `.atom` file by running a per-block oscillator and integrating (prefix-sum) within each
block, and a `MatchedFilterLayer` that runs the forward pass with **zero heap
allocations** in the hot loop. Because each block is independent (anchored), decode is
embarrassingly parallel across blocks and rows.

## User Stories
1. As a systems engineer, I want a Rust decoder that loads a `.atom` file and synthesizes weights on demand (oscillator + prefix-sum) so no dense matrix is ever stored.
2. As a performance engineer, I want the forward pass to make zero heap allocations so the L1-residency claim is testable.
3. As a performance engineer, I want per-block decode to be independent so I can parallelize across blocks/rows without coordination.

## Acceptance Criteria
- [ ] Workspace compiles on **stable** Rust (`cargo build` succeeds).
- [ ] `BlockAtom` + `WeightGenerator` trait in `phasor-synthesis`.
- [ ] `MatchedFilterLayer`, `HalfWaveRectifier`, `AutomaticGainControl` in `phasor-dsp`.
- [ ] `decode_weight_scalar` numerically matches Python `decode()` per-element (verified in Epic 03).
- [ ] `process()` performs **0 heap allocations** (verified in Epic 03).

## Tasks
- [ ] Create the workspace skeleton
  - Files: `Cargo.toml`, `crates/phasor-synthesis/Cargo.toml`, `crates/phasor-dsp/Cargo.toml`, `crates/phasor-inference/Cargo.toml`
  - Details: `resolver = "2"`, edition 2021. **Remove the phantom `simdno` dep.**
- [ ] Implement `BlockAtom` in `phasor-synthesis`
  - Files: `crates/phasor-synthesis/src/lib.rs`
  - Details: `struct BlockAtom { anchor: f32, amplitudes: Vec<f32>, frequencies: Vec<f32>, phases: Vec<f32> }` (per-block, K harmonics). A `TensorAtom` holds `rows`, `cols`, `block_size`, `num_blocks`, and `Vec<BlockAtom>` in row-major block order. `impl WeightGenerator for TensorAtom` with `fn weight_at(&self, r: usize, c: usize) -> f32`.
- [ ] Implement the block integrator (decode hot path)
  - Files: `crates/phasor-synthesis/src/lib.rs`
  - Details: `weight_at(r, c)` locates `(block, x)` from `c / block_size`, `c % block_size`, then computes `anchor + Σ_{i<x} G(block, i)` where `G(block, i) = Σ_k A_k·sin(2πf_k·i + φ_k)`. Provide `decode_block(row, block, out: &mut [f32])` that fills a block via running prefix-sum (one accumulator, no per-element re-summation). **No `Vec` allocation in the per-element path.**
- [ ] Implement `.atom` file loader
  - Files: `crates/phasor-synthesis/src/loader.rs`
  - Details: Read binary header (`magic`, `version=2`, `K`, `block_size`, `num_blocks`, `rows`, `cols`), then per row/block the anchor + `3K` f32 values. Return a `TensorAtom`.
- [ ] Implement `MatchedFilterLayer` in `phasor-dsp`
  - Files: `crates/phasor-dsp/src/lib.rs`
  - Details: `process(&mut self, input: &[f32], output: &mut [f32])` — iterates over (r, c), calls `weight_at`, accumulates `output[r] += input[c] * w`. Prefer block-contiguous iteration (decode a block into a small stack buffer, then matmul against it) to amortize the prefix-sum. **No `Vec` in the hot path.**
- [ ] Implement `HalfWaveRectifier` + `AutomaticGainControl`
  - Files: `crates/phasor-dsp/src/lib.rs`
  - Details: Simple `f32` transforms, no state beyond struct fields.
- [ ] Write `bench_decode` benchmark
  - Files: `crates/phasor-synthesis/benches/bench_decode.rs`
  - Details: Criterion benchmark measuring decode throughput (elements/sec) for a representative `block_size`/`K`. Report both per-element `weight_at` and block-wise `decode_block` throughput.
- [ ] Add `zero_alloc` test harness
  - Files: `crates/phasor-dsp/tests/zero_alloc.rs`
  - Details: Use `#[global_allocator]` counter to assert 0 allocations during `process()`.

## Notes
- **Decode cost per weight:** one block decode is `O(block_size · K)` `sin` calls but amortizes to `O(K)` per weight via the running prefix-sum. Per-element random access `weight_at` is `O(x·K)` within a block (must re-integrate from the anchor) — so the matmul should always iterate block-contiguously, never random-access individual weights.
- **Parallelism:** blocks are independent (each anchored), so `decode_block` calls for different blocks can run on separate threads with no shared state. This is a decode-contract advantage over any global-cumsum scheme.
