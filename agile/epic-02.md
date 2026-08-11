# Epic 02: Rust Hot-Path Decoder + Zero-Alloc Matched Filter

## Status
Ready

## Goal
A Rust workspace with a `PhasorAtomIter<K>` that procedurally synthesizes weight elements
from a `.atom` file and a `MatchedFilterLayer` that runs the forward pass with **zero heap
allocations** in the hot loop.

## User Stories
1. As a systems engineer, I want a Rust decoder that loads a `.atom` file and synthesizes weights on demand so no dense matrix is ever stored.
2. As a performance engineer, I want the forward pass to make zero heap allocations so the L1-residency claim is testable.

## Acceptance Criteria
- [ ] Workspace compiles on **stable** Rust (`cargo build` succeeds).
- [ ] `PhasorAtom<K>` + `WeightGenerator` trait in `phasor-synthesis`.
- [ ] `MatchedFilterLayer`, `HalfWaveRectifier`, `AutomaticGainControl` in `phasor-dsp`.
- [ ] `decode_weight_scalar` numerically matches Python `decode()` per-element (verified in Epic 03).
- [ ] `process()` performs **0 heap allocations** (verified in Epic 03).

## Tasks
- [ ] Create the workspace skeleton
  - Files: `Cargo.toml`, `crates/phasor-synthesis/Cargo.toml`, `crates/phasor-dsp/Cargo.toml`, `crates/phasor-inference/Cargo.toml`
  - Details: `resolver = "2"`, edition 2021. **Remove the phantom `simdno` dep.**
- [ ] Implement `PhasorAtom<K>` in `phasor-synthesis`
  - Files: `crates/phasor-synthesis/src/lib.rs`
  - Details: `#[repr(C)] struct PhasorAtom<const K: usize> { amplitudes: [f32; K], frequencies: [f32; K], angles: [f32; K], phases: [f32; K], decay: f32 }`. `impl WeightGenerator for PhasorAtom<K>` with `fn weight_at(&self, r: f32, c: f32) -> f32`.
- [ ] Implement `.atom` file loader
  - Files: `crates/phasor-synthesis/src/loader.rs`
  - Details: Read binary header (`tile_dim`, `K`), then `4K+1` f32 values. Return `PhasorAtom<K>`.
- [ ] Implement `MatchedFilterLayer` in `phasor-dsp`
  - Files: `crates/phasor-dsp/src/lib.rs`
  - Details: `process(&mut self, input: &[f32], output: &mut [f32])` — iterates over (r, c) grid, calls `weight_at`, accumulates `output[r] += input[c] * w`. **No `Vec` in the hot path.**
- [ ] Implement `HalfWaveRectifier` + `AutomaticGainControl`
  - Files: `crates/phasor-dsp/src/lib.rs`
  - Details: Simple `f32` transforms, no state beyond struct fields.
- [ ] Write `bench_decode` benchmark
  - Files: `crates/phasor-synthesis/benches/bench_decode.rs`
  - Details: Criterion benchmark measuring decode throughput (elements/sec) for K=16.
- [ ] Add `zero_alloc` test harness
  - Files: `crates/phasor-dsp/tests/zero_alloc.rs`
  - Details: Use `#[global_allocator]` counter to assert 0 allocations during `process()`.
