# Epic 08: Rust LLM Forward Pass — Attention + FFN via Phasor Decode

## Status
Draft

## Goal
Build the full transformer forward pass in Rust where every weight matrix is synthesized
on-the-fly from block-wise `Θ` sets (oscillator + prefix-sum per block). No dense weight
matrix is ever fully materialized — the `WeightGenerator` trait from Epic 02 is the only
path from `Θ` to a scalar weight value.

## User Stories
1. As a systems engineer, I want a `TransformerBlock` that runs attention and FFN using only block-wise `Θ` weights so no dense matrix is ever stored.
2. As a performance engineer, I want per-layer buffer pre-allocation so the forward pass allocates zero memory after model load.
3. As a researcher, I want the Rust forward pass to match a PyTorch reference forward pass on the same (dense) weights within tolerance so I know the phasor decode isn't introducing errors beyond the codec MSE.

## Acceptance Criteria
- [ ] `PhasorWeightMatrix` struct that wraps a `TensorAtom` (per-row, per-block anchors + harmonics) and implements `weight_at(row, col) -> f32` via the block integrator.
- [ ] Full `TransformerBlock`: RMSNorm, multi-head attention (QKV via phasor matmul, scaled dot-product, output projection), FFN (gate/up/down via phasor matmul), residual connections.
- [ ] All matmuls use a `phasor_matmul(input, weight_matrix, output)` function that decodes block-contiguously (never random-access) — no dense `W` buffer.
- [ ] All intermediate buffers (activations, attention scores, FFN hidden) are pre-allocated at model load; the forward pass performs **0 heap allocations**.
- [ ] **Parity test:** load Gemma 2B dense weights in PyTorch, run a single transformer block forward pass on a fixed input; run the same block in Rust with phasor-encoded weights; assert outputs match within `1e-3` (looser than codec parity because we stack codec error + float assoc).
- [ ] Model loads from the `manifest.json` + `.atom` directory produced by Epic 07.

## Tasks
- [ ] Implement `PhasorWeightMatrix`
  - Files: `crates/phasor-synthesis/src/matrix.rs`
  - Details: holds a `TensorAtom` (`rows`, `cols`, `block_size`, `num_blocks`, `Vec<BlockAtom>`). `weight_at(r, c)` locates the block and integrates from its anchor. Provide `decode_block_into(row, block, out: &mut [f32])` for amortized block-contiguous decode. This is the bridge between the codec and the matmul.
- [ ] Implement `phasor_matmul`
  - Files: `crates/phasor-dsp/src/matmul.rs`
  - Details: `fn phasor_matmul(input: &[f32], weight: &PhasorWeightMatrix, output: &mut [f32])`. Iterate block-contiguously: for each row, decode each block into a small stack buffer via running prefix-sum, then accumulate `output[row] += Σ input[block_range] · block_weights`. Caller supplies buffers. No allocation, no per-element re-integration.
- [ ] Implement RMSNorm
  - Files: `crates/phasor-dsp/src/norm.rs`
  - Details: Gemma uses RMSNorm. `fn rms_norm(x: &mut [f32], weight: &[f32], eps: f32)` in-place. Norm weights are 1D dense (loaded from manifest sidecar).
- [ ] Implement attention forward
  - Files: `crates/phasor-inference/src/attention.rs`
  - Details: `Attention::forward(&self, x: &[f32], kv_cache: &mut KvCache, pos: usize, output: &mut [f32])`. QKV projections via `phasor_matmul`, RoPE positional encoding (precomputed sin/cos tables), scaled dot-product attention over cached KV, output projection via `phasor_matmul`. All buffers pre-allocated.
- [ ] Implement FFN forward
  - Files: `crates/phasor-inference/src/ffn.rs`
  - Details: `Ffn::forward(&self, x: &[f32], output: &mut [f32])`. gate_proj → GELU → multiply with up_proj → down_proj. All via `phasor_matmul`. Pre-allocated hidden buffer.
- [ ] Implement `TransformerBlock`
  - Files: `crates/phasor-inference/src/block.rs`
  - Details: pre-norm architecture: `x += attention(rms_norm(x))`, `x += ffn(rms_norm(x))`. Wires attention + FFN + norms together.
- [ ] Implement `Model::forward` (single token)
  - Files: `crates/phasor-inference/src/model.rs`
  - Details: embedding lookup (embedding matrix is encoded as block-wise `Θ` too), then N × `TransformerBlock::forward`, final RMSNorm, logits via `phasor_matmul` against the output projection (or tied embedding transpose). Returns `&[f32]` logits slice.
- [ ] PyTorch reference parity test
  - Files: `python/parity_forward.py`, `crates/phasor-inference/tests/forward_parity.rs`
  - Details: PyTorch loads dense Gemma 2B, extracts one block, runs forward on a fixed `randn` input, saves input + output as `.npy`. Rust loads the same block's phasor weights, runs forward, asserts `max_abs_diff < 1e-3`. Uses the Epic 07 encoding for the block weights.

## Dependencies
- Epic 02 (Rust decoder, zero-alloc DSP primitives)
- Epic 07 (encoded model directory + manifest)
