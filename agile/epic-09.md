# Epic 09: Tokenizer + KV Cache + Autoregressive Generation Loop

## Status
Draft

## Goal
Add the remaining pieces needed for real text generation: a tokenizer to convert text ↔
token IDs, a KV cache for efficient autoregressive decoding, and a sampling loop that
turns logits into generated tokens one at a time.

> **Note:** This epic is codec-agnostic — it operates on logits and token IDs, not on the
> block-wise `Θ` weights. It is unchanged by the architecture pivot except that the model
> it drives decodes weights via the block integrator (Epic 08).

## User Stories
1. As a user, I want to type a prompt and get a token-by-token response so I can interact with the model conversationally.
2. As a systems engineer, I want a KV cache so each new token only requires one forward pass (not recomputing the full sequence).
3. As a researcher, I want configurable sampling (temperature, top-k, top-p) so I can control generation quality.

## Acceptance Criteria
- [ ] Gemma tokenizer loads from the HuggingFace `tokenizer.json` (via `tokenizers` crate) and encodes/decodes text correctly.
- [ ] `KvCache` struct pre-allocates `(num_layers, num_heads, max_seq_len, head_dim)` for K and V. Each forward pass at position `t` appends to the cache and attends over `0..=t`.
- [ ] `generate()` loop: encode prompt → forward each prompt token (prefill) → sample next token → forward single token (decode) → repeat until EOS or `max_tokens`.
- [ ] Sampling supports: greedy (argmax), temperature scaling, top-k, top-p (nucleus). Configurable at runtime.
- [ ] Generated text matches a PyTorch reference generation for the same prompt + greedy decoding (validates the full pipeline end-to-end).
- [ ] All KV cache and sampling buffers are pre-allocated; the generation loop allocates **0 heap allocations** after setup.

## Tasks
- [ ] Integrate `tokenizers` crate
  - Files: `crates/phasor-inference/Cargo.toml`, `crates/phasor-inference/src/tokenizer.rs`
  - Details: add `tokenizers` dep (with `onig` or `unstable_wasm` disabled for simplicity). Wrap `Tokenizer::from_file("tokenizer.json")` with `encode(text) -> Vec<u32>` and `decode(ids) -> String`. The tokenizer file comes from the Gemma 2B HF snapshot downloaded in Epic 07.
- [ ] Implement `KvCache`
  - Files: `crates/phasor-inference/src/kv_cache.rs`
  - Details: struct with pre-allocated `Vec<f32>` for K and V per layer, shaped `(num_layers, 2, max_seq_len, num_kv_heads * head_dim)`. Methods: `append(layer, k, v, pos)`, `get_k(layer, up_to_pos)`, `get_v(layer, up_to_pos)`. Handles GQA (Gemma uses grouped-query attention — fewer KV heads than Q heads).
- [ ] Wire KV cache into attention
  - Files: `crates/phasor-inference/src/attention.rs`
  - Details: modify `Attention::forward` to accept `&mut KvCache` and `pos: usize`. After computing QKV, append K/V to cache. Compute attention scores only against cached positions `0..=pos`. This replaces the Epic 08 forward which was single-position-only.
- [ ] Implement RoPE (Rotary Position Embedding)
  - Files: `crates/phasor-inference/src/rope.rs`
  - Details: precompute `cos`/`sin` tables at model load for `max_seq_len` positions. Apply rotation to Q and K vectors in-place before caching. Gemma uses `theta = 10000` (verify from config).
- [ ] Implement sampling strategies
  - Files: `crates/phasor-inference/src/sampling.rs`
  - Details: `fn sample_greedy(logits: &[f32]) -> u32`, `fn sample_temperature(logits: &[f32], temp: f32, rng: &mut impl Rng) -> u32`, `fn sample_top_k(logits, k, temp, rng)`, `fn sample_top_p(logits, p, temp, rng)`. Use a seeded RNG for reproducibility.
- [ ] Implement `generate()` loop
  - Files: `crates/phasor-inference/src/generate.rs`
  - Details: `fn generate(model, tokenizer, prompt, config) -> String`. Prefill: forward each prompt token, fill KV cache. Decode: loop sampling + single-token forward. Yield each decoded token string via callback (for streaming). Stop on EOS token ID or `max_tokens`.
- [ ] Greedy generation parity test
  - Files: `python/parity_generate.py`, `crates/phasor-inference/tests/generate_parity.rs`
  - Details: PyTorch: load dense Gemma 2B, greedy-generate 20 tokens from a fixed prompt, save token IDs. Rust: load phasor model, greedy-generate 20 tokens from the same prompt, assert token IDs match (allowing divergence after N tokens due to accumulated float error — assert first 5 tokens match exactly).

## Dependencies
- Epic 08 (full forward pass)
