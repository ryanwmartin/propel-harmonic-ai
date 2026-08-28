# Epic 07: Full-Model Encoder — Encode Every Weight of Gemma 2B

## Status
Draft

## Goal
Encode **every weight tensor** of Gemma 2B into a manifest of `.atom` files using the
block-wise derivative codec. This epic has two encoding sources depending on the path
taken:
- **Direct codec path** (if Epic 03 gate passed): encode each weight matrix with `HWaveCodec` (per-row, per-block derivative fitting).
- **Distillation path** (if Epics 04–06 were executed): Stage 2 `.atom` files are already in the correct format — this step handles any layers not covered by distillation (embeddings, etc.) and assembles the unified manifest.

Either way, the output is a `model_atoms/` directory of `Θ` blobs plus a JSON manifest
that the Rust inference engine loads — no `.safetensors` or dense weights needed at
inference time.

## User Stories
1. As a researcher, I want to encode all ~2B parameters of Gemma into block-wise `Θ` sets so I can measure total model size vs. fidelity across the whole network.
2. As a systems engineer, I want a deterministic, resumable encoding pipeline so I can re-encode individual layers without redoing the full model.
3. As a systems engineer, I want a manifest format that maps (layer, role) → atom file so the Rust engine can look up any weight on demand.

## Acceptance Criteria
- [ ] Downloads Gemma 2B weights (via `huggingface_hub` / `safetensors`) and iterates all 2D weight matrices (attention QKV/O, FFN up/down/gate, embeddings).
- [ ] **Direct path:** each matrix is encoded row-wise: each row split into `num_blocks = ceil(cols / block_size)` blocks, each block stored as an anchor + K-harmonic derivative fit.
- [ ] **Distillation path:** Stage 2 `.atom` files are loaded directly (same format — no conversion). For layers not covered by distillation (embeddings, non-target roles), encode via the direct codec path. The manifest merges both sources.
- [ ] `block_size` and `K` per layer are configurable — allow per-layer-type profiles (e.g., smaller `block_size` / higher `K` for attention, larger for FFN) via a config file.
- [ ] Output is a `model_atoms/` directory: one `.atom` file per weight matrix, plus `manifest.json` mapping `(layer_idx, matrix_role)` → `{file, block_size, K, num_blocks, original_shape, mse, source}`.
- [ ] Pipeline is resumable: skips already-encoded matrices, logs per-layer MSE and total compression ratio.
- [ ] Total wall-clock encoding time and per-layer fidelity are reported.

## Tasks
- [ ] Add model download + weight iteration
  - Files: `python/encode_model.py`
  - Details: use `huggingface_hub.snapshot_download` for Gemma 2B, load with `safetensors`, iterate named 2D tensors. Skip biases and 1D tensors (norms, etc.) — those stay dense (small enough to not matter).
- [ ] Implement block-wise encoding loop (direct codec path)
  - Files: `python/encode_model.py`
  - Details: for each weight matrix `W` of shape `(rows, cols)`, split each row into `ceil(cols / block_size)` blocks. Encode each block (anchor + derivative harmonic fit) with `HWaveCodec` at the configured `K`. Track per-matrix MSE. Edge blocks shorter than `block_size` are handled per the Epic 01 contract.
- [ ] Implement distillation path — merge Stage 2 `.atom` files into unified manifest
  - Files: `python/encode_model.py`
  - Details: load `stage2_manifest.json` from Epic 06. For each `(layer_idx, role)` entry, copy the `.atom` file into `model_atoms/` and add a manifest entry with `source: "distilled"`. For layers not covered by distillation (embeddings, etc.), fall through to the direct codec path with `source: "direct"`. Both sources produce the same `.atom` binary format — no conversion needed.
- [ ] Per-layer `block_size`/`K` configuration
  - Files: `python/encode_config.py` (or a JSON/TOML config)
  - Details: allow specifying `block_size` and `K` per matrix role (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, embed_tokens). Default: uniform `block_size=32`, `K=8` (direct) or values from the Stage 1/2 manifest (distillation).
- [ ] Manifest writer
  - Files: `python/encode_model.py`
  - Details: write `manifest.json` with schema: `{model_name, entries: [{layer_idx, role, file, block_size, K, num_blocks, original_shape, mse, rel_error, source: "direct"|"distilled"}]}`. Also store 1D tensors (layernorm weights, biases) as raw `f32` blobs referenced in the manifest.
- [ ] Resumability + progress logging
  - Files: `python/encode_model.py`
  - Details: check for existing `.atom` files before encoding; skip if present. Use `tqdm` for progress. Log per-layer aggregate MSE and running compression ratio to stdout and an `encoding_report.json`.
- [ ] Round-trip validation harness
  - Files: `python/validate_encoding.py`
  - Details: loads the manifest, decodes every matrix (oscillator + prefix-sum per block), reconstructs each full weight matrix, compares to the original `safetensors` tensor. Reports per-layer and global relative error. This is the smoke test before Rust inference.

## Dependencies
- Epic 01 (codec) must be complete — for the direct path
- Epic 03 (fidelity gate) must have passed — `block_size`/`K` per layer is informed by the gate results
- **OR** Epics 04–06 (distillation) must have passed — `.atom` files are the encoding source
