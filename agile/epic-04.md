# Epic 04: Distillation Data Pipeline — Activation Extraction & Calibration

## Status
Draft

## Goal
Build the data infrastructure that feeds the distillation pipeline: stream calibration text
from a large corpus, run it through the teacher model (dense Gemma 2B), and cache the
input/output activation pairs `(X, Y)` for every target weight layer. The output is an
on-disk activation store that Stage 1 (Epic 05) consumes.

## User Stories
1. As a researcher, I want a streaming data loader that pulls tokenized sequences from FineWeb-Edu so I don't need to download a full dataset.
2. As a researcher, I want activation hooks on every FFN and attention projection layer so I can capture exactly what the teacher computes at each matrix multiply.
3. As an engineer, I want the cached activations stored as memory-mappable files so Stage 1 can load them without re-running the teacher.

## Acceptance Criteria
- [ ] Streams text from `HuggingFaceFW/fineweb-edu` (sample-10BT) via `datasets` streaming API.
- [ ] Tokenizes with the Gemma 2B tokenizer, producing sequences of shape `(batch, 512)` for Stage 1 calibration.
- [ ] Collects **1,000 sequences of 512 tokens** (~512K token activations) for Stage 1.
- [ ] `ActivationCatcher` class registers `forward_hook` on every target layer, captures `(input, output)` pairs, detaches to CPU.
- [ ] Targets **all** 2D projection layers: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` for every transformer block. Embedding layer is excluded (handled separately).
- [ ] Saves activations to disk as `.npy` or `.safetensors` files with a manifest mapping `(layer_idx, role)` → file path.
- [ ] Data loading and caching are separate scripts — cache once, distill many times.

## Tasks
- [ ] Implement `src/data_loader.py` — streaming calibration data loader
  - Files: `src/data_loader.py`
  - Details: `get_calibration_dataset(model_name, num_samples=1000, seq_len=512) -> Tensor[N, 512]`. Stream from `HuggingFaceFW/fineweb-edu` (name=`sample-10BT`), tokenize with `AutoTokenizer.from_pretrained(model_name)`, truncate/pad to `seq_len`, stop after `num_samples` complete sequences. Return as a single `torch.Tensor`.
- [ ] Implement `src/cacher.py` — activation hook & caching engine
  - Files: `src/cacher.py`
  - Details: `ActivationCatcher` class wrapping `register_forward_hook`. On each forward pass, append `input[0].detach().cpu()` and `output.detach().cpu()` to per-layer lists. `close()` removes hooks. Provide `cache_all_activations(model, tokenizer, dataset, output_dir)` that iterates the model's named modules, attaches catchers to every `nn.Linear` inside transformer blocks, runs the full calibration dataset through the model in eval mode (no grad), and saves results.
- [ ] Define the layer-targeting manifest
  - Files: `src/cacher.py`
  - Details: `TARGET_ROLES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`. Walk `model.model.layers[i].self_attn.{role}` and `model.model.layers[i].mlp.{role}`. Record `(layer_idx, role, in_dim, out_dim)` for each target.
- [ ] Save/load activation store
  - Files: `src/cacher.py`
  - Details: save to `data/activations/layer_{i:02d}_{role}_X.npy` and `layer_{i:02d}_{role}_Y.npy`. Write `data/activations/manifest.json` with per-layer shapes, dtypes, and file paths. Implement `load_activation_pair(layer_idx, role) -> (Tensor, Tensor)`.
- [ ] CLI entry point
  - Files: `src/cache_activations.py`
  - Details: `python -m src.cache_activations --model google/gemma-2b --num-samples 1000 --seq-len 512 --output data/activations/`. Uses `data_loader.py` + `cacher.py`. Prints progress and summary of cached layers/shapes.

## Dependencies
- None (runs against dense teacher model, no phasor code needed)

## Notes
- Teacher model runs in `torch.no_grad()` + `model.eval()` mode — no gradient computation needed during caching.
- Keep activations in `float32` to avoid precision loss during Stage 1 optimization.
- For Gemma 2B: 18 layers × 7 projections = 126 activation pairs. Total disk usage ≈ 126 × 2 × (1000 × 512 × dim × 4 bytes). Manage by batching if needed.
