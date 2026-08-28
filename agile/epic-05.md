# Epic 05: Stage 1 — Layer-Wise Activation Alignment Distillation

## Status
Draft

## Goal
Fit block-wise `Θ` parameters (per-row, per-block anchors + harmonic sets) for **every
target weight layer** by minimizing the MSE between teacher activations `Y_teacher` and
student activations `Y_student = X @ W(Θ).T`, where `W(Θ)` is synthesized by the block
integrator (oscillator + prefix-sum). This is a per-layer, embarrassingly parallel
optimization that produces a `.atom` file for each weight matrix.

Unlike Epic 01's direct codec (which fits `Θ` to weight *derivative values*), Stage 1 fits
`Θ` to match *behavior* — the layer's input→output mapping. This is a smoother objective
and is the primary encoding path when the direct codec's fidelity gate (Epic 03) is
insufficient.

## User Stories
1. As a researcher, I want to fit each layer independently so I can parallelize across CPU cores / GPUs.
2. As a researcher, I want per-layer MSE reporting so I can identify which layers are hardest to distill.
3. As an engineer, I want the output saved as `.atom` files with a manifest so Stage 2 can load them into a student model.

## Acceptance Criteria
- [ ] `distill_single_layer(X_cache, Y_teacher, block_size, K)` optimizes `Θ` via Adam to minimize `MSE(X @ W(Θ).T, Y)`, where `W(Θ)` is the block-integrator synthesis.
- [ ] Processes all 126 layer/role pairs from the activation store.
- [ ] Per-layer convergence: MSE < 5.0 × 10⁻⁴ target threshold while serialized state remains within the assigned RAM budget. Layers that fail are retried across `K` and `block_size` without silently exceeding that budget.
- [ ] Primary state budget: serialized procedural state plus mandatory inference scratch is no more than 50% of the corresponding dense FP16/BF16 layer footprint. For an FP16 weight-only comparison, 8 bits per represented dense weight is the nominal ceiling before scratch and metadata.
- [ ] Output: one `.atom` file per layer (anchors + per-block harmonics), plus `stage1_manifest.json` mapping `(layer_idx, role)` → `{file, block_size, K, final_mse, state_bytes, dense_baseline_bytes, ram_ratio, steps, wall_time}`.
- [ ] Parallel execution: use `multiprocessing.Pool` or `concurrent.futures` to run multiple layers concurrently.
- [ ] Deterministic: fixed random seed, same input → same output.

## Tasks
- [ ] Implement `src/phasor_atom.py` — the differentiable block-integrator synthesis
  - Files: `src/phasor_atom.py`
  - Details: `BlockAtomLayer(nn.Module)` with parameters `anchors(rows, num_blocks)`, `amplitudes(rows, num_blocks, K)`, `frequencies(rows, num_blocks, K)`, `phases(rows, num_blocks, K)`. Method `synthesize_weight() -> Tensor[rows, cols]` runs the oscillator per block and `torch.cumsum` within each block (using the stored anchor as the block's initial value). Method `forward(x) -> x @ synthesize_weight().T`. The cumsum must be differentiable (it is — `torch.cumsum` has a well-defined gradient).
- [ ] Implement `src/stage1_distill.py` — single-layer distillation loop
  - Files: `src/stage1_distill.py`
  - Details: `distill_single_layer(X_cache, Y_teacher, block_size=32, K=8, lr=1e-2, steps=500) -> dict`. Initialize `Θ` (warm-start anchors from the true block-first-weights if available, else small random init; harmonics small random), optimize with Adam, track MSE. Return the `Θ` dict + `final_mse`. Support GPU if available.
- [ ] Implement the full-model orchestrator
  - Files: `src/stage1_distill.py`
  - Details: `run_stage1(activation_dir, output_dir, block_size_profile, K_profile, max_workers)`. Load `manifest.json` from Epic 04, iterate all `(layer_idx, role)` pairs, call `distill_single_layer` for each. Use `concurrent.futures.ProcessPoolExecutor` for parallelism. Log per-layer results. Save `.atom` files + manifest.
- [ ] Implement adaptive retry on `K` and `block_size`
  - Files: `src/stage1_distill.py`
  - Details: if `final_mse > 5e-4` after initial run, retry with `K *= 2` (up to `K_max`) and/or `block_size //= 2` (more anchors, less drift). Record the escalation in the manifest. This handles layers where the default config is insufficient.
- [ ] Save/load `.atom` format (shared with Epic 01 codec)
  - Files: `src/phasor_atom.py`
  - Details: reuse the **same** `save_atom(theta, path)` / `load_atom(path)` binary format defined in Epic 01 — header (`K`, `block_size`, `num_blocks`, `rows`, `cols`) then per row/block anchor + `3K` `f32` values. One format, one extension, used by both the direct codec and the distillation pipeline. No conversion step needed downstream.
- [ ] CLI entry point
  - Files: `src/stage1_distill.py`
  - Details: `python -m src.stage1_distill --activations data/activations/ --output data/atoms/ --block-size 32 --K 8 --lr 1e-2 --steps 500 --workers 4`. Progress bar via `tqdm`. Summary table of per-layer MSE at the end.

## Dependencies
- Epic 04 (activation store with cached `(X, Y)` pairs)

## Quality Gate
Before proceeding to Stage 2, review the Stage 1 manifest. A layer passes only when it meets
both the MSE threshold and its 50%-of-dense RAM budget. If **> 20% of layers** exceed either
threshold after adaptive retry, the candidate representation is insufficient for Stage 2.
Increasing `K_max` or decreasing `block_size` is allowed only while the state budget still
passes; otherwise select a different procedural family or accept a documented experimental
failure. Record the decision in `agile/stage1_verdict.md`.
