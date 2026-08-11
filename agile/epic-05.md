# Epic 05: Stage 1 — Layer-Wise Activation Alignment Distillation

## Status
Draft

## Goal
Fit PhasorAtom parameters `Θ = {A_k, f_k, θ_k, φ_k, γ}` for **every target weight layer**
by minimizing the MSE between teacher activations `Y_teacher` and student activations
`Y_student = X @ G(Θ, r, c).T`. This is a per-layer, embarrassingly parallel optimization
that produces a `.atom` file (phasor parameter set) for each weight matrix.

## User Stories
1. As a researcher, I want to fit each layer independently so I can parallelize across CPU cores / GPUs.
2. As a researcher, I want per-layer MSE reporting so I can identify which layers are hardest to distill.
3. As an engineer, I want the output saved as `.atom` files with a manifest so Stage 2 can load them into a student model.

## Acceptance Criteria
- [ ] `distill_single_layer(X_cache, Y_teacher, K)` optimizes `Θ` via Adam to minimize `MSE(X @ G(Θ,r,c).T, Y)`.
- [ ] Processes all 126 layer/role pairs from the activation store.
- [ ] Per-layer convergence: MSE < 5.0 × 10⁻⁴ target threshold. Layers that fail are logged and retried with higher K.
- [ ] Output: one `.atom` file per layer containing `Θ` as `f32` array of `4K+1` values, plus `stage1_manifest.json` mapping `(layer_idx, role)` → `{file, K, final_mse, steps, wall_time}`.
- [ ] Parallel execution: use `multiprocessing.Pool` or `concurrent.futures` to run multiple layers concurrently.
- [ ] Deterministic: fixed random seed, same input → same output.

## Tasks
- [ ] Implement `src/phasor_atom.py` — the differentiable `G(Θ, r, c)` forward evaluation
  - Files: `src/phasor_atom.py`
  - Details: `PhasorAtomLayer(nn.Module)` with parameters `amplitudes[K], frequencies[K], angles[K], phases[K], decay[1]`. Method `synthesize_weight(out_dim, in_dim) -> Tensor[out_dim, in_dim]` builds the meshgrid, evaluates the multi-tone wave sum, applies the decay envelope. Method `forward(x) -> x @ synthesize_weight().T`. This module is the PyTorch student building block.
- [ ] Implement `src/stage1_distill.py` — single-layer distillation loop
  - Files: `src/stage1_distill.py`
  - Details: `distill_single_layer(X_cache, Y_teacher, K=16, lr=1e-2, steps=500) -> dict`. Initialize `Θ` (small random init), optimize with Adam, track MSE. Return `{"amplitudes", "frequencies", "angles", "phases", "decay", "final_mse"}`. Pre-compute the `(r, c)` meshgrid once. Support GPU if available.
- [ ] Implement the full-model orchestrator
  - Files: `src/stage1_distill.py`
  - Details: `run_stage1(activation_dir, output_dir, K_profile, max_workers)`. Load `manifest.json` from Epic 04, iterate all `(layer_idx, role)` pairs, call `distill_single_layer` for each. Use `concurrent.futures.ProcessPoolExecutor` for parallelism. Log per-layer results. Save `.atom` files + manifest.
- [ ] Implement K-adaptive retry
  - Files: `src/stage1_distill.py`
  - Details: if `final_mse > 5e-4` after initial run, retry with `K *= 2` (up to `K_max=128`). Record the K escalation in the manifest. This handles layers where the default K is insufficient.
- [ ] Save/load `.atom` format (shared with Epic 01 codec)
  - Files: `src/phasor_atom.py`
  - Details: reuse the **same** `save_atom(theta, path)` / `load_atom(path)` binary format defined in Epic 01 — raw `f32` array `[amplitudes(K), frequencies(K), angles(K), phases(K), decay(1)]` with a header recording `K`, `in_dim`, `out_dim`. One format, one extension, used by both the direct codec and the distillation pipeline. No conversion step needed downstream.
- [ ] CLI entry point
  - Files: `src/stage1_distill.py`
  - Details: `python -m src.stage1_distill --activations data/activations/ --output data/atoms/ --K 16 --lr 1e-2 --steps 500 --workers 4`. Progress bar via `tqdm`. Summary table of per-layer MSE at the end.

## Dependencies
- Epic 04 (activation store with cached `(X, Y)` pairs)

## Quality Gate
Before proceeding to Stage 2, review the Stage 1 manifest. If **> 20% of layers** exceed
the MSE threshold even after K-adaptive retry, the phasor basis may be insufficient for
those layers. Options: (a) increase K_max, (b) switch to independent `(f_r, f_c)` per
harmonic, (c) accept higher error on those layers and rely on Stage 2 to compensate.
Document the decision in `agile/stage1_verdict.md`.
