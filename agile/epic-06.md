# Epic 06: Stage 2 — Global End-to-End Logit Alignment

## Status
Draft

## Goal
Fine-tune **all** block-wise `Θ` parameters jointly by minimizing KL-divergence between
teacher and student logits on a large corpus. Stage 1 aligns each layer independently;
Stage 2 eliminates the error accumulation that happens when slightly-off layers compound
across 18 transformer blocks. This is the step that makes the student model actually
*behave* like the teacher.

## User Stories
1. As a researcher, I want to fine-tune all Θ parameters end-to-end using KL-divergence on logits so the student's output distribution matches the teacher's.
2. As an engineer, I want a student model assembled from `.atom` files that can run a full forward pass in PyTorch so I can backpropagate through the block-integrator synthesis.
3. As a researcher, I want perplexity and KL-divergence metrics tracked during training so I know when to stop.

## Acceptance Criteria
- [ ] `PhasorStudentModel` assembles a full Gemma 2B forward pass using `BlockAtomLayer` for every target weight, loaded from Stage 1 `.atom` files.
- [ ] Uses **50,000 sequences of 2,048 tokens** (~100M tokens) from FineWeb-Edu for Stage 2 training data.
- [ ] KL-divergence loss: `L = α · T² · KL(softmax(z_t/T) ‖ softmax(z_s/T))` with `T=2.0`, `α=0.7`, plus optional cross-entropy on ground-truth tokens with weight `(1−α)`.
- [ ] Training loop runs for **1,000 steps** with Adam, logging per-step loss, KL-divergence, and perplexity delta.
- [ ] Final KL-divergence ≤ 0.05 on held-out validation set.
- [ ] Final perplexity delta ≤ +0.35 vs teacher on wikitext-2 test set.
- [ ] Measured peak inference model RAM, including procedural state and mandatory scratch, is ≤ 50% of the dense FP16/BF16 teacher baseline.
- [ ] Off-chip model-state traffic per generated token is lower than the dense baseline and reported with the added compute and throughput.
- [ ] No dense projection weights are resident or reconstructed in the inference hot path.
- [ ] Updated `.atom` files saved after Stage 2 fine-tuning, versioned as `stage2_manifest.json`.
- [ ] Evaluation suite (`eval_comparison.py`) produces the comparison table: KL-div, perplexity, latency, peak RAM, state traffic, and throughput.

## Tasks
- [ ] Implement `src/student_model.py` — full phasor student model
  - Files: `src/student_model.py`
  - Details: `PhasorStudentModel(nn.Module)` that mirrors Gemma 2B architecture but replaces every `nn.Linear` in target roles with `BlockAtomLayer` (block-integrator synthesis from Epic 05). Loads Θ from Stage 1 `.atom` files via the manifest. Keeps embeddings, layernorms, and biases dense (small, non-target). Implements `forward(input_ids) -> logits` matching the teacher's output shape.
- [ ] Implement `src/stage2_distill.py` — end-to-end KL-divergence training loop
  - Files: `src/stage2_distill.py`
  - Details: `run_stage2(teacher_model, student_model, dataset, lr=1e-4, steps=1000, T=2.0, alpha=0.7)`. For each batch: run teacher (no_grad) to get logits, run student to get logits, compute KL-div loss at temperature T, backprop through student Θ parameters only (freeze embeddings/norms), optimizer step. Log loss, KL-div, PPL every 10 steps. Note: gradients flow through `torch.cumsum` in the block integrator — this is differentiable but means low-frequency/anchor parameters receive accumulated gradient; monitor for instability.
- [ ] Implement Stage 2 data loader
  - Files: `src/stage2_distill.py` (or extend `src/data_loader.py`)
  - Details: separate from Stage 1 calibration data. Stream 50K sequences of 2,048 tokens from FineWeb-Edu. Batch size configurable (default 4 for GPU memory). Shuffle buffer for streaming data.
- [ ] Implement `eval_comparison.py` — evaluation suite
  - Files: `eval_comparison.py`
  - Details: `evaluate_models(teacher, student, test_dataloader, tokenizer)` computing: (1) Mean KL-divergence on logits, (2) Perplexity on wikitext-2 test split for both models, (3) Latency per batch (teacher vs student), (4) Memory bandwidth estimate (bytes of weight data read per forward pass). Print comparison table. Assert gating criteria pass/fail.
- [ ] Save Stage 2 output
  - Files: `src/stage2_distill.py`
  - Details: after training, save updated Θ for every layer as `.atom` files in `data/atoms_stage2/`. Write `stage2_manifest.json` with per-layer initial MSE (from Stage 1), final KL contribution, and training stats.
- [ ] CLI entry point
  - Files: `src/stage2_distill.py`
  - Details: `python -m src.stage2_distill --teacher google/gemma-2b --atoms data/atoms/ --output data/atoms_stage2/ --steps 1000 --lr 1e-4 --batch-size 4`. Then: `python eval_comparison.py --teacher google/gemma-2b --atoms data/atoms_stage2/`.

## Dependencies
- Epic 05 (Stage 1 `.atom` files + manifest)
- Epic 04 (data loader infrastructure — extended for Stage 2 volume)

## Quality Gate
This is the **distillation go/no-go decision**. Proceed to Epic 07 only if the student meets
all behavioral gates (KL-divergence ≤ 0.05 and PPL delta ≤ +0.35) and the systems gate:
measured peak inference model RAM is ≤ 50% of the dense FP16/BF16 baseline, off-chip state
traffic is lower, and no dense projection is resident or reconstructed. If the gate fails,
options are: (a) increase Stage 2 steps or tune optimization without increasing inference
state, (b) revisit the Stage 1 representation and budget allocation, or (c) test a hybrid
whose complete state still fits the 50% RAM ceiling. Document the verdict in
`agile/stage2_verdict.md`.
