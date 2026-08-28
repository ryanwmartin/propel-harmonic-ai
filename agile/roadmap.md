# Phasor Inference — Roadmap

**Vision:** Prove a real small LLM can execute from compact procedural state rather than
repeatedly fetching dense weight matrices. A CPU/GPU function either generates weights into
immediate multiply-accumulates or transforms activations through a structured DSP pipeline.
The aha moment is coherent terminal chat from a model whose dense weights are never resident
or reconstructed, with measured reductions in model RAM and off-chip state traffic.

**Memory-wall framing (locked in):** Batch-1 LLM decode reads every weight byte once per
token for ~1 MAC — arithmetic intensity ≈ 1–2 FLOPs/byte, while modern accelerators need
~150–300 FLOPs/byte to saturate (e.g. RTX 4090 ≈ 165 TFLOPS FP16 / ~1 TB/s ≈ 165;
H100 ≈ 990 TFLOPS / 3.35 TB/s ≈ 295). Compute units sit >99% idle during dense decode;
that idle FLOP capacity is the resource procedural generation spends. Two regimes:

1. **Bandwidth-bound (model fits in RAM):** the win is fewer bytes moved per token —
   throughput scales with the state-reduction factor.
2. **Capacity-bound (model exceeds RAM — the motivating case):** even quantized frontier
   models exceed most hardware's RAM; the fallback is SSD streaming (~3–14 GB/s), which is
   orders of magnitude below DRAM and makes batch-1 decode unusable. Here procedural state
   only has to *fit in RAM* and decode faster than SSD streaming — a far softer compute bar.

**Per-weight decode-FLOP budget:** each dense FP16 byte-pair not read frees
≈ 2 bytes × (FLOP:byte ratio) ≈ **300–600 FLOPs of headroom per generated weight** before
the generator itself becomes the bottleneck. Candidate check: harmonic rotation recurrence
≈ 6–7 FLOPs/sinusoid/weight (K=8 ≈ 50 ✓, K=32 ≈ 220 borderline, K=block_size/2 ✗);
low-rank rank-r ≈ 2r FLOPs/weight (r=64 ≈ 128 ✓); low-rank + sparse residual ✓. Any family
requiring per-weight transcendentals/special functions in the inner loop is disqualified —
decode must run on the wide MAC/tensor units (GEMM-shaped), not SFUs, and must not fragment
into launch-overhead-dominated small kernels.

**Guiding objective (locked in):** This is **procedural execution**, not conventional
compression. Success means replacing repeated dense-weight RAM reads with reusable CPU/GPU
calculation while avoiding dense-weight materialization. The block-wise derivative codec is
the first baseline: each block stores an anchor plus a harmonic first-difference formula,
then decodes through an oscillator and running prefix sum. It is not the only candidate
function family. Shared bases, learned generators, structured DSP operators, cross-layer
transforms, and procedural-plus-residual representations are evaluated in
[`procedural-inference-experiments.md`](procedural-inference-experiments.md). Every method
must count all model-specific operands and residuals, not only executable function code.

**Why block-wise derivative cumsum (the architecture):**
- **1D fits are tractable.** Fitting K sinusoids to a 1D block is near-closed-form
  (Prony/ESPRIT-type harmonic retrieval), avoiding the fragile non-convex 2D Adam fit of
  the prior directional-field approach.
- **Chunking bounds integration drift.** A DC error `ε` in a block of length `L` becomes
  an `ε·L` error at the block's far end. `block_size` caps `L`, so drift is a tunable
  fidelity-vs-size knob, not a catastrophic failure mode.
- **Anchors re-synchronize.** Each block stores its first weight at full precision, so
  error never propagates across block boundaries.
- **Decode is cheap and parallel.** Per weight: a few `sin` calls + one running add, one
  accumulator register, and blocks are independent (embarrassingly parallel).

**Tunable `block_size`:** The encoder accepts a `block_size` parameter. Smaller blocks =
less drift + more anchors/formulas to store; larger blocks = fewer parameters + more
within-block drift. The encoder chooses `block_size` per tensor (or per row) to satisfy a
target fidelity metric, then records it in the `.atom` header. This is the primary
fidelity control alongside per-block `K`.

**Discovery then distillation strategy:** Epics 01–03 measure direct harmonic fitting to raw
weight derivatives. The procedural experiment program compares additional function families
under identical state-traffic and compute accounting. Epics 04–06 then fit the strongest
families to teacher *behavior*—activations and logits—because functional equivalence may not
require numerical reproduction of every teacher weight.

**Explicit representation formats:** Harmonic `Θ` remains in `.atom`. Other procedural
families may use family-specific, versioned formats that count every coefficient, seed,
permutation, operand, lookup table, and residual. There is one loader per explicit format;
no candidate may hide model state in executable code or unreported constants.

**Target model:** Gemma 2B (or a similarly sized open-weights model) for the proof of concept,
with Pythia-14m used for fast structure discovery. Exact weight fidelity is measured where
available, but advancement is decided by activation/logit fidelity, model-state traffic,
peak RAM, added compute, and throughput.

## Epics

| # | Epic | Status | File |
|---|------|--------|------|
| 01 | Python Reference Codec — Block-wise Derivative Encoder + Decoder | **Done** | `agile/epic-01.md` |
| 02 | Rust Hot-Path Decoder + Zero-Alloc Block Integrator | Ready | `agile/epic-02.md` |
| 03 | End-to-End Fidelity Gate & Parity Test | Ready | `agile/epic-03.md` |
| 04 | Distillation Data Pipeline — Activation Extraction & Calibration | Draft | `agile/epic-04.md` |
| 05 | Stage 1: Layer-Wise Activation Alignment Distillation | Draft | `agile/epic-05.md` |
| 06 | Stage 2: Global End-to-End Logit Alignment | Draft | `agile/epic-06.md` |
| 07 | Full-Model Encoder — Encode Every Weight of Gemma 2B | Draft | `agile/epic-07.md` |
| 08 | Rust LLM Forward Pass — Attention + FFN via Phasor Decode | Draft | `agile/epic-08.md` |
| 09 | Tokenizer + KV Cache + Autoregressive Generation Loop | Draft | `agile/epic-09.md` |
| 10 | Terminal Chat Interface — The Aha Moment | Draft | `agile/epic-10.md` |

**Execution order:** 01 → 03 diagnostics → **procedural representation experiments** →
04 activation data → distill the strongest candidates → 02/08 fused kernels → 07 full-model
artifact → 09 → 10. Epic 02's Rust decoder remains useful as the harmonic baseline, but
optimization follows representation discovery rather than preceding it.

Epic 03 is the go/no-go gate for the current direct harmonic codec, not for procedural
inference as a whole. Its broadband-weight result triggers the broader experiments in
[`procedural-inference-experiments.md`](procedural-inference-experiments.md). Epics 04–06
then distill the Pareto-optimal procedural families, and Epics 07–10 build inference around
the candidate that best reduces measured model-state traffic while retaining behavior.

**Distillation gating criteria** (must pass before Epic 07 uses distilled weights):
| Metric | Target |
|--------|--------|
| Per-layer activation MSE (Stage 1) | < 5.0 × 10⁻⁴ |
| Logit KL-divergence (Stage 2) | ≤ 0.05 |
| Perplexity delta vs teacher | ≤ +0.35 on wikitext-2 |
| Peak inference model RAM | ≤ 50% of dense FP16/BF16 baseline (≥ 2× reduction) |
| Measured off-chip state traffic | Lower than dense baseline; ≥ 2× reduction is a stretch target |
| Dense weight materialization | 0 bytes in inference hot path |
| Per-weight decode cost | Within the 300–600 FLOP/weight roofline budget; no per-weight transcendentals in the inner loop |

**Baselines (every candidate is scored against all three):**
| Baseline | What it tests |
|----------|---------------|
| Dense FP16/BF16 in RAM | Bandwidth-bound win: bytes moved per token |
| 4-bit quantized in RAM | The real incumbent — procedural state must beat or compose with ~4× quantization to matter |
| Dense streamed from SSD (~7 GB/s effective) | Capacity-bound fallback — the throughput floor any resident procedural decode must exceed |

**Definition of "aha moment working":**
- Every 2D projection of the proof-of-concept model is represented by a measured procedural
  artifact; `.atom` remains the harmonic format and other families use equally explicit,
  fully accounted formats.
- The CPU/GPU forward pass generates weights into immediate MACs or applies structured DSP
  transforms directly, with no dense weight matrices resident or reconstructed.
- Hardware measurements demonstrate lower peak model RAM and lower off-chip model-state
  traffic, not merely a smaller executable function body.
- A tokenizer, KV cache, sampling loop, and terminal REPL produce coherent multi-turn chat.

**Explicitly out of scope:** 27B models, WASM, distributed inference, unmeasured "27 MB"
claims, and production serving. CPU kernels are required; targeted GPU kernels are in scope
for finalists because GPU memory traffic is part of the core hypothesis.

## Progress Log

- **Epic 01 — Done.** Reference codec complete: BlockAtom container, block extractor,
  FFT warm-start + best-so-far Adam harmonic fitter (DC-aware), `torch.cumsum` decode
  with **f32 accumulator pinned as the Epic 02 parity contract**, `.atom` v2 reader/writer
  with geometry validation, auto-fitter with per-configuration breakdown, CLI. 27/27
  contract tests passing. At `K = block_size/2` the format reconstructs any block exactly
  (f32 rounding) at ~1.5× dense parameter cost — the lossless ceiling is proven; the
  Epic 03 gate must now measure fidelity-per-parameter on **real** weights. See
  `agile/epic-01.md` Completion Notes for deviations (Prony/ESPRIT deferred to Epic 07).
- **Next:** Build the common benchmark harness from
  `agile/procedural-inference-experiments.md`, then compare transformed/shared procedural
  families before selecting finalists for distillation and fused kernel work.
- **Harness + Experiment 1 (harmonic baseline) — Done.** `src/harness/` adds the shared
  `ProceduralRepresentation` interface (per-field state-bit accounting, fused
  `transform` contract, decoded-weight in-flight bound, no-materialization gate,
  required `ExperimentReport` template) and wraps the BlockAtom codec behind it with
  oscillator rotation recurrences (transcendentals once per block) and a fused
  generate+MAC hot path — decoded weights live only in a per-row accumulator.
  Precision sweep (anchors/coefficients FP32/FP16) included. 14 new contract tests;
  63/63 pass. Baseline cost confirmed at ~1.6x dense FP16 at the lossless ceiling —
  the number Experiments 3 (shared bases) and 5 (low-rank + sparse residual) must beat.
  Next candidates ranked: Exp 5 (low-rank+residual, closed-form, no distillation
  needed) > Exp 3 (shared bases) > Exp 7 (butterfly/DSP, needs distillation).
- **Experiment 2 (orderings/transforms) — Done, gate failed on real weights.**
  `src/harness/orderings.py` + `transformed_representation.py` +
  `experiment_2_orderings.py` add exactly reversible canonicalizations
  (column-norm / spectral-seriation / greedy-NN orderings, sign flips, FP16
  diagonal scales; all state counted, column transform moved onto activations
  in the hot path). 48 new contract tests. Synthetic weights: up to 7.1x
  coefficient reduction (machinery works). Real Pythia-14m: best ~1.00x —
  permutations are orthogonal transforms so low-rank error is provably
  invariant, and block-locality seriation found no hidden order. Decision:
  drop canonicalization state from finalists; the next lever is Experiment 12
  (fit low-rank/shared-basis to cached activations instead of weights).
- **Experiment 12 stage 1 (activation-aware closed-form fitting) — Done.**
  `src/harness/activation_aware.py` (whitened-SVD low-rank, importance-ranked
  residuals, block-averaged whitener for shared bases) +
  `activation_cache.py` (Epic 04 small-scale: real Pythia-14m forward passes,
  pre-hook capture of QKV/MLP-down inputs at layers 0 and 3, disjoint
  fit/eval split, disk cache) + `experiment_12_behavioral.py` (paired
  weight-space vs. activation-aware sweep). 38 new contract tests; 163
  harness+codec tests pass. **Result:** activation-aware fitting beats
  weight-space in all 72 paired configs on real weights (mean ~1.25x, best
  2.14x lower held-out on-distribution error); layer-3 QKV reaches 0.126
  rel err at 0.334x dense state (vs 0.24 at 0.82x in Exp 3/5). Still short
  of the 1e-2 stage-1 gate, so no "advance" at the strict criterion — the
  measured path forward is stage 2 joint multi-layer fitting and end-to-end
  logit-KL measurement, where the model-level gates (KL <= 0.05, ppl delta
  <= +0.35) replace the per-layer 1e-2 requirement. Added `transformers`
  dep (teacher forward passes only).

- **Stress test (real weights) — Done.** `tests/test_real_safetensor.py` downloads the
  real `model.safetensors` of `EleutherAI/pythia-14m` (14M-param GPT-NeoX, F16) and
  encodes/decodes its trained weight matrices, reporting on-disk `.safetensors` vs.
  `.atom` size, parameter ratio, and fidelity (rel_err / max_abs / MSE). **Gate signal:**
  a trained weight row's first difference is **spectrally broadband** (near white).
  Sub-spectral fits (K=8 at 0.20× dense, K=32 at 0.76× dense) diverge (rel_err ≈ 1.0–1.3);
  only `K = block_size/2` (a complete real DFT per block) reconstructs, at rel_err ≈
  1–3×10⁻⁵ (f32 rounding) and ~1.51× dense params. Confirms the Epic 01 lossless ceiling
  holds on real weights but the direct codec is **not a compressor** for trained LLM
  weights — the fidelity-per-parameter curve has no useful middle ground. This is the
  go/no-go evidence pushing toward the distillation path (Epics 04–06). 31/31 tests pass
  (27 contract + 4 real-safetensor). Added `safetensors` + `huggingface_hub` deps.
