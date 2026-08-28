# Procedural Inference — Experiment Roadmap

## Status

Ready

## Mission

Discover whether neural-network weight reads can be replaced by reusable computation:

```text
small model-specific function state + shared executable code
    -> procedural transform or generated weight
    -> immediate multiply-accumulate
```

The primary objective is **not file compression**. The objective is to reduce model-state
RAM and off-chip memory traffic during inference by spending additional CPU/GPU arithmetic.
A successful representation must execute a layer without materializing its dense weight
matrix.

This roadmap explores both forms of procedural inference:

1. **Procedural weight generation:** generate each required weight from compact state and
   consume it immediately in a fused matrix multiplication.
2. **Procedural layers:** replace a dense matrix multiplication with a sequence of
   structured DSP operations that acts directly on activations.

The current block-wise harmonic derivative codec remains the first measured baseline. Its
negative low-harmonic result excludes only independent Fourier fits of raw row derivatives;
it does not exclude shared bases, transformed coordinates, cross-row structure,
cross-layer structure, learned generators, or structured operators.

## Accounting Rules

Every experiment must report all three components separately:

- **Shared executable bytes:** compiled CPU/GPU kernels and decoder bytecode. These are
  loaded once and are not multiplied by model size.
- **Unique model-state bytes:** coefficients, seeds, anchors, permutations, quantization
  scales, expression operands, lookup tables, and residuals. Constants embedded in code
  count as model state.
- **Transient bytes:** scratch buffers and generated values. A dense decoded matrix is
  forbidden in the final hot path.

A representation has not shifted inference to compute if it hides weights in source code,
constant memory, a large permutation, a lookup table, or an incompressible residual.

### Primary metrics

| Metric | Definition |
|---|---|
| Model-state bits per generated weight | Total unique model-state bits divided by represented dense weights |
| Off-chip bytes per generated weight | Model-state and residual bytes fetched from DRAM during a warm inference pass |
| Arithmetic intensity | Useful and generator operations divided by off-chip bytes |
| State reuse | Generated weights or outputs produced per model-state byte fetched |
| Peak model RAM | Resident model state plus mandatory scratch, excluding OS/framework overhead |
| Dense materialization | Peak bytes allocated for reconstructed dense weights; target is zero |
| Added compute | Generator/transform operations relative to dense matrix multiplication |
| Layer fidelity | Relative output error, activation MSE, cosine similarity |
| Model fidelity | Logit KL divergence, perplexity delta, and task score delta |
| Throughput | Tokens per second and latency compared with dense FP16/BF16 baselines |

Disk size is reported for reproducibility but is not the primary decision metric.

### Roofline budget and required baselines

Batch-1 decode has arithmetic intensity ≈ 1–2 FLOPs/byte against hardware balance points of
~150–300 FLOPs/byte, so every dense FP16 weight not fetched frees ≈ **300–600 FLOPs** of
idle compute. That is the hard per-weight decode budget: a candidate whose generator exceeds
it is compute-bound and has merely moved the bottleneck. Generators must be GEMM-shaped
(wide MAC/tensor-core friendly, no per-weight transcendentals, no launch-overhead-dominated
per-block kernels).

Every experiment reports its state and throughput numbers against **three baselines**, not
just dense FP16:

1. **Dense FP16/BF16 in RAM** — measures the bandwidth-bound win.
2. **4-bit quantized in RAM** — the incumbent; procedural state must beat ~4× reduction or
   demonstrably compose with quantization (quantized coefficients/bases/residuals).
3. **Dense streamed from SSD (~7 GB/s effective)** — the capacity-bound floor; resident
   procedural decode need only exceed this throughput to win when the model cannot fit in
   RAM at all, even quantized.

### Measurement levels

- **Analytical:** exact parameter/state count and operation count from the representation.
- **Reference:** Python/PyTorch correctness and fidelity measurements on Pythia-14m.
- **Kernel:** fused Rust CPU and CUDA/Triton prototype with no dense materialization.
- **Model:** autoregressive inference with measured peak RAM, hardware counters, latency,
  throughput, and quality.

The reference implementation may materialize a tensor for research comparisons. Passing a
reference experiment does not count as procedural inference until a fused hot path passes
the kernel gate.

### Completed diagnostic: adaptive entropy coding of exact FP16 bytes

A self-contained adaptive order-0 arithmetic coder was run on the original little-endian
FP16 bytes of two Pythia-14m projection matrices. The decoder reproduced every source byte
exactly; headers and CRC32 checksums are included in the encoded sizes.

| tensor | raw bytes | whole-tensor ratio | bits / FP16 weight | independent-row ratio |
|---|---:|---:|---:|---:|
| attention QKV `(384,128)` | 98,304 | 0.9320x | 14.913 | 1.0319x |
| MLP down `(128,512)` | 131,072 | 0.9179x | 14.686 | 0.9612x |
| **aggregate** | 229,376 | **0.9240x** | **14.784** | **0.9915x** |

This byte-frequency model removes only about 7.6% of exact FP16 storage in a continuous
stream. Resetting the model per row, as required for simple parallel/random-access decode,
reduces the aggregate saving to about 0.85%. This is a useful lossless baseline but does not
approach the 50% inference-RAM reduction target. Future prediction tests must reduce
conditional residual entropy, not merely model marginal byte frequencies.

## Program-Wide Success Criteria

### Minimum viable procedural win

A candidate advances to distillation and kernel work when it meets all of the following on
at least two attention projections and two MLP projections:

- Compact state plus mandatory inference scratch is **no more than 50%** of the dense
  FP16/BF16 layer state.
- Measured off-chip weight/state traffic is lower than the dense baseline in a warm fused
  layer run; its exact reduction is reported separately from RAM.
- **Zero dense-weight materialization** in the measured hot path.
- Added arithmetic is finite, deterministic, and expressible with vectorizable operations.
- Either exact reconstruction, or layer output relative error `<= 1e-2` before
  distillation.

### Model-level success

A representation becomes the default inference path when it meets:

- **Peak inference model RAM is no more than 50%** of the dense FP16/BF16 baseline,
  including all unique model state and mandatory scratch (a reduction of at least 50%).
- Measured off-chip model-state traffic per generated token is lower than the dense
  baseline and is reported alongside the added compute cost.
- Logit KL divergence `<= 0.05` against the teacher.
- Perplexity delta `<= +0.35` on WikiText-2.
- No dense weights resident or reconstructed during inference.
- End-to-end throughput no worse than `0.5x` dense inference on the same hardware.

### Stretch goals

- At least **4x lower peak inference model RAM** than dense FP16/BF16.
- At least **2x lower off-chip model-state traffic** while preserving the model-level
  fidelity gates. These are stretch targets, not assumptions baked into early experiments.

Candidates that miss fidelity but demonstrate a strong traffic reduction remain eligible
for behavioral distillation. Candidates that require reading an effectively dense residual
or coefficient set are stopped even if their executable function is small.

## Common Experiment Harness

Before comparing representations, implement one shared benchmark harness.

### Proposed implementation

- Load deterministic slices and complete 2D projection tensors from
  `EleutherAI/pythia-14m`, then scale finalists to Gemma 2B.
- Measure attention Q/K/V/O and MLP gate/up/down projections separately.
- Define a `ProceduralRepresentation` interface with:
  - encode or fit;
  - exact state serialization;
  - state-bit accounting by field;
  - reference reconstruction for diagnostics;
  - direct activation transform or generated-weight iterator;
  - operation-count estimate;
  - optional exact residual.
- Benchmark fixed quality/state budgets rather than allowing each candidate to choose a
  favorable metric.
- Produce Pareto curves for state bytes, off-chip bytes, operations, and fidelity.
- Add a fused-kernel contract that rejects implementations allocating a dense output-weight
  buffer.

### Harness success criteria

- Repeated runs produce identical state accounting and fidelity measurements.
- Every serialized byte is assigned to code, model state, residual, metadata, or scratch.
- Dense FP16/BF16 and current `.atom` decoding are included as baselines.
- At least one CPU hardware-counter backend reports cache misses and DRAM traffic; GPU
  finalists report profiler-observed global-memory traffic.
- Tests detect hidden dense constants and dense reconstruction buffers.

---

## Experiment 1 — Harmonic Derivative Baseline

### Status: Reference implementation complete

Implemented in `src/harness/`:

- `representation.py` — the shared `ProceduralRepresentation` interface
  (state accounting by field, diagnostic reconstruction, fused `transform`,
  operation counts, decoded-weight in-flight bound, no-materialization gate).
- `accounting.py` — `StateAccounting` + the required `ExperimentReport`
  template with dense-FP16 baseline ratios.
- `harmonic_baseline.py` — `HarmonicDerivativeRepresentation`: BlockAtom
  behind the harness interface with (a) an oscillator rotation recurrence
  (f64 state, transcendentals once per block; f32 derivative feeds the pinned
  f32 prefix-sum accumulator) replacing per-sample `sin()`, and (b) a fused
  generate+MAC `transform` where decoded weights exist only in a per-row
  accumulator (`rows` in-flight decoded elements, never a dense buffer), and
  independent FP16/FP32 quantization of anchors vs. coefficients.
- `experiment_1_harmonic.py` — sweep runner over block size, harmonic count,
  anchor precision, and coefficient precision; emits the required report per
  configuration and gates cross-algorithm oscillator drift (relative 1e-5;
  the absolute 1e-6 same-recurrence parity contract moves to Epic 02
  Python-vs-Rust).

14 contract tests in `tests/test_harness_experiment_1.py`; 63/63 total pass.

**Reference measurements (synthetic smooth tensor):** the K = L/2 lossless
ceiling reproduces at layer-output error ≈ 4e-7 with zero dense
materialization; FP16 coefficient storage cuts state ≈ 2x at ≈ 3e-3 layer
output error. State ratio remains ≈ 1.6x dense FP16 at the ceiling —
confirming this baseline is the cost floor that shared-state families
(Experiments 3/5) must beat, per the Epic 03 broadband result.

### Hypothesis

The existing block oscillator establishes the cost and fidelity baseline for procedural
weight generation. Shared or learned variants must outperform its independent per-block
state traffic.

### Proposed implementation

- Preserve the existing anchor plus harmonic first-difference representation.
- Replace repeated transcendental calls in the hot path with oscillator recurrences using
  precomputed sine/cosine step values.
- Fuse block generation with dot-product accumulation; never write decoded blocks to RAM.
- Sweep block size, harmonic count, coefficient precision, and anchor precision.

### Success criteria

- Python/Rust parity within `1e-6` for sampled weights.
- Zero hot-path allocation and zero dense matrix materialization.
- Report state bytes and operations per generated weight.
- Serves as a baseline even if it fails the 4x advancement gate.

---

### Experiment 3/5 first measurement (reference level, no distillation)

Implemented `SharedBasisRepresentation` (Exp 3: one DCT or SVD basis per tensor,
per-block coefficients, closed-form projection fit) and
`LowRankResidualRepresentation` (Exp 5: truncated-SVD factors + top-magnitude
sparse residual, closed-form fit) behind the common
`ProceduralRepresentation` interface, both with fused `transform` paths and
`max_decoded_weight_elements() == 0`.

**Synthetic (64x128, Epic 01 generator):** low-rank r=6 at FP16 factors hits
layer-output rel err 3.1e-4 at **0.14x dense FP16 state** — passes Gate A with
a wide margin. Confirms harness + gates work end to end.

**Real Pythia-14m (attention QKV 384x128, MLP down 128x512):** no candidate
passes the `<=1e-2` layer-output gate below 1.0x dense state. Best points:
low-rank r=64 + 5% residual = 0.24 rel err at 0.82x state (QKV); shared SVD
basis beats DCT at every K but plateaus near 0.38 rel err at 0.87x state
(MLP). The spectrum of trained matrices is heavy-tailed but not *sharply*
low-rank at this scale: singular values decay smoothly, so truncation error
stays high until near-full rank.

**Interpretation:** structure exists (SVD basis consistently beats DCT;
residuals help monotonically) but closed-form weight-space fitting alone does
not reach the pre-distillation fidelity gate on real trained weights. This is
the expected motivation for Experiment 12: fit these same structured families
to layer *behavior* (activations/logits), where the `<=1e-2` output-error
requirement applies only on the data distribution, not for all of R^n.
Candidates remain eligible for distillation per the program rules.

Decision: **retain both families as distillation candidates; proceed to
Experiment 2 (orderings/transforms) and Experiment 12 (behavioral fit).**

## Experiment 2 — Reversible Ordering and Transform Search

### Status: Reference implementation complete — measured, gate failed on real weights

Implemented in `src/harness/`:

- `orderings.py` — reversible canonicalizations with O(rows + cols) state:
  column-norm sort, spectral (Fiedler-vector) seriation, greedy
  nearest-neighbor (TSP-like) chaining under a sign-invariant distance,
  per-column sign flips, and per-column FP16 RMS diagonal scales. All state
  bits (16-bit permutation indices, 1-bit signs, 16-bit scales) counted.
- `transformed_representation.py` — `TransformedRepresentation` wrapper: any
  inner `ProceduralRepresentation` fitted on the canonicalized matrix; the
  column transform is applied to the **activations** and the row permutation
  to the output vector in the hot path (no per-call weight sort/unsort),
  preserving `max_decoded_weight_elements() == 0`.
- `experiment_2_orderings.py` — sweep runner: 14 canonicalizations x
  {low-rank, shared-SVD-basis} x capacity ladders, exact-inverse verification
  per fit, and gate scoring (>= 2x coefficient reduction at matched
  layer-output error; canonicalization bytes < 25% of dense FP16).

48 contract tests in `tests/test_harness_experiment_2.py`.

**Synthetic (64x128, Epic 01 generator):** orderings pass easily — col-norm
ordering gives up to **7.1x** coefficient reduction for low-rank and **2.9x**
for the shared basis at 1.6–4% permutation overhead. Confirms the machinery
detects real exploitable ordering structure when it exists.

**Real Pythia-14m (attention QKV 384x128, MLP down 128x512):** every
canonicalization fails the 2x gate. Best coefficient reduction is **~1.00x**
(no improvement) for both families across all 13 non-identity
canonicalizations; several orderings actively hurt (col-norm on the shared
basis drops to 0.75x). Permutation budgets were never the problem
(0.3–1.6% of dense).

**Interpretation:** for the low-rank family this is the expected mathematics —
permutations and sign flips are orthogonal transforms, so singular values are
invariant and truncation error cannot improve; only diagonal scaling could
have helped, and trained column RMS is too uniform to matter. The block-local
shared-basis family *could* have gained from locality-improving seriation but
did not: trained-weight block structure is not hidden by training-time
column order at this scale. Consistent with the Exp 3/5 finding that the
obstacle is smooth singular-value decay, not coordinate order.

Decision: **stop cheap canonicalization search; do not carry ordering state
into finalists. Proceed to Experiment 12 (behavioral/activation-space fit),
where the fitting target — not the coordinate system — is what changes.**

### Hypothesis

A cheap shared change of coordinates may expose local, spectral, or block structure that is
hidden in the matrix's training-time row/column order.

### Proposed implementation

Test transformations whose state is shared across many weights:

- Joint row and column seriation using spectral graph ordering.
- Nearest-neighbor/TSP-like ordering of column vectors on reduced-dimensional embeddings.
- Co-clustering and block permutations.
- Fixed DCT, wavelet, Hadamard, and learned orthogonal transforms.
- Learned permutations optimized for top-K spectral energy or downstream output error.
- Compositions of a generated permutation, sign flips, and diagonal scaling.

Move shared column permutations into the input activation and propagate row permutations
into the next compatible layer. Do not physically sort and unsort every row in the hot
path. Per-row sorting is retained only as an upper-bound diagnostic because its permutation
state and irregular gathers are expensive.

### Success criteria

- Exact inverse verified for lossless transformations.
- All permutation/transform state included in the model-state count.
- At least **2x reduction** in the coefficients required to reach the same reconstruction
  or layer-output error as the untransformed candidate.
- Activation-space permutation/transform overhead is less than 10% of layer runtime.
- No candidate advances if permutation bytes consume more than 25% of dense FP16 bytes.

---

## Experiment 3 — Shared Spectral and Multiresolution Bases

### Hypothesis

Frequencies may be broadband per row but reusable across rows, blocks, or a whole layer.
Sharing a basis can remove frequency and phase traffic while retaining procedural synthesis.

### Proposed implementation

- Learn one frequency bank per tensor, projection role, or transformer layer.
- Store only per-block coefficients and anchors.
- Compare Fourier, DCT, wavelet packets, chirps, and multiresolution sinusoidal bases.
- Evaluate coefficient prediction across adjacent rows/blocks instead of storing every
  coefficient independently.
- Generate basis values through oscillator recurrences and fuse coefficient accumulation
  directly into matrix multiplication.

A representative form is:

```text
weight[row, block, position] = anchor
    + sum(coefficient_cos * shared_cos_basis
        + coefficient_sin * shared_sin_basis)
```

### Success criteria

- Shared basis state amortizes to less than 0.05 bits per represented weight.
- Total state bytes are **no more than 50%** of dense FP16 at layer output relative error
  `<= 1e-2`; 4x fewer bytes is recorded as a stretch result.
- Fused kernel reads each coefficient set once and generates at least one complete block.
- Outperforms independent harmonics at matched state bytes and operation count.

---

## Experiment 4 — Cross-Row and Cross-Block State-Space Prediction

### Hypothesis

Even if individual rows look unstructured, the matrix may have predictable evolution over
row and block coordinates.

### Proposed implementation

Fit compact predictors such as:

- Two-dimensional autoregression over row and block indices.
- Linear recurrences and low-order state-space models.
- Shared dictionary atoms with procedurally generated coefficients.
- Small correction codes only where predictor error exceeds a threshold.
- Morton/Hilbert traversals over matrix blocks to test alternate spatial locality.

The decoder advances a small recurrent state and emits block coefficients or weights. Exact
mode stores a correction residual; approximate mode omits or quantizes corrections.

### Success criteria

- Predictor state plus corrections is **no more than 50%** of dense FP16.
- Recurrent state fits in registers or local/shared memory for the fused kernel.
- Corrections account for less than 25% of dense FP16 bytes.
- Layer output relative error `<= 1e-2`, or a clear Pareto improvement over shared spectral
  bases at matched state traffic.

---

## Experiment 5 — Low-Rank, Sparse Residual, and Shared Dictionary

### Hypothesis

A dense layer may decompose into a compute-friendly global component and a small residual:

```text
weight_matrix = low_rank_or_dictionary_component + sparse_or_quantized_residual
```

### Proposed implementation

- Sweep truncated SVD and randomized low-rank factorization.
- Fit low-rank plus sparse residual with explicit sparse-index accounting.
- Learn a basis shared across Q/K/V/O or gate/up/down projections.
- Test basis reuse across transformer layers with small layer-specific coefficients.
- Execute factors as consecutive matrix-vector/matrix-matrix operations; add sparse residual
  directly without reconstructing the matrix.

### Success criteria

- Factors plus residual use **no more than 50%** of dense FP16 state bytes.
- Sparse indices, scales, and metadata are fully counted.
- At least 80% of represented output energy is produced by the structured component.
- Layer output relative error `<= 1e-2` without distillation, or model quality gates after
  distillation.
- Measured off-chip bytes are lower than the dense baseline; a low-rank method that
  repeatedly reloads factors without reuse does not pass.

---

## Experiment 6 — Kronecker, Tensor-Train, Toeplitz, and Circulant Operators

### Hypothesis

Whole-matrix algebraic structure can replace per-weight generation with smaller tensor
cores and direct structured operations.

### Proposed implementation

- Fit sums of Kronecker products across shape-compatible factorizations.
- Fit tensor-train and tensor-ring decompositions with rank sweeps.
- Fit block-Toeplitz and block-circulant components plus optional residual.
- Implement direct contraction/FFT execution on activations.
- Compare independent per-layer factors with factors shared by projection role.

### Success criteria

- State bytes are **no more than 50%** of dense FP16 for advancement; stronger reductions
  must justify the additional execution complexity.
- No intermediate allocation larger than the input or output activation tensor.
- Layer output relative error `<= 2e-2`, or `<= 5e-2` with a demonstrated distillation path.
- Direct structured execution is faster than procedural scalar weight generation at matched
  fidelity.

---

## Experiment 7 — Butterfly and DSP Procedural Layers

### Hypothesis

A learned chain of fixed mixing operators and small diagonal functions can replace dense
matrices without generating individual weights:

```text
input -> permutation -> diagonal -> butterfly/Hadamard/FFT
      -> diagonal/nonlinearity -> butterfly/Hadamard/FFT -> output
```

This is the most direct test of shifting inference from RAM bandwidth to arithmetic.

### Proposed implementation

- Train products of learned diagonal matrices and fixed Hadamard/FFT transforms.
- Test learned butterfly factors and block-butterfly operators.
- Support rectangular projections using padding, truncation, and low-rank adapters.
- Fuse adjacent transforms where possible and keep intermediate activations in registers,
  cache, or GPU shared memory.
- Evaluate replacing one projection, one transformer sublayer, and then a complete block.

### Success criteria

- Parameter/state complexity scales near `O(n log n)` rather than `O(n^2)`.
- State bytes are **no more than 50%** of the replaced dense projections.
- No dense weight generation or materialization at any stage.
- Layer activation MSE `< 5e-4` after layer-wise distillation.
- Kernel achieves at least 50% of dense layer throughput while reducing measured
  model-state traffic below the dense baseline.

---

## Experiment 8 — Coordinate-Network Weight Generators

### Hypothesis

A small neural function can share information across layer, row, column, and block
coordinates:

```text
generator(layer, role, row, column) -> weight
```

The generator parameters can remain cached while arithmetic produces many weights.

### Proposed implementation

- Compare SIREN, Fourier-feature MLP, polynomial, and small state-space generators.
- Use normalized layer/role/row/column coordinates and learned low-dimensional embeddings.
- Generate tiles, not isolated scalars, to amortize generator state and instruction cost.
- Distill against teacher activations while constraining generator parameter count.
- Fuse tile generation with matrix multiplication; generated tiles stay in registers or
  shared memory and are discarded after accumulation.

### Success criteria

- Generator and coordinate embeddings use **no more than 50%** of dense FP16 state bytes
  for represented layers.
- Each generator-state byte supports at least 16 generated weights per warm layer call.
- Layer activation MSE `< 5e-4` after distillation.
- Generated tiles are never written to global memory.
- Added computation remains below 20x the dense multiply operation count and reaches at
  least 0.5x dense end-to-end throughput on target hardware.

---

## Experiment 9 — Cross-Layer Base Functions and Transformation Propagation

### Hypothesis

Layers with compatible roles may be transformations of shared procedural bases, allowing
one function definition to serve many layers:

```text
layer_weight = left_transform(layer) * shared_base(role)
             * right_transform(layer) + correction(layer)
```

Alternatively, an initial procedural state may be transformed through the network by a DSP
pipeline rather than storing independent weights for every layer.

### Proposed implementation

- Align same-role projections across layers using permutation, sign, scale, and orthogonal
  Procrustes transforms.
- Learn shared bases per projection role with small layer embeddings.
- Test recurrent evolution of generator state across layer index.
- Test activation-coordinate propagation so a layer's output order/basis becomes the next
  layer's native input order/basis, avoiding repeated inverse transforms.
- Compare independent residuals with low-rank or sparse layer corrections.

### Success criteria

- Shared base plus all layer transforms and corrections is **no more than 50%** of the
  represented dense FP16 layer state.
- At least 75% of model state is shared across two or more layers.
- Basis/permutation propagation eliminates explicit inverse transforms between at least two
  consecutive layers.
- Stage-1 activation MSE `< 5e-4` after joint multi-layer fitting.
- Corrections consume less than 20% of dense FP16 bytes.

---

## Experiment 10 — Program-Synthesis and Recurrence Search

### Hypothesis

Some rows, blocks, coefficient fields, permutations, or residual masks may be exactly or
approximately generated by tiny arithmetic programs even when raw weights are not.

### Proposed implementation

Retarget the existing lossless search away from decimal-string compression and toward a
small GPU/CPU-friendly generator bytecode:

- Constant, affine, polynomial, modular, and nonlinear recurrences.
- Periodic and nested-periodic generators.
- Small expression trees with integer/fixed-point arithmetic.
- Generated permutations and sparse masks.
- Exact residuals for diagnostic mode and bounded-error residuals for approximate mode.

Search transformed coefficient streams and metadata as well as raw FP16 row bits. Reject
ordinary LZ/LZMA fallbacks for procedural execution because they primarily replace one form
of memory decoding with another and do not provide direct fused arithmetic.

### Success criteria

- Decoder bytecode has a bounded, branch-light GPU/CPU implementation.
- Program operands and residuals use **no more than 50%** of their dense target's state
  bytes.
- At least 90% of emitted values come from arithmetic rather than literal operands.
- Exact programs regenerate source bytes byte-for-byte; approximate programs satisfy the
  layer output gate.
- Programs with effectively one literal per emitted value are classified as failures.

---

## Experiment 11 — Exact Procedural Component Plus Residual

### Hypothesis

Exact fidelity may be practical when a procedural predictor handles most layer behavior and
a small residual carries the irreducible information.

### Proposed implementation

For every preceding candidate, evaluate:

```text
weight_matrix = procedural_component + residual
```

Encode residuals as sparse FP16/FP8 values, entropy-coded integers, low-rank corrections, or
small block exceptions. Execute the procedural component and residual contribution as
separate fused paths without constructing the full matrix.

This experiment is the information-accounting guardrail: it determines whether structure
was discovered or merely approximated while a dense residual retained the original data.

### Success criteria

- Exact mode reproduces original FP16/BF16 layer outputs within deterministic accumulator
  tolerance.
- Residual values, indices, decoder state, and traffic are fully measured.
- Residual traffic is less than 25% of dense weight traffic for an advancement win.
- Approximate mode reports quality curves as residual precision and density decrease.
- A candidate fails the compute-shift goal if residual traffic dominates total traffic.

---

## Experiment 12 — Behavioral Distillation Into Procedural Families

### Status: Stage 1 (closed-form activation-aware fitting) complete — measured

Implemented in `src/harness/`:

- `activation_aware.py` — the closed-form stage-1 math: damped symmetric
  `C^{1/2}` / `C^{-1/2}` of the teacher activation second moment,
  activation-aware low-rank factors via whitened SVD
  (`min ||(W - UV) C^{1/2}||_F`, the exact minimizer of the on-distribution
  output error `E_x ||(W - UV) x||^2`), per-column activation importance
  `sqrt(C_jj)` for output-aware residual ranking, and the block-averaged
  sub-moment used to keep the shared-basis whitener shared across blocks.
- `activation_cache.py` — Epic 04 at small scale: runs the real
  `EleutherAI/pythia-14m` teacher over a bundled ~50-passage corpus, captures
  projection **inputs** with forward pre-hooks (QKV + MLP-down from layers 0
  and 3 — the two roles the advancement gate requires), and produces a
  fitting second moment plus **held-out** evaluation activations
  (deterministic disjoint split, disk-cached). Second moments are
  fitting-time state only, never inference model state.
- `low_rank_residual.py` / `shared_basis.py` — both families accept an
  optional `second_moment`; the hot path, state accounting, and
  zero-materialization contract are unchanged (verified by tests — the
  activation-aware DCT variant honestly counts its re-weighted stored basis).
- `experiment_12_behavioral.py` — sweep runner fitting every capacity point
  **twice** (weight-space vs. activation-aware) and reporting held-out
  on-distribution error, random-probe error, activation MSE, and paired
  gain per configuration.

38 contract tests in `tests/test_harness_experiment_12.py`; full harness
suite 163 tests, all passing.

**Synthetic control (full-rank 64x128 W, rank-12 activation subspace):**
activation-aware rank-16 hits 5.0e-3 on-distribution error at **0.38x dense
state** — passes both gates — while the weight-space fit at identical
capacity sits at 0.70 error. Random-probe error stays ~0.76 for the
activation-aware fit, confirming the error mass moved off-distribution
exactly as intended. Machinery validated.

**Real Pythia-14m (QKV + MLP-down, layers 0 and 3; fit N=1229 tokens,
held-out eval):** activation-aware fitting improves **every one of the 72
paired configurations** — gains 1.08–2.14x, mean ~1.25x, largest on the
deeper layer (layer-3 QKV mean 1.40x, best 2.14x at rank 51). Best points at
the 0.5x-state budget: layer-3 QKV **0.126 err at 0.334x state** (was 0.24 at
0.82x weight-space in Exp 3/5 — a real Pareto shift), layer-0 QKV 0.258 at
0.484x, MLP-down 0.30–0.40 at 0.498x. **No configuration reaches the 1e-2
gate**, so 0 advance under the strict stage-1 criterion.

**Interpretation:** the fitting-target hypothesis is confirmed directionally
— on-distribution structure is real, exploitable, and free (closed form),
and the QKV projections are markedly more compressible on-distribution than
in weight space. But Pythia-14m's layer inputs are not low-dimensional
enough (post-LayerNorm spectra are flat-ish at dim 128) for a *single-layer,
closed-form* fit to cross 1e-2 below 0.5x state. The remaining 10-30x error
gap is what stage 2/3 (joint multi-layer fitting, end-to-end logit
alignment) and the roadmap's model-level KL/perplexity gates — which do not
require 1e-2 per layer — are for. Note the roadmap's own model gates measure
logit KL <= 0.05 and perplexity delta <= +0.35, not per-layer 1e-2; a 0.13
relative layer error may well be tolerable end-to-end once later layers are
jointly fitted to absorb it.

Decision: **advance the activation-aware low-rank family to stage 2 (joint
consecutive-layer fitting with error feed-forward) and end-to-end KL
measurement; keep weight-space fits as the control arm. The gap to close is
~10x in layer error or a demonstration that model-level gates pass despite
it.**

### Hypothesis

Teacher behavior may be representable by compact procedural layers even when exact teacher
weights are not. Distillation can move information from dense literals into shared
functional structure.

### Proposed implementation

- Use Epic 04's cached teacher activation pairs.
- Stage 1: fit each procedural candidate to layer outputs with state and operation budgets.
- Stage 2: jointly optimize consecutive procedural layers to absorb local approximation
  errors.
- Stage 3: end-to-end logit KL and language-model loss alignment.
- Compare harmonic, shared-basis, butterfly, coordinate-network, and hybrid candidates under
  identical state-byte budgets.
- Penalize state traffic and residual density in the objective, not only parameter count.

### Success criteria

- Per-layer activation MSE `< 5e-4`.
- Logit KL divergence `<= 0.05`.
- WikiText-2 perplexity delta `<= +0.35`.
- Peak inference model RAM is no more than 50% of the dense baseline, and measured
  off-chip state traffic is lower than the dense baseline.
- No dense teacher weights are required at inference.

---

## Experiment 13 — Fused CPU/GPU Execution and Hardware Validation

### Hypothesis

A compact mathematical representation only achieves the mission if hardware executes it
without turning generated values back into memory traffic.

### Proposed implementation

- Build Rust CPU kernels for finalists using register-resident recurrent state and direct
  accumulation.
- Build CUDA or Triton prototypes for tile generation and structured activation transforms.
- Keep generator parameters in cache, constant memory, registers, or shared memory where
  capacity permits.
- Profile global-memory reads, cache hit rates, occupancy, instruction mix, arithmetic
  intensity, peak RAM, and generated-value stores.
- Compare batch size 1 autoregressive inference and larger prefill batches separately.

### Success criteria

- Zero global-memory writes of generated dense weights.
- Zero dense-weight allocation after model load.
- Measured off-chip state traffic agrees with analytical accounting within 20%.
- At least 50% lower peak inference model RAM and lower measured state traffic for the
  selected model.
- End-to-end generation throughput is at least 0.5x dense baseline; stretch target is parity
  or faster on bandwidth-bound hardware.

## Execution Order and Gates

### Phase A — Instrumentation and broad discovery

1. Common experiment harness.
2. Harmonic baseline.
3. Reversible ordering/transforms.
4. Shared spectral bases.
5. Cross-row/block predictors.
6. Low-rank/dictionary and Kronecker/tensor candidates.

**Gate A:** retain only Pareto-optimal methods whose compact state plus mandatory scratch is
no more than 50% of dense FP16/BF16 at `<= 1e-2` layer-output error. Methods slightly above
this budget may be retained only if distillation has a measured path to the 50% target.

### Phase B — Procedural-layer discovery

7. Butterfly/DSP layers.
8. Coordinate-network generators.
9. Cross-layer bases and transformation propagation.
10. Program-synthesis search on transformed state.
11. Exact residual accounting.

**Gate B:** select no more than three families with a credible path to at least 50% lower
peak inference model RAM, bounded compute, lower state traffic, and direct fused execution.

### Phase C — Learn behavior, not literal weights

12. Distill finalists using cached activations and end-to-end logits.

**Gate C:** require activation, KL, perplexity, state-byte, and residual criteria. Fidelity
alone is insufficient; state reduction alone is insufficient.

### Phase D — Prove the hardware claim

13. Fused CPU/GPU kernels and full autoregressive inference.

**Final gate:** demonstrate lower peak model RAM and lower measured off-chip model-state
traffic with no dense weights materialized, while maintaining acceptable quality and
throughput.

## Required Experiment Report

Every completed experiment must append a result using this template:

```text
Representation:
Tensor/model:
Shared executable bytes:
Unique model-state bytes:
Residual bytes:
Transient scratch bytes:
Dense FP16/BF16 baseline bytes:
Off-chip bytes per layer/token:
Generator/transform operations:
Dense weight materialized: yes/no
Weight reconstruction error (diagnostic):
Layer output error:
Activation MSE:
Logit KL:
Perplexity delta:
Latency and throughput:
Decision: advance / retain as baseline / stop
Reason:
```

This prevents a small function body from being mistaken for a small model when its operands,
residuals, or generated intermediates still carry dense-weight-scale information.
