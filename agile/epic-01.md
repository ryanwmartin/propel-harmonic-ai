# Epic 01: Python Reference Codec — Block-wise Derivative Encoder + Decoder

## Status
**Done** — all acceptance criteria met, 27/27 contract tests passing. See Completion Notes below.

## Goal
A runnable `HWaveCodec` that encodes a weight tensor `W` (shape `(rows, cols)`) into a
set of per-row, per-block harmonic parameter sets `Θ` via **block-wise derivative
fitting**, and decodes `Θ` back into `Ŵ` procedurally by running an oscillator and
integrating within each block. This is the **source of truth** the Rust decoder must
match.

**The encoding model.** Each row of `W` is split into `num_blocks = ceil(cols /
block_size)` blocks. Within a block of length `L`:
- Store the block's first weight as an exact **anchor** `a = W[row, block_start]`.
- Compute the first difference `d[x] = W[row, block_start+x+1] − W[row, block_start+x]`
  for `x ∈ [0, L−1)`.
- Fit a 1D harmonic formula `G(x) = Σ_k A_k·sin(2πf_k·x + φ_k)` to `d[x]`.
- Decode: `Ŵ[0] = a`; `Ŵ[x+1] = Ŵ[x] + G(x)` (running prefix-sum).

**Tunable `block_size`.** The encoder takes `block_size` as a parameter. It may select
`block_size` per tensor to satisfy a target relative-error threshold: decrease
`block_size` (more anchors, less drift) or increase per-block `K` (more harmonics) until
the fidelity metric is met. This is the primary fidelity knob.

## User Stories
1. As a researcher, I want to encode any weight tensor into per-block anchors + harmonic params so I can quantify the fidelity vs. `block_size`/`K` trade-off.
2. As a systems engineer, I want a deterministic decode function (oscillator + prefix-sum) so the Rust port has an exact reference to test against.
3. As a researcher, I want the block fits to be near-closed-form (Prony/ESPRIT) so encoding is fast and reproducible without a fragile global 2D Adam fit.

## Acceptance Criteria
- [x] `HWaveCodec.encode()` + `HWaveCodec.decode()` round-trip a weight tensor for a range of `block_size` and `K`.
- [x] MSE and relative error printed for every encode/decode cycle, broken down per `block_size`/`K` configuration.
- [x] CLI runs: `python -m src.codec --rows 128 --cols 128 --block-size 32 --K 8`
- [x] `Θ` serialized to disk as `.atom` binary file (anchors + per-block harmonics).
- [x] `block_size` auto-selection: given a target relative error, the encoder reports the `(block_size, K)` it chose.

## Tasks
- [x] Create Python project skeleton
  - Files: `requirements.txt`, `src/__init__.py`
  - Details: `numpy`, `scipy`, `torch`, `matplotlib` in requirements (plus `pytest` for the contract suite)
- [x] Implement `BlockAtom` parameter container
  - Files: `src/phasor_atom.py`
  - Details: `@dataclass` holding `anchors` `(rows, num_blocks)`, `amplitudes` `(rows, num_blocks, K)`, `frequencies` `(rows, num_blocks, K)`, `phases` `(rows, num_blocks, K)`, plus `block_size`, `num_blocks`, `rows`, `cols` as `torch.Tensor`/ints.
- [x] Implement 1D harmonic evaluation + block integrator
  - Files: `src/phasor_atom.py` (delegates to `src/decoder/block_decoder.py`)
  - Details: `eval_block(atom, row, block) -> Tensor[L]` computes `G(x) = Σ_k A_k·sin(2πf_k·x + φ_k)` for `x ∈ [0, L)`. `decode_row(atom, row) -> Tensor[cols]` does `anchor`, then `torch.cumsum(G)` per block, concatenating blocks. This is the decode contract Epic 02 must match.
- [x] Implement block-wise derivative encoder
  - Files: `src/encoder.py` → implemented as `src/encoder/harmonic_fitter.py`
  - Details: `encode_block(d, K) -> (A, f, φ)` fits K sinusoids to the first-difference vector `d`. Implemented path: FFT warm-start on `d` → top-K peaks (DC-aware) → best-so-far Adam refinement on the 1-D block only. Prony/ESPRIT deferred (see Deviations).
- [x] Implement derivative + anchor extraction
  - Files: `src/encoder.py` → implemented as `src/encoder/block_extractor.py`
  - Details: `extract_blocks(W, block_size)` splits each row into blocks, records each block's anchor (first weight, exact f32) and first-difference vector. Handles edge blocks shorter than `block_size` (zero-pad the difference vector; the anchor is always exact).
- [x] Implement `block_size`/`K` auto-selection
  - Files: `src/encoder.py` → implemented as `src/encoder/auto_fitter.py`
  - Details: `fit_tensor(W, target_rel_error, block_size_init, K_init)` — encode, measure relative error, and adjust: if error too high, first try `K *= 2` (up to `K_max`), then `block_size //= 2` (more anchors). Returns the chosen `(block_size, K)`, final error, and a per-configuration `ConfigEvaluation` breakdown.
- [x] Implement `HWaveCodec` facade
  - Files: `src/codec.py`
  - Details: `.encode(W, config) → Θ`; `.decode(Θ) → Ŵ`; `.save(Θ, path)` / `.load(path) → Θ`; `.fit(W, config, search_space) → FitResult`.
- [x] Add serialization to `.atom` binary format
  - Files: `src/codec.py` → implemented as `src/io/atom_writer.py` + `src/io/atom_reader.py`
  - Details: header (`ATOM` magic, version, `K`, `block_size`, `num_blocks`, `rows`, `cols`) then per row, per block: `f32` anchor + `3K` little-endian `f32` values (`A_0..A_{K-1} | f_0..f_{K-1} | φ_0..φ_{K-1}`). Loader validates magic, version, truncation, and header geometry consistency (`num_blocks == ceil(cols/block_size)`).
- [x] CLI entry point
  - Files: `src/codec.py` (`__main__`), `src/__main__.py`
  - Details: `--rows 128 --cols 128 --block-size 32 --K 8` → encode synthetic tensor, decode, print MSE + relative error + parameter count (with ratio vs dense) + chosen parameters. `--target-rel-error` triggers auto-selection with a per-configuration breakdown table. `--out` saves the `.atom` file and verifies a bit-exact reload round-trip.

## `.atom` binary layout (little-endian)

```
u32 magic "ATOM" | u32 version=3 | u32 K | u32 block_size | u32 num_blocks | u32 rows | u32 cols | u32 flags
then for each row r in [0, rows):
  for each block b in [0, num_blocks):
    f32 anchor
    f32[K] amplitudes
    f32[K] frequencies
    f32[K] phases
if flags & 1:
  u32[cols] shared_column_order
```

Per block: `1 + 3K` floats. Total per tensor: `rows · num_blocks · (1 + 3K)` floats plus
the 32-byte header and, when shared column ordering is enabled, `4 · cols` permutation
bytes. Version 3 adds optional transform flags; the loader remains backward-compatible
with the unflagged 28-byte version 2 format.

## Decode contract (must match Epic 02 exactly)

```
for r in 0..rows:
  for b in 0..num_blocks:
    acc = anchor[r][b]                 # f32 accumulator — see precision note
    W[r][b*block_size] = acc
    for x in 0..block_len(b):          # block_len = block_size except last block
      acc += G(r, b, x)                # G = Σ_k A_k·sin(2πf_k·x + φ_k)
      W[r][b*block_size + x + 1] = acc
```

**Precision contract (pinned during this epic):** the running prefix-sum accumulates in
**float32** (`torch.cumsum` on an f32 tensor in the Python reference). The Rust decoder
must accumulate in `f32` — not `f64` — or block-tail values will diverge beyond the
1e-6 parity tolerance mandated by Epic 03. Enforced by
`tests/test_codec.py::TestBlockMath::test_decode_accumulates_in_float32`.

## Completion Notes

**Final structure** (the flat `src/encoder.py` planned above became a domain package):

```
src/
  encoder/         block_extractor, harmonic_fitter, tensor_encoder, auto_fitter, config
  decoder/         block_decoder (the decode contract — single source of truth for Epic 02)
  io/              atom_writer, atom_reader (.atom binary format)
  batch/           batch_processor (multi-tensor encoding)
  phasor_atom.py   BlockAtom container
  codec.py         HWaveCodec facade + CLI
```

**Verification** (`python -m pytest tests/ -q` → 27 passed):
- Decode contract: block geometry, prefix-sum integration, f32 accumulator precision, determinism.
- Binary layout: byte-level header validation, save/load bit-exact round-trip, rejection of bad magic / wrong version / truncated payload / inconsistent geometry / zero dimensions.
- Fitting: pure-sinusoid recovery, constant-derivative (linear trend) exactness, edge-block handling.
- Auto-fitter: convergence to target, per-configuration evaluation history.

**Sample CLI results** (synthetic structured 32×64 tensor, `--target-rel-error 0.05`):

| block_size | K | MSE | rel_error | params |
|---|---|---|---|---|
| 32 | 2 | 7.19e-02 | 5.59e-01 | 448 |
| 32 | 4 | 7.93e-02 | 5.87e-01 | 832 |
| 32 | 8 | 1.99e-02 | 2.94e-01 | 1600 |
| 32 | 16 | **5.36e-13** | **1.53e-06** | 3136 (1.53× dense) |

At `K = L/2` the FFT warm start is a complete real DFT of the block and reconstruction
is exact to f32 rounding — confirming the format can represent any block losslessly at
`~1.5×` dense parameter cost. Fidelity below that budget is what the Epic 03 gate must
measure on **real** weights.

**Notable fixes landed while closing this epic:**
1. **DC-bin handling** — the warm start previously zeroed the DC bin of the derivative
   spectrum. A constant first-difference is a *linear trend* in the weights, which the
   anchor cannot absorb; DC is now a first-class peak candidate encoded as
   `A·sin(2π·0·x + π/2) = A`, with correct 1/N (vs 2/N) scaling for DC and Nyquist bins.
   Linear rows now round-trip exactly (`test_linear_trend_roundtrip`).
2. **Best-so-far Adam refinement** — refinement now tracks the lowest-loss parameters
   across all steps including the warm start itself, so gradient steps can never return
   a worse fit than the FFT initialization (previously Adam could walk away from an
   exact DFT reconstruction).
3. **Vectorized decode** — the per-element Python accumulation loop was replaced with
   `torch.cumsum` per block, as this epic's task list specified, and the f32
   accumulator precision was pinned as the Epic 02 parity contract.
4. **Loader geometry validation** — `.atom` headers with `num_blocks ≠
   ceil(cols/block_size)` or zero dimensions are now rejected at load time.
5. **Per-configuration breakdown** — `FitResult.evaluations` records
   `(config, MSE, rel_error, param_count)` for every encode/decode cycle of the
   auto-selection search; the CLI prints the table.
6. **Shared-column ordering experiment** — added an optional deterministic PCA column
   order shared by all rows, inverse restoration during full decode, and `.atom` v3
   serialization with a `u32[cols]` permutation payload. Contract tests prove the
   transform and file round-trip are reversible. On real Pythia-14m matrices it did
   **not** make low-K derivatives more compact: K=8 relative error worsened from
   1.2707 to 1.3566 (attention QKV) and from 1.2031 to 1.2622 (MLP down-projection).
   Full-spectrum fidelity remains intact, but the permutation adds `4·cols` bytes.
   This rules out shared PCA sorting as a direct-codec fidelity improvement; it remains
   available as an explicitly selected experimental transform rather than a default.
7. **Lossless per-row program-search experiment** — implemented a self-describing exact
   search compressor (`src/search_compressor.py`) with raw radix packing, periodic forms,
   continued-fraction rationals, power-plus-offset expressions, decimal modular generators,
   and ordinary zlib/LZMA candidates. For real safetensor rows the adapter correctly searches
   original FP16 bytes rather than decimal renderings and verifies every decoded IEEE-754 bit.
   On the representative Pythia-14m matrices, 423/512 rows selected raw bytes and 89 selected
   zlib; complete per-row framing produced an aggregate **1.0111×** FP16 ratio (1.11% larger).
   Thus exact program search works well on generated/periodic inputs but finds no aggregate
   storage or RAM win on trained weight rows. It remains an isolated experiment and is not
   part of `.atom` or the Rust hot-path contract.

**Deviations from plan (accepted):**
- **Prony/ESPRIT not implemented.** The FFT warm-start + best-so-far Adam path meets all
  acceptance criteria. Closed-form harmonic retrieval remains a performance optimization
  tracked for Epic 07 (full-model encoding), where the per-block Python/Adam loop must be
  vectorized/batched regardless.
- **Encoder is per-block sequential.** Fine for reference correctness and Epic 03 grid
  experiments on small tensors; batched/vectorized encoding is an Epic 07 prerequisite.

**Hand-off to Epic 02/03:**
- Decode contract file: `src/decoder/block_decoder.py` (f32 accumulator pinned).
- Binary contract tests: `tests/test_codec.py::TestAtomFileFormat` (byte-level layout).
- Open question for the Epic 03 gate: parameter budget. Auto-selection currently
  escalates `K` without a cap relative to dense size; the gate should report
  params-vs-dense ratio per configuration (now surfaced in `FitResult.evaluations`
  and the CLI) and define the acceptable fidelity threshold.

## Note for Epic 03

The fidelity gate must run on a **real** `nn.Linear` weight, not synthetic data, and must
sweep `block_size` as well as `K`. The decisive experiment is whether the *first
difference* of a real weight row is spectrally compact (few harmonics per block) — if the
derivative is broad-band/white, no `block_size` will yield compression and the gate should
fail toward distillation. Contract tests live in `tests/test_codec.py`.
