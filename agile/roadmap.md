# Phasor Inference — Roadmap

**Vision:** Prove the HWave codec works end-to-end — encode every weight tensor of a real
small LLM into Phasor Patches `Θ`, procedurally decode them on the Rust hot path, and run
full autoregressive inference in a terminal chat. The aha moment: a user types a message and
gets a coherent reply from a model whose weights are **never stored as dense matrices** —
they are synthesized on the fly from tiny `Θ` parameter sets.

**Guiding pivot (locked in):** This is **procedural execution via an implicit wave function
`G(Θ, r, c)`**, not "compression." The goal is replacing VRAM reads with a deterministic
decoder that synthesizes weights in registers. Success = reconstruction fidelity + zero-alloc
decode, *not* a headline compression ratio.

**Distillation-first strategy:** The direct codec (Epics 01–03) fits `G(Θ, r, c)` to raw
weight *values*. The distillation pipeline (Epics 04–06) fits `Θ` to match teacher
*behavior* — activations and logits. Distillation is the **primary encoding path** because
it optimizes for functional equivalence, not numerical approximation. Epic 03's gate
determines whether direct encoding suffices; if not (or if best quality is desired),
distillation takes over.

**Single format:** All `Θ` parameter sets — whether produced by direct encoding or
distillation — are stored in the same `.atom` binary format (flat `f32` array of
`4K+1` values with a header: `K`, `in_dim`, `out_dim`). One format, one extension, one
loader on the Rust side. No conversion step exists anywhere in the pipeline.

**Target model:** Gemma 2B (or similarly sized open-weights model) for the proof of concept.
Full fidelity encode/decode is expected — no information loss tolerance beyond the agreed
relative-error threshold from the Epic 03 gate.

## Epics

| # | Epic | Status | File |
|---|------|--------|------|
| 01 | Python Reference Codec (Encoder + Decoder) | Ready | `agile/epic-01.md` |
| 02 | Rust Hot-Path Decoder + Zero-Alloc Matched Filter | Ready | `agile/epic-02.md` |
| 03 | End-to-End Fidelity Gate & Parity Test | Ready | `agile/epic-03.md` |
| 04 | Distillation Data Pipeline — Activation Extraction & Calibration | Draft | `agile/epic-04.md` |
| 05 | Stage 1: Layer-Wise Activation Alignment Distillation | Draft | `agile/epic-05.md` |
| 06 | Stage 2: Global End-to-End Logit Alignment | Draft | `agile/epic-06.md` |
| 07 | Full-Model Encoder — Encode Every Weight of Gemma 2B | Draft | `agile/epic-07.md` |
| 08 | Rust LLM Forward Pass — Attention + FFN via Phasor Decode | Draft | `agile/epic-08.md` |
| 09 | Tokenizer + KV Cache + Autoregressive Generation Loop | Draft | `agile/epic-09.md` |
| 10 | Terminal Chat Interface — The Aha Moment | Draft | `agile/epic-10.md` |

**Execution order:** 01 → 02 → 03 (gate) → **04 → 05 → 06** (distillation) → 07 → 08 → 09 → 10.

Epic 03 remains the go/no-go gate for the direct codec. If it passes with acceptable
relative error, Epic 07 can use the direct encoder as a baseline. If it fails — or if
the fidelity target demands it — Epics 04–06 (distillation) produce the `.atom` files
instead. Either way, Epics 07–10 build the inference stack on top of whatever encoding
produces the best fidelity.

**Distillation gating criteria** (must pass before Epic 07 uses distilled weights):
| Metric | Target |
|--------|--------|
| Per-layer activation MSE (Stage 1) | < 5.0 × 10⁻⁴ |
| Logit KL-divergence (Stage 2) | ≤ 0.05 |
| Perplexity delta vs teacher | ≤ +0.35 on wikitext-2 |
| Memory bandwidth reduction | > 100× vs dense weight reads |

**Definition of "aha moment working":**
- Every weight tensor of Gemma 2B is encoded into `.atom` files (Epic 07, using distilled or direct encoding).
- The Rust forward pass synthesizes all weights procedurally — no dense weight matrices in memory (Epic 08).
- A tokenizer, KV cache, and sampling loop generate tokens autoregressively (Epic 09).
- A terminal REPL accepts user input, streams tokens, and produces coherent multi-turn chat (Epic 10).

**Explicitly out of scope:** 27B models, WASM, SIMD intrinsics, distributed inference,
GPU acceleration, multi-tile sharing / "27 MB" claims, production serving.
