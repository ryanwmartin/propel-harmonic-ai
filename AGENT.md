# AGENT.md

## Core Engineering & Architectural Principles

When working on this repository, all code changes and additions must strictly adhere to the following principles:

### 1. Domain Boundary Separation
- Modules must be structured strictly according to their domain responsibility.
- **Encoder** (`src/encoder/`): Fitting weight derivatives into 1D harmonic parameters ($\Theta$).
- **Decoder** (`src/decoder/`): Procedural synthesis of original tensors via oscillator prefix sums on hot paths.
- **I/O** (`src/io/`): Reading and writing binary `.atom` files. No codec logic should reside in I/O handlers.
- **Batch Processing** (`src/batch/`): Model-level layer loops and dictionary-level tensor operations.

### 2. Naming Conventions & Human-Readable Code
- **No Abbreviations or Cryptic Variable Names**: Use explicit, self-documenting semantic identifiers.
  - Bad: `k`, `lr`, `d`, `acc`, `w`, `r`, `b`, `G`, `L`
  - Good: `harmonic_count`, `learning_rate`, `first_difference_vector`, `accumulator_weight`, `weight_matrix`, `row_index`, `block_index`, `harmonic_derivative_estimate`, `block_length`
- Write code that reads naturally like technical specifications.

### 3. Single-Argument & Value Object Functions
- Avoid large positional parameter lists or overloaded functions.
- Prefer single parameter signatures using immutable configuration objects or dataclasses (e.g., `EncoderConfig`, `AtomHeaderInformation`).

### 4. Pattern Matching over Deep Nested Conditional Trees
- Use structural pattern matching (`match ... case` in Python 3.10+) or lookup dispatchers rather than deeply nested `if/elif/else` statements.

### 5. Procedural Zero-Allocation Standard
- High-performance execution paths (decoding, oscillator loops) must operate procedurally without unnecessary intermediate memory allocations or matrix reshaping steps.

---

## Repository Map

```
propel-harmonic-ai/
├── AGENT.md                       # Coding standards and architectural guidance for AI and developers
├── README.md                      # Project overview and high-level description
├── agile/                         # Agile project management, epics, and roadmap specifications
│   ├── roadmap.md                 # Master project roadmap and execution plan
│   └── epic-01.md ... epic-10.md  # Detailed specifications per milestone
├── requirements.txt               # Dependency definitions (torch, numpy, pytest, etc.)
├── src/                           # Main source directory
│   ├── __init__.py                # Package initialization
│   ├── __main__.py                # Entry point CLI runner (`python -m src`)
│   ├── codec.py                   # Unified high-level facade for encoding and decoding
│   ├── phasor_atom.py             # Data classes representing block-wise parameters and tensors
│   ├── encoder/                   # Domain: Parameter fitting and derivative analysis
│   │   ├── config.py              # Configuration dataclasses (EncoderConfig)
│   │   ├── block_extractor.py     # Row chunking and exact anchor extraction
│   │   ├── harmonic_fitter.py     # Prony/Adam 1D sinusoid curve fitting
│   │   ├── tensor_encoder.py      # Weight matrix encoding orchestrator
│   │   └── auto_fitter.py         # Adaptive grid-search parameter optimizer
│   ├── decoder/                   # Domain: Procedural weight reconstruction
│   │   └── block_decoder.py       # Oscillator evaluation and prefix-sum integration
│   ├── io/                        # Domain: Binary file storage
│   │   ├── atom_writer.py         # Binary file serialization (`.atom`)
│   │   └── atom_reader.py         # Binary file deserialization with magic byte validation
│   └── batch/                     # Domain: Multi-tensor model batch processing
│       └── batch_processor.py     # Concurrent/batch encoding and decoding of model weights
└── tests/                         # Unit and integration test suites
    └── test_codec.py              # End-to-end codec validation tests
```

---

## How to Navigate and Extend

1. **Adding New Encoding Strategies**: Modify `src/encoder/harmonic_fitter.py` or extend `src/encoder/auto_fitter.py`.
2. **Improving Decoder Performance / Porting to C/Rust/CUDA**: Refer to `src/decoder/block_decoder.py` for the reference mathematical integration.
3. **Modifying File Serialization**: Update binary layout specifications in `src/io/atom_writer.py` and `src/io/atom_reader.py`.
4. **Running Tests**: Run `PYTHONPATH=. pytest` from the root directory to verify all contract and parity tests.
