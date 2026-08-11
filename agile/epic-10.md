# Epic 10: Terminal Chat Interface — The Aha Moment

## Status
Draft

## Goal
A terminal REPL where a user types a message, watches tokens stream back in real time, and
has a multi-turn conversation with Gemma 2B — all weights procedurally decoded from phasor
patches. This is the **ChatGPT aha moment**: a working chatbot with no dense weight matrices.

## User Stories
1. As a user, I want to type a message and see the response stream token-by-token so the interaction feels alive.
2. As a user, I want multi-turn conversation (the model remembers prior turns) so I can have a real dialogue.
3. As a developer, I want a startup banner showing model stats (parameters, phasor size vs. original, K profile) so the achievement is visible.
4. As a developer, I want `/reset`, `/stats`, `/tokens`, `/quit` commands so I can inspect and control the session.

## Acceptance Criteria
- [ ] `cargo run --bin phasor-chat -- --model model_phasor/` launches a terminal REPL.
- [ ] User types a message, hits enter, sees tokens appear one at a time (streaming, not batch).
- [ ] Multi-turn: the conversation history is maintained in the KV cache (or re-encoded each turn) — the model responds in context.
- [ ] Startup banner displays: model name, total parameters, phasor disk size vs. safetensors size, per-layer K profile, tokens/sec.
- [ ] REPL commands: `/reset` (clear history), `/stats` (session token count, avg tokens/sec, memory usage), `/tokens` (show last N token IDs), `/quit`.
- [ ] Graceful handling of: empty input, very long input (truncate to `max_seq_len`), Ctrl+C (clean exit with stats summary).
- [ ] A scripted demo mode (`--demo "Tell me a joke"`) that runs a single prompt non-interactively for testing/screenshots.

## Tasks
- [ ] Create the binary crate + CLI args
  - Files: `crates/phasor-chat/Cargo.toml`, `crates/phasor-chat/src/main.rs`
  - Details: use `clap` for arg parsing: `--model <path>` (required), `--max-tokens <n>`, `--temperature <f>`, `--top-k <n>`, `--top-p <f>`, `--demo <prompt>`. Depend on `phasor-inference` for model + generation.
- [ ] Implement streaming output
  - Files: `crates/phasor-chat/src/main.rs`
  - Details: the `generate()` loop from Epic 09 accepts a callback `FnMut(&str)`. In the callback, print each token string immediately with `io::stdout().flush()`. This gives the typewriter effect.
- [ ] Implement conversation history management
  - Files: `crates/phasor-chat/src/session.rs`
  - Details: maintain a `Vec<Message>` with roles (user/assistant). On each turn, format as Gemma chat template (`<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n`). Track total token count; when approaching `max_seq_len`, truncate oldest turns (or reset KV cache and re-prefill with truncated history).
- [ ] Implement REPL loop
  - Files: `crates/phasor-chat/src/repl.rs`
  - Details: `rustyline` for line editing + history. Print `You> ` prompt, read line, if starts with `/` dispatch command, else send to model and stream response with `Model> ` prefix. Track stats (total tokens, elapsed time).
- [ ] Implement REPL commands
  - Files: `crates/phasor-chat/src/commands.rs`
  - Details: `/reset` → clear history + KV cache. `/stats` → print session stats (tokens generated, avg tok/s, peak memory via `sysinfo` or manual tracking). `/tokens` → show last 20 token IDs + decoded strings. `/quit` → print summary, exit 0.
- [ ] Startup banner
  - Files: `crates/phasor-chat/src/banner.rs`
  - Details: read `manifest.json` + `encoding_report.json` from the model dir. Display: model name, total params, dense size (MB), phasor size (MB), ratio, per-layer K, tile count. ASCII art optional but encouraged.
- [ ] Integration test: scripted conversation
  - Files: `crates/phasor-chat/tests/chat_script.rs`
  - Details: run `--demo "What is 2+2?"` with a phasor-encoded model, capture stdout, assert it contains a plausible response (e.g., the string "4" or non-empty output of >10 chars). This is the smoke test for the full pipeline.

## Dependencies
- Epic 09 (tokenizer + generation loop)

## Definition of Done (The Aha Moment)
Running:
```
$ cargo run --release --bin phasor-chat -- --model model_phasor/
╔══════════════════════════════════════╗
║  Phasor Inference — Gemma 2B        ║
║  Weights: 2.1B params               ║
║  Dense: 5,016 MB → Phasor: XXX MB  ║
║  Every weight synthesized from Θ    ║
╚══════════════════════════════════════╝

You> What is the capital of France?
Model> The capital of France is Paris.

You> /quit
Session: 12 tokens generated, 3.2 tok/s
```
A working chatbot. Every weight synthesized from wave functions. No dense matrices.
That's the aha moment.
