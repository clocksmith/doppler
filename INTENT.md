# INTENT: Doppler

Parent: none

## Need

Developers need a dependable, zero-egress way to package, verify, and run AI models in standard JavaScript environments without cloud APIs or native compilation toolchains.

## Target

Deliver an evidence-backed WebGPU model foundry and inference runtime that executes signed model capsules with high performance, deterministic precision, and inspectable receipts.

## Invariants

- Model execution runs locally using declared WebGPU or validated fallback runtimes; prompt data never leaves the host.
- Model capsules are immutable and cryptographically bound to their conversion source, tokenizer, and target plan.
- Unsupported device features, precision mismatches, or missing extensions fail closed with explicit receipts.
- Token generation exposes observable evidence (probabilities, surprisal, top candidates) without altering model weights.
- Standalone library utility remains independent of external network coordination or commercial licensing.

## Evidence

- Clean test execution via declared package verification suites (`npm test` and `npm run check:green`).
- Objective parity passing against reference model output oracles within declared tolerances.
- Goal completion matrix recorded in `src/config/goal-completion-matrix.json`.

## Non-goals

- General application UI and agent task orchestration.
- Hosting cloud API endpoints or multi-tenant model serving.
- Unvalidated model conversion without reproducible evaluation evidence.

## Truth

Execution proof and verifiable receipts determine capability. Benchmarks and compatibility claims are valid only when backed by reproducible local test evidence.

---

Links:
- Root strategy: [GOALS.md](GOALS.md)
- Technical charter: [CATSCAN.md](CATSCAN.md)
