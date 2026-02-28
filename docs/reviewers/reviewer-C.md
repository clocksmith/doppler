# Reviewer C Assignment (Tools, Benchmarks, Demos, Tests)

See central process in `docs/reviewers/coordination.md` before starting.

## Scope
- Own:
  - `tools/**`
  - `benchmarks/**`
  - `demo/**`
  - `tests/**`
- Prioritize files already marked with `console-io` risk in these trees because they are frequent policy violations.

## Primary objectives
1. Make CLI/tooling/demos deterministic with explicit config/timestamp inputs.
2. Convert exception behavior from implicit runtime fallback to explicit user-visible policy decisions.
3. Keep demo/tooling behavior aligned with core command contracts.

## Mandatory checks per file
- Identify and normalize:
  - implicit boolean fallback (`||`), nullish ambiguity, random timestamp naming, and compat tokens.
- Ensure output modes and debug labels remain explicit and deterministic.
- For CLI/tooling:
  - map behavior to runtime config and policy flags when practical,
  - avoid hidden semantic fallback.
- For tests:
  - preserve assertions and expected semantics after each deterministic refactor.

## Deliverables
- Update these tracking artifacts for every reviewed file:
  - `docs/tracking/execution-plane-review-queue.md`
  - `docs/tracking/execution-plane-review-queue.js-runtime.md`
  - `docs/tracking/execution-plane-audit.md`
  - `docs/tracking/execution-plane-audit.json`
  - `docs/tracking/execution-plane-audit.csv`
  - `docs/tracking/execution-plane-review-queue.json`
  - `docs/tracking/execution-plane-review-queue.js-runtime.json`
- For `tools` and `benchmarks` edits, run the nearest relevant script/validator only if explicitly requested by the owner; otherwise do not run external test commands without confirmation.

## Hard constraints
- No hidden runtime changes outside visible config knobs.
- Do not edit WebGPU kernel internals (`src/gpu/kernels/**`) unless it’s an explicit cross-file regression you are routing with reviewer B/D.
- Keep demo UX deterministic (timestamps, defaults, mode selection).

## Stop criteria
- Stop when all files in `tools`, `benchmarks`, `demo`, and `tests` that you claimed are marked `reviewed` in both queues.
