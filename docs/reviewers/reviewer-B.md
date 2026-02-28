# Reviewer B Assignment (GPU, Memory, Kernel Runtime)

See central process in `docs/reviewers/coordination.md` before starting.

## Scope
- Own: all pending files under:
  - `src/gpu/**`
  - `src/memory/**`
- Shared ownership (only when reviewer A/D cannot claim):
  - `benchmarks/runners/transformersjs-bench.js`
  - `src/gpu/command-recorder.js` (if not already handled)

## Primary objectives
1. Keep GPU orchestration deterministic and policy-driven.
2. Enforce explicit unsupported-path failures for kernel-path and capability checks.
3. Remove implicit defaults by moving policy into JSON/config where practical.

## Mandatory checks per file
- Search for:
  - `||` fallback for operational values.
  - `try/catch` that silently recovers from deterministic failures.
  - random-time-source usage (`Date.now`, `new Date()`) without deterministic input.
- Convert to strict checks:
  - explicit config values,
  - explicit nullish checks for optional structures,
  - explicit error codes/messages for unsupported GPU/path.
- Ensure kernel selection and dispatch setup uses the same policy-driven behavior for browser/Node paths.

## Deliverables
- For every assigned file, update these tracking rows:
  - `docs/tracking/execution-plane-review-queue.md`
  - `docs/tracking/execution-plane-review-queue.js-runtime.md`
  - `docs/tracking/execution-plane-audit.md`
  - `docs/tracking/execution-plane-audit.json`
  - `docs/tracking/execution-plane-audit.csv`
  - `docs/tracking/execution-plane-review-queue.json`
  - `docs/tracking/execution-plane-review-queue.js-runtime.json`
- Choose a concrete decision label (`converted-to-policy`, `timestamp-override-config`, `reviewed-with-no-code-change`, etc.) for each row.

## Hard constraints
- No behavior drift across browser/node command surfaces.
- Do not add ad-hoc logging in runtime paths; use existing debug trace hooks.
- Never catch and ignore capability/dispatch errors; return explicit failures.

## Stop criteria
- Stop when all `src/gpu/**` and `src/memory/**` files are in `reviewed` state in both queues.
