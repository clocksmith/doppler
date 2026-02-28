# Reviewer A Assignment (Inference & Text Pipelines)

See central process in `docs/reviewers/coordination.md` before starting.

## Scope
- Own: all pending files under:
  - `src/inference/**`
  - `src/training/**`
  - `src/inference/pipelines/energy/**`
- Also own `tools/` files that are explicitly inference-facing:
  - `tools/convert-safetensors-node.js`
  - `tools/refresh-converted-manifest.js` (only if inference conversion semantics touched)

## Primary objectives
1. Replace config-forced or deterministic fallbacks with explicit policy checks.
2. Remove implicit behavior switches and random/time-based drift in inference/runtime paths.
3. Keep JS as orchestration glue; avoid introducing extra inline runtime branching unless required.

## Mandatory checks per file
- Confirm `||` fallback usage is replaced by:
  - explicit config keys, or
  - `== null`/`?.` checks when checking intentional nullish states.
- Ensure any `try/catch` has a deterministic failure path and does not swallow unsupported capability cases.
- Replace any non-deterministic timing source (`Date.now`, new Date without input) with deterministic config/time-source injection or injected test clocks.
- Ensure unsupported cases fail fast with actionable errors.

## Deliverables
- For every assigned file, update these tracking rows:
  - `docs/tracking/execution-plane-review-queue.md`
  - `docs/tracking/execution-plane-review-queue.js-runtime.md`
  - `docs/tracking/execution-plane-audit.md`
  - `docs/tracking/execution-plane-audit.json`
  - `docs/tracking/execution-plane-audit.csv`
  - `docs/tracking/execution-plane-review-queue.json`
  - `docs/tracking/execution-plane-review-queue.js-runtime.json`
- Mark state as `reviewed` only after all codepath decisions are documented in row `decision`.
- If no code change is required, use `reviewed-with-no-code-change` and explicitly state why.

## Hard constraints
- Never edit files outside your scope unless you coordinate ownership handoff.
- Do not introduce new helper logic outside shared inference utility modules.
- Do not modify `tools/vendor-bench.js` unless the change is required by inference conversion semantics.

## Stop criteria
- Stop when all `src/inference/**` and `src/training/**` files are marked reviewed in both queues and all required tracking updates are applied.
