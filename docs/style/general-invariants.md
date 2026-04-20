# General Invariants (Quick Reference)

Rules that cause bugs when violated. Full rationale and examples: [General Style Guide](./general-style-guide.md).

## Execution Plane Contract

- **JSON** owns policy: manifests, execution graphs, rule assets define runtime decisions before execution.
- **JS** owns orchestration: merge/validate config, allocate buffers, dispatch work, read back.
- **WGSL** owns compute only: deterministic arithmetic, no policy branching.
- Missing or ambiguous contract must fail fast. No hidden fallbacks.

## No Runtime Defaults in Code

Runtime code reads resolved config values directly. No literal fallbacks for tunables in JS.
If policy is missing, raise a typed configuration error — do not silently select an alternate behavior.

## Nullable Required Fields

- `null` = explicitly disabled (valid).
- `undefined` = not specified (validation error, fail fast).

## Manifest as Source of Truth

- Converter embeds all model-specific inference params in `manifest.json`.
- Runtime never detects model family in pipeline code.
- Pipeline reads config values directly, no architecture-string inference.

## Kernel Selection

- Fully explicit in the manifest execution graph. Each step pins exact WGSL file, entry point, and content digest.
- `defaultKernelPath` does not exist in v1 manifests. The execution graph is the sole dispatch contract.

## Performance Invariants (F32 Policy)

- F32 is a correctness fallback, not a performance default.
- When `shader-f16` is available, prefer `f16` for activations, KV cache, and intermediate buffers.
- Any `f32` path must be explicitly configured and logged once per session.

## No Ad-Hoc Debug Logging

Do not add temporary log statements. Use existing trace categories, config-driven probes, or permanent trace extensions.
Use the debug module (`src/debug/index.js`), not raw `console.*` in runtime code.

## Single Source of Truth

When the same metadata appears in multiple files, exactly one must be canonical.
Mirrors must be generated from that source and covered by a sync check.

## Failure-Path Regression Requirement

Any fix for buffer lifecycle, readback cleanup, or failure-path-only behavior must include a regression test exercising the failing path.

## Inventory Before Edit

When a failure indicates a repeated drift class, run or create the broadest inventory check before editing individual files. Classify the full set first, then batch fixes by decision type: fix to runtime, fix to declarations, remove stale surface, or quarantine as pending-feature/debt with owner and expiry.

One-off repairs for recurring drift should become checkable tooling with a `--check` mode before the task is considered complete.
