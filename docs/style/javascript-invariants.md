# JavaScript Invariants (Quick Reference)

Rules that cause bugs when violated. Full rationale and examples: [JavaScript Style Guide](./javascript-style-guide.md).

## Role Boundaries

- JSON owns policy and selection state (manifests, runtime profiles, rule maps).
- JS owns orchestration: validation, pipeline setup, resource lifecycle, dispatch, readback.
- WGSL owns only math and memory transforms.

## JSON Rule Maps for Selection Logic

Any selection of kernel variants, dtype strings, or op names must use JSON rule maps via `selectRuleValue()`.
No inline ternaries/if-else for choosing variant strings or dtypes in production code.

## Manifest-First Contract

Any new inference knob must be wired end-to-end:
1. Add to `ManifestInferenceSchema`
2. Populate in converter (explicit conversion config + HF config mapping)
3. Merge in `src/config/merge.js` with `_sources`
4. Validate in `parseModelConfigFromManifest()`
5. Add tests for manifest values and override precedence

Do not reintroduce runtime model detection or hidden family fallbacks.

## Nullable Required Fields

- `null` = explicitly disabled (valid).
- `undefined` = missing (validation error).
- Check `=== undefined` for nullable fields, `== null` for non-nullable fields.

## Kernel Path Only

Kernel selection overrides use `kernelPath`. `kernelPlan` is removed and must not be reintroduced.
Kernel path overrides are config-only; harness/UI must not set kernel selection via ad-hoc flags.

## Runtime Configuration (Performance Invariants)

- Do not hardcode `f32` fallbacks when `shader-f16` is available.
- If `f32` is required, require an explicit config flag and log once per session.
- Put dtype/variant decisions in rule maps or schema-driven config, not ad-hoc conditionals.

## Runtime Failure-Path Invariants

- Every acquired resource must have one clear owner and one deterministic cleanup path.
- `createBuffer()`, pooled-buffer acquire, `mapAsync()`, staging readback — all must release on every throw path.
- Prefer `try/finally` or shared helpers; do not duplicate cleanup sequences inline.

## Harness Override Rules

- Runtime tunables are config-only. Harness/UI controls must not override tunables.
- Harness URLs accept only `runtimeProfile`, `runtimeConfig`, `runtimeConfigUrl`, or `configChain`.
