# Add a Manifest or Runtime Contract Field

## Goal

Add a new behavior-controlling field without breaking Doppler's manifest-first and config-first contracts.

## When To Use This Guide

- You need a new inference knob, cache setting, or model-behavior field.
- Larger journeys depend on a new field before they can be implemented cleanly.

## Blast Radius

- Schema + merge + parser + tests

## Required Touch Points

- The owning schema file under `src/config/schema/*.schema.js` and `.d.ts`
- `src/config/merge.js` when manifest and runtime layers need explicit precedence tracking
- The runtime consumer, often `src/inference/pipelines/text/config.js`
- If manifest-owned: `src/converter/manifest-inference.js` and possibly model presets or conversion configs
- Tests under `tests/config/`, `tests/converter/`, or the relevant runtime area

## Recommended Order

1. Decide whether the field is runtime-owned, conversion-owned, or both.
2. Add the field to the schema and declaration files, including defaults where appropriate.
3. Propagate it through `src/config/merge.js` with `_sources` tracking if it participates in manifest-vs-runtime precedence.
4. Parse and validate it in the runtime consumer.
5. If the field is manifest-owned, author it in conversion output through presets or `src/converter/manifest-inference.js`.
6. Add regression tests that prove the field survives the intended path without silent rewrite.

## Verification

- `npm run test:unit`
- `npm run onboarding:check`
- If manifest-owned, run one real conversion or the targeted converter test that exercises the field

## Common Misses

- Treating a runtime-only tuning knob as if it belongs in converted manifests.
- Adding the field to schema but not to merge or parser code, leaving it as a silent no-op.
- Forgetting `_sources` tracking when field precedence matters.
- Using `undefined` where the contract requires explicit `null`.
- Updating behavior without updating docs and tests in the same change.

## Related Guides

- [03-model-preset.md](03-model-preset.md)
- [09-sampling-strategy.md](09-sampling-strategy.md)
- [13-attention-variant.md](13-attention-variant.md)
- [14-quantization-format.md](14-quantization-format.md)
- [15-kvcache-layout.md](15-kvcache-layout.md)

## Canonical References

- [../style/general-style-guide.md](../style/general-style-guide.md)
- [../style/javascript-style-guide.md](../style/javascript-style-guide.md)
- [../conversion-runtime-contract.md](../conversion-runtime-contract.md)
- `src/config/merge.js`
- `src/inference/pipelines/text/config.js`
