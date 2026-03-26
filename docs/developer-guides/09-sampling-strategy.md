# Add a Sampling Strategy or Sampling Knob

## Goal

Add a new sampling parameter or change how token selection works after logits are produced.

## When To Use This Guide

- You need a new runtime sampling option such as an extra filter or penalty.
- The change belongs in generation-time policy, not in model conversion.

## Blast Radius

- Schema + runtime + tests

## Required Touch Points

- `src/config/schema/doppler.schema.js` and `.d.ts`
- If the shared benchmark contract exposes the knob:
  `src/config/schema/benchmark.schema.js` and `.d.ts`
- `src/inference/pipelines/text/sampling-config.js`
- Tests for the new sampling behavior

## Recommended Order

1. Add the field to the sparse runtime inference schema and declaration files.
2. Update any command or benchmark schema that needs to validate the new knob.
3. Implement the new behavior in `src/inference/pipelines/text/sampling.js`.
4. Add tests that cover default behavior, non-default behavior, and edge cases.

## Verification

- `npm run test:unit`
- Run a debug pass with the new knob set to a non-default value and confirm the sampling behavior changes

## Common Misses

- Putting the default in a checked-in config asset instead of the runtime schema.
- Changing benchmark-visible sampling semantics without updating the shared benchmark contract.
- Mutating the filtering order in a way that silently changes top-k or top-p behavior.
- Forgetting deterministic cases such as `temperature = 0`.

## Related Guides

- [07-manifest-runtime-field.md](07-manifest-runtime-field.md)
- [12-command-surface.md](12-command-surface.md)

## Canonical References

- `src/config/schema/doppler.schema.js`
- `src/config/schema/benchmark.schema.js`
- `src/inference/pipelines/text/sampling-config.js`
- [../config.md](../config.md)
