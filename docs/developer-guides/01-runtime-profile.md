# Add a Runtime Profile

## Goal

Add a named runtime profile that changes behavior through JSON only.

## When To Use This Guide

- You want a reusable `runtimeProfile` for CI, production, low-memory, trace, or model-specific tuning.
- You do not need new schema fields or runtime code.

## Blast Radius

- JSON only

## Required Touch Points

- `src/config/runtime/profiles/<bucket>/<id>.json`
- Optional docs if the profile becomes a canonical workflow

## Recommended Order

1. Copy the closest existing profile under `src/config/runtime/profiles/`.
2. Set `id`, `name`, `description`, `intent`, `stability`, `owner`, `createdAtUtc`, and `extends`.
3. Keep the `runtime` block minimal. Only set fields that differ from defaults or the parent profile.
4. Confirm discovery lists the new ID with `node src/cli/doppler-cli.js profiles --json`.
5. Use the new ID through `request.runtimeProfile` in a CLI or harness run.

## Verification

- `npm run onboarding:check`
- Run one command with the new profile ID:

```bash
node src/cli/doppler-cli.js debug --config '{
  "request": {
    "modelId": "gemma-3-270m-it-q4k-ehf16-af32",
    "workload": "inference",
    "runtimeProfile": "profiles/your-profile"
  },
  "run": { "surface": "auto" }
}'
```

Equivalent CLI shorthand:

```bash
node src/cli/doppler-cli.js debug \
  --config '{"request":{"modelId":"gemma-3-270m-it-q4k-ehf16-af32","workload":"inference"},"run":{"surface":"auto"}}' \
  --runtime-profile profiles/your-profile
```

## Common Misses

- Copying `default.json` wholesale instead of keeping the profile as a small override.
- Setting the wrong intent. Harnessed flows expect `runtime.shared.tooling.intent` to match the command contract.
- Putting behavior defaults in JS instead of schema or checked-in config JSON.
- Adding ad hoc URL/UI knobs when the setting belongs in runtime config.

## Related Guides

- [03-model-family-config.md](03-model-family-config.md)
- [04-conversion-config.md](04-conversion-config.md)
- [07-manifest-runtime-field.md](07-manifest-runtime-field.md)

## Canonical References

- [../config.md](../config.md)
- [../style/config-style-guide.md](../style/config-style-guide.md)
- `src/config/runtime/default.json`
- `src/config/schema/doppler.schema.js`
