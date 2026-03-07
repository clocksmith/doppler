# Add a Runtime Preset

## Goal

Add a named runtime preset that changes behavior through JSON only.

## When To Use This Guide

- You want a reusable `runtimePreset` for debug, bench, CI, or model-specific tuning.
- You do not need new schema fields or runtime code.

## Blast Radius

- JSON only

## Required Touch Points

- `src/config/presets/runtime/<bucket>/<id>.json`
- Optional docs if the preset becomes a canonical workflow

## Recommended Order

1. Copy the closest existing preset under `src/config/presets/runtime/`.
2. Set `id`, `name`, `description`, `intent`, `stability`, `owner`, `createdAtUtc`, and `extends`.
3. Keep the `runtime` block minimal. Only set fields that differ from defaults or the parent preset.
4. Use the new ID through `request.runtimePreset` in a CLI or harness run.

## Verification

- `npm run onboarding:check`
- Run one command with the new preset ID:

```bash
node tools/doppler-cli.js debug --config '{
  "request": {
    "modelId": "gemma-3-270m-it-q4k-ehf16-af32",
    "runtimePreset": "modes/your-preset"
  },
  "run": { "surface": "auto" }
}'
```

## Common Misses

- Copying `default.json` wholesale instead of keeping the preset as a small override.
- Setting the wrong intent. Harnessed flows expect `runtime.shared.tooling.intent` to match the command contract.
- Putting behavior defaults in JS instead of schema or preset JSON.
- Adding ad hoc URL/UI knobs when the setting belongs in runtime config.

## Related Guides

- [03-model-preset.md](03-model-preset.md)
- [04-conversion-config.md](04-conversion-config.md)
- [07-manifest-runtime-field.md](07-manifest-runtime-field.md)

## Canonical References

- [../config.md](../config.md)
- [../style/config-style-guide.md](../style/config-style-guide.md)
- `src/config/presets/runtime/default.json`
- `src/config/schema/doppler.schema.js`
