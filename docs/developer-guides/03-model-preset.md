# Add a Model Preset

## Goal

Add a new model preset for a family that already fits Doppler's existing schema and runtime behavior.

## When To Use This Guide

- The model still fits an existing pipeline family such as the transformer text path.
- You need preset-driven architecture, inference, tokenizer, and detection metadata.
- You do not need a new pipeline family.

## Blast Radius

- JSON + loader registry

## Required Touch Points

- `src/config/presets/models/<id>.json`
- `src/config/loader.js`
- Conversion configs or tests that should use the new preset

## Recommended Order

1. Copy the closest existing preset from `src/config/presets/models/`.
2. Fill in `id`, `name`, `extends`, `architecture`, `inference`, `tokenizer`, and `detection`.
3. Add a static import, `PRESET_REGISTRY` entry, and `PRESET_DETECTION_ORDER` position in `src/config/loader.js`.
4. Wire the preset into a real conversion config or directly exercise `detectPreset()`.

## Verification

- `npm run onboarding:check:strict`
- Verify detection returns the new preset ID:

```bash
node --input-type=module -e "import { detectPreset } from './src/config/loader.js'; console.log(detectPreset({ model_type: 'your-model-type' }, 'YourArchitecture'))"
```

- Run one real convert/debug pass if the preset is immediately intended for use

## Common Misses

- Adding the JSON file but not registering it in `src/config/loader.js`.
- Putting detection patterns in the wrong order and shadowing a more specific family.
- Leaving `inference.kernelPaths` incomplete for the quantization and activation-dtype combinations you expect to support.
- Reintroducing runtime family detection in pipeline code instead of authoring the preset correctly.

## Related Guides

- [02-assign-chat-template.md](02-assign-chat-template.md)
- [04-conversion-config.md](04-conversion-config.md)
- [06-kernel-path-preset.md](06-kernel-path-preset.md)
- [07-manifest-runtime-field.md](07-manifest-runtime-field.md)

## Canonical References

- `src/config/presets/models/gemma3.json`
- `src/config/loader.js`
- [../style/general-style-guide.md](../style/general-style-guide.md)
- [../config.md](../config.md)
