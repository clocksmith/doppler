# Assign an Existing Chat Template

## Goal

Point a model preset at one of Doppler's built-in chat template types.

## When To Use This Guide

- The model already fits one of the built-in template styles.
- You do not need to implement a new formatter.

## Blast Radius

- JSON only

## Required Touch Points

- `src/config/presets/models/<model>.json`
- Re-converted artifacts if the template should appear in emitted manifests

## Recommended Order

1. Check the built-in keys in `src/inference/pipelines/text/chat-format.js`.
2. Set `inference.chatTemplate.type` and `inference.chatTemplate.enabled` in the model preset.
3. Re-run conversion if you need existing artifacts to pick up the new manifest value.
4. Run a debug pass with multi-turn messages and inspect the resulting prompt formatting.

## Verification

- `npm run onboarding:check`
- Confirm the preset uses one of the built-in keys: `gemma`, `llama3`, `gpt-oss`, `chatml`, `qwen`, or `translategemma`
- Run a chat-style debug pass and inspect the prompt in verbose output or the emitted manifest

## Common Misses

- Using a template key that is not in `CHAT_FORMATTERS`. Unknown keys fall back to plaintext formatting.
- Forgetting `"enabled": true`.
- Updating the preset but not re-converting an existing artifact that should carry the template in `manifest.json`.

## Related Guides

- [03-model-preset.md](03-model-preset.md)
- [04-conversion-config.md](04-conversion-config.md)
- [08-chat-template-formatter.md](08-chat-template-formatter.md)

## Canonical References

- `src/inference/pipelines/text/chat-format.js`
- `src/config/presets/models/gemma3.json`
- `src/converter/manifest-inference.js`
