# Add a Built-In Chat Template Formatter

## Goal

Add a new built-in chat template type that can be referenced by `manifest.inference.chatTemplate.type`.

## When To Use This Guide

- The model needs a prompt format that the built-in registry does not already support.
- You want the formatter to be part of Doppler's reusable template set.

## Blast Radius

- JS + type declarations + tests

## Required Touch Points

- `src/inference/pipelines/text/chat-format.js`
- `src/inference/pipelines/text/chat-format.d.ts`
- If you expose a named helper on the provider surface:
  `src/client/doppler-provider/generation.js`
  `src/client/doppler-provider/generation.d.ts`
  `src/client/doppler-provider.js`
  `src/client/doppler-provider.d.ts`
- A conversion config or test fixture that uses the new template key
- A targeted test under `tests/`

## Recommended Order

1. Implement the formatter in `src/inference/pipelines/text/chat-format.js`.
2. Register the new key in `CHAT_FORMATTERS`.
3. Update `chat-format.d.ts` and any provider exports you want to keep public.
4. Point a conversion config or test case at the new template key.
5. Add tests for supported roles, rejected roles, and output shape.

## Verification

- `npm test`
- Run one debug pass or direct formatter call with multi-turn messages and inspect the exact emitted prompt

## Common Misses

- Registering the formatter in JS but forgetting the `.d.ts` updates.
- Forgetting the provider export when following an existing public formatter pattern.
- Silently dropping unsupported roles instead of throwing an explicit error.
- Not re-converting artifacts whose manifests should now carry the new template type.

## Related Guides

- [02-assign-chat-template.md](02-assign-chat-template.md)
- [03-model-family-config.md](03-model-family-config.md)

## Canonical References

- `src/inference/pipelines/text/chat-format.js`
- `src/inference/pipelines/text/chat-format.d.ts`
- `tests/integration/qwen-chat-template.test.js`
- `tests/integration/translate-request-shape.test.js`
