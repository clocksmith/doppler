# Root API

## Purpose

Primary application-facing API for loading models and generating text with the `doppler` facade.

## Import Path

```js
import { doppler } from '@simulatte/doppler';
```

## Audience

Application authors who want the simplest stable surface.

## Stability

Preferred public surface.

## Primary Exports

- `doppler`
- `doppler.load(model, options)`
- `doppler.text(prompt, options)`
- `doppler.chat(messages, options)`
- `doppler.chatText(messages, options)`
- `doppler.evict(model)`
- `doppler.evictAll()`
- `doppler.listModels()`

The root package also re-exports advanced loaders, pipelines, and adapter helpers.
Those advanced exports are documented separately in [Advanced Root Exports](advanced-root-exports.md), not as the primary quick-start contract.

## Model Inputs

`doppler.load()` accepts:

- registry ID string, for example `'gemma3-270m'`
- `{ url }`
- `{ manifest, baseUrl? }`

A bare string is treated as a bundled/known registry ID, not a path heuristic.

## Core Behaviors

### Loading

- `doppler.load()` creates an explicit model instance
- instance ownership is explicit; call `model.unload()` when done
- Node quick-start runs emit basic progress logs by default

### Convenience calls

- `doppler(prompt, { model })` reuses a convenience cache
- `doppler.text(...)` is the non-streaming wrapper
- `doppler.chat(...)` and `doppler.chatText(...)` format chat-style input
- `doppler.evict(model)` and `doppler.evictAll()` clear the convenience cache

### Fail-fast rules

- `doppler()` requires `options.model`
- load-affecting options belong on `doppler.load()`, not the convenience call
- unsupported resolution inputs fail fast rather than silently falling back

## Primary Symbol Notes

### `doppler.load(model, options)`

Returns a `DopplerModel` instance with:

- `generate(...)`
- `generateText(...)`
- `chat(...)`
- `chatText(...)`
- `loadLoRA(...)`
- `unloadLoRA()`
- `unload()`
- `advanced.prefillKV(...)`
- `advanced.generateWithPrefixKV(...)`

### `doppler(prompt, options)`

Returns an `AsyncGenerator<string>` and caches the loaded model by resolved model key.

### `doppler.text(prompt, options)`

Convenience wrapper that consumes the stream and returns a final string.

### `doppler.listModels()`

Returns bundled registry IDs known to the quick-start facade.

## Minimal Example

```js
import { doppler } from '@simulatte/doppler';

const model = await doppler.load('gemma3-270m');

for await (const token of model.generate('Hello, world')) {
  process.stdout.write(token);
}
```

## Advanced Example

```js
import { doppler } from '@simulatte/doppler';

const model = await doppler.load('gemma3-270m', {
  onProgress: ({ message }) => console.log(`[doppler] ${message}`),
});

const reply = await model.chatText([
  { role: 'user', content: 'Write one sentence about WebGPU.' },
]);

console.log(reply.content);
await model.unload();
```

## Code Pointers

- facade implementation: [src/client/doppler-api.js](../../src/client/doppler-api.js)
- facade types: [src/client/doppler-api.d.ts](../../src/client/doppler-api.d.ts)
- root export surface: [src/index.js](../../src/index.js)
- root type surface: [src/index.d.ts](../../src/index.d.ts)

## Related Surfaces

- [Advanced Root Exports](advanced-root-exports.md)
- [Provider API](provider.md)
- [Generation API](generation.md)
- [Generated export inventory](reference/exports.md)
