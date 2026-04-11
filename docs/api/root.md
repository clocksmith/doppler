# Root API

## Purpose

Primary application-facing API for loading models and generating text with the `doppler` facade.

## Import Path

```js
import { doppler } from 'doppler-gpu';
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

Advanced runtime helpers now live on dedicated subpaths such as
`doppler-gpu/loaders`, `doppler-gpu/orchestration`, `doppler-gpu/generation`,
and `doppler-gpu/tooling`.

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
- `doppler.text(...)` requires `options.model` and returns the final string
- `doppler.chat(...)` requires `options.model` and returns an `AsyncGenerator<string>`
- `doppler.chatText(...)` requires `options.model` and returns `{ content, usage }`
- `doppler.evict(model)` and `doppler.evictAll()` clear the convenience cache

### Fail-fast rules

- `doppler()`, `doppler.text()`, `doppler.chat()`, and `doppler.chatText()` all require `options.model`
- load-affecting options belong on `doppler.load()`, not the convenience call
- `runtimeConfig`, `runtimeProfile`, and `runtimeConfigUrl` are rejected on the convenience-call surface
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
- `advanced.prefillWithLogits(...)`
- `advanced.decodeStepLogits(...)`
- `advanced.generateWithPrefixKV(...)`

### `doppler(prompt, options)`

Returns an `AsyncGenerator<string>` and caches the loaded model by resolved model key.

### `doppler.text(prompt, options)`

Convenience wrapper that consumes the stream and returns a final string.

### `doppler.chat(messages, options)`

Formats chat input and returns an `AsyncGenerator<string>`.

### `doppler.chatText(messages, options)`

Formats chat input and returns a final `{ content, usage }` object.

### `doppler.listModels()`

Returns canonical quick-start `modelId` values known to the root facade.

## Minimal Example

```js
import { doppler } from 'doppler-gpu';

const model = await doppler.load('gemma3-270m');

for await (const token of model.generate('Describe WebGPU briefly')) {
  process.stdout.write(token);
}
```

## Advanced Example

```js
import { doppler } from 'doppler-gpu';

const model = await doppler.load('gemma3-270m', {
  onProgress: ({ message }) => console.log(`[doppler] ${message}`),
});

const reply = await model.chatText([
  { role: 'user', content: 'Write one sentence about WebGPU.' },
]);

console.log(reply.content);
await model.unload();
```

## Advanced Telemetry Example

Use the `advanced` handle when you need logits-backed instrumentation instead of
the standard generation surface.

```js
import { doppler } from 'doppler-gpu';

const model = await doppler.load('gemma3-270m');

const prefill = await model.advanced.prefillWithLogits('Write one word for GPU.');
const topLogit = prefill.logits[0];

const step = await model.advanced.decodeStepLogits(prefill.tokenIds);

console.log({
  prefillTokens: prefill.tokenIds.length,
  vocabSize: step.vocabSize,
  firstLogit: topLogit,
});

await model.unload();
```

## Code Pointers

- facade implementation: [src/client/doppler-api.js](../../src/client/doppler-api.js)
- facade types: [src/client/doppler-api.d.ts](../../src/client/doppler-api.d.ts)
- root export surface: [src/index.js](../../src/index.js)
- root type surface: [src/index.d.ts](../../src/index.d.ts)

## Related Surfaces

- [Loaders API](loaders.md)
- [Orchestration API](orchestration.md)
- [Generation API](generation.md)
- [Generated export inventory](reference/exports.md)
