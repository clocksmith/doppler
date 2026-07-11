# Root API

## Purpose

Primary application-facing API for loading models and generating text with the `dr` facade.
The older `doppler` name remains a compatibility alias.

## Import Path

```js
import { dr } from 'doppler-gpu';
```

## Audience

Application authors who want the simplest stable surface.

## Stability

Preferred public surface.
See [Subsystem Support Matrix](../subsystem-support-matrix.md) for how advanced
instance features such as LoRA loading relate to the tier1 contract.

## Primary Exports

- `dr`
- `doppler` (compatibility alias)
- `dr.load(model, options)`
- `dr.text(prompt, options)`
- `dr.chat(messages, options)`
- `dr.chatText(messages, options)`
- `dr.evict(model)`
- `dr.evictAll()`
- `dr.listModels()`

Advanced runtime helpers now live on dedicated subpaths such as
`doppler-gpu/loaders`, `doppler-gpu/orchestration`, `doppler-gpu/generation`,
and `doppler-gpu/tooling`.

## Model Inputs

`dr.load()` accepts:

- registry ID string, for example `'qwen3-0.8b'`
- `{ url }`
- `{ manifest, baseUrl? }`

A bare string is treated as a bundled/known registry ID, not a path heuristic.

## Core Behaviors

### Loading

- `dr.load()` creates an explicit model instance
- instance ownership is explicit; call `model.unload()` when done
- Node quick-start runs emit basic progress logs by default

### Convenience calls

- `dr(prompt, { model })` reuses a convenience cache
- `dr.text(...)` requires `options.model` and returns the final string
- `dr.chat(...)` requires `options.model` and returns an `AsyncGenerator<string>`
- `dr.chatText(...)` requires `options.model` and returns `{ content, usage }`
- `dr.evict(model)` and `dr.evictAll()` clear the convenience cache

### Fail-fast rules

- `dr()`, `dr.text()`, `dr.chat()`, and `dr.chatText()` all require `options.model`
- load-affecting options belong on `dr.load()`, not the convenience call
- `runtimeConfig`, `runtimeProfile`, and `runtimeConfigUrl` are rejected on the convenience-call surface
- unsupported resolution inputs fail fast rather than silently falling back

## Primary Symbol Notes

### `dr.load(model, options)`

Returns a `DopplerModel` instance with:

- `generate(...)`
- `generateText(...)`
- `chat(...)`
- `chatText(...)`
- experimental `loadLoRA(...)`
- experimental `unloadLoRA()`
- `unload()`
- `manifestHash`
- `advanced.tokenizeText(...)`
- `advanced.prefillKV(...)`
- `advanced.prefillWithLogits(...)`
- `advanced.prefillWithTokenLogits(...)`
- `advanced.prefillWithTokenLogitsFromKV(...)`
- `advanced.decodeStepLogits(...)`
- `advanced.generateWithPrefixKV(...)`

The text generation and `advanced.*` telemetry helpers are part of the promoted
root-facade story. LoRA instance methods are available on the same model object,
but remain outside the tier1 proof contract.

### `dr(prompt, options)`

Returns an `AsyncGenerator<string>` and caches the loaded model by resolved model key.

### `dr.text(prompt, options)`

Convenience wrapper that consumes the stream and returns a final string.

### `dr.chat(messages, options)`

Formats chat input and returns an `AsyncGenerator<string>`.

### `dr.chatText(messages, options)`

Formats chat input and returns a final `{ content, usage }` object.

### `dr.listModels()`

Returns canonical quick-start `modelId` values known to the root facade.

## Minimal Example

```js
import { dr } from 'doppler-gpu';

const model = await dr.load('qwen3-0.8b');

for await (const token of model.generate('Describe WebGPU briefly')) {
  process.stdout.write(token);
}
```

## Advanced Example

```js
import { dr } from 'doppler-gpu';

const model = await dr.load('qwen3-0.8b', {
  onProgress: ({ message }) => console.log(`[dr] ${message}`),
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
import { dr } from 'doppler-gpu';

const model = await dr.load('qwen-3-reranker-0-6b-q4k-ehf16-af32');
const yesTokenId = model.manifest.inference.rerank.trueTokenId;
const noTokenId = model.manifest.inference.rerank.falseTokenId;

const prefill = await model.advanced.prefillWithLogits('Write one word for GPU.');
const topLogit = prefill.logits[0];

const selected = await model.advanced.prefillWithTokenLogits(
  'Answer yes or no.',
  [yesTokenId, noTokenId],
  { useChatTemplate: false }
);

const step = await model.advanced.decodeStepLogits(prefill.tokenIds);

console.log({
  prefillTokens: prefill.tokenIds.length,
  vocabSize: step.vocabSize,
  firstLogit: topLogit,
  yesLogit: selected.logitsByTokenId[yesTokenId],
  noLogit: selected.logitsByTokenId[noTokenId],
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
