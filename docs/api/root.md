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
- `dr.open(model, options)`
- `dr.generate(model, input, options)`
- `dr.text(prompt, options)`
- `dr.chat(messages, options)`
- `dr.chatText(messages, options)`
- `dr.evict(model)`
- `dr.evictAll()`
- `dr.listModels()`
- `dr.listModelDetails()`
- `dr.listPersistentModels()` (browser)
- `dr.removePersistentModel(model)` (browser)

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
- browser callers can set `options.cache: 'opfs'` to download and verify the
  complete RDRR artifact before loading it from persistent browser storage
- `options.cache: false` keeps the streaming HTTP load path; unsupported cache
  values and Node-side OPFS requests fail before model resolution

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
- `cache` is rejected on the convenience-call surface; persistent caching is
  explicit model-instance policy on `dr.load()`
- unsupported resolution inputs fail fast rather than silently falling back

## Primary Symbol Notes

### `dr.load(model, options)`

Returns a `DopplerModel` instance with:

- `generate(...)`
- `generateText(...)`
- `generateWithEvidence(...)`
- `chat(...)`
- `chatText(...)`
- experimental `loadLoRA(...)`
- experimental `unloadLoRA()`
- `unload()`
- `manifestHash`
- `persistentCache` (`null` or the verified OPFS cache receipt)
- `inspect.listPolicies()`
- `inspect.generate(prompt, options)`
- `advanced.tokenizeText(...)`
- `advanced.prefillKV(...)`
- `advanced.resetToSeqLen(...)`
- `advanced.prefillWithLogits(...)`
- `advanced.prefillWithTokenLogits(...)`
- `advanced.prefillWithTokenLogitsFromKV(...)`
- `advanced.decodeStepLogits(...)`
- `advanced.generateWithPrefixKV(...)`

The text generation and `advanced.*` telemetry helpers are part of the promoted
root-facade story. LoRA instance methods are available on the same model object,
but remain outside the tier1 proof contract.

### `dr.open(model, options)`

Returns a scoped session with explicit ownership and capability discovery:

```js
const session = await dr.open('qwen3-0.8b', { cache: 'opfs' });
session.require('generate');

const result = await session.generate('Describe WebGPU briefly');
console.log(result.outputText, result.fingerprint);

await session.close();
```

The session exposes `generate`, `stream`, `embed`, `encodeSequence`, `inspect`,
`supports`, `require`, and idempotent `close`. Unsupported capabilities fail
before execution. `generate` always returns
`doppler.generation-result/v1`; `stream` always emits versioned semantic events.
The observation policy is explicit and recorded in the result fingerprint.

`always` records identity and coarse execution evidence without requesting
additional logits. `guided-quality` and `deep-xray` label the requested
inspection tier and report unavailable observations instead of inventing them.
The result's `executionChanged` field states whether observation changed the
execution path.

### `dr.generate(model, input, options)`

One-shot scoped generation. Doppler opens the model, returns the same stable
generation result as `session.generate`, and closes the GPU session in a
`finally` block. Use `dr.open` when multiple calls should share one loaded
session.

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
  cache: 'opfs',
  onProgress: ({ message }) => console.log(`[dr] ${message}`),
});

const reply = await model.chatText([
  { role: 'user', content: 'Write one sentence about WebGPU.' },
]);

console.log(reply.content);
await model.unload();
```

## Generation Evidence Example

Use `generateWithEvidence()` when a caller needs the generated output and a
browser-safe receipt that binds the transcript to the resolved generation
config, runtime profile, WebGPU backend, and execution-plan identity.

```js
import { dr } from 'doppler-gpu';

const model = await dr.load('qwen3-0.8b');
const evidence = await model.generateWithEvidence('Describe WebGPU briefly', {
  maxTokens: 64,
  temperature: 0,
});

console.log({
  outputText: evidence.outputText,
  tokenIds: evidence.tokenIds,
  transcriptHash: evidence.transcriptHash,
  generationConfigHash: evidence.generationConfigHash,
  runtimeProfileHash: evidence.runtimeProfileHash,
  backendIdentityHash: evidence.backendIdentityHash,
});

await model.unload();
```

The receipt records what ran. It does not by itself establish semantic
correctness or output quality.

## Inspection contract

`model.inspect.generate()` resolves a checked-in observation policy and returns
`doppler.model-inspection-receipt/v1`. It binds artifact, tokenizer, prompt
token IDs, sampling, observation policy, execution plan, browser, and adapter
into canonical comparison fingerprints.

```js
const receipt = await model.inspect.generate('Describe WebGPU briefly', {
  policyId: 'demo/guided-quality',
  generation: {
    maxTokens: 64,
    temperature: 0,
  },
});

console.log(receipt.quality.words.map((word) => ({
  word: word.text,
  summedSurprisal: word.summedSurprisal,
  rollingPerplexity: word.rollingPerplexity,
  cumulativePerplexity: word.cumulativePerplexity,
})));
```

The always-on policy is the only demo policy whose wall timing is performance
representative. Guided quality and Deep X-Ray modify execution. Perplexity
comparisons fail closed when tokenizer identity differs.

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
