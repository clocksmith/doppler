# Compatibility API

## Purpose

`doppler-gpu/compat` preserves the former manifest-loading application facade
while supported production models migrate to signed Capsules. It exports `dr`,
`doppler`, `load`, `open`, `generate`, `openCapsule`, and provider construction.

```js
import { dr } from 'doppler-gpu/compat';

const session = await dr.open('qwen3-0.8b');
const result = await session.generate('Describe WebGPU briefly');
await session.close();
```

This route is explicit compatibility, not the default production authority.
New integrations should import `doppler-gpu` and execute signed Capsules. The
compatibility `openCapsule()` still requires explicit signer trust and rejects
behavior-changing `modelLoadOptions`.

## Streaming inspection

Models returned by `dr.load()` expose
`model.inspect.generate(prompt, { policyId, generation, onEvent })`. While the
promise is pending, `onEvent` receives ordered `{ type: 'token', tokenId, index }`
events, including the first generated token and stop tokens. These observations
preserve the selected decode batching and do not request extra GPU readbacks.
Policies that already capture selected-token probabilities add an optional `token`
record to each live event; it is the same record retained by the completed receipt.

Buffer IDs and decode them together with `model.advanced.decodeTokenIds(ids)`;
individual tokens can contain incomplete Unicode bytes or context-sensitive
spacing. The [demo renderer](../../demo/output.js) batches decoding and updates
the existing text node at most once per animation frame.

The promise returns the completed inspection receipt, also delivered as
`{ type: 'inspection-complete', receipt }`. Its `outputText` is authoritative.
Aborted and failed runs do not emit completion. Callbacks are synchronous; a
callback exception rejects generation and uses its normal cleanup path. Pass
an `AbortSignal` through `generation.signal` to cancel, and retain any partial
display separately from completed evidence.

## Code pointers

- [Compatibility entrypoint](../../src/index.js)
- [Capsule Runtime API](root.md)
