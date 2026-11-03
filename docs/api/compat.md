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
`receipt.inputTokenIds` holds the exact IDs prefilled before the first generated
token, or `null` when a pipeline does not report them.

Inspection accepts text, a chat-message array, or `{ messages: [...] }` using the
same chat formatter as generation. Structured history is snapshotted before
execution. Its fingerprint uses the actual prefilled IDs when reported, otherwise
the existing tokenizer/template helper with the resolved chat-template setting.
Plain-string fingerprints retain their earlier raw-text convention. A structured
receipt's exact prefix can be replayed for a branch without re-tokenizing history.


To continue from an exact token boundary, pass `inputTokenIds` instead of relying
on `prompt` tokenization: `model.inspect.generate(prompt, { inputTokenIds })`.
Every generation starts from cleared attention and recurrent state, so prefilling
`receipt.inputTokenIds`, the generated IDs before a position and a replacement ID
restores the state after that replacement and samples what follows; `prompt`
remains the receipt label. The run fails if the pipeline did not prefill exactly
those IDs, and the fingerprint's `promptTokenIds` records them. Never rebuild the
prefix from displayed text, which may tokenize differently. Replay prefills the
prefix in one pass rather than one token at a time, so numerics may differ
slightly from the original decode.
Aborted and failed runs do not emit completion. Callbacks are synchronous; a
callback exception rejects generation and uses its normal cleanup path. Pass
an `AbortSignal` through `generation.signal` to cancel, and retain any partial
display separately from completed evidence.

## Code pointers

- [Compatibility entrypoint](../../src/index.js)
- [Doppler Run API](root.md)
