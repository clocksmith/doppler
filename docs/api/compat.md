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


### Recording part of a run

Policies that record model activity (`activationTrace.activityMaxBytes`) declare
what they record in `activationTrace.activitySelection` and, per capture kind, in
`activationTrace.activityRoleMaxBytes`. A request can record less with
`model.inspect.generate(prompt, { policyId, capture })`:

| `capture` field | Meaning |
| --- | --- |
| `roles` | Capture kinds to keep, for example `['attention-query', 'kv-key']`. `[]` keeps none. The kinds are listed in [`activity-stages.json`](../../src/config/inspection/activity-stages.json). |
| `layers` | Layer indexes to keep. Model-level captures such as `logits` have no layer and are unaffected. |
| `tokens` | `{ from, to }` token positions to keep, counted from the first prompt token; `to` is exclusive and `null` means no end. |
| `maxBytes` | A smaller budget for the whole activity recording. |
| `roleMaxBytes` | A budget for one kind, for example `{ logits: 4194304 }`. |

Omitted or `null` fields keep the policy's value. A request only narrows: a kind,
layer, token range or budget outside the policy fails before the run starts, as
does `capture` on a policy that records no activity. Layer outputs are recorded
by every activity policy and are not part of this selection. Selection changes
what is read back and kept, not how the model executes: the same prompt and
sampling give the same tokens with any selection.

Two kinds of capture are not cut exactly as asked, because cutting them would make
a view wrong rather than smaller:

- Keys and values (`kv-key`, `kv-value`) keep every token before a token range's
  start. Attention at a later token reads all of them.
- Fixed per-layer parameters (`linear-a-log`, `linear-dt-bias`) are recorded once
  per layer, whatever the token range, and marked `metadata.constant`.

The receipt's `activationTrace.activity` says what was left out. `selection` is
the resolved selection. `coverage` has one entry per kind the run attempted:
`captured` and `capturedBytes`; `dropped` and `firstDroppedToken` for captures
that did not fit a budget (each dropped frame's `metadata.limit` names
`activityMaxBytes` or `activityRoleMaxBytes`); `failed`; `reused` for repeats of
a fixed value already recorded; and `notSelected` counts by the rule that left
them out (`role`, `layer` or `token`). A capture left out by selection takes no
frame and no budget.

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

### Projecting recorded layer outputs

`model.inspect.projectLayerOutputs({ vectors, topK })` passes hidden-state
vectors through the loaded model's own final normalization and output head and
returns the `topK` candidates for each, with softmax probabilities over the
whole vocabulary. Pass the `values` of recorded `layer-output` frames as
`Float32Array`s of the model's hidden size, up to 64 per call, with `topK` from
1 to 64.

The result has `schema: "doppler.layer-projection/v1"` and
`projection.kind: "projected"`. A projection estimates what a vector leans
toward; it is not a token the model produced at that layer, and earlier layers
were not trained to be read this way. The last layer's vector at a position
reproduces that step's logits. The call does not read or change sequence state
and fails for hyper-connection models and paired-precision execution plans.
Physical evidence: `tests/inference/hidden-projection-gpu.test.js` on a
synthetic head; no full-model qualification is claimed.

## Code pointers

- [Compatibility entrypoint](../../src/index.js)
- [Doppler Run API](root.md)
