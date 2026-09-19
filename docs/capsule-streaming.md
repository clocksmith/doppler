# Incremental Capsule operation streams

This candidate adds an explicit v2 transport contract. Published `0.6.1` bytes do
not contain this change. Adopt a tested archive by digest; the source version
alone does not identify its API. Capsule signatures, operation semantic version,
selected programs and application trust policy retain their existing meanings.

## Select the format

| Request schema | Event schema | Receipt schema | Partial payload |
| --- | --- | --- | --- |
| `doppler.capsule-operation-request/v1` | `doppler.capsule-operation-event/v1` | `doppler.capsule-operation-receipt/v1` | Cumulative `output`, existing `delta` |
| `doppler.capsule-operation-request/v2` | `doppler.capsule-operation-event/v2` | `doppler.capsule-operation-receipt/v2` | New values in `delta`, no `output` or `receipt` |

The source of truth is `src/config/capsule-operations.json`. Unknown formats fail.
Existing request constants and omitted Reploid format selections retain v1.
HTTP uses the existing `/v1/operations` endpoint: its request body's schema
selects the event contract. The handler sends one JSON event per line and honors
response backpressure. No page refresh is involved.

Generation partials contain `{ tokenIds: number[], text: string }`. IDs are newly
generated raw IDs. Text is a stable, well-formed Unicode append. A raw token may
produce no display text while a byte sequence is incomplete. A final decoder
flush can produce a text-only partial with an empty ID array. Do not assume one
token equals one character, one displayed update, or one network packet.

Embedding partials contain `{ itemIndex, item }`, where `item` is one completed
embedding result and indices start at zero. Reranking and sequence encoding
retain their completion-only output behavior. All operations return one complete
final output and receipt. Generation preserves complete text, IDs, resolved
options, generated token count and actual stopping reason.

## Decode and stop semantics

Bundled tokenizer decoders preserve byte-level, byte-fallback and WordPiece
behavior across token boundaries. The implementation uses a streaming UTF-8
decoder and retains only its incomplete byte boundary. It never full-decodes the
accumulated IDs in v2. Other tokenizer backends must supply an incremental
decoder or fail explicitly before generation; there is no cumulative fallback.

Stop checks retain a bounded text suffix and compare the same canonical prefix
as v1. A pending malformed UTF-8 prefix can contain a replacement character in
that comparison; it is not emitted as unstable display text. Final flushing
preserves canonical full decoding, including malformed trailing bytes. Stopping
tokens and matched stop strings remain included, as in v1. This migration does
not silently trim output or change sampling. Prepared prompts still disable
repeated chat-template formatting in the pipeline option projection.

## Consume and render

The public root and `doppler-gpu/host` export `createCapsuleStreamAccumulator` and
`capsuleOperationSnapshots`. Use the accumulator to verify and reconstruct once:

```js
import { createCapsuleStreamAccumulator } from 'doppler-gpu/host';

const request = {
  ...reviewedOperationRequest,
  schema: 'doppler.capsule-operation-request/v2',
};
const stream = createCapsuleStreamAccumulator(request);
for await (const event of session.executeOperation(request, { signal })) {
  stream.accept(event);
  if (event.status === 'partial' && request.operation.name === 'generate') {
    appendDisplayText(event.delta.text); // Batch DOM Text-node appends per animation frame.
  }
}
const completed = stream.finish();
saveResult(completed.output, completed.receipt);
```

The helper validates sequence numbers, hash chaining, request identity, delta
bounds, completion receipt integrity and the final reconstructed output hash.
The receipt's `stream` field binds event schema, partial count and last partial
digest. A duplicate, omitted, reordered, corrupt or post-completion event rejects
the accumulator permanently. A truncated or cancelled stream cannot finish.
Hash integrity does not authenticate a remote producer or approve model trust;
applications retain their existing identity and acceptance checks.

`snapshot()` explicitly materializes cumulative output. The async
`capsuleOperationSnapshots(events, request)` helper provides that compatibility
view for every event. Calling either per token restores cumulative copying cost;
ordinary consumers should append deltas and obtain complete output at completion.

Output byte limits are accounted incrementally, with final metadata checked at
completion. Token count cannot exceed the requested `maxTokens`; embedding item
count cannot exceed input count. Deadlines, cancellation, iterator cleanup and
single-operation session exclusion apply to both formats. Cancellation prevents
later results from being accepted; it does not interrupt already-submitted GPU
commands. Closing a consumer iterator releases operation-owned resources.

## Reploid adoption

Reploid's installed runtime service imports the same public helper from the
configured archive. Local execution takes an explicit `requestSchema`; peer jobs
bind it in the signed intent. Providers, requesters and offline replay use one
accumulator per stream and keep the existing final identity and acceptance checks.
Retries replay retained signed responses without executing the model again.

The native peer journal uses database version 2, storing attempt metadata and
individual responses separately. Legacy cumulative records migrate atomically;
signed messages retain their values and order. Provider append requests
`{ snapshot: false }`, so normal delivery never reloads prior responses. Explicit
claim/replay materializes history once. Writer fencing, byte/record bounds,
expiry, retention and corruption rejection remain in force.

## Reproduce the focused evidence

```sh
node tests/inference/tokenizer-incremental-decode.test.js
node tests/runtime/capsule-incremental-stream.test.js
node tools/measure-capsule-streaming.js tests/fixtures/capsule-streaming-measurement.json /path/to/report.json
node tools/check-packed-package.js --retain /path/to/new-candidate-directory
```

Run Reploid against that retained installation, including the signed browser
transport and native replay check:

```sh
export DOPPLER_TEST_CONSUMER=/path/to/new-candidate-directory/consumer
node /path/to/reploid/tests/fixtures/doppler-installed-generation.js
node /path/to/reploid/tests/fixtures/doppler-installed-peer-browser-check.js
```

The designated cross-repository workflow requires both checks and installs its
Chromium dependency. Missing installation or fixtures fail the job. The browser
check uses an injected model program, real WebRTC between browser contexts and
native IndexedDB. It drops completion, replaces the provider and requires saved
responses to replay without another model execution. Physical model checks are
separate: `tools/check-installed-capabilities.js` consumes an explicit descriptor
config and records the archive, model identities, environment and failures.

The measurement uses synthetic ASCII IDs and the actual bundled decoder,
operation adapters, executor, JSON transport and consumer reconstruction. It
records producer CPU, serialization CPU, consumer parsing/verification CPU,
transferred bytes, logical partial payload volume, and separate V8 allocation
sampling. Sampling includes collected temporary allocations and is approximate.
It excludes GPU work, peer signatures, HTTP/WebRTC overhead and IndexedDB; those
paths have separate integration tests. Unicode and stop correctness use canonical
full-decoding comparisons, independently of the performance probe.
