# Generation API

## Purpose

Lower-level text pipeline construction and utilities for consumers who need direct pipeline access instead of the root `doppler` facade.

## Import Path

```js
import { createPipeline, InferencePipeline } from 'doppler-gpu/generation';
```

## Audience

Advanced runtime consumers and experiments that need direct pipeline objects.

## Stability

Public, but advanced. Prefer the root facade unless you need direct pipeline control.
Current support tier: `tier1` advanced surface. See [Subsystem Support Matrix](../subsystem-support-matrix.md).

## Primary Exports

- `createPipeline(manifest, contexts?)`
- `InferencePipeline`
- `EmbeddingPipeline`
- core text-pipeline option/result/types from the shipped `.d.ts`

## Core Behaviors

- explicit manifest + context driven pipeline construction
- direct access to text inference pipeline objects
- lower-level than the root `doppler` facade

`generate()`, `generateTokens()`, and `generateTokenIds()` start a new prompt
context while retaining loaded model weights. Prior calls cannot supply hidden
KV or recurrent state to that prompt. Explicit prefix-KV and manual decode APIs
remain available for continuation, with their existing lifecycle contracts.

Seeded generation advances a separate deterministic draw sequence for each
request or resident attempt. Generated-token index zero uses the original seed
and preserves the existing first draw. Later indices use
`((seed % 4294967296) + (index % 4294967296)) % 4294967296` with the existing
scalar sampler. This defines wrapping and advancement for all accepted finite
non-negative seeds, including fractional and large seeds. Negative and null
public seeds are invalid; an omitted seed remains unseeded where supported.
A token selected from a one-candidate distribution still advances the output
index. Retrying the same index repeats its draw; a new attempt starts at zero.
The immutable generation settings and their digest retain the original seed.

CPU and GPU selection receive the same effective step seed. Low-level sampling
primitives still describe a single draw; owners of manual token loops must
advance their draw input explicitly. Equal numeric seeds do not promise token
sequence equality with another runtime's PRNG, such as PyTorch. Greedy
generation does not depend on the draw sequence. Older seeded multi-token
results used a repeated draw and must not be represented as evidence for this
corrected sequence behavior.

## Symbol Notes

### Construction

- `createPipeline(...)`

### Pipeline classes and advanced types

- `InferencePipeline`
- `EmbeddingPipeline`
- `GenerateOptions`
- `GenerationResult`
- `PipelineContexts`
- `KVCacheSnapshot`

## Minimal Example

```js
import { createPipeline } from 'doppler-gpu/generation';

const pipeline = await createPipeline(manifest, contexts);
const text = await pipeline.generateText('Hello');
console.log(text);
```

## Advanced Example

## Code Pointers

- generation export surface: [src/generation/index.js](../../src/generation/index.js)
- generation types: [src/generation/index.d.ts](../../src/generation/index.d.ts)
- text pipeline implementation: [src/inference/pipelines/text.js](../../src/inference/pipelines/text.js)

## Related Surfaces

- [Root API](root.md)
- [Loaders API](loaders.md)
- [Orchestration API](orchestration.md)
- [Generated export inventory](reference/exports.md)

## Generation evidence migration

`generateWithEvidence()` records the resolved `presencePenalty` alongside the
other sampling controls in `generationConfig`. Its configuration hash includes
that value even when two runs happen to return identical tokens. Older receipts
without this field do not establish the executed presence penalty and must be
requalified before comparison with the current contract. The source defaults and
validation remain owned by the generation contract.
