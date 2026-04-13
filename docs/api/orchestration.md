# Orchestration API

## Purpose

Advanced orchestration helpers that sit above loading but below the root `doppler` facade.

## Import Path

```js
import { KVCache, mergeMultipleLogits, Tokenizer } from 'doppler-gpu/orchestration';
```

## Audience

Advanced consumers who need direct access to KV cache management, routers, adapter helpers, structured/energy heads, or logit-merge primitives.

## Stability

Public, but advanced. Prefer the root facade unless you need direct orchestration primitives.
Current support tier: `experimental`. This subpath is available, but it is not
part of the canonical tier1 demo or quickstart proof path. See
[Subsystem Support Matrix](../subsystem-support-matrix.md).

## Primary Exports

- `KVCache`
- `Tokenizer`
- `SpeculativeDecoder`
- `ExpertRouter`
- `MoERouter`
- `MultiModelNetwork`
- `MultiPipelinePool`
- structured/energy head pipelines
- LoRA adapter helpers
- `LogitMergeKernel`, `mergeLogits(...)`, `mergeMultipleLogits(...)`

## Related Surfaces

- [Root API](root.md)
- [Loaders API](loaders.md)
- [Generation API](generation.md)
- [Generated export inventory](reference/exports.md)
