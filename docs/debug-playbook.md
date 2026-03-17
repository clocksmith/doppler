# Debug Playbook

Canonical workflow for inference regressions, incoherent output, and conversion-vs-runtime triage.

## Goal

Reduce debug sprawl by forcing the shortest path to the first divergent boundary.

## 1. Classify the failure first

Pick one primary bucket before changing code:

- `tokenization / chat-template`
- `conversion / artifact integrity`
- `runtime numerics`
- `surface / harness parity`
- `benchmark-only`

Do not mix categories until the current one is either confirmed or ruled out.

## 2. Establish one trusted reference

Before changing runtime behavior, collect one trusted reference from the source runtime when possible.

Capture:

- exact prompt text
- exact token IDs
- deterministic sampling tuple
- one early activation slice
- one output/logits slice

If you do not have these, you do not yet have a stable comparison target.

## 3. Use boundary diffs only

Compare the same token and stop at the first divergence:

1. embeddings
2. post input norm
3. Q/K/V pre-RoPE
4. Q/K post-RoPE
5. attention output
6. FFN output
7. final logits

Interpretation:

- divergence before RoPE usually points to weight loading, dequantization, tensor layout, or matmul
- divergence after RoPE but before attention output usually points to RoPE application or masking
- divergence only after attention output points to residual/FFN/downstream path

## 4. Stop harness churn when early boundaries match

If token IDs match, stop changing prompt formatting.

If embeddings match, stop changing tokenization theories.

If post-input-norm matches, stop changing RMSNorm theories.

Move to the next unresolved boundary instead of reopening solved ones.

## 5. Conversion triage is separate

For fresh conversions:

1. verify source dtypes
2. verify manifest quantization/default kernel path
3. verify sampled shard hashes
4. verify sampled tensor bytes against source
5. then and only then classify runtime divergence

## 6. Conversion completion is fail-closed

Do not report conversion success unless all of these exist and agree:

- successful process exit
- `manifest.json`
- expected shard set
- valid conversion report

A directory with shards but no manifest is an interrupted conversion.

## 7. Add the smallest permanent probe

Prefer:

- existing trace categories
- config-driven probes
- one permanent probe-stage extension

Avoid:

- throwaway `console.*`
- broad benchmark rewrites before classification
- speculative config changes without a boundary diff

## 8. Required deliverables for a serious debug pass

- failing command/config
- trusted reference values
- first divergent boundary
- classification of likely owner (`tokenization`, `conversion`, `runtime`, `surface`)
- next binary split or fix
