# Batched projections and current 35B replay control

These are component and exploratory integration results. They do not promote a
27B serving configuration, qualify task quality, or establish a cross-platform
advantage over Pooled. The generated variants are registered explicitly; no
model profile or existing shader is changed to select them automatically.

## Current 35B replay control

On Jupiter (Ryzen AI MAX+ 395 / Radeon 8060S, Chrome 146), source parent
`531d567648cffdb01eecbff03a7996c9e125a991` with the retained immutable source
inventory completed the fixed 28-input-token, 128-output-token short request:

| Measured repeat | Doppler ordinary replay | Pooled MTP depth 3, lookup off |
| --- | ---: | ---: |
| 1 | 1689.7 ms | 1730.0 ms |
| 2 | 1668.4 ms | 1727.2 ms |

Both prompts and all 128 output IDs match in each pair. Repeat zero is warmup.
Pooled uses the explicitly configured `prefillMath: f32`; loaded materializations
and KV precision are not established as identical. Two pairs are an exploratory
current-source check, not a new statistical or task-quality qualification.
Initialization is separate in the raw results; it is not a cold OS-cache test.

This exercises `pipeline.generate`, without the separate experimental native
MTP generator. A slower experimental generator is not evidence that this replay
route regressed. The retained long-request control was not rerun here.

The `35b-replay-77--*.gz` files preserve the exact configuration, contract,
execution receipt, source tree identity, prompts, token IDs, and timings.

## Registered four-activation-row kernels

Native Q4_0/Q8_0 pair columns share a workgroup across up to four activation
rows. Native Q5_K columns reuse decoded block values across those rows while
retaining the scalar F32 per-row accumulation and reduction order. Generated
patches derive from the existing arithmetic owners; the source shaders and
their existing entry digests remain unchanged.

Jupiter operator timings use six predeclared rounds, alternating variant order,
and 12 dispatches per variant per round. The table gives median GPU compute time
per dispatch, including every round. The tested M dimension is four.

| Type and N/K | Existing variant | Four-row variant |
| --- | ---: | ---: |
| Q4_0, 17408/5120 | 0.887467 ms | 0.425984 ms |
| Q8_0, 5120/6144 | 0.169301 ms | 0.147456 ms |
| Q5_K, 5120/6144 | 0.401408 ms | 0.259413 ms |

These are isolated projection savings, not whole-model savings. Independent CPU
projection checks precede timing. In these large fixtures, each candidate's
maximum absolute difference from its existing GPU variant was zero. Small
fixtures additionally cover offsets and incomplete row/output tiles. Q5_K
immediate and recorded output is bit-identical to both established variants on
AMD and Apple M3. Q4_0/Q8_0 use the existing frozen F32 reference tolerance;
bit identity is not required for the new column variants. Existing pair-variant
bit checks are retained.

Focused checks:

- `node tests/gpu/matmul-gguf-pair-gpu.test.js` on Jupiter and M3.
- `node tests/gpu/q5k-kernels-gpu.test.js` on Jupiter and M3, including guarded output offsets.
- `node tools/check-kernels.js` (generation, registry, uniforms, digests, language contract).
- Source style, unused declarations, checked source types, and `git diff --check`.

## 27B integration remains open

The 128-token experimental Node history-based MTP screen initially took
22045.04 ms, against its 13384.74 ms sequential control. Combining target
recording and the registered paired target head produced 19487.40 ms, against
13303.51 ms sequential. Both generated all 128 expected tokens, also matching
the external Pooled token reference. These are exploratory screens, not an
alternating end-to-end comparison; they must not replace the separate retained
Chrome serving comparison.

Adding the four-row Q5_K candidate produced 19250.84 ms against its 12918.35 ms
sequential control, with the same 128 target/external token IDs and 80 of 138
drafts accepted. The preceding 19487.40 ms result was not an alternating pair;
this does not establish a complete-request speedup. The candidate remains
unpromoted.

A standalone, single-submission draft/target/recovery/refill experiment emitted
32 matching reference tokens, with one 36-byte feedback map per window and no
GPU errors. Its 31 continuation tokens took 4931.89 ms; it is not a speed win.
The first accepted-prefix logit comparison had maximum absolute difference
4.7684e-6, within the frozen 0.001 check. Cancellation, public-generator
integration, longer-run resource accounting, and paired complete-request
performance remain separate gates. The receipt deliberately says
`qualified: false`.

After the backup volume became read-only, all 236 existing 27B artifact shards
were reconstructed directly on healthy internal storage from the healthy
original GGUF and checked against their original SHA256 values. The manifest
and source GGUF identities were retained. No model identity was changed.

`index.json` lists decoded receipt hashes. The raw diagnostic scripts and source
inventories preserve experimental deltas; they are not production runtime APIs.
