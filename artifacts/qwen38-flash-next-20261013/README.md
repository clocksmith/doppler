# Flash-Next pinned operator reference preparation

Component: `doppler`. Intent: preserved. Boundary effects: CPU reference
preparation only; no runtime implementation, model acquisition or promotion.

`source-binding.json` binds Qwen/Qwen3.8-Flash-Next metadata at
`de4b8e4d43b917e7706784d8bb445c9af86a3540` and the Transformers qwen4_exp
implementation at `e0eb2810ac1b2caf6d082f247d8d37162ecf67a0`. Every retained
source file has its exact URL, length and SHA256. The binding's `executed:false`
describes initial source acquisition; execution receipts are the JSON files below.
The 2026-10-14 amendments add `tokenizer.json`, `modeling_rope_utils.py` (the
`dynamic_rope_update` decorator), the expert implementations
(`integrations_moe.py`), kernel and accelerate decorators, `activations.py`,
`cache_utils.py` (linear-attention cache) and `modeling_qwen3_5.py` (comparison
only), all at the pinned Transformers revision.

No checkpoint weight was downloaded. Every script runs on CPU from the Doppler
root, validates all source hashes first, and extracts unchanged upstream
definitions through Python AST. Run each with `python3 -I <script>`.

## Current state

| Receipt | Script | Result |
| --- | --- | --- |
| `reference-fixtures.json` | `prepare-reference.py` | pass (torch 2.9.1 run) |
| `ngram-incremental-reference.json` | `prepare-ngram-incremental-reference.py` | pass (torch 2.9.1 run) |
| `expert-reference-fixtures.json` | `prepare-expert-reference.py` | pass (torch 2.11.0, Accelerate, arm64) |
| `sparse-attention-reference.json` | `prepare-sparse-attention-reference.py` | pass (torch 2.11.0, Accelerate, arm64) |
| `moe-block-reference.json` | `prepare-moe-block-reference.py` | pass (torch 2.11.0, Accelerate, arm64) |
| `linear-attention-reference.json` | `prepare-linear-attention-reference.py` | pass (torch 2.11.0, Accelerate, arm64) |
| `tokenizer-reference.json` | `check-tokenizer.py` | pass (tokenizers 0.22.2) |
| `checkpoint-headers.json` | `acquire-checkpoint-headers.py` | pass (2026-11-04; headers and three buffers only) |
| `qsa-stage-reference.json` | `prepare-qsa-stage-reference.py` | pass (2026-11-05, torch 2.11.0, arm64) |
| `ple-reference.json` | `prepare-ple-reference.py` | pass (2026-11-05, torch 2.11.0, arm64) |
| `decoder-composition-reference.json` | `prepare-decoder-composition-reference.py` | pass (2026-11-05, torch 2.11.0, arm64) |
| `acquisition-receipt.json` | `acquire-text-checkpoint.py` | pass (2026-11-05; 131 files, 360,000,192,888 bytes) |
| `storage-precision-scan.json` | `storage-precision-scan.py` | measured (2026-11-05, numpy 2.4.4) |

No retained fixture fails. None of them qualifies Flash-Next or establishes a
Doppler operator; all use synthetic weights or fixed strings.

### Checkpoint headers and the n-gram table (2026-11-04)

The earlier sixteen-file inventory lacked the loader rule that assembles the
n-gram table. Three more pinned Transformers files are now bound:
`conversion_mapping.py` (qwen4_exp_text `Concatenate(dim=0)` over
`ngram_embedding.shard_*.weight`), `core_model_loading.py` (`dot_natural_key`
load ordering) and `masking_utils.py` (causal and recurrent masks).

`acquire-checkpoint-headers.py` reads only SafeTensors headers of all 131
checkpoint files plus the bytes of three small PLE buffers. Every header
matches the index. Executing the pinned `dot_natural_key` orders shards
numerically (`shard_2` before `shard_11`), and shard numbers rise with file
numbers (files 00005 to 00037), so per-file and global ordering agree. Each of
the 128 shards is `[2500012, 160]` BF16, giving 320,001,536 rows (95.37 GiB):
global row `r` is shard `floor(r / 2500012)`, local row `r mod 2500012`. The
receipt records every shard's file and absolute byte span.

The checkpoint's `layer_multipliers` (23703573157769, 20109073645365,
8052911324071), `ngram_heads_offsets` and `ngram_heads_vocab_sizes` equal the
values derived by the unchanged upstream functions with the class-default seed
1234 (`config.json` omits `seed`). The multipliers are below 2^45, so
token-times-multiplier products reach about 2^63; exact address arithmetic needs
64-bit integers, not JavaScript Number.

Text-only BF16 bytes by category: routed experts 225.00 GiB, n-gram table
95.37 GiB, other text tensors 9.22 GiB; excluded vision 0.84 GiB and MTP
4.86 GiB.

### Checkpoint acquisition (2026-11-05)

`acquire-text-checkpoint.py` downloaded every file holding a text tensor (all
131; no file is vision-only) to the models drive of the qualification host,
never to the Mac. Each file's length and SHA-256 equal the Hugging Face LFS
`X-Linked-Size` and `X-Linked-ETag` at the pinned revision, and its header
bytes equal `checkpoint-headers.json`. `acquisition-receipt.json` lists all
131 files (360,000,192,888 bytes); the run took 2,545 s after one restart from
a DNS failure. The conversion profile pins the 33 n-gram table files by these
SHA-256 values. The seven metadata files beside the weights were copied from
`source/` and checked against `source-binding.json`.

### Storage precision (2026-11-05)

`storage-precision-scan.py` reads every BF16 tensor that the conversion profile
stores unquantized (everything except experts, attention and shared-expert
projections, MTP, vision and the n-gram table) and counts values F16 storage
cannot hold exactly. No value overflows F16. Norm vectors, `A_log` and
`dt_bias` are exact. Projection classes have 0.01% to 0.05% inexact values, all
in the F16 subnormal range below 6.1e-5, so each such value moves by at most
2^-25 (about 3.0e-8) in absolute terms; a few hundred per class flush to zero
(for example 91 of 157,286,400 attention hyper-connection down-projection
values). The profile therefore stores these tensors as F16; qualification
tolerances must carry this absolute bound. The n-gram table keeps its BF16
bytes.

### PLE layer reference (2026-11-05)

`prepare-ple-reference.py` runs the unchanged upstream `Qwen4ExpTextPLELayer`
(with its n-gram embedding and RMSNorm) on dyadic synthetic weights for 24
tokens with boundary tokens. A row spy replaces the 95 GiB table with
BF16-exact rows defined by address, so consumers regenerate any row. Token
history and short-convolution state use an explicit port that follows the
pinned `update_conv_state` (record_past disabled). The full sequence and the
chunked schedule 5, 1, 1, 17 produce identical addresses and token history;
outputs differ by at most 1.7e-6 (F32 batching), and the final convolution
states are equal. The fixture records addresses, the signed-square-root gate
input before the sigmoid, every 16th output channel of `gated + conv` (the
decoder adds the streams), and the final states. Padding masks and learned
weights are not covered.

### QSA stage reference and RoPE tables (2026-11-05)

`prepare-qsa-stage-reference.py` reuses the sparse-attention recipe (weights
verified by SHA256) and runs the unchanged upstream indexer on row 0 through an
appending cache port in chunks 1024, 1024, 509, 1, 1, 1. For 13 sampled queries
(5 dense, 8 sparse) it records upstream selected tokens, float64 block scores,
and first-order F32 bounds of the form A + B * eps, where eps bounds the
executing implementation's cos/sin table error; float64 query vectors and block
keys use the same form. It also records the upstream f32 inverse frequencies.

Doppler's theta-generated RoPE tables deviate from torch's f32-angle cos/sin by
1.2e-4 at 2,560 positions, 9.8e-4 at 32,768 and 7.8e-3 at 262,144 (Apple
metal-3), because the GPU derives the inverse frequencies itself. With the
upstream f32 inverse frequencies retained (`inference.rope.ropeInverseFrequencies`)
the same precompute matches within 1.2e-7 and 6.4e-7. A Flash-Next conversion
must author those 32 values; this affects every theta-generated model at long
positions and is reported separately.

### Gated residual and n-grams

Gated residual fixtures use the actual 2560 hidden width and four residual
streams with deterministic synthetic weights; malformed width rejects. N-gram
fixtures execute upstream address computation for eight tokens including EOS
boundaries, checked against an independent Python-integer calculation of all
sixteen addresses per token; a lookup spy replaces the learned embedding. All
128 chunk partitions and two interleaved histories match. Upstream Cache
internals, learned embeddings and Doppler state are not qualified.

### Router and experts

The router logits, softmax, top-10 and renormalization are now checked against a
forward error bound derived for the executed F32 operations, not a chosen
tolerance (`routerErrorAnalysis` holds the derivation):

- Logits: `|fl(x.w) - x.w| <= gamma_n * sum|x_i w_i|` for any summation order,
  n = 2560, u = 2^-24, gamma_n = 1.52611e-4. At the worst element (row 0,
  expert 510) sum|x_i w_i| = 1.090944, so the bound is 1.66490e-4. Observed:
  6.0797e-6 = 102 * 2^-24, 3.65% of the bound.
- Scores: PyTorch's float softmax (max, exp, one shared reciprocal, multiply)
  and the top-k renormalization cancel the shared normalizer, leaving
  `|v_i - s_i| <= s_i * expm1(N_i + D)`, about 3.3e-4 relative at these logit
  bounds. Bound at the worst element 3.35e-5; observed 6.12e-7 (1.8%).
- Selection is proven stable: the 10th/11th exact logit gap 2.135e-3 exceeds
  the combined perturbation, leaving a margin of 1.81e-3.
- Exactness control: 980 of 1024 logits are provably exact in any F32 order
  and are bitwise exact; all 44 nonzero errors are in the 22 even experts
  468..510 per row where rounding is possible.
- The executed Accelerate result equals left-to-right F32 summation bitwise
  for all 1024 logits, so a correct sequential implementation already produces
  6.08e-6.

The initial 2e-6 / 2e-7 router tolerances were not derived and are superseded
by these elementwise bounds. No model acceptance threshold changed. The small
eager expert body (hidden 8, 4 experts, sentinel slot) matches independent scalar
equations within 2.86e-10. Its underived 2e-7 tolerance is replaced by the
derived elementwise bound of the MoE block below, 3.0e-9 to 9.9e-9 here (tighter
than 2e-7); the observation is 3.0% of it.

### MoE block at full geometry

`prepare-moe-block-reference.py` runs the unchanged `SparseMoeBlock`: router over
512 experts with top-10 renormalization, 640-wide SiLU experts, the 640-wide
shared expert and its sigmoid gate, and scatter/combine. The eager body runs
through the pinned `use_experts_implementation` dispatcher, which also runs the
pinned `batched_mm` and `grouped_mm` implementations. A test port generates
synthetic expert e on access, because the dense F32 expert tensors would take
10 GB; `grouped_mm` needs one dense tensor, so it runs on the 56 experts hit,
with a monotonic id remap, through torch's CPU `grouped_mm` kernel. Eight tokens
include a near-duplicate and an exact duplicate; 22 experts serve more than one
token.

The output bound uses the router-style derivation: gamma_n|W||x| projections,
silu and sigmoid as executed by PyTorch's CPU kernels (Lipschitz constants plus
relative rounding), product rounding, gamma_9 for the 10-way combine, and float64
truth covered by a 1 + 2^-28 scale. The bound is order-independent, so it covers
all three implementations:

| Output | Max observed / bound |
| --- | --- |
| eager experts | 6.5e-6 |
| batched_mm | 1.4e-6 (1.07e-6 from eager) |
| grouped_mm | 6.5e-6 (1.2e-7 from eager) |
| shared expert | 2.6e-5 |
| block output | 9.8e-6 (observed 2.7e-6, bound up to 0.30) |
| router logits | 9.1e-4 |

These worst-case bounds are about 1e5 times the observed error, because random
signs make |W||x| much larger than |Wx| and the first projection's bound passes
through the second. They bound rounding rigorously but are not regression
tolerances. To keep the fixture discriminating, five alternative semantics are
checked to exceed twice the bound somewhere: silu shared gate, no shared gate,
swapped gate/up halves, routing weights ignored, and top-k without
renormalization (the thinnest margins are 1.75x and 1.86x). An expert id beyond the sentinel
and a 2559-wide input reject; the sentinel slot contributes nothing. One token's
10th/11th router logits are too close to prove selection stable; the experts
stage uses the captured routing as its input. Not established: learned routing,
bfloat16 (eager `index_add_` accumulates in the output dtype), CUDA
deepgemm/sonicmoe/grouped_mm, expert parallelism.

### Linear attention (gated delta, 36 of 48 layers)

`prepare-linear-attention-reference.py` runs the unchanged
`Qwen4ExpTextGatedDeltaNet` at the pinned geometry. That is 16 key and 48 value
heads (grouped with `repeat_interleave`), dimension 128, a 10240-channel
kernel-4 causal conv, the sigmoid-gated RMSNorm and 64-token chunks. Synthetic
A_log spans the init range, so per-token decay runs from 0.987 to about 1e-9. The
pinned `LinearAttentionLayer` cache and the unchanged `Cache` methods carry state
over three paths on 150 tokens:

- full sequence;
- prefill 70 + 50 tokens, then 30 single-token decodes;
- a 3-token prefill (the left-padded conv-state branch), then 147 decodes.

Every stage is checked against float64 truth from that call's captured inputs
with a derived bound:

| Stage | Max observed / bound |
| --- | --- |
| in_proj qkv / z / b / a, out_proj | 2.3e-3 or less |
| causal conv + silu | 0.39 |
| beta = sigmoid(b) | 0.32 |
| g = -exp(A_log) softplus(a + dt_bias) | 0.26 |
| delta-rule outputs (chunk and recurrent) | 5.7e-3 |
| final recurrent state | 0.081 |
| sigmoid-gated RMSNorm | 0.071 |

The chunk bound follows the executed UT-transform algebra: cumulative decay,
pairwise decays, the unit lower triangular solves (backward error gamma_64|T|),
then a normwise scan whose state error is propagated through the exact chunk
transition D I - K'^T W. The recurrent bound uses the token transition
d(I - beta k k^T). The float64 chunk algebra matches the independent
token-by-token recurrence within 1.2e-15 relative.

Semantics established exactly:

- The conv state is the last 4 raw pre-conv qkv columns, zero-left-padded after a
  short prefill.
- A continuation prefill convolves that state followed by the new columns.
- Each call's initial recurrent state is the previous call's final state.

End to end, all three paths are within 3.6e-6 relative of the float64 pipeline
and within 5.6e-6 of each other. Final states are within 6.0e-7 relative.

Six alternative semantics are each distinguished by more than twice a stage
bound: silu norm gate, (1 + w) norm weight, no q/k l2norm, tiled instead of
interleaved heads, beta ignored, decay ignored. A 2559-wide input and a
non-linear cache layer reject.

Not established:

- learned weights and bfloat16;
- the FLA chunk and fused-recurrent kernels and the causal_conv1d CUDA kernels
  (GPU only);
- padding masks and `record_past`/`crop` rollback;
- the export-only forward-substitution solve.

Compared with pinned Qwen3.5 at the same revision, `l2norm`, both delta-rule
bodies, both conv functions and the padding helper are identical. The only
semantic difference is the gated RMSNorm activation: Qwen3.5 hardcodes silu,
while qwen4_exp uses `output_gate_type` (Flash-Next: `sigmoid`). Qwen3.5 also has
a layer-level hub-kernel decorator. Read from Doppler source at `a40f02a1`
(not executed): the original gated-delta kernels hardcode a silu norm gate
(`src/gpu/kernels/gated_delta_recurrent.wgsl:470`,
`gated_delta_fused_decode.wgsl:477`, their `_f16` variants, and the paired
`gated_delta_gate_paired.wgsl`, applied after `gated_delta_norm_paired.wgsl`).
Head grouping (`head / q_rep`), the softplus threshold of 20 and the
kernel-length conv state match upstream.

Doppler source now declares the gate explicitly. The kernels below ran on one
physical GPU (Apple metal-3) on 2026-10-14; receipts are in
[`gpu-verification-20261014/`](gpu-verification-20261014/):

- The manifest field `inference.normalization.linearNormGateActivation` takes
  `silu` or `sigmoid`. An absent field resolves to the schema default `silu`,
  so existing Qwen3.5 artifacts are unchanged.
- The converter maps qwen4_exp `output_gate_type or hidden_act` into the field
  and authors silu for qwen3_5. It no longer records the norm-gate meaning as
  `attention.outputGateType`.
- Five generated sigmoid kernels differ from their unchanged silu sources only
  in the gate call: `gated_delta_recurrent_sigmoid_gate{,_f16}.wgsl`,
  `gated_delta_fused_decode_sigmoid_gate{,_f16}.wgsl` and
  `gated_delta_gate_sigmoid_paired.wgsl`. On Apple metal-3,
  `tests/inference/linear-attention-core-gpu-regression.test.js` passes its
  scalar recurrent and fused-decode sigmoid cases (Node/WebGPU), and
  `tests/kernels/paired-linear-physical.js` passes the paired path with
  sigmoid and, unchanged, with silu (Chrome 155, tolerance 2e-11). On
  `amd-host` (AMD RDNA3, Vulkan) both commands also pass: the Node test through
  the final-9 Dawn module, and the paired physical test with sigmoid and silu in
  Chrome 146 ([`gpu-verification-20261014/amd-host/`](gpu-verification-20261014/amd-host/);
  amdgpu error count unchanged). On `intel-host` (Intel Gen12LP, Mesa 24.2.8
  ANV) both pass too: the Node test, and the paired test with sigmoid and silu in
  Chrome 136 at tolerance 2e-11
  ([`gpu-verification-20261014/intel-host/`](gpu-verification-20261014/intel-host/)).
  Both ran with a private Mesa shader cache (`MESA_SHADER_CACHE_DIR`) so an
  interrupted compile could not affect the shared cache, and the i915
  hang/reset count was unchanged. CI has no WebGPU adapter.
- Rule-based kernel selection and registry `normGateActivation` metadata fail
  closed when a kernel's gate activation differs from the manifest's.

The contract is in `docs/conversion-runtime-contract.md`. The CPU reference
(`tests/kernels/recurrent-reference.js`) computes both gates, and its test
checks the sigmoid gate against this receipt's `normGate` and the pinned
`modeling_qwen4_exp.py`.

### Sparse attention and QSA indexer

The unchanged indexer, rotary embedding, RMSNorm and eager attention run at the
pinned geometry (layer 3; 24 heads, 2 KV heads, head dim 256; indexer 4 heads,
1 KV head, dim 128, budget 2048, compression 4) on 2560 tokens, batch 2 (one row
left-padded by 5), with dyadic synthetic F32 weights. An independent float64
recomputation in numpy reproduces every selection row. Established:

- A query with at most 2051 visible tokens selects its full visible set; the
  first sparse query has 2052 visible tokens.
- Blocks are consecutive groups of 4 visible tokens from the first visible
  token, not absolute slots; the trailing `visible mod 4` tokens are always
  selected; 512 blocks are kept, at most 2051 tokens.
- Block key: k_layernorm of the mean of 4 raw (pre-norm, pre-RoPE) keys, RoPE
  at the block's first-token slot. Score: sum over heads of relu(q.k) / sqrt(128).
  Six alternative semantics (RoPE at the last token, norm then mean, no key norm,
  no relu, max over heads, absolute alignment) each change selections and are
  distinguished.
- A query's own token is not guaranteed: 17 and 23 queries exclude it.
- Bool and float masks select identically; eager attention receives
  causal + selection, masked probabilities are exactly 0, and attention uses
  uncompressed per-token keys and values.
- Chunked prefill (1024, 1024, 509) plus three decode steps through an
  appending cache port equals the full-sequence selection.
- Sampled attention probabilities and outputs, and q/k/v/o projections, lie
  within derived F32 bounds (largest ratio 0.006).
- Rejected: width 2559, mask wider than keys, positions shorter than keys.
  Accepted without validation: positions longer than keys (changes selection),
  non-causal masks (selects future tokens), masks narrower than keys.

Not established: tie order (259 boundaries tie at relu zero; torch.topk picks
neither lowest nor highest indices, so only tie-invariant parts are checked);
near-tie decidability (9 untied boundaries have relative gaps below 1e-5);
bfloat16 checkpoint dtype (254 of 509 sparse queries select differently);
create_causal_mask, sdpa, DynamicCache internals, multimodal or generate-style
positions; positions beyond 2560 (F32 angle rounding changes cos by up to
7.8e-3 near position 262144); standalone norm, RoPE and output-gate numerics.

### Tokenizer

`tokenizer.json` (12,809,320 bytes, SHA256
`0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`) matches
the Hugging Face X-Linked-Size and X-Linked-ETag at the pinned revision.
`tokenizer_config.json` matches its pinned SHA256 and the upstream git blob
SHA1. It is redistributed alongside the retained LICENSE. Byte-level BPE:
248,044 base tokens, 247,587 merges, 33 added tokens (248044..248076) in a
248,320-row embedding. Thirteen fixed strings and two rendered chat-template
conversations encode identically with the tokenizers runtime and an
independent Python BPE, and decode back after NFC normalization.

EOS differs by source: config.json uses 248044 `<|endoftext|>` (also the n-gram
EOS boundary); generation stops on [248046 `<|im_end|>`, 248044];
tokenizer_config's eos_token is `<|im_end|>`; no BOS is added. With thinking
enabled (the default) the template adds an xhigh reasoning-effort system
instruction and opens the generation prompt with `<think>`. Parity with the
Transformers template renderer and Doppler's tokenizer is not established.

## Remaining work, in order

The CPU expert execution reference is now in place at full geometry (MoE block above).

1. Learned-checkpoint comparison: an independent reference on learned weights
   for selected layers, in the checkpoint's bfloat16 dtype, covering indexer
   selection (ties, near ties), routing and gated-delta decay ranges.
2. Conversion: the profile
   `src/config/conversion/qwen3/qwen-3-8-flash-next-text-q4k-ehf16-af32.json`
   (2026-11-05) excludes MTP (31) and vision (333) by explicit ignore rules,
   authenticates and then ignores the three I64 buffers, retains the 128
   n-gram table tensors in their 33 pinned files, and authors the three
   qwen4_exp contracts, the sigmoid norm gate and the 32 source inverse
   frequencies. The RDRR artifact and conversion-quality controls are not yet
   established.
3. GPU verification of the sigmoid gated-delta kernels against the CPU
   reference: done 2026-10-14 on Apple metal-3, AMD RDNA3 Vulkan and Intel
   Gen12LP Vulkan (see above). Neither establishes Flash-Next numerics on
   learned weights.
4. License suitability: the checkpoint files are under the Qwen Community
   License 1.0 and the nine Transformers files under Apache-2.0
   ([source/LICENSES.md](source/LICENSES.md)). Qwen's clause 2: "If the
   licensee or any of its affiliates conducts a Model as a Service or AI Work
   Assistant business, the licensee shall obtain a separate license from Qwen
   before Using the Software or its derivative works for any commercial
   purpose." The trigger is the operator's business, not hosting alone, and
   once triggered it covers any commercial purpose. Clause 1 requires the
   notice in all copies, and requires the model name to be shown prominently
   above 100,000,000 monthly active users or US$ 20,000,000 monthly revenue.
   Whether the intended service is such a business is unresolved; it needs
   Qwen's determination or a separate license before any commercial release.
5. Physical qualification on coordinated GPU time: hardware, memory-tier
   placement (upstream notes a ~95 GiB n-gram embedding), concurrent state,
   admitted budgets and installed distributed acceptance under the existing
   full-model gate. GPU-only upstream implementations (FLA, causal_conv1d,
   CUDA grouped_mm, deepgemm, sonicmoe) can be compared to the eager references
   only there.

`decoder-composition-reference.json` covers decoder-layer composition for layers 0-3 and the root
mixer with low-rank linear spies in place of attention, linear attention and MoE (composition order
only; those branches have their own references). No fixture yet covers learned-weight composition,
the MTP layer or the vision tower. These fixtures allow source preparation to
proceed during runtime repair without making Flash-Next a dependency for chat.
