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

No retained fixture fails. None of them qualifies Flash-Next or establishes a
Doppler operator; all use synthetic weights or fixed strings.

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
  amdgpu error count unchanged). Intel Vulkan is not yet run; CI has no WebGPU
  adapter.
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
2. Conversion: RDRR conversion and conversion-quality controls. The norm-gate
   contract now exists: the field, the qwen4_exp mapping and conversion
   fail-fast guards are in place. A Flash-Next conversion config must author
   `linearNormGateActivation: "sigmoid"`; conversion rejects omission or silu.
   No Flash-Next conversion config or RDRR artifact exists yet.
3. GPU verification of the sigmoid gated-delta kernels against the CPU
   reference on Intel Vulkan (Apple metal-3 and AMD RDNA3 Vulkan done
   2026-10-14, see above). Neither establishes Flash-Next numerics on
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

No fixture yet covers the PLE layer (convolution with the n-gram embedding),
decoder-layer composition (hyper-connection mixing around attention and MoE),
the MTP layer or the vision tower. These fixtures allow source preparation to
proceed during runtime repair without making Flash-Next a dependency for chat.
