# Flash-Next pinned operator reference preparation

Component: `doppler`. Intent: preserved. Boundary effects: CPU reference
preparation only; no runtime implementation, model acquisition or promotion.

`source-binding.json` binds Qwen/Qwen3.8-Flash-Next metadata at
`de4b8e4d43b917e7706784d8bb445c9af86a3540` and the Transformers qwen4_exp
implementation at `e0eb2810ac1b2caf6d082f247d8d37162ecf67a0`. Every retained
source file has its exact URL, length and SHA256. The binding's `executed:false`
describes initial source acquisition; execution receipts are the JSON files below.
The 2026-10-14 amendment adds `tokenizer.json` and the pinned
`modeling_rope_utils.py` (for the `dynamic_rope_update` decorator).

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
eager expert body still matches independent scalar equations within 2.86e-10;
its 2e-7 tolerance is an initial fixture tolerance, not a derived bound.

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

1. Expert execution path: full 512-expert/top-10 geometry with 640-wide
   experts, shared-expert sigmoid gate composition (`SparseMoeBlock`),
   accelerated expert implementations against the eager body, and a derived
   expert-output bound.
2. Learned-checkpoint comparison: an independent reference on learned weights
   for selected layers, including bfloat16 indexer selection, ties and near ties.
3. Conversion: RDRR conversion and conversion-quality controls.
4. License suitability: Qwen Community License 1.0 requires a separate license
   for commercial Model-as-a-Service or AI Work Assistant use; suitability for
   the intended service is unresolved.
5. Physical qualification on coordinated GPU time: hardware, memory-tier
   placement (upstream notes a ~95 GiB n-gram embedding), concurrent state,
   admitted budgets and installed distributed acceptance under the existing
   full-model gate.

No fixture yet covers the linear-attention (GatedDeltaNet, 36 of 48 layers),
PLE convolution, MTP or vision paths. These fixtures allow source preparation to
proceed during runtime repair without making Flash-Next a dependency for chat.
