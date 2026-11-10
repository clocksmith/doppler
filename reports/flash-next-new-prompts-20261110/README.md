# Flash-Next: two new prefill prompts

This packet freezes two prompts not used to tune the preceding numerical
repairs. The independent CPU source reference completed all 48 layers and is
retained with verified payload hashes and finite F32 arrays. The new GPU
comparison **fails** the unchanged policy on the long prompt. The previously
passing evidence remains in
[the bounded decode report](../flash-next-decode-20261110/README.md).

| New prefill case | Tokens | Checked logits | Maximum absolute error | Violations | Greedy token |
| --- | ---: | ---: | ---: | ---: | ---: |
| marbles | 22 | 248,320 | 0.0000262260 | 0 | 8646 |
| inventory | 2,187 | 248,320 | 0.4672708511 | **30,664** | 271 |

Both greedy tokens match the source. That does not override the score failure.
[comparison.json](comparison.json) preserves the failed gate and
[preparation.json](preparation.json) binds the run to its unchanged parent
configuration, independent reference and frozen inputs. No repair or new
qualification is established here.

## First routing boundary

The observation-only GPU trace produces byte-identical final scores to the
original failed run. Full Layer 2 and 3 outputs differ from the source by at most
1.14e-6, with relative L2 differences below 9.3e-7. Sampled QSA selections agree
through Layer 35; sampled differences appear at Layer 39. The initial incomplete
QSA-block hypothesis does not explain the earliest observed routing difference.

The independent F32 source observer retains all router rows and periodic layer
checkpoints. Every retained layer sample and router metadata item matches the
original reference bit-for-bit; its identity is in
[source-trace-receipt.json](source-trace-receipt.json). The observer adapts the
existing F32 observer to an explicit sequence ID without changing source math.

[routing-trace.json](routing-trace.json) compares all 2,187 positions across 48
layers. It first infers a different expert set at Layer 24, token 112: expert 225
instead of 60. Those rankings are inferred from captured logits, rather than
live observations of the model's expert-index buffers.

[topk-replay.json](topk-replay.json) separately executes the production
`softmax_topk_strided.wgsl` shader on the physical AMD GPU for Layers 0 through
24. All 25 source-logit cases reproduce the source expert sets. The GPU-logit
cases first differ at Layer 24, token 112, with the same 225/60 change. This
places the difference before the selection operator; it does not establish
that all upstream variation is permitted or repair the final-score failure.

[topk-weight-replay.json](topk-weight-replay.json) extends that physical replay
to compare selected weights **by expert ID** on identical source logits:
546,750 weights across 25 layers. Maximum absolute difference is 1.79e-7,
maximum relative difference 5.42e-7, and the largest per-layer RMS difference
is 1.15e-8. All source expert sets still agree. This does not establish that
small weight differences are harmless after expert combination; recombination
on identical actual expert outputs remains a separate control.

[upstream-stages.json](upstream-stages.json) compares 106 matched stage slices
through Layers 20–24 at positions 112 and 2186. The new GPU observer reproduces
the retained Layer-24 router inputs bit-for-bit; independently resumed source
traces likewise reproduce their original source router inputs. Reference RMS
and per-stream metrics keep absolute normalization amplification visible.
Unmatched boundaries are listed explicitly. These are observations of two
different upstream trajectories, not identical-input operator controls or
proof that the accumulated variation is permitted.

## Gated-delta algorithm control

[gdn-algorithm-control.json](gdn-algorithm-control.json) compares the pinned
source's chunked and recurrent implementations at Layer 24. They receive
identical Q/K/V/decay/beta tensors constructed by the source convolution from
the captured GPU projection outputs. After source normalization and gating:

| Comparison across 13,436,928 values | Relative RMS difference |
| --- | ---: |
| Source recurrent versus source chunked | 9.73e-7 |
| GPU versus source chunked on GPU projections | 9.94e-7 |
| GPU versus source recurrent on GPU projections | 2.81e-7 |

The source replay matches all 12,288 retained source core values at positions
112 and 2186 exactly. The repeated chunk call matches its full raw output
bit-for-bit, and zero beta with zero initial state produces zero memory output.
These checks establish coupling and an algorithm-order contribution; they do
not establish complete-computation correctness or an unavoidable tradeoff.
The GPU comparisons also include CPU/GPU convolution, decay and normalization
differences: this control has **not** captured identical GPU post-convolution
Q/K/V inputs.

Separating source Q normalization from head scaling versus combining their
scales changes 4,748,325 of 13,436,928 common-input CPU values, with relative
RMS difference 5.06e-8. That is another bounded arithmetic sensitivity, not a
demonstrated model fix.

[compare-gdn-algorithms.py](compare-gdn-algorithms.py) loads the unchanged
pinned source definitions and independently decoded artifact weights. It runs
CPU-only, exports a hash-bound optional standalone fixture, and never feeds
modified tensors into model execution. An initial attempt incorrectly assumed
that the source core capture contained every position; it stopped at its
coverage assertion. The corrected tool records the actual two-position source
coverage explicitly. No failed model result or acceptance threshold changed.

An initial exploratory replay accidentally selected the generic `fused`
variant, which is not the model's declared 512-expert routing kernel. Its output
is excluded from attribution. The corrected producer requires both the compiled
shader identity and declared expert/top-k limits before dispatch.

## Causal-prefix control (token 112)

[causal-prefix-control.json](causal-prefix-control.json) compares the 113-token
causal prefix `inventory_prefix113` (`tokens[0..112]`, `M = 113`) against the
full 2,187-token `inventory` prompt (`M = 2,187`) at position 112:

- **Physical AMD GPU (`Layers 0–24`)**: all 25 router-input rows
  (`mlp_hc.branchInput` at position 112, `64,000` F32 values) match
  **bit-for-bit** (`0` differing values) between `M = 113` and `M = 2,187`.
  Prompt-length-dependent GPU dispatch or tail-tile leakage does not cause the
  Layer-24 difference.
- **Independent CPU F32 source (`Layers 0–47`)**: ordered top-10 expert
  selections at position 112 match across all 48 layers (`0` differences,
  selecting expert 60 at rank 10 in Layer 24), while CPU PyTorch SGEMM itself
  drifts by up to `4.053e-6` on shared sample positions (`Layer 46`) and shifts
  Layer 24's rank-10 margin (`expert 60 - expert 225`) from `1.049e-5`
  (`M = 2,187`) to `8.11e-6` (`M = 113`) from `M`-tile blocking alone.

## Full-prefix same-input operator controls (Layers 0–24)

[same-input-operator-controls.json](same-input-operator-controls.json) and
[audit-same-input-operators.py](audit-same-input-operators.py) capture full
`[113, ...]` GPU stage boundaries across Layers 0–24 on `inventory_prefix113`
under `profiles/flash-next-routed-f32kv-qualification` (matching the retained
prepared-plan prefix router inputs bit-for-bit across all `64,000` F32 values at
position 112) and replay every sub-operator on CPU with identical GPU inputs:

| Operator family across `inventory_prefix113` (`Layers 0–24`) | Scope | Worst local same-input difference |
| --- | --- | ---: |
| Layer 0 token embedding + 4-stream repeat | `113 x 10,240` | `0.0` (bit-exact) |
| Layer 1 PLE (`ple_layer_ids = [2]`) | pos 112 (`10,240` values) | `relL2 = 1.79e-7` (`maxAbs = 8.94e-8`) |
| 50 hyper-connection norms / mixes / injections | all `113` tokens | `maxAbs <= 7.63e-6` / `4.05e-6` / `1.07e-6` |
| 19 Gated DeltaNet layers vs source recurrent / chunked | pos 112 (`2,560` values) | `relL2 <= 9.47e-7` / `1.02e-6` |
| 6 full indexed-attention layers (`L3, 7, 11, 15, 19, 23`) | pos 112 (`2,560` values) | `relL2 <= 2.05e-6` |
| 24 upstream MoE + shared-expert blocks (`L0–L23`) | pos 112 (`2,560` values) | `relL2 <= 1.14e-6` (`0` routing flips on all `113` tokens) |

Across all `2,712` upstream token-layer routing decisions (`113` tokens across
Layers 0–23), GPU inputs produce **zero** expert-set or top-10 rank-order
differences relative to the CPU source trajectory. Layer 23's `8.64e-6`
full-attention trajectory difference at position 112 decomposes into `2.05e-6`
local same-input rounding and `8.78e-6` propagated upstream input sensitivity.

At `(Layer 24, token 112)`, exact rational (`Fraction`) dot products give
$s(60) - s(225) = +7.855 \times 10^{-6}$ on $x_{\text{cpu}}[112]$ (expert 60
wins) and $-7.073 \times 10^{-6}$ on $x_{\text{gpu}}[112]$ (expert 225 wins):
an upstream shift $(x_{\text{gpu}} - x_{\text{cpu}}) \cdot (w_{60} - w_{225}) = -1.493 \times 10^{-5}$
accumulated from `4.59e-6` relative F32 evaluation-order / fused-Q4_K FMA
rounding across Layers 0–24. On identical inputs, `matmul_f32_row_subgroup.wgsl`
(`wave64`) and PyTorch `F.linear` produce bit-identical scores for both experts
(`roundingVsExact` is `2.51e-7` on $x_{\text{cpu}}$ and `-7.95e-8` on
$x_{\text{gpu}}$; `wave32` selects the same winner on each input). Selecting
rank-10 expert 225 instead of 60 injects a `0.00482` (`relL2 = 1.41` single-slot)
branch step at Layer 24. No single-operator shader defect or missing F32
rounding boundary is present in Layers 0–24; the `inventory` full-logit gate
remains **failed** (`30,664` violations) under unchanged `quantized-v1`.

- `marbles`: 22 tokens, arithmetic completion.
- `inventory`: 2,187 tokens, structured inventory beyond the sparse-attention
  threshold.
- Raw token inputs, without a chat template; numerical tests only, not an
  answer-quality benchmark or a statistically representative prompt sample.

[spec.json](spec.json) contains the text and declared size bounds.
[inputs.json](inputs.json) contains exact tokens. [input-receipt.json](input-receipt.json)
was created before either reference or GPU execution. [protocol.json](protocol.json)
fixes the unchanged tolerance, two complete last-position vectors, greedy token
agreement and clean execution requirements. It was recorded while the CPU
reference was running and before GPU execution; the inputs were frozen earlier.

The source producer is the unchanged pinned
`artifacts/qwen38-flash-next-20261013/prepare-learned-forward-reference.py`
(SHA-256 `f6bc0f15a6545ab1d205cfc3ec412442415220e541190b63a9c712fa238ba5b0`).
It recomputes each complete prefix on CPU with F32 state, independently decoded
converted Q4_K/F16 weights, and source BF16 n-gram rows. It does not execute
Doppler or use GPU results. The model artifact and original F32 numerical policy
remain unchanged. [reference-index.json](reference-index.json) binds the complete
reference JSON and its sampled-position logit arrays. Source next-token IDs are
8646 for `marbles` and 271 for `inventory`; no Doppler output was used to choose
them. This is source reference evidence, not a new GPU comparison.

## Resume

Use the retained source reference and the exact frozen repaired runtime from
the bounded decode report. Current main and an installed package have not been
qualified by that frozen-runtime result. Do not swap source snapshots while
comparing new prompts.

```sh
python3 reports/flash-next-new-prompts-20261110/prepare-run.py \
  PROTOCOL.json INPUT_RECEIPT.json INPUTS.json \
  REFERENCE/learned-forward-reference-float32-artifact.json \
  PARENT_GPU_RUN.json NEW_GPU_DIRECTORY
```

Then run the existing provisioned `tests/integration/flash-next-learned-physical.test.js`
from that runtime, with `DOPPLER_FLASH_NEXT_PROBE` pointing to the newly prepared
`run.json`. Preserve the shared GPU lock, the active 27B provider and the existing
GPU memory limit. Use a fresh output directory; the original observation remains
retained and failed. The preparation script itself starts no inference.

```sh
python3 artifacts/qwen38-flash-next-20261013/compare-learned-reference.py \
  GPU_OBSERVATION.json REFERENCE/learned-forward-reference-float32-artifact.json \
  NEW_COMPARISON.json --kv-dtype f32 --inputs INPUTS.json --policy POLICY.json
```

The comparator now supports F32 prefill without a decode protocol; F32 decode
still requires one. Missing prompt cases fail. Thirteen focused comparator tests
and seven trace-integrity tests pass. No runtime, policy, catalog, service, memory
limit or deployment change is part of this packet.

The [installed candidate report](../flash-next-installed-20261110/README.md)
retains the `6da05a25` named-profile failure alongside the `e1a02e41` repaired
named-profile pass (byte-identical to the prepared plan across all six vectors)
on the original two prompts. This new inventory failure remains unresolved.
Longer generation including QSA block completion, intended release
configuration, F16 cache and other platforms remain separate work.

Component: `doppler.repository-tooling`, `doppler.tests`. Intent: preserved.
Boundary effects: input preparation and evidence validation only.
