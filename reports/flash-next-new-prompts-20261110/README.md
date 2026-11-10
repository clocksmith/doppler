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

An initial exploratory replay accidentally selected the generic `fused`
variant, which is not the model's declared 512-expert routing kernel. Its output
is excluded from attribution. The corrected producer requires both the compiled
shader identity and declared expert/top-k limits before dispatch.

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
now separates the named-profile failure from the explicit prepared-plan pass
on the original two prompts. This new inventory failure remains unresolved.
Longer generation including QSA block completion, intended release
configuration, F16 cache and other platforms remain separate work.

Component: `doppler.repository-tooling`, `doppler.tests`. Intent: preserved.
Boundary effects: input preparation and evidence validation only.
