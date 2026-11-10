# Flash-Next: two new prefill prompts

This packet freezes two prompts not used to tune the preceding numerical
repairs. The independent CPU source reference completed all 48 layers and is
retained with verified payload hashes and finite F32 arrays. No GPU
comparison has been performed for these inputs, so there is no new model
qualification result. The previously passing evidence remains in
[the bounded decode report](../flash-next-decode-20261110/README.md).

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
GPU memory limit. No new run is started by this packet.

```sh
python3 artifacts/qwen38-flash-next-20261013/compare-learned-reference.py \
  GPU_OBSERVATION.json REFERENCE/learned-forward-reference-float32-artifact.json \
  NEW_COMPARISON.json --kv-dtype f32 --inputs INPUTS.json --policy POLICY.json
```

The comparator now supports F32 prefill without a decode protocol; F32 decode
still requires one. Missing prompt cases fail. Thirteen focused comparator tests
and seven trace-integrity tests pass. No runtime, policy, catalog, service, memory
limit or deployment change is part of this packet.

After these new prompts: longer generation including QSA block completion, then
installed-package verification and intended release configuration. F16 cache and
other platforms remain separate qualification lanes.

Component: `doppler.repository-tooling`, `doppler.tests`. Intent: preserved.
Boundary effects: input preparation and evidence validation only.
