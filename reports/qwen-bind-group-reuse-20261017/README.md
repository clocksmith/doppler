# Verified Qwen successor and recorded-matmul binding observation

Component: `doppler` qualification and `doppler.docs`. Intent preserved.
No runtime implementation, shader, historical model, catalog, release metadata,
Reploid file, or public contributor process changed. No optimized package was
promoted. The outcome is a verified local successor, a frozen baseline, and a
negative binding-reuse observation on one execution path.

## Exact identities

- Baseline source: `0e75e3c4f51c806c8af6d8bdf473b063fe62532f`.
- Frozen npm archive SHA-256:
  `2662a8b9923dbeb4a455556bf6536a5b1291595a4f5846800eff312f6c24c60f`.
- Model: Qwen3.5-2B, Q4_K weights, F16 embeddings/KV, F32 compute, context 2048.
- Successor manifest SHA-256:
  `c3364f7fecc01b70ce63b4de3347d8305a1a6437ac63bb943edece7fa279fc0e`.
- Historical manifest SHA-256, unchanged:
  `83e5a01a21e234d238ea3ed5b0b234cc52781ddfce51861abd0fdf0ae3199028`.
- Local successor: `.cache/qwen-2b-standard-blake3-20261017/`.
- Physical surface: Chrome 155.0.8059.80, Apple M3, Metal, driver 26.6.2.

[Run 39085845335](https://github.com/clocksmith/doppler/actions/runs/39085845335)
passed for that exact baseline commit. All workflow stages executed.
[Optional skips](ci-baseline-skips.json) retain the 17 unexecuted optional lanes.
Lavapipe is software execution evidence, separate from the M3 run below.

## Successor preparation

`prepare-successor.py` verified every source piece against the independent,
commit-pinned SHA-256 inventory from Reploid `dd5af6cfbac57ab8749b8306e6b7714d0724a9cd`.
It copied 32 shards plus the tokenizer (2,087,223,456 bytes) into a new directory
as separate files, not hard links. It verified the copied SHA-256 and Python
standard-BLAKE3 values. Frozen Doppler independently reproduced every BLAKE3
digest and Node SHA-256 check. [Verification](successor-verification.json) and
[JavaScript verification](successor-js-verification.json) retain both identities.

Only shard hash fields, the manifest variant identity, and explicit migration
provenance changed. Tensor bytes, tokenizer, execution graph, precision, and
sampling semantics did not. A new piece index binds the new manifest identity.
The historical manifest and all old receipts remain frozen. This unsigned local
qualification artifact is not a signed successor release or catalog activation.

The successor reproduced the retained M3 candidate-v3 generated tokens/text for
short and long prompts and the complete public prefill-logit byte hash. This is
scoped execution agreement, not a new answer-quality acceptance.

## Binding-only hypothesis: rejected before implementation

The frozen probe observes actual matmul bind groups and recorded dispatches. It
keys identity by device, selected pipeline, layout, every bound GPUBuffer, binding
number, offset, and size. Pipeline identity captures selected shader/constants;
the selected uniforms retain their existing immutable content-addressed ownership.
No uniforms, ring slots, allocation policy, or GPU commands were changed.

Across the attribution journey, 20,832 recorded decode matmul calls had **zero
exact repeats within a recorder**. Only 38 identities repeated across the full
journey; those are not proof of safe cross-owner reuse. The hot `ffn_down` path
alone had 5,208 calls and zero same-recorder repeats.

For every successive per-layer decode call, input and output buffers changed.
Pipeline, layout, uniform buffer, weight buffer, offsets, and sizes remained
stable. The required bindings therefore were not reusable in the proposed scope.
[Summary](summary.json) retains counts for each role and changed binding.

No cache candidate was implemented or timed. This is a negative eligibility
result, not a claim that an implemented cache regressed or that all bind-group
caching is useless. Adding allocation reuse to make the probe succeed would
combine two experiments. Scratch lifetimes remain a separate campaign.

## Baseline lifecycle and observations

A separate pass without binding instrumentation passed:

- Opening, short generation, warm reuse, public logits, and long prefill.
- Cancellation followed by identical output.
- Two resident sessions with distinct prompts, overlapping generation, and
  exact agreement with each session's isolated output.
- Isolated cancellation while the unrelated session completed, then successful
  reuse of the cancelled session.
- Unloading one session while the other remained usable.
- Device destruction after submission, visible failure, cleanup, recreation,
  and identical output afterward. This deliberately induced loss does not
  establish spontaneous-loss coverage.

Baseline observations: opening 5,345.5 ms; three warm samples had median first
delivered output 198 ms, complete generation 1,537.3 ms, decode 22.96 tokens/s,
and CPU recording 380 ms. Generation produced 32 tokens (31 decode tokens).
Warm delivered-text chunk gaps had p95 619.7 ms. Long-prompt first output was
7,896 ms. There is no candidate or competitor comparison in these numbers.

Peak observed pool allocation across two sessions was 4,432,379,648 bytes;
peak sampled browser-process RSS sum was 2,731,917,312 bytes. Pool accounting is
not physical VRAM, and RSS can double-count shared mappings. After final unload,
retained-model and active allocations were zero; 9,443,328 bytes remained in the
reusable pool. The browser was fresh but OS disk cache was not flushed. The host
was nonexclusive; timing and CPU samples retain that limitation. Three warm runs
are insufficient for a publication-grade speed claim.

## Evidence and reproduction

[Raw bundle](raw-evidence.tar.gz) contains the immutable npm archive, exact
configuration and driver bytes, original and successor manifests, successor
piece index, both binding probes, complete lifecycle receipts, CPU/RSS samples,
and complete CI logs/status. [SHA256SUMS](SHA256SUMS) pins retained controls.
No weights are included; use the provisioned historical bytes.

Extract the bundle into this report directory, and extract the earlier
[representation bundle](../weight-representation-20261017/README.md#evidence-and-reproduction)
for its pinned `model-2b-pieces.json` and `extract-runtime.py`. Run from the repo
root with installed Playwright/Chrome and Python `blake3` (the version is in the
verification receipt). Use fresh destination and output paths; preserve old
receipts. The preparation script refuses to overwrite an existing artifact.

```sh
python3 reports/qwen-bind-group-reuse-20261017/prepare-successor.py models/local/qwen-3-5-2b-q4k-ehaf16 .cache/qwen-successor-new reports/qwen-successor-new
python3 reports/weight-representation-20261017/extract-runtime.py reports/qwen-bind-group-reuse-20261017/baseline/doppler-gpu-0.6.32.tgz .cache/qwen-baseline-new
node reports/qwen-bind-group-reuse-20261017/baseline-journey.js .cache/qwen-baseline-new .cache/qwen-successor-new reports/qwen-bind-group-reuse-20261017/metal-config.json reports/qwen-successor-new/journey.json
python3 reports/qwen-bind-group-reuse-20261017/summarize.py
```

The reducer checks frozen archive, model, configuration, and driver identities.
Its final command reduces the retained campaign, not an arbitrary new run.
`binding-attribution.js` accepts the same runtime/model/config/output arguments
as the lifecycle driver. Its overhead excludes its timings from primary evidence.

Still open: this successor's physical Vulkan lane, optional M3 27B closure,
separate scratch planning, and any optimized package acceptance. Representation
planning stays closed within its tested scope. The rejected four-token delivery
experiment remains rejected. Reploid and WAN work remain outside this campaign.
