# Shared reduction follow-up

The Qwen GQA backward softmax reuses `shared_reduce` for two reductions. Every
invocation reads slot zero as the maximum before computing its partial sum, but
there was no barrier preventing lane zero from replacing that maximum before
other invocations read it. A uniform `workgroupBarrier()` now separates the
phases. The repair and regenerated demo shell are on main in `ebb35f98`.

## Operator checks and evidence limits

The regression covers sequence lengths 33, 58 and 129, four query heads sharing
two KV heads, and both causal and noncausal attention. Each surface executes 48
recomputed-softmax dispatches and six complete backward calls. Probabilities,
row sums, repeatability and query/key/value gradients are checked against the
existing independent CPU references. The observation explicitly converts the
reference's head-major order to the shader's token-major row order.

Node/WebGPU and Chrome/Metal passed on Apple M3. Maximum probability error was
below 5.1e-7 and maximum gradient error below 3.7e-7. The **unpatched** M3
baseline also passed: this packet documents a source-level race repair, not a
reproduced numerical corruption on that host. No training campaign, AMD test or
performance comparison ran.

```sh
node tests/inference/qwen-gqa-softmax-race.test.js
node tests/integration/qwen-gqa-softmax-race-browser.js --physical
```

The Node command reports a skip if its GPU probe fails; a skip is not a pass.
The browser runner rejects software and unidentified adapters. Both call the
same case helper. Shared browser ownership/cleanup mechanics also run the
existing F16 RMSNorm regression, which passed again after extraction.

## Gemma guard hypothesis: tested, not established

There is an important scope correction to the earlier
[Gemma repeatability packet](../gemma-repeatability/README.md). Its isolated
snapshot edited `inference/ffn.rules.json` without regenerating the consumed
`rule-bundle.json`. Those source edits did not enable fusion. The three identical
16-token runs and 715 matching boundary arrays remain valid observations of the
unfused path; they do not prove fused-FFN behavior.

The new isolated snapshot regenerates the rule bundle, checks that the exact
Q4K/F16/double-width context selects fusion, and observes the actual shader and
dispatches. Each run reports 30 fused gate/up dispatches over the two configured
trace layers, using `fused_ffn_q4k_f16.wgsl#main` and F16 fused post-normalization.
Source dtype repair `e8b30237` is present in both controls.

- With the RMSNorm barrier: two fresh 16-token continuations are identical.
- Removing only that barrier: two fresh 16-token continuations differ.
- All four commands completed; the historical NaN failure was not reproduced.

This establishes that the RMSNorm race can also affect the genuinely fused
diagnostic path's repeatability. It does **not** establish that a zeroed row
caused the historical intermittent NaN. Production guards remain unchanged.
The explicit diagnostic overrides, regenerated FFN source rules and selected
command-envelope fields are retained here. Complete envelopes remain local;
their projections bind the originals by SHA-256.

## Demo shell

The source refresh retains the 838-module ceiling and the ten-million-byte
limit. A check against isolated committed source `ebb35f98` passes with 902 files,
838 modules and 9,957,864 bytes. This excludes another session's concurrent,
uncommitted activation-grid UI edits; those edits were not staged or regenerated
into this follow-up. No deployment is claimed.

Component: `doppler.runtime-source.gpu.kernels`, `doppler.demo`, `doppler.tests`.
Intent: preserved.
Acceptance evidence: operator references, browser regressions, shader/package
identity checks, committed-source shell check and tracked receipt integrity.
Boundary effects: shader digest and demo cache identity; no training, production
capability-policy, catalog or hosted-model change.
