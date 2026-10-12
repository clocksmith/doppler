# Explicit compensated recurrent state

Component: `doppler.runtime-source.inference` and `doppler.runtime-source.gpu.kernels`. User-approved bounded precision exception: see `approval.json` and its exact proposal hash. Approved F64, 0.001, model weights, F16 KV, and the 1,420,000,000-byte contributor limit remain unchanged.

The canonical implementation keeps `f32` as the default and accepts explicit `inference.session.linearRecurrentStatePrecision: "two-f32"`. Each scalar stores interleaved high/low F32 components. Both prefill recurrence and fused decode preserve correction components through decay, memory reduction, delta, update and readout. A schema-3 state carries precision and component count through CPU/GPU ownership and clones; incompatible snapshots and active-attempt precision changes fail explicitly. Reset/cleanup retain the existing owners.

The extra allocation is 9,437,184 bytes per nine-linear-layer partition per attempt. Actual candidate concurrent/long-prompt memory acceptance has not yet run. The current full-model diagnostic uses its existing six-billion-byte diagnostic budget; it cannot establish the contributor memory requirement.

## Physical controls and first complete result

Apple Metal and AMD Vulkan each pass 20,014 independently referenced pair arithmetic cases, including cancellation and subnormal carry. Both pass existing independent recurrent-core controls extended to two-F32, F16/F32 inputs, and immediate/recorded fused decode. Operator tolerances are separate from the unchanged whole-model tolerance. Finite intermediate sums/products are the helper domain; this is not expanded exponent storage. Overflow remains nonfinite.

Standard `npm pack --ignore-scripts` archive `d0d53e8f6d72ef7de63433d5935b2e89a717d3b4d6e8d890bd19add499e5e85e` contains canonical source plus the bounded state implementation, without historical projection or RoPE candidates. Later boundary-validation/probe-metadata edits are not in this first archive. Exact model/manifest bindings are retained separately. Every run checks 55 unique prefixes and two resident-reuse controls against the approved independent F64 vectors.

| Device | State | Failed / 55 | Maximum | RMS | Values above 0.001 |
|---|---|---:|---:|---:|---:|
| mac | f32-control | 29 | 0.00295966257719 | 0.000229633042286 | 17987 |
| mac | candidate | 17 | 0.00230737297927 | 0.000200780747521 | 10562 |
| amd | f32-control | 27 | 0.00349515070585 | 0.000226435031292 | 25151 |
| amd | candidate | 18 | 0.0028786238948 | 0.000197323483903 | 10795 |

**Both candidates remain rejected.** Aggregate improvement does not qualify the runtime. On Mac, 36 prefix maxima improve and 19 worsen; 14 old failures clear but decode 7 and 8 newly fail. The first remaining AMD failure is request 2/decode 4; Mac is request 2/decode 7. Numerical investigation continues at those boundaries.

The new AMD F32 control exactly matches all 55 logits vectors and initial input sequences from the retained `c67badcd` source control, archive `d6d007d2`; see `amd-historical-source-control.json`. That source baseline differs from the accepted installed archive's historical 22/55 result. No bitwise cross-platform or historical-Mac compatibility claim follows.

`independent-implementation-review.json` records a separate source review. `operators-*.json`, `core-*-complete.log`, `layout-allocation.log`, and generated-kernel/type checks bind their stated scopes. Old probe code assumed dense cache indexing and installed its observer after proxy method binding; the repaired test-only observer now maps declared allocation indices and installs before method binding. Failed diagnostic attempts remain retained. No model output was substituted.

Both retained shortest-prefix audits exactly reproduce the approved F64 vectors before comparison. At AMD decode 4, final error is 0.00100821 with inherited error 0.00100630; the local final projection contributes less than 0.000007. At Mac decode 7 the corresponding errors are 0.00163967 and 0.00163896. Same-input recurrent-core RMS errors in the first three layers are approximately 1e-8. The first large amplification occurs at layer 3's F16 cache materialization; most differing stored entries inherit upstream differences. This narrows the next causal investigation; it does not qualify either candidate.

The initial 922-file unit sweep had two failures: a legacy schema-1 test fixture, now constructed through the canonical runtime factory, and a stale remote demo test against newer HTML. Both focused reruns pass; the initial failure log remains retained. The subsequent complete current-checkout sweep passes all 922 files (`unit-current-checkpoint.log.gz`). Current strict types, source style, architecture, generated kernel checks and focused layout/probe/failure-path tests pass. These checks do not establish model qualification. Raw archive, vectors and captures are hash-bound in `local-raw-index.json` and `remote-raw-index.json`; compressed audits and check logs are retained here.

## One projection intervention

`qkv-counterfactual-comparison.json` tests only layer 0's prefill QKV output. Copying its original captured output back produces exactly identical downstream logits at all five tested prefixes. Replacing that one output with the independent same-input projection rounded to F32 lowers RMS at all five prefixes and clears decode 4 (0.00100821 → 0.000783249). It also introduces a decode-3 failure (0.000962517 → 0.00122937). This is diagnostic causality, not an accepted correction; injected reference tensors can never qualify production.

The resulting single-shader candidate changes the wide-tile Q4 prefill reduction to retain product/sum residuals before storing the same F32 output. Encoded/materialized weights, dimensions, addressing, other operators and storage remain unchanged. `q4-archive-delta.json` proves precisely one file differs in archive `91b7c94d` relative to `d0d53e8f`. Independently referenced physical projection checks pass on both AMD and Mac, with identical outputs: maximum error falls from 2.88986e-6 to 9.43378e-7 and RMS from 1.29550e-7 to 3.32699e-8. The complete 55-prefix trials are pending; this shader remains an artifact experiment and is not in canonical runtime source.

Nothing is promoted, published or deployed by this checkpoint. The accepted Reploid package is unchanged.
