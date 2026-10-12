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

The resulting single-shader candidate changes the wide-tile Q4 prefill reduction to retain product/sum residuals before storing the same F32 output. Encoded/materialized weights, dimensions, addressing, other operators and storage remain unchanged. `q4-archive-delta.json` proves precisely one file differs in archive `91b7c94d` relative to `d0d53e8f`. This is a diagnostic shader substitution and repack, not a standard package built from canonical source and not promotable. Independently referenced physical projection checks pass on both AMD and Mac, with identical outputs: maximum error falls from 2.88986e-6 to 9.43378e-7 and RMS from 1.29550e-7 to 3.32699e-8.

Both complete trials still **fail 17/55 prefixes**, maximum 0.00235314935, RMS 0.000201294801, 7,364 logits above tolerance. All 55 complete logits vectors and both resident-reuse vectors are byte-identical between Mac and AMD, which establishes agreement only. Mac's maximum and RMS slightly regress; its prefix maxima improve in 35 cases and worsen in 20, clearing three failures and introducing three. AMD improves 30 maxima and worsens 25, clearing five failures and introducing four. The first remaining failure is request 2/decode 7. This shader remains a rejected artifact experiment and is not in canonical runtime source. A release would require canonical implementation, a new standard package and qualification of those exact bytes.

Nothing is promoted, published or deployed by this checkpoint. The accepted Reploid package is unchanged.

## Bounded frequency replay

The first failed projection-trial prefix was captured without changing its output. Its independent audit again reproduces the approved F64 vectors exactly. Layer 3 key rotation introduces a local maximum error of 1.59244e-5, versus inherited 7.12423e-6. This is the same frequency/angle boundary already investigated in `source-prefix-correction-20261012`, not a newly discovered implementation defect.

The diagnostic archive `7be8358b` reuses that retained corrected-frequency shader, propagating only the later `position_count` ABI. Its sole delta against `91b7c94d` is `rope_precompute.wgsl`. Independent physical table comparison over the actual theta 10,000,000, unscaled 64-dimensional geometry and positions 0–255 reproduces maximum error 7.87126e-8, versus the retained canonical table's 8.42844e-6. The known wider-range first-order approximation limitation remains.

The eight-prefix causal trial is rejected: decode 7 maximum worsens from 0.00104084 to 0.00118385, and new failures appear at decode 3 (0.00108178) and decode 5 (0.00119266). RMS improves at steps 3–7 and worsens at 0–2. `frequency-prefix-comparison.json` retains every prefix; no full-55 run or canonical RoPE change follows from this result. An initial extraction attempted during an incomplete archive transfer failed before GPU execution; the incomplete directory and log remain on the diagnostic host, and the completed archive was hashed before retry.

## One final-token decode control

Retained earlier work already independently characterized the decode Q4 reduction; this is not a new cross-GPU tuning campaign. The current capture reports local layer-0 QKV error 6.27643e-6 at decode 7. The existing independent operand exporter is reused on that capture, with the same deployed weights and recorded `main_gemv` dispatch.

Two diagnostic archives change only that final token's layer-0 QKV output. `dfd2cdf3` copies the captured output; `b248ab98` substitutes the independently evaluated same-input projection rounded to F32. Copying reproduces all eight complete logits vectors exactly. The independent intervention leaves steps 0–6 byte-identical but changes decode-7 maximum 0.00104083690 → 0.00104274425, RMS 0.000198025652 → 0.000196998010, and above-tolerance logits 1 → 2. This local error is not a sufficient cause of the current acceptance failure. No additional decode shader substitution follows.

`decode-counterfactual-comparison.json`, both archive bindings, the compressed operand fixture and both compressed raw results retain the negative outcome. The capture-derived tensors are diagnostic interventions only and cannot serve as production computation or independent acceptance references.

## Ideal global-buffer materialization sensitivity

`ideal-materialization-map.json` and its independent review freeze a test-only control for request 2 through decode 7. The approved evaluator is never changed. An inactive diagnostic copy must first reproduce all 51 approved request-2 vectors exactly; that guard passed. The active copy then uses ideal F64 arithmetic between the reviewed global tensor-buffer stores, nearest-F32 non-state buffers and RoPE tables, two-F32 recurrence stores after both decay and update, and the unchanged nearest-even F16 cache store. Private/workgroup arithmetic and scratch stores are intentionally omitted, so this is not an attainable GPU oracle, an error lower bound, or a proof of impossibility.

The phase map preserves real fusion: decode FFN gate/up values remain internal; prefill stores them separately. Prefill convolution and raw recurrent output materialize separately, unlike fused decode. The post-attention normalization uses the ideal internal residual sum while separately storing the rounded residual; prefill recurrent normalization uses an ideal denominator with its rounded output-buffer numerator. Source hashes and actual dispatch observations bind those distinctions.

Even this ideal global-storage control **fails 2/8 prefixes**: decode 3 reaches **0.00119896819**, and decode 7 **0.00150251166**. Overall RMS is **0.000181356929**, with **119** logits above 0.001. The actual parent at decode 7 was closer (0.00104083690), illustrating cancellation rather than a monotonic arithmetic-error hierarchy. No further shader substitution, reference change, or precision expansion follows automatically. The summary and compressed raw audit retain every compared vector and storage count. Ten focused audit checks pass, including mismatched archive/capture/dispatch rejection. The first remote launch stopped on a missing local binding file; that failure remains retained separately.

The unresolved numerical question is which actual F32 materialization or F16 cache-storage boundary drives the remaining error under this reviewed map. Existing controls establish sensitivity and cancellation but do not yet identify a sufficient production correction. Any next investigation should attribute that boundary using the retained operands and controls before another arithmetic substitution; reference authority, 0.001, weights, F16 KV and the approved precision scope remain unchanged.
