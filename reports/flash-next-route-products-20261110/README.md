# Flash-Next: preserve MoE weighted-product rounding

The frozen 2,304-token F32-KV final-logit check now passes its **unchanged**
`doppler.boundary-tolerance/quantized-v1` policy. All 248,320 values are within
tolerance; maximum absolute difference is **0.0001218319**, RMS **1.3924e-5**.
This is the named long-prefill check on the provisioned AMD Node/WebGPU lane,
not full model, browser, decode, performance or release qualification.

| Ordinary execution on the same artifact, input and F32-KV reference | Maximum absolute difference | Violations |
| --- | ---: | ---: |
| Before the softplus repair | 0.10902119 | 13 |
| After softplus, `970a9a7b` candidate | 0.05958652 | 1 |
| After softplus and separately rounded route products | 0.00012183 | **0** |

The preceding evidence remains unchanged in
[the origin investigation](../flash-next-origin-20261109/README.md). No old
failure is relabeled as a pass. The new candidate has its own shader and run
identities and uses no source-tensor or router-selection replacement.

## Defect and source boundary

The pinned source MoE first computes an expert output, materializes
`current_hidden_states * top_k_weights` as F32, then performs `index_add_`.
The route scatter shader instead expressed `sum + weight * expert_output`.
Both tested compilers can contract the latter into an FMA, omitting the source's
product rounding boundary.

A cancellation control distinguishes the two operations without relying on a
loose numerical allowance. With two routes, weights `0.5` and
`0.5 * (1 + 2^-23)`, and outputs `-1` and `1 - 2^-23`, separately rounded F32
products sum to **zero**. Contracting the second product with the running sum
returns **−2^-47**. The original production shader fails this control on Metal
and AMD. The broader original Metal fixture records 632 output mismatches across
six immediate/recorded cases; see `operator-before-metal.json`.

This does not claim that every differing summation order is a defect. The change
preserves the existing route-slot addition order and specifically restores the
source's separate weighted-product materialization.

## Implementation and physical controls

`scatter_add_routes_f32.wgsl` passes each rounded product to an adjacent
invocation through workgroup memory, then sums it. A barrier precedes the read;
a second barrier prevents reuse until every lane has consumed the product.
Padded invocations participate in both barriers, and bounds checks guard the
product read and final output. The host binding layout and route geometry are
unchanged. A 257-channel fixture crosses token and workgroup boundaries and
exercises the partial final workgroup.

A rejected local multiply helper passed Metal but still contracted on AMD.
Its source and failed AMD receipt are retained as
`rejected-local-helper.wgsl` and `rejected-local-helper-amd.json`. It is not the
shipped repair. Cross-invocation storage uses the same materialization mechanism
already exercised by the hyper-connection branch/injection kernels.

The final physical test exercises the production wrapper in immediate and
recorded modes, with top-k values 2 and 10. Each device checks **1,060 outputs
across six cases**, exactly matching an independent reference that rounds each
F32 product before each F32 addition. Both physical Metal and AMD runs pass,
with no uncaptured GPU errors. See `operator-metal.json` and `operator-amd.json`.
This is operator evidence, not all-model qualification.

Existing Qwen route/executor tests also pass on physical Metal. The Chrome
browser integration passes its 12 route/scatter cases. Kernel registry,
generated shaders, uniform interfaces and content digests remain aligned.

## Full network and remaining qualification

`prepare-route-product-candidate.py` copies the previously identified softplus
candidate to a new operator-local directory, changes the one scatter shader and
its content digest, and records the new identity. The source artifact manifest,
learned tensors, independent reference, KV dtype, projection bindings and
acceptance policy remain unchanged. `preparation.json` records these changes;
`long-comparison.json` binds the completed observation and reference.

At Layer 25, position 1717, the captured margin for experts 493 minus 330 is now
**+1.4305e-6**. The exact dyadic margin on the new GPU input is **+9.3002e-7**,
agreeing in sign with the source input's **+1.5143e-6**. Position 2271 also agrees
in sign. See `router-margin.json`.

Ranked captured logits have the same top-10 sets through Layer 25; the only
inferred set difference across all 48 layers is Layer 43, position 735. The
unchanged final-logit policy passes despite that diagnostic difference. These
are rankings inferred from captured logits, not fresh observations of GPU
expert-index buffers. Expert-set agreement is not substituted for the logit
gate.

The 15-token short check also passes the unchanged prefill policy, with maximum
absolute difference **2.8610e-5**. Three generation rows are finite and cleanup
has no errors. See `short-comparison.json`. Independent prefill-logit comparison
and finite decode rows are not independent decode-logit parity. No learned-weight Metal full-model run, unseen-prompt
campaign, original F16-KV lane promotion, timing result or production rollout is
claimed. The existing provider remains running; no service is restarted and no
GPU memory setting changes.

`source-delta.json` verifies identical inventories across all 3,100 source files:
only the scatter shader and generated digest table differ from the softplus
candidate. Numeric runtime configuration is equal after removing the two
kernel-path identity labels. Artifact, reference, inputs and requested sequence
are unchanged. Its producer is `compare-source-delta.py`.

## Reproduction

The final `receipt.json` binds the code revision and Git blobs of the targeted
softplus and scatter shaders to their source hashes and entry-point digests.
It also binds the unchanged policy, comparator sources, reference, inputs,
candidate source tree and retained long-run observation. `seal-receipt.py`
reads the committed sources and refuses mismatches, substituted execution or a
failed gate. Its checks remain active with Python assertions disabled.

```sh
node tests/kernels/moe-route-product-rounding-physical.test.js OPERATOR_RECEIPT.json
node tests/inference/qwen35-route-contract.test.js
node tests/inference/qwen35-route-gpu.test.js
CHROME_PATH=CHROME_EXECUTABLE node tests/integration/qwen35-route-browser.test.js
python3 artifacts/qwen38-flash-next-20261013/prepare-route-product-candidate.py \
  SOFTPLUS_RUNTIME SOFTPLUS_RUN.json \
  src/gpu/kernels/scatter_add_routes_f32.wgsl \
  tests/kernels/moe-route-product-rounding-physical.test.js NEW_RUNTIME NEW_DATA
python3 reports/flash-next-route-products-20261110/seal-receipt.py \
  REPOSITORY CODE_REVISION LONG_OBSERVATION.json SHORT_OBSERVATION.json NEW_RECEIPT.json
```

From the prepared runtime, run the provisioned learned-model harness with
`DOPPLER_FLASH_NEXT_PROBE` naming `NEW_DATA/run.json`. The prior
`compare-softplus.py` comparator is reused unchanged: it checks the new
preparation, original policy and reference and does not authorize promotion.
The learned weights and raw captures remain operator-local; this report is not
a self-contained model distribution.
