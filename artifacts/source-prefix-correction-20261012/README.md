# Current-source first-prefix investigation

Component: `doppler.repository-tooling`, `doppler.tests`. Intent preserved. No canonical runtime shader, package, reference, precision, or application pin changed.

The installed unchanged-source archive (`d6d007d2…`) reproduces request 2, decode step 3, vocabulary 18050: GPU **5.773292541504**, approved independent F64 **5.772070706681**, error **0.001221834823**. The same-input independent final projection is 5.773292830979: almost all error is inherited before the head. All 51 independently recomputed request-2 vocabulary vectors exactly match the approved retained vectors before captured GPU data enters the diagnostic.

The first attention cache has 352 differing K values and 329 differing V values after F16 storage. Independent same-input producer replay attributes 274 K / 281 V differences to inputs inherited before that producer; 78 K / 48 V are locally introduced. These counts describe the captured current source and do not prove one coordinate causes the final failure.

## Correction under investigation

Compensated F32 projection accumulation has **not** passed its isolated independent bound. The captured prefill baseline maximum is 2.889857e-6; product/sum compensation reaches 2.606869e-6, still above the existing 2e-6 bound and below the required twofold improvement. No candidate proceeds to package promotion.

A narrower trace demonstrates why ordered compensation expressions are unreliable here. For 785 of 1,024 terms at one output, `fma(-1, total, sum)` differs from independently evaluating the captured F32 sum minus total; the result equals the product. Products, product residuals, and actual decoded weights match their independent calculations. The instrumented and uninstrumented candidate outputs match exactly.

For example: total -0.137742310762, product 0.029750371352, sum -0.107991941273. The rounded subtraction is 0.029750369489; GPU `z` is 0.029750371352. This invalidates this candidate's error-free-sum assumption; it does not identify an error in the unchanged production shader or establish a compiler specification violation.

Frexp/ldexp, per-invocation atomic materialization, and the consumed developer-only `strictMath` option do not improve these outputs. Other retained expression and integer-residual attempts also fail. `projection-summary.json` records each exact shader, fixture, physical adapter, full receipt hash, and unfavorable result. The subsequent integer-residual trace and correction are recorded below.

## Retention and reproduction

- `control-audit.json.gz`: independently evolved and actual-input comparisons.
- `executed-audit.py.gz`: exact source used for that receipt; its SHA matches the report. The current tool adds incomplete-history rejection and clearer selected-prefix labeling.
- `observed-products-projection.json.gz`, `residual-differences.json.gz`: captured intermediate values and comparison to specified F32 evaluation.
- `raw-identities.json`: complete raw capture/audit/projection identities. Raw large receipts remain in `reports/local/source-prefix-correction-20261012/` on `x@128.tail995236.ts.net:/home/x/deco/doppler`; the initial capture and compensation controls are also retained in the Mac's same relative directory.
- Six capture-pairing/error tests pass (`tests.log`). The projection harness rejects ignored developer `strictMath`; actual consumption is recorded and never claimed as a production fix.

```sh
PYTHONPATH=/tmp/doppler-reference-deps OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 python3 tools/qwen35-short-prefix-audit.py --capture reports/local/source-prefix-correction-20261012/control-capture.json --contract artifacts/whole-model-f64-complete-20261011/request-2-contract.json --binding artifacts/short-prefix-f64-20261012/control-binding.json --out /tmp/current-source-audit.json
```

Numerical acceptance remains open. No new all-55-prefix success, Mac candidate acceptance, performance improvement, or deployment is claimed.

## Explicit correction reset

The observed integer-residual candidate evaluates every captured product, product residual, F32 sum, integer-derived sum residual, and low-accumulator update correctly. However, the first low accumulator in blocks 1–3 contains the preceding block’s correction. Adding it to the outer total again overcounts that correction. `integer-block-boundary.json` records all 16 block/lane boundaries; the reproduction script checks actual operands and continuity. This observation concerns this experimental shader, not the unchanged production implementation.

Explicitly resetting the per-block correction alongside its high sum fixes this local defect. The complete captured 184×6144 projection passes the unchanged independent bound: maximum **9.433781e-7**, RMS **3.326989e-8**, versus baseline maximum **2.889857e-6**, RMS **1.295499e-7**. Storage, weights, products, and outputs remain F32/Q4_K; integer significands recover the F32 addition residual without a wider floating-point type.

A standard npm archive built from the isolated source snapshot contains only that shader change and generated digest changes. Its identity is in `reset-binding.json`. Full 55-prefix execution, including resident reuse, confirms all declared shader pins. The candidate still fails **15/55**, maximum **0.003245788507**, RMS **0.000194760629**, with **21,497** logits exceeding 0.001. Source control fails 27/55, maximum 0.003495150706, RMS 0.000226435031, with 25,151 logits exceeding tolerance. The complete distributions remain retained; the candidate has not qualified. First remaining failure is request 2 decode 9, largest is decode 28. The earlier compensation candidate fails 25/55, maximum 0.002184175259, and remains separate.

The next trace captures decode 9 with the corrected candidate to separate inherited error from locally introduced error. Canonical runtime source and the Reploid dependency remain unchanged.
