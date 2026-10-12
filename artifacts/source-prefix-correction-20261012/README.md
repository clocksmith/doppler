# Current-source first-prefix investigation

Component: `doppler.repository-tooling`, `doppler.tests`. Intent preserved. No canonical runtime shader, package, reference, precision, or application pin changed.

The installed unchanged-source archive (`d6d007d2…`) reproduces request 2, decode step 3, vocabulary 18050: GPU **5.773292541504**, approved independent F64 **5.772070706681**, error **0.001221834823**. The same-input independent final projection is 5.773292830979: almost all error is inherited before the head. All 51 independently recomputed request-2 vocabulary vectors exactly match the approved retained vectors before captured GPU data enters the diagnostic.

The first attention cache has 352 differing K values and 329 differing V values after F16 storage. Independent same-input producer replay attributes 274 K / 281 V differences to inputs inherited before that producer; 78 K / 48 V are locally introduced. These counts describe the captured current source and do not prove one coordinate causes the final failure.

## Correction under investigation

Compensated F32 projection accumulation has **not** passed its isolated independent bound. The captured prefill baseline maximum is 2.889857e-6; product/sum compensation reaches 2.606869e-6, still above the existing 2e-6 bound and below the required twofold improvement. No candidate proceeds to package promotion.

A narrower trace demonstrates why ordered compensation expressions are unreliable here. For 785 of 1,024 terms at one output, `fma(-1, total, sum)` differs from independently evaluating the captured F32 sum minus total; the result equals the product. Products, product residuals, and actual decoded weights match their independent calculations. The instrumented and uninstrumented candidate outputs match exactly.

For example: total -0.137742310762, product 0.029750371352, sum -0.107991941273. The rounded subtraction is 0.029750369489; GPU `z` is 0.029750371352. This invalidates this candidate's error-free-sum assumption; it does not identify an error in the unchanged production shader or establish a compiler specification violation.

Frexp/ldexp, per-invocation atomic materialization, and the consumed developer-only `strictMath` option do not improve these outputs. Other retained expression and integer-residual attempts also fail. `projection-summary.json` records each exact shader, fixture, physical adapter, full receipt hash, and unfavorable result. An additional observed integer-residual shader is prepared but unexecuted while the native integration owns the GPU.

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
