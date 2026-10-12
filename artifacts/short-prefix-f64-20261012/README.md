# Short-prefix accuracy investigation

Component: `doppler.repository-tooling` and `doppler.tests`. Intent preserved. Boundary effects: none; no runtime arithmetic, package pin, reference, precision, or deployment changed after the rejected experiments.

Fresh AMD/RDNA3 Chrome 146 execution reproduces the installed archive’s first failure: request 0, decode step 1, vocabulary 9072, error **0.0010009163826676937**. This capture covers only its 19-token prefill and one decode token. It serves the archive’s exact shaders and verifies that instrumented and ordinary logits match exactly.

## What the shorter sequence establishes

The immutable approved F64 evaluator is observed through Python line events; no evaluator code, function, local, or state is replaced. Both independently recomputed vocabulary vectors exactly match the approved retained vectors before GPU observations enter comparison. The audit pairs observations by generation step and operation, checks duplicate observations, and explicitly recognizes prefill-only FFN gate/up captures.

At vocabulary 9072, local final projection error is approximately -0.000000693255; inherited input error is -0.001000223128. The final projection is not the material cause of this failure.

All 24 direct K/V storage/read comparisons pass: physical F16 bytes exactly round their actual captured inputs, and attention reads the same cache identities. This rules out storage conversion/indexing at these observed boundaries, not all runtime defects.

The first full-attention layer has 54 K and 71 V stored values differing from independently evolved F64 state. Recomputing that entire producer on the same GPU-normalized input leaves 44 K and 59 V differences: these are inherited before the producer. Producer arithmetic introduces the other 10 K and 12 V differences; it also cancels some inherited differences. A concrete K example is token 7, dimension 275: GPU pre-store 2.276366949 versus F64 2.276367363 becomes stored 2.275390625 versus 2.27734375. Its same-input F64 producer still rounds to the GPU side. This example does **not** establish that this coordinate alone causes the final failure.

The largest early local projection error remains the first recurrent A projection, already investigated in the retained F16 accumulation repair. No completed dispatch/cache investigation was reopened.

## One shader at a time, then every prefix

Two Q4 reduction candidates were built with ordinary `npm pack` from source `c67badcd`, alongside an unchanged-source control. Package comparisons prove that each candidate changes only `fused_matmul_q4_widetile.wgsl` plus its generated digest/configuration entries. Upstream F16 projection, normalization, recurrence, weights, precision and generation choices remain fixed. Diagnostic model manifests repin the corresponding exact shader bytes; no runtime substitution bypasses the manifest.

| Configuration | Failed / 55 | Maximum error | RMS over all logits | Logits over 0.001 |
| --- | ---: | ---: | ---: | ---: |
| Pinned archive, retained full comparison | 22 | 0.002666610 | See approved-reference receipt | See approved-reference receipt |
| Fresh unchanged source control | 27 | 0.003495151 | 0.000226435 | 25,151 |
| 64-element pair subtotals | 16 | 0.001928114 | 0.000178789 | 2,162 |
| Balanced pair/block reduction | 25 | 0.002715724 | 0.000213710 | 13,294 |

Each new full run covers 55 unique prefixes and two resident-reuse controls. Full distributions retain every prefix’s maximum, RMS, percentiles, histogram and count above tolerance across 13,657,600 vocabulary values. Historical comparisons emitted by the old replay harness are retained forensic output, **not another acceptance target**; `*-distribution.json` compares only the approved F64 vectors.

On the short capture’s first-layer QKV operands, local maximum errors were baseline 7.072372e-6, pair subtotals 6.742528e-6, balanced reduction 2.927831e-6. Both fail the existing isolated harness’s 2e-6 bound. The balanced candidate improves local arithmetic yet worsens whole-model error relative to the pair candidate. Both are rejected; their shaders and physical receipts remain retained. The canonical runtime shader and generated identities were restored.

**Important source/package distinction:** current source already passes the installed archive’s short failing prefix (maximum 0.000489723), but has larger failures elsewhere. Both new candidates also pass that prefix and still fail the full set. The source control’s first remaining failure is request 2, step 3; the unchanged-source maximum is at step 15. Fixing the first archived failure alone cannot qualify a replacement.

## Evidence and reproduction

- `standard-capture.json.gz`, `audit.json.gz`, `physical-storage.json`: short physical observations, independent/same-input comparisons and actual cache-byte checks.
- `*-distribution.json`: all 55 prefixes and complete vocabulary error distributions.
- `*-binding.json`, `*-archive-differences.json`, `summary.json`: exact archive/model/source identities and raw receipt locations/hashes.
- `rejected-*.wgsl`, `*-projection.json.gz`, logs: failed candidates and observations. The initial audit failure on prefill-only FFN captures is retained separately.
- Raw 54 MiB replay receipts and the three standard archives are retained under `reports/local/short-prefix-f64-20261012/` on **both** the Mac and `x@128.tail995236.ts.net:/home/x/deco/doppler/`. `summary.json` records compressed and uncompressed hashes. They are not duplicated into Git history.

Run from Doppler with NumPy/gguf available:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 tools/qwen35-short-prefix-audit.py --capture artifacts/short-prefix-f64-20261012/standard-capture.json.gz --approved reports/local/whole-model-f64-complete-20261011/independent-vectors/request-0-steps-0-1.json.gz --out /tmp/short-prefix-audit.json
python3 artifacts/short-prefix-f64-20261012/compare-distribution.py --capture reports/local/short-prefix-f64-20261012/candidate-prefixes.json.gz --binding artifacts/short-prefix-f64-20261012/candidate-binding.json --out /tmp/pair64-distribution.json
```

Thirteen focused Python checks pass (five capture/error controls and eight existing F64 equation controls), alongside restored kernel/configuration digest checks. These are correctness diagnostics, not latency or contributor-memory acceptance. No Mac candidate run, package adoption, signed application acceptance, or deployment is claimed.

Next: use the retained current-source request-2/step-3 boundaries and its step-15 maximum to identify a causal producer correction; do not repeat the already resolved short-prefix final-head/storage checks or promote an average-only improvement.
