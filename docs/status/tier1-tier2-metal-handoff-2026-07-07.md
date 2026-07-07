# Tier 1 And Tier 2 Metal Handoff - 2026-07-07

Scope: collect Apple Metal browser evidence for current Tier 1 and Tier 2 lanes,
then regenerate the README/support/release docs from saved receipts. Keep older
supported models such as Gemma 3 270M in the supported-legacy tier unless the
roadmap changes.

## Current Vulkan Receipts

Use these Linux/Vulkan receipts as the reference set:

| Lane | Workload | Receipt | State |
| --- | --- | --- | --- |
| Qwen 3.5 0.8B Q4K | p512-d128-t0-k1 | `benchmarks/vendors/results/compare_20260707T153509.json` | release-claimable, exact 128-token match, 69.07 vs 41.79 decode tok/s |
| Qwen 3.5 2B Q4K | p064-d064-t0-k1 | `benchmarks/vendors/results/compare_20260707T154847.json` | local-comparable, exact 64-token match, 60.95 vs 40.10 decode tok/s |
| Qwen 3.5 2B Q4K | p256-d128-t0-k1 | `benchmarks/vendors/results/compare_20260707T155858.json` | local-comparable, exact 128-token match, 58.98 vs 40.64 decode tok/s |
| Qwen 3.5 2B Q4K | p512-d128-t0-k1 | `benchmarks/vendors/results/compare_20260707T161623.json` | local-comparable, exact 128-token match, 56.50 vs 41.50 decode tok/s |
| Gemma 4 E2B INT4-PLE AF32 | p064-d064-t0-k1 | `benchmarks/vendors/results/compare_20260707T170557.json` | local-comparable under explicit decode-valid mismatch policy, 16.32 vs 9.24 decode tok/s, total run 4417.1 vs 7036.4 ms |

Gemma 4 plain Q4K AF32 is blocked before benchmark execution on this machine:
`models/local/gemma-4-e2b-it-q4k-ehf16-af32/manifest.json` is stale and fails
required inference preflight. Missing fields include `output.embeddingScale`,
`output.logitInputScale`, `layerPattern.residualBranchScale`, and LongRoPE
nullable fields. Refresh or reconvert the artifact; do not patch runtime
fallbacks or hand-edit the manifest as a benchmark fix.

## Stage TJS Models On Mac

```bash
node tools/stage-tjs-model.js --model-id onnx-community/Qwen3.5-0.8B-ONNX --preset full --dtype q4f16
node tools/stage-tjs-model.js --model-id onnx-community/Qwen3.5-2B-ONNX --preset full --dtype q4f16
node tools/stage-tjs-model.js --model-id onnx-community/gemma-4-E2B-it-ONNX --preset text-generation --dtype q4f16
```

Use the staged root as `--tjs-local-model-path`, for example
`/Users/<user>/.cache/doppler/tjs-models`.

## Run Tier 1/Tier 2 Text Lanes

Run Qwen 0.8B p512 first because it is the hosted quickstart release-claim lane:

```bash
node tools/compare-engines.js --model-id qwen-3-5-0-8b-q4k-ehaf16 --workload p512-d128-t0-k1 --mode compute --warmup 1 --runs 15 --decode-profile throughput --doppler-surface browser --tjs-local-model-path /Users/<user>/.cache/doppler/tjs-models --save --timeout-ms 1800000
```

Run Qwen 2B p064/p256/p512:

```bash
node tools/compare-engines.js --model-id qwen-3-5-2b-q4k-ehaf16 --workload p064-d064-t0-k1 --mode compute --warmup 1 --runs 15 --decode-profile throughput --doppler-surface browser --tjs-local-model-path /Users/<user>/.cache/doppler/tjs-models --save --timeout-ms 1800000
node tools/compare-engines.js --model-id qwen-3-5-2b-q4k-ehaf16 --workload p256-d128-t0-k1 --mode compute --warmup 1 --runs 15 --decode-profile throughput --doppler-surface browser --tjs-local-model-path /Users/<user>/.cache/doppler/tjs-models --save --timeout-ms 1800000
node tools/compare-engines.js --model-id qwen-3-5-2b-q4k-ehaf16 --workload p512-d128-t0-k1 --mode compute --warmup 1 --runs 15 --decode-profile throughput --doppler-surface browser --tjs-local-model-path /Users/<user>/.cache/doppler/tjs-models --save --timeout-ms 1800000
```

Run Gemma 4 INT4-PLE p064 under the performance-comparable INT4-PLE policy.
Exact output parity is not required for this lane; the acceptance gate is
`fairness.claimGrade=true`, `fairness.correctnessOk=true`, shared prompt
validity, and non-empty decode under `outputParityPolicy.matchMode=decode-valid`.

```bash
node tools/compare-engines.js --model-id gemma-4-e2b-it-q4k-ehf16-af32-int4ple --workload p064-d064-t0-k1 --mode compute --warmup 1 --runs 15 --decode-profile throughput --doppler-surface browser --tjs-local-model-path /Users/<user>/.cache/doppler/tjs-models --save --timeout-ms 1800000
```

If Gemma 4 INT4-PLE reports `fairness.claimGrade=false`,
`fairness.correctnessOk=false`, or decode validity failure, keep the receipt as
diagnostic evidence only and do not add win/loss README language. If the receipt
is claim-grade with output mismatch, disclose that it is a product-format
throughput comparison, not an exact-output claim. If plain Q4K has a refreshed
artifact and passes manifest preflight, run the same command with
`--model-id gemma-4-e2b-it-q4k-ehf16-af32`.

## Optional Tier 1 Retrieval Lanes

Run these if Mac Metal retrieval receipts are needed in the same commit:

```bash
node tools/compare-embeddings.js --model-id qwen-3-embedding-0-6b-q4k-ehf16-af32 --warmup 1 --runs 15 --doppler-source quickstart-registry --doppler-surface browser --cache-mode warm --load-mode http --save --json
node tools/compare-rerankers.js --model-id qwen-3-reranker-0-6b-q4k-ehf16-af32 --warmup 1 --runs 15 --doppler-source quickstart-registry --doppler-surface browser --cache-mode warm --load-mode http --save --json
```

## Regenerate And Check

After saving Mac receipts:

```bash
node tools/vendor-bench.js matrix --include-local-results
npm run support:inventory:sync
npm run support:competition:sync
npm run bench:vendors:validate
npm run claims:evidence:check
npm run support:inventory:check
npm run support:competition:check
npm run check:green
node tools/vendor-bench.js matrix --include-local-results --check
```

Then update `README.md` only from claim-grade receipts. For Gemma 4 INT4-PLE,
claim decode/total-run wins only when the receipt passes the explicit
decode-valid policy, and disclose any output mismatch.

## Commit And Sync

Include saved Mac receipts, `benchmarks/vendors/local-inference-claim-matrix.json`
when a static claim lane changes, generated release/support docs, and the README.
Before pushing:

```bash
git status --short
git diff --stat
git add README.md docs/model-roadmap.md docs/status/tier1-tier2-metal-handoff-2026-07-07.md benchmarks/vendors/local-inference-claim-matrix.json benchmarks/vendors/release-matrix.json docs/release-matrix.md benchmarks/vendors/model-support-inventory.json docs/model-support-inventory.md benchmarks/vendors/model-competition-scoreboard.json docs/model-competition-scoreboard.md benchmarks/vendors/results
git commit -m "Refresh Tier 1 and Tier 2 benchmark evidence"
git pull --rebase
git push
```
