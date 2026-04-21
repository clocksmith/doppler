# Vendor Benchmark Registry

This folder is the source of truth for cross-product benchmark comparisons.
It is intentionally separate from style guides and from Doppler-only benchmark notes.
Claim methodology and fairness policy are canonical in [docs/benchmark-methodology.md](../../docs/benchmark-methodology.md).

## Quick Links

- Shared workloads: [workloads.json](./workloads.json)
- Example compare artifact: [fixtures/g3-p064-d064-t0-k1.compare.json](./fixtures/g3-p064-d064-t0-k1.compare.json)
- Target registry: [registry.json](./registry.json)
- Capability matrix: [capabilities.json](./capabilities.json)
- Harness definitions: [harnesses/](./harnesses)
- Latest generated matrix: [release-matrix.json](./release-matrix.json)
- Published matrix doc: [docs/release-matrix.md](../../docs/release-matrix.md)

## Purpose

- Track vendor targets in one machine-readable registry.
- Keep one harness definition per vendor.
- Normalize external benchmark outputs into a shared comparison record.
- Gate benchmark claims through reproducible CI checks and normalized result artifacts.

## How to interpret results

What these benchmarks prove:
- Phase timing split under one shared contract (`modelLoadMs`, `firstTokenMs`, `firstResponseMs`, `prefillMs`, `decodeMs`, throughput, and decode p50/p95/p99).
- Relative behavior for the same workload/sampling/cache/load settings on the same browser + machine profile.
- Whether configuration and load-path changes map to measurable differences.

What these benchmarks do not prove:
- Internet/WAN performance (local loopback/LAN load-path tests are not WAN tests).
- Cross-hardware absolute rankings.
- Quality parity beyond the explicit correctness checks captured in the run.

Claim format to keep reports auditable:
- State the workload and cache/load mode.
- State engine/version (`Doppler`, `Transformers.js (v4)`).
- Include the exact command plus artifact paths:
  - generated outputs under [results/](./results/)
  - committed fixture evidence under [fixtures/](./fixtures/) when publishing stable claims

## Claimable Evidence Rules

Use these before turning a compare result into README copy, a chart, or a release-facing statement:

- Correctness must be clean on the promoted lane.
  A mismatch compare artifact is diagnostic evidence, not claimable speed evidence.
- Prompt rendering must be explicit and shared.
  `tools/compare-engines.js` records `promptContract` with raw/rendered prompt
  text and passes the same rendered model-input prompt to both engines whenever
  compare-owned rendering is used.
- Zero-decode runs are invalid evidence.
  If an engine returns no decode tokens or an empty generated text sample for a
  multi-token benchmark, the compare artifact must mark the lane non-comparable
  instead of reporting a speed claim.
- Verify the measured artifact source.
  Published HF artifacts, local manifests, and compare-profile defaults can diverge; claim only from the source the artifact actually used.
- Use the real performance lane.
  Do not promote timing from a debug-only `f32`, `http`, or otherwise investigation-specific profile as if it were the warmed compare lane.
- Separate product-engine wins from format-identical wins.
  A Doppler RDRR or direct-source LiteRT lane may be compared with the best
  available Transformers.js ONNX/WebGPU lane, but the artifact and claim must
  say the formats differ.
- Keep one traceable artifact per claim.
  A claim should point back to one saved compare JSON and one reproducible command.
- If a chart mixes legacy `warm` fixtures with newer `compute/parity` fixtures, the chart tool must normalize that intentionally and have a regression test covering it.

## Registry Files

- [registry.json](./registry.json): canonical list of vendor products and harness links.
- [workloads.json](./workloads.json): shared workload IDs used for apples-to-apples comparisons.
  - includes `defaults.compareEngines`, used by `tools/compare-engines.js` when no explicit workload/prompt/token lengths are passed.
  - `prefillTokens` is an enforced model-input token target, not a raw word-count hint.
- [capabilities.json](./capabilities.json): capability matrix for bench/profiling coverage by target.
  Feature values are tri-state: `supported`, `unsupported`, `unknown`.
- [harnesses/](./harnesses): one harness definition per vendor.
- [schema/](./schema): schemas for registry, workloads, harness, capabilities, metric contract, and normalized result records.
- [schema/compare-engines-config.schema.json](./schema/compare-engines-config.schema.json): schema for `compare-engines.config.json`.
- [schema/release-matrix.schema.json](./schema/release-matrix.schema.json): schema for generated release/support matrix payload.
- [results/](./results/): generated normalized outputs and committed chart snapshots. Some JSON compare artifacts may also be committed here when they are part of published evidence.
- [fixtures/](./fixtures/): committed sample compare payloads for clean-checkout chart and matrix smoke checks.
- [compare-metrics.json](./compare-metrics.json): shared compare metric contract for CLI and harness-driven extraction.
- Compare artifacts intentionally use apples-to-apples prompt metrics:
  `firstTokenMs` and `promptTokensPerSecToFirstToken`.
  Raw engine payloads may still carry engine-defined `prefillMs` / `prefillTokensPerSec`, but those are not compare-contract claims when semantics differ.
- [release-matrix.json](./release-matrix.json): generated release/support matrix from registry + workloads + capabilities + model catalog (+ optional latest compare JSON).

## Format Comparison Matrix

The benchmark system supports a **2x2 format-fairness matrix** that isolates
engine performance from format optimization advantages.

Each engine gets its optimized format **and** a shared neutral baseline (SafeTensors).
This prevents "unfair format" objections and enables four distinct claims from one model.

```
                              Doppler                    Transformers.js
                    ┌─────────────────────────┐ ┌─────────────────────────┐
                    │  WebGPU Inference Engine │ │  ONNX Runtime / Native  │
                    └────────┬────────────────┘ └────────┬────────────────┘
                             │                           │
               ┌─────────────┴─────────────┐  ┌──────────┴──────────┐
               │                           │  │                     │
         ┌─────┴─────┐             ┌───────┴──┴───┐          ┌─────┴─────┐
         │   RDRR    │             │ SafeTensors  │          │   ONNX    │
         │ optimized │             │   neutral    │          │ optimized │
         └─────┬─────┘             └───────┬──┬───┘          └─────┬─────┘
               │                      ▲    │  │    ▲               │
               │                      │    │  │    │               │
               │              ┌───────┘    │  │    └───────┐       │
               │              │            │  │            │       │
         ┌─────┴──────────────┴──┐   ┌─────┴──┴────────────┴──────┴┐
         │    DOPPLER LANES      │   │  TRANSFORMERS.JS LANES      │
         │                       │   │                              │
         │  doppler + rdrr       │   │  tjs + onnx                 │
         │  doppler + safetensors│   │  tjs + safetensors           │
         └───────────────────────┘   └──────────────────────────────┘

  ════════════════════════════════════════════════════════════════════════
                         COMPARISON MATRIX
  ════════════════════════════════════════════════════════════════════════

  ┌──────────────────┬──────────────────────┬──────────────────────────┐
  │                  │  SafeTensors         │  Optimized Format        │
  │                  │  (neutral ground)    │  (best-case per engine)  │
  ├──────────────────┼──────────────────────┼──────────────────────────┤
  │                  │                      │                          │
  │  Doppler         │  direct-source/v1    │  RDRR                    │
  │                  │  F16 weights         │  Q4K shards              │
  │                  │                      │                          │
  ├──────────────────┼──────────────────────┼──────────────────────────┤
  │                  │                      │                          │
  │  Transformers.js │  native backend      │  ONNX / ORT             │
  │                  │  F16 weights         │  Q4F16 graph             │
  │                  │                      │                          │
  └──────────────────┴──────────────────────┴──────────────────────────┘

  What each quadrant proves:

    neutral x neutral     Pure engine comparison. Same weights, same
    (ST vs ST)            precision, same format. No format advantage.

    optimized x optimized Best-vs-best. Each engine with its format
    (RDRR vs ONNX)        advantage. Real-world practical comparison.

    optimized vs neutral  Format uplift. How much does RDRR help
    (per engine)          Doppler? How much does ONNX help TJS?

  ════════════════════════════════════════════════════════════════════════
                      BACKEND MATRIX (orthogonal)
  ════════════════════════════════════════════════════════════════════════

  Format comparison uses one backend (browser WebGPU).
  Backend comparison uses optimized format per engine.

  ┌──────────────────────┬─────────┬──────────────────┐
  │  Backend             │ Doppler │ Transformers.js   │
  ├──────────────────────┼─────────┼──────────────────┤
  │  Browser WebGPU      │   ✓     │   ✓               │
  │  Node @simulatte     │   ✓     │   experimental    │
  │  Node webgpu-npm     │   ✓     │   experimental    │
  │  Bun WebGPU          │   exp   │   experimental    │
  │  Deno WebGPU         │   exp   │   experimental    │
  └──────────────────────┴─────────┴──────────────────┘

  Total lanes per model:
    Format matrix:  4 combos x 1 backend  =  4
    Backend matrix: 1 combo  x 5 backends =  5
                                          ─────
                                     9 lanes/model
```

CLI flags for format selection:

```bash
# Default: optimized formats (rdrr vs onnx)
node tools/compare-engines.js --model-id gemma-3-1b-it-q4k-ehf16-af32

# Neutral ground: both engines on SafeTensors
node tools/compare-engines.js --model-id gemma-3-1b-it-q4k-ehf16-af32 \
  --doppler-format safetensors --tjs-format safetensors

# Mixed: Doppler optimized vs TJS neutral
node tools/compare-engines.js --model-id gemma-3-1b-it-q4k-ehf16-af32 \
  --tjs-format safetensors
```

Config in [compare-engines.config.json](./compare-engines.config.json):
the root `defaults` block defines:
- `warmLoadMode` and `coldLoadMode`, so warm/cold compare lanes have explicit load semantics even without `--load-mode`

each model profile has:
- `defaultDopplerSource` (`quickstart-registry|local`) so compare runs resolve the same artifact source on every machine
- `compareLane` (`performance_comparable|capability_only`) so support-only rows do not turn into accidental speed claims
- `dopplerRuntimeProfileByDecodeProfile` so model-scoped Doppler tuning is explicit for parity/throughput lanes
- `defaultLoadMode` and `defaultLoadModeReason` when a model needs a
  shared load-mode override to keep both engines runnable under the same
  recorded cache/load contract
- `defaultDopplerFormat`, `defaultTjsFormat`, and `safetensorsSourceId`
  (the HF repo with the original F16/BF16 weights)

Canonical Transformers.js compare metadata comes from [models/catalog.json](../../models/catalog.json):
- `vendorBenchmark.transformersjs.repoId` is the canonical repo mapping fallback when a compare profile omits `defaultTjsModelId`
- `vendorBenchmark.transformersjs.dtype` is the canonical default `--tjs-dtype` for claim lanes unless the caller overrides it explicitly

`tools/compare-engines.js` resolves the Doppler model from that declared source
and preflights the selected manifest before timing. Hosted quickstart models
therefore use the published HF artifact by default and fail closed when that
artifact is stale. Capability-only lanes are rejected unless you pass
`--allow-non-comparable-lane`.

Capabilities in [capabilities.json](./capabilities.json): the `format`
feature category tracks `rdrr_runtime`, `onnx_runtime`, `safetensors_runtime`,
and `format_matrix_compare` per target.

## LiteRT/TFLite Goal Lane

Doppler has an experimental direct-source path for `.tflite`, `.task`, and
`.litertlm` inputs. Treat that as a capability/best-available-web goal lane,
not as part of the default claim matrix until the harness produces a saved
compare artifact.

The useful product claim, if the receipt supports it, is:

> Doppler runs Gemma 4 E2B in the browser faster than the best available
> Transformers.js WebGPU/ONNX path under the same prompt, token, sampling,
> cache, and hardware contract.

That claim is valid only if the artifact shows:

- Doppler is running the declared browser artifact path (`rdrr` or direct-source
  LiteRT/TFLite), not a Node-only surrogate.
- Transformers.js is running its supported browser path (`onnx` or
  `safetensors` in this harness), not an unavailable LiteRT surrogate.
- The `promptContract` is shared and both engines produce non-empty text with
  non-zero decode tokens.
- The compare section explicitly records the differing formats and does not
  imply a format-identical kernel benchmark.

## Closed Workstream Snapshot (2026-02-22 UTC)

- Gemma 3 Q4K `f32a` now auto-selects `gemma3-q4k-dequant-f32a-online` on subgroup-capable devices ([src/rules/inference/kernel-path.rules.json](../../src/rules/inference/kernel-path.rules.json)).
- Kernel path registry marks `gemma3-q4k-dequant-f32a-online` as canonical/default ([src/config/kernel-paths/registry.json](../../src/config/kernel-paths/registry.json)).
- Structural CI sweep for Gemma 3 1b kernel-path invariants is enforced by [tests/inference/gemma3-1b-kernel-sweep.test.js](../../tests/inference/gemma3-1b-kernel-sweep.test.js).
- Inference guard workflow now triggers on inference rule changes and executes the sweep gate ([.github/workflows/inference-guard.yml](../../.github/workflows/inference-guard.yml)).
- Historical local performance numbers are stale after kernel-path routing updates; re-run apples-to-apples benchmark suites before publishing comparative claims.

Execution tracking now lives in:

- generated normalized artifacts under [results/](./results/)
- committed compare fixtures under [fixtures/](./fixtures/)
- harness + workload contracts in this folder
- CI gates in [.github/workflows/inference-guard.yml](../../.github/workflows/inference-guard.yml)

## CLI

Use [tools/vendor-bench.js](../../tools/vendor-bench.js):

- `node tools/vendor-bench.js list`
- `node tools/vendor-bench.js validate`
- `node tools/vendor-bench.js capabilities`
- `node tools/vendor-bench.js capabilities --target transformersjs`
- `node tools/vendor-bench.js gap --base doppler --target transformersjs`
- `node tools/vendor-bench.js matrix`
- `node tools/vendor-bench.js matrix --compare-result benchmarks/vendors/fixtures/g3-p064-d064-t0-k1.compare.json`
- `node tools/vendor-bench.js matrix --include-local-results`
- `node tools/vendor-bench.js show --target webllm`
- `node tools/vendor-bench.js import --target webllm --input /tmp/webllm-result.json`
- `node tools/vendor-bench.js run --target webllm --workload p064-d064-t0-k1 -- node ./path/to/runner.js`

`import` and `run` both produce normalized records under [results/](./results/) unless `--output` is specified.
`matrix` writes [release-matrix.json](./release-matrix.json) and [docs/release-matrix.md](../../docs/release-matrix.md).
By default, `matrix` auto-discovers committed compare fixtures ending in `.compare.json` under [fixtures/](./fixtures/) only.
Use `--include-local-results` to also scan [results/](./results/) for additional compare JSON artifacts.
Use `--strict-compare-artifacts` to fail generation when any auto-discovered compare artifact is invalid.
Workload rows in the markdown include a `GPU/OS/Platform` column derived from each linked compare artifact's runtime environment metadata.
When multiple compare artifacts exist, markdown selects the latest artifact per `(workloadId, modelId, runtime)` tuple.
Runtime labeling prefers explicit GPU adapter metadata, then falls back to host CPU model when adapter fields are missing.
For fixtures tagged `.apple-m3pro.compare.json`, missing runtime fields default to `Apple M3 / metal / darwin / chromium`.
Browser labeling prefers explicit browser executable and falls back to `userAgent` parsing; browser platform is not used as browser identity.
When `--compare-result` is provided, matrix generation also captures host/browser/GPU specs from that compare payload.
`tools/compare-engines.js --save` refreshes release-matrix artifacts automatically using fixture-only discovery by default (use `--skip-matrix-update` to opt out).

## Normalization Notes

- Canonical timing contract includes:
  - `decodeTokensPerSec`
  - `prefillTokensPerSec`
  - `promptTokensPerSecToFirstToken` (compare artifacts only)
  - `firstTokenMs`
  - `firstResponseMs`
  - `prefillMs`
  - `decodeMs`
  - `totalRunMs`
  - `modelLoadMs`
  - `decodeMsPerTokenP50`
  - `decodeMsPerTokenP95`
  - `decodeMsPerTokenP99`
- Compare artifacts rename prompt throughput to `promptTokensPerSecToFirstToken` and exclude raw `prefillMs` from the compare contract when engines expose different prefill semantics.
- Capability matrices expose `promptTokensPerSecToFirstToken` separately from raw `prefillTokensPerSec` so compare-contract coverage stays explicit.
- Shared synthetic workload prompts are resolved against the selected tokenizer before timing, so `prefillTokens` means actual model-input prompt tokens after any enabled chat template.
- Compare sections are invalid when prompt-token counts are missing, differ between engines, or miss the shared `prefillTokens` target.
- `cacheMode` and `loadMode` are required under each run's `timing` object (`cacheMode`: `cold|warm`, `loadMode`: `opfs|http|memory`).
- Normalized result records now require a canonical `environment` block (`host`, `browser`, `gpu`, `runtime`) so platform/hardware context is always captured in benchmark JSON.
- For `vendor-bench run`, missing core environment capture fields fail normalization (`host`, browser identity, GPU identity, backend, runtime device/library).
- Harness mappings allow ordered fallback path arrays (`normalization.metricPaths` / `metadataPaths`).
- Path order is canonicalized in harness files and validated before comparison.
- Metric paths are canonicalized through [harnesses/](./harnesses) and validated as required before any comparison.
- [tools/compare-engines.js](../../tools/compare-engines.js) defaults to `--decode-profile parity` with the release decode cadence from [benchmark-policy.json](./benchmark-policy.json) (currently Doppler `batchSize=4`, `readbackInterval=4`, `disableMultiTokenDecode=false`, `session.speculation.mode=none`); use `--decode-profile throughput` for the same cadence with throughput-lane labeling.
- When a compare model profile declares `dopplerRuntimeProfileByDecodeProfile`, [tools/compare-engines.js](../../tools/compare-engines.js) loads that runtime profile first and then reapplies compare-managed prompt/sampling/cadence fields on top, so model-specific tuning stays explicit without silently changing the lane contract.
- [tools/compare-engines.js](../../tools/compare-engines.js) records the exact installed Transformers.js / ONNX Runtime stack in each compare artifact and validates that the v4 runner is pinned to the same nested ORT modules before timing starts.
- [tools/compare-engines.js](../../tools/compare-engines.js) applies the explicit Doppler compare-lane `runtime.inference.kernelPathPolicy` from [benchmark-policy.json](./benchmark-policy.json); capability-aware remaps used for known platform/runtime constraints are therefore part of the recorded engine overlay, not a hidden runtime fallback.
- [tools/compare-engines.js](../../tools/compare-engines.js) also applies the explicit Doppler browser channel from [benchmark-policy.json](./benchmark-policy.json) unless `--doppler-browser-channel` overrides it, so compare runs do not silently drift across locally installed browser channels.
- [tools/compare-engines.js](../../tools/compare-engines.js) applies the explicit Doppler load-mode runtime overlays from [benchmark-policy.json](./benchmark-policy.json). The `http` overlay pins direct HTTP range reads, disables eager full-shard hash verification, coalesces ranges into 64 MiB blocks with a 512 MiB range cache and 8 concurrent loads, and opts RDRR manifests into bounded 8-shard whole-shard prefetch for cold browser compare runs; the compare artifact records the overlay JSON and hash.
- [tools/compare-engines.js](../../tools/compare-engines.js) resolves warm/cold load modes from one explicit contract only: `--load-mode` or the root `defaults` block in [compare-engines.config.json](./compare-engines.config.json). It does not derive load mode from `cacheMode`.
- [tools/compare-engines.js](../../tools/compare-engines.js) resolves the default Transformers.js repo/dtype from [models/catalog.json](../../models/catalog.json) vendor-benchmark metadata before falling back to generic defaults, so claim lanes stay pinned to the cataloged comparable baseline.
- The Transformers.js warm `opfs` lane now performs an untimed one-token generation prime, not just a bare model load, before the offline timed pass. This is required for models that lazily fetch generation assets on first decode.
- For large browser-side Transformers.js compare lanes, prefer a staged local snapshot over live HF/Xet fetches. Use [tools/stage-tjs-model.js](../../tools/stage-tjs-model.js) and pass the snapshot root with `--tjs-local-model-path`. The staging helper now validates the required decoder/embed shards for the selected dtype and fails closed when the local snapshot is incomplete.
- [tools/compare-engines.js](../../tools/compare-engines.js) does not mutate compare-lane semantics on retry. If an engine fails, the section is recorded with `pairedComparable: false` and an `invalidReason`.
- Compare artifacts pin harness + metric-contract hashes; stale compare JSON is dropped from `vendor-bench matrix` unless you refresh it.
- Doppler surface is now explicit in compare runs: `--doppler-surface auto|node|browser` (default from `compare-engines.config.json` per model profile via `defaultDopplerSurface`, fallback `auto`).

## Visualization

Use [benchmarks/vendors/compare-chart.js](./compare-chart.js) to turn a saved compare result file into an SVG:

```bash
node benchmarks/vendors/compare-chart.js --input benchmarks/vendors/fixtures/g3-p064-d064-t0-k1.compare.json
node benchmarks/vendors/compare-chart.js --input benchmarks/vendors/fixtures/g3-p064-d064-t0-k1.compare.json --chart stacked
node benchmarks/vendors/compare-chart.js --input benchmarks/vendors/fixtures/g3-p064-d064-t0-k1.compare.json --chart radar --section compute/parity
```

Use `--section` to choose the section, `--chart` (`bar|stacked|radar`) to pick the renderer, and `--metrics` to limit metric IDs.

## Change Checklist

Add a vendor target:
- Update `benchmarks/vendors/registry.json` with the product entry and harness path.
- Add/update capability statuses (`supported`/`unsupported`/`unknown`) + evidence in `benchmarks/vendors/capabilities.json`.
- Add a harness definition in `benchmarks/vendors/harnesses/<vendor>.json`.
- Run `node tools/vendor-bench.js validate` and fix schema/shape violations.
- Update this README if workflow/coverage expectations changed.

Add a workload:
- Add the workload row in `benchmarks/vendors/workloads.json`.
- Ensure it passes `benchmarks/vendors/schema/workloads.schema.json`.
- If it should be the default, update `defaults.compareEngines`.
- Run `node tools/vendor-bench.js validate`.

Add or rename a compare metric:
- Update `benchmarks/vendors/compare-metrics.json` (id/label/unit/higherBetter/required).
- Ensure harness path mappings are present in both Doppler and Transformers.js harness files.
- Run `node tools/compare-engines.js --help` sanity checks and a sample compare run.
- Regenerate chart artifacts if metric display is expected in committed visuals.
