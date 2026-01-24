# Doppler Benchmarking (2026 Baseline)

This doc defines the default benchmark workflow and baseline targets for
2026-ready measurements. It is the canonical entrypoint for TTFT/TPS tracking.

## Default Baselines

Target devices for initial baselines:
- Apple M2/M3 class (unified memory, 16-32GB)
- NVIDIA RTX 4070/4090 class (discrete, 12-24GB)

Primary baseline model:
- `gemma-3-1b-it-q4`

Optional follow-ups:
- `gemma-3-1b-it-f16` (when assets are available)
- `mixtral-8x7b-q4` (once model assets are ready)

## Quick Run (Demo UI)

1) Open `/demo`.
2) Set Runtime Preset: `experiments/gemma3-bench-q4k`.
3) Diagnostics -> Suite = `bench`.
4) Run and export the report (JSON).

Reports are stored via `src/storage/reports.js` and can be exported from the UI.

## Kernel Warmup (Optional)

To reduce cold-start variance, enable kernel prewarm or auto-tuning before
benchmark runs via runtime config. When `tooling.intent = "calibrate"`, the
runtime now defaults `kernelWarmup.prewarm` to `true` unless explicitly set.

```json
{
  "shared": {
    "kernelWarmup": {
      "prewarm": true,
      "prewarmMode": "parallel",
      "autoTune": false
    },
    "tooling": { "intent": "calibrate" }
  }
}
```

## Manifest Run (Console)

The demo ships a manifest file at `/demo/bench-manifest.json`.
You can run it directly from DevTools:

```js
const manifest = await (await fetch('/demo/bench-manifest.json')).json();
const { runBrowserManifest } = await import('/src/inference/browser-harness.js');
const result = await runBrowserManifest(manifest);
console.log(result.report);
```

## Report Storage

Save exported JSON reports under `tests/results/` with a stable naming scheme:

- `tests/results/bench_<modelId>_<prompt>_<preset>.json`
- `tests/results/pipeline_<modelId>_<preset>_<timestamp>.json` (legacy suite)

Update `docs/performance.md` with the measured TTFT and TPS values.

## Schema Validation

Validate stored results against the benchmark schema:

```bash
node tests/validate-benchmark-results.js
```

## Expected Outputs

Bench runs return the schema in `docs/BENCHMARK_SCHEMA.json`, including:
- Required: `schemaVersion`, `timestamp`, `suite`
- `env` (browser/OS/GPU/WebGPU features)
- `model` (modelId, quantization, sizes)
- `workload` (prompt name, token counts, sampling)
- `metrics` (ttft/prefill/decode timings, GPU submits, VRAM usage)
