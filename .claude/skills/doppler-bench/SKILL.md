---
name: doppler-bench
description: Run DOPPLER performance benchmarks in the browser (demo UI or harness), capture JSON reports, and compare against baselines. Use when validating speed changes or collecting benchmark artifacts. (project)
---

# DOPPLER Bench Skill (Browser)

Use this skill to measure DOPPLER inference performance **in the browser**. This repo does not use npm/CLI benchmark commands.

## Quick Start (Demo UI)

1) Serve the repo root:
```bash
python3 -m http.server 8080
```
2) Open `http://localhost:8080/demo/`.
3) Select the target model from the local list.
4) Set Runtime Preset to `experiments/gemma3-bench-q4k` (or the model-specific bench preset).
5) Diagnostics -> Suite = `bench` -> Run.
6) Export the report JSON and save it under `tests/results/`.

## Browser Console (Repeatable)

In DevTools (Demo UI or `tests/harness.html`):
```js
const manifest = await (await fetch('/demo/bench-manifest.json')).json();
const { runBrowserManifest } = await import('/src/inference/browser-harness.js');
const result = await runBrowserManifest(manifest);
console.log(result.report);
```

Optional suite runner:
```js
const { runBrowserSuite } = await import('/src/inference/browser-harness.js');
const result = await runBrowserSuite({
  suite: 'bench',
  runtimePreset: 'experiments/gemma3-bench-q4k'
});
console.log(result.report);
```

## Kernel Microbenchmarks (Browser)

- Open `http://localhost:8080/tests/kernels/browser/test-page.js`
- Use the perf/bench mode to measure kernel timings.

## Discovery

```bash
ls models/
ls src/config/presets/models/
```

## Key Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| `decode_tokens_per_sec` | Main throughput metric | Higher is better |
| `ttft_ms` | Time to first token | Lower is better, measures prefill latency |
| `prefill_tokens_per_sec` | Prompt processing speed | Higher is better for long prompts |
| `decode_ms_per_token_p99` | 99th percentile per-token decode latency (ms) | Flag if >100ms for interactive use |
| `estimated_vram_bytes_peak` | Peak GPU memory usage (bytes) | Lower means more headroom |

## Report Storage + Validation

- Save exported JSON reports under `tests/results/`:
  - `tests/results/bench_<modelId>_<prompt>_<preset>.json`
- Validate against schema:
```bash
node tests/validate-benchmark-results.js
```

## Regression Detection Protocol (Browser)

1) Capture a baseline report JSON under `tests/results/`.
2) Re-run the same preset + prompt + token count.
3) Compare metrics in the reports (TTFT, decode tok/s, p99).
4) **<5% difference**: within noise; **5-10%**: investigate; **>10%**: likely regression.

## Interpretation Guidelines

| Scenario | What to Check |
|----------|---------------|
| Prefill slow | Memory bandwidth, embedding gather |
| Decode slow | Matmul kernels, attention implementation |
| High variance | Background processes, thermal throttling |
| Memory regression | Buffer pool usage, tensor allocation |

## Reference Files

- **Benchmark harness**: `src/inference/browser-harness.js`
- **Demo manifest**: `demo/bench-manifest.json`
- **Benchmark guide**: `docs/style/benchmark-style-guide.md`
- **Performance workflow**: `docs/performance.md`
- **Historical results**: `tests/results/*.json`

## Related Skills

- Use `doppler-debug` if benchmark fails or produces errors
- Use `doppler-convert` to test different quantization levels
