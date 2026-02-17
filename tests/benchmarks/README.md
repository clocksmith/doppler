# Doppler Benchmarks

This directory holds benchmark artifacts and manifests for the browser harness.
The actual benchmark runner lives in `src/inference/browser-harness.js` and is
invoked via the demo diagnostics UI or the test harness.

Recommended paths:
- Demo UI: `/doppler/demo` -> Diagnostics -> Suite = `bench`
- Harness console: `runBrowserSuite({ suite: 'bench', modelId })`
- Manifest runs: `runBrowserManifest(manifest)`

Benchmark outputs should be saved under `tests/results/` (JSON) and referenced
from your benchmark logbook. Cross-product comparison records live in
`benchmarks/competitors/results/`.
