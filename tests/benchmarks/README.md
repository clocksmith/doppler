# Doppler Benchmarks

This directory holds contract tests for the benchmark tooling (receipt schema,
comparison charts, compute evidence, Transformers.js staging). The browser
benchmark runner lives in `src/tooling/browser-harness.js`.

Recommended paths:
- CLI: `npm run bench -- --config '{"request":{"modelId":"..."},"run":{"surface":"auto"}}'`
  (see [tests/GUIDE.md](../GUIDE.md))
- Harness console (page served with `node tools/serve-static.js --port 8080`):
  `const { runBrowserSuite } = await import('/src/tooling/browser-harness.js');`
  then `runBrowserSuite({ suite: 'bench', modelId })`
- Manifest runs: `runBrowserManifest(manifest)`, or open
  `tests/harness.html?manifestUrl=...`

Suite reports are written by `saveReport` (`src/storage/reports.js`): OPFS or
IndexedDB in a browser, and `reports/<modelId>/` (or `$DOPPLER_REPORTS_DIR`)
under Node. Cross-product comparison records live in
`benchmarks/vendors/results/`.
