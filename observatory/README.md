# Doppler Observatory

The browser site for exploring how Doppler forms an answer, published at
`https://d4da.com/observatory/` (and `https://d4da.com/doppler/`, which
redirects there). It is not part of the npm package: `package.json` publishes
from an allowlist that does not include this folder.

## Pages

- **Explore** (`/`, `public/index.html`, `public/workspace.js`): one prompt box
  with the answer streaming below (Send becomes Stop while it streams); a
  signal strip with surprise, confusion (full-distribution entropy when the
  runtime records it, otherwise the labelled spread of the recorded top
  candidates), repetition and perplexity with its coverage; the selected
  token's candidates and forks; the inside-the-model views (layer journey,
  attention, memory, gearbox, branches, candidates, time) behind one switcher
  and one layer picker, offering the nearest recorded token where a capture is
  sparse; saved-run open and save; and two-run comparison (first different
  token, first numerical difference by layer). Everything follows one shared
  selection (`public/selection.js`). Without WebGPU it opens the demo's
  recorded answer, and its 23 MB activity recording on request.
- **Benchmark** (`/benchmark.html`): runs one fixed workload (protocol in
  `config/protocols/device-benchmark.json`) without inspection, and compares
  benchmark results from several devices or runtimes (`public/compare-results.js`).
  Benchmark-result comparison and answer comparison are different operations.

Retired addresses redirect and keep the visitor's query: `explore.html` and
`workspace.html` to `/`, `index-preview.html` to `/` on the preview runtime,
`compare.html` to `benchmark.html#compare`.

## Pieces

- `public/shared-run.js`: the saved-run format (`doppler.observatory.run/v1`),
  its validation, forks and comparisons.
- `public/views/`, `public/viz/`: the views and the drawing helpers they share.
  `public/viz/signals.js` holds the signal math.
- `public/runtimes.js`, `config/runtimes.json`, `config/flags.json`: which
  Doppler runtime a page uses (the hosted one, or an opt-in labelled preview).
- `server/api.js`: a local-only gate for uploads, which stay off.

The pages load Doppler's runtime from `/doppler/src/` and the demo's styles and
recorded example from `/doppler/demo/`, so they always match the runtime d4da
hosts.

## Run and check

From this folder, with the repository's dependencies installed:

```sh
npm run dev               # http://127.0.0.1:8791/, runtime from this repository
npm test                  # unit tests
npm run check:workspace   # Explore in a real browser; with a GPU adapter also a live answer, Stop, a fork, a comparison (--no-gpu skips)
npm run check:site        # GPU: Benchmark and its comparison, then Explore, with one run identity
npm run check:benchmark   # GPU: the benchmark itself
npm run check:offline     # GPU: Explore answers with the network cut after one visit
npm run check:resume      # GPU: an interrupted first model download recovers
npm run check:projection  # GPU: layer projection matches recorded tokens
```

`DOPPLER_ROOT=<d4da>/sites/d4da-com/doppler npm run dev` serves the runtime d4da
hosts instead, which is the published arrangement. On macOS the checks start
Chromium with `--enable-unsafe-webgpu --use-angle=metal`; the port is fixed so a
model saved by the browser is found again.

## Publishing

d4da publishes copies with its `scripts/sync-world-observatory.mjs`, which reads
this folder. The preview runtime it copies must be a pinned, released Doppler
commit (set `DOPPLER_WORLD_SOURCE` to a clean checkout of it); the hosted
runtime at `/doppler/` is published separately.

Status: working prototype. A "Passed" benchmark is evidence about one device,
browser and workload, not a published score; nothing is ranked or uploaded.
