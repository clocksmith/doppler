# DOPPLER Setup

This page covers environment setup and troubleshooting.

For first-run workflow, use [getting-started.md](getting-started.md).
For sizing and expected performance, use [performance-sizing.md](performance-sizing.md).

## Prerequisites

- Node.js 20+
- repo dependencies installed
- WebGPU-capable runtime

## Browser requirements

Supported:
- Chrome/Edge (recommended)
- Safari with WebGPU support
- Firefox Nightly (experimental)

Check WebGPU:
- browser diagnostics page (`chrome://gpu` on Chromium)
- or runtime check in console

```javascript
const adapter = await navigator.gpu.requestAdapter();
console.log(Boolean(adapter));
```

## Setup methods

### CLI-first

Use [getting-started.md](getting-started.md) to run `verify`, optional `convert`, then `bench`.

### Browser harness

```bash
python3 -m http.server 8080
```

Open:
- `http://localhost:8080/tests/harness.html`
- `http://localhost:8080/demo/`

## Troubleshooting

### WebGPU unavailable

- verify browser version and flags
- update GPU drivers
- check that adapter/device creation succeeds

### Model loading failed

- confirm `modelId` and `modelUrl`
- verify manifest and shard availability
- clear stale cache when needed

### Out of memory

- choose a smaller model/quantization
- lower workload sizes
- use sizing guidance in [performance-sizing.md](performance-sizing.md)

### Slow performance

- validate warm/cold conditions
- ensure benchmark profile consistency
- compare against canonical methodology in [benchmark-methodology.md](benchmark-methodology.md)

## Related

- [getting-started.md](getting-started.md)
- [cli-quickstart.md](cli-quickstart.md)
- [operations.md](operations.md)
