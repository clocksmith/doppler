# DOPPLER Setup (Browser WebGPU path)

DOPPLER is the local WebGPU inference engine for medium/large models. It uses the `.rpl` format (manifest + shard_*.bin) and supports three ways to load models.

For Ollama and other local server options, see [local-models.md](./local-models.md).

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | WebGPU-capable | Discrete GPU with 8GB+ VRAM |
| RAM | 8GB | 16GB+ |
| Storage | 4GB free | 20GB+ for multiple models |
| Browser | Chrome 113+ | Chrome/Edge with WebGPU enabled |

**Notes:**
- Unified memory (Apple Silicon, AMD Strix) is ideal for dense models
- Discrete GPUs benefit from MoE architectures or smaller shards
- Check `chrome://gpu` for WebGPU status

---

## Setup Methods

1) Serve local models (all browsers)
- Place an RDRR model under `/models/<model-id>/` with `manifest.json` and shard files.
- Serve the repo root with `python3 -m http.server 8080`.
- In the diagnostics UI (`/demo/`), pick the model from the local list.

2) Import GGUF or safetensors in-browser (Chrome/Edge)
- Open the demo UI (`/demo/`) and use the import flow.
- Streams GGUF/safetensors → RDRR directly into OPFS with progress UI. No CLI/server needed.

3) Convert via remote URLs
- Use the demo UI URL import to stream remote safetensors shards (with HTTP range + download fallback).
- Converted output is cached in OPFS.

Notes:
- Manifests include tensor locations and shard hashes; `hashAlgorithm` may be `sha256` or `blake3`.
- Unified memory (Apple/Strix) is ideal for dense models; discrete GPUs benefit from MoE or smaller shards.

## Troubleshooting

### WebGPU Not Available

1. Check browser version (Chrome 113+ required)
2. Enable WebGPU flag: `chrome://flags/#enable-unsafe-webgpu`
3. Check GPU status: `chrome://gpu`
4. Update GPU drivers
5. Try Firefox Nightly with WebGPU flag

### Model Loading Failed

| Symptom | Solution |
|---------|----------|
| "Out of memory" | Try smaller model or close other tabs |
| "Shader compilation failed" | Update GPU drivers, check WebGPU status |
| "Network error" | Check CORS settings on the static server or remote host |
| "Hash mismatch" | Re-download model, check disk integrity |

### Storage Management

- Models live in OPFS/IndexedDB
- Clear via: DevTools → Application → Storage → Clear site data
- Check usage: `navigator.storage.estimate()`

---

## Related Documentation

- [local-models.md](./local-models.md) - Ollama and server-based local models
- [troubleshooting.md](./troubleshooting.md) - General troubleshooting
- [security.md](./security.md) - Security considerations for local execution

---

*Last updated: December 2025*
