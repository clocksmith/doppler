# DOPPLER Setup (Browser WebGPU path)

DOPPLER is the local WebGPU inference engine for medium/large models. It uses the `.rpl` format (manifest + shard_*.bin) and supports three ways to load models.

For Ollama and other local server options, see [LOCAL_MODELS.md](./LOCAL_MODELS.md).

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

1) Serve CLI (all browsers)
- From the DOPPLER repo: `node tools/serve-cli.ts /path/to/model.gguf`
- Converts GGUF → .rpl (temp dir) and serves with CORS (default http://localhost:8765).
- In the boot UI, pick provider "DOPPLER" and paste the Model URL. DOPPLER downloads into OPFS and caches it.

2) Import GGUF in-browser (Chrome/Edge)
- In the model form, choose "DOPPLER" → click "Import GGUF from Disk".
- Streams GGUF → .rpl directly into OPFS with progress UI. No CLI/server needed.

3) Native Bridge (extension + host, with browse modal)
- Load the Chrome extension from `bridge/extension/` in the DOPPLER repo (dev mode).
- Run `bridge/native/install.sh <extension-id>` from the DOPPLER repo to install the native host.
- In the model form, choose "DOPPLER"; a Local Path field and browse button appear. Browse to a local `.rpl` directory; shards stream from disk with hash verification.

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
| "Network error" | Check CORS settings on serve-cli |
| "Hash mismatch" | Re-download model, check disk integrity |

### Extension/Bridge Issues

1. **Extension install:** Load from `bridge/extension/` in the DOPPLER repo (dev mode)
2. **Native host:** Run `bridge/native/install.sh <extension-id>` in the DOPPLER repo
3. **Connectivity test:** Use `bridge/native/test-host.ts` in the DOPPLER repo (PING/READ/LIST)
4. **Check logs:** DevTools console for importer/bridge logs

### Storage Management

- Models live in OPFS/IndexedDB
- Clear via: DevTools → Application → Storage → Clear site data
- Check usage: `navigator.storage.estimate()`

---

## Related Documentation

- [LOCAL_MODELS.md](./LOCAL_MODELS.md) - Ollama and server-based local models
- [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) - General troubleshooting
- [SECURITY.md](./SECURITY.md) - Security considerations for local execution

---

*Last updated: December 2025*
