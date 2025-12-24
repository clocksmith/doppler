# WebGPU Inference Test Results

Index of DOPPLER validation sessions across different hardware and browsers.

This file is a human-readable log. Store machine-readable benchmark outputs as JSON using
`docs/spec/BENCHMARK_HARNESS.md` so results can be compared automatically.

See also:
- `docs/spec/BENCHMARK_HARNESS.md` for benchmark methodology and JSON result schema.
- `docs/spec/KERNEL_TESTING.md` for WGSL kernel and pipeline segment testing.
- `doppler/kernel-tests/TODO.md` for the implemented kernel test harness.
- `doppler/kernel-tests/BENCHMARKS.md` for kernel microbenchmark baselines.

## Result Artifacts (Recommended)

| Artifact | Purpose | Suggested Path |
|----------|---------|----------------|
| Pipeline benchmark JSON | TTFT, tok/s, submits, readback, memory | `doppler/reploid/doppler/tests/results/` |
| Kernel correctness JSON/HTML | per-kernel correctness | `doppler/kernel-tests/results/` |
| Kernel benchmark JSON/HTML | per-kernel timings | `doppler/kernel-tests/results/` |

If a run does not have a JSON artifact yet, record the session here and file it as follow-up work.

## Test Sessions

### Session 2025-12-14: AMD Strix Halo + Gemma 3 1B

**Tester**: Linux AMD machine
**GPU**: AMD Strix Halo integrated GPU (Radeon 8050S/8060S Graphics)
**Browser**: Google Chrome 142.0.7444.162
**OS**: Linux 6.17.0-7-generic
**Model**: Gemma 3 1B IT (Q4_K_M quantization)

#### Status: IN PROGRESS

**Steps completed**:
1. ✓ Located Gemma 3 1B model in HuggingFace cache
2. ✓ Converted to RDRR format (Q4_K_M quantization) - 965MB, 15 shards
3. ✓ Model served locally for browser load
4. ❌ **BLOCKED**: Headless browser cannot access WebGPU (no GPU in headless environment)

**Test Limitation**: The Linux environment runs headless without X server or GPU access. WebGPU requires either:
- A headed browser with GPU drivers (X11/Wayland + working GPU)
- OR Manual testing in a desktop environment

**Model is ready** - just needs a desktop browser to test.

**Model path**:
- Source: `/home/clocksmith/.cache/huggingface/hub/models--google--gemma-3-1b-it/snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752`
- RDRR output: `doppler/reploid/doppler/models/gemma-3-1b-q4/`

**Expected model size**: ~1.2GB (340 tensors, 26 layers, 1152 hidden size)

---

### Session 2025-12-14: Mac M3 + GPT-OSS 20B (parallel session)

**Tester**: MacBook with M3
**GPU**: Apple M3 (unified memory)
**Model**: GPT-OSS 20B MoE (Q4_K_M, 32 experts, topK=4)
**Status**: PARTIAL - Router fixed, expert loading in progress

#### Bug Fix 1: MoE Gather Kernel (FIXED)
- Root cause: WebGPU `layout: 'auto'` only includes bindings used by each entry point
- `count_and_map` used 4/6 bindings, `gather_tokens` used 6/6
- Bind group creation with mismatched layout caused silent failure
- Fix: Created explicit bind group layout with all 6 bindings
- See: `docs/MOE-EXPLICIT-LAYOUT-POSTMORTEM.md`

#### Bug Fix 2: Router Weight Quantization (FIXED)
- Root cause: Router weights quantized to Q4_K_M despite HuggingFace config `modules_to_not_convert`
- Symptom: Router logits extreme (56 vs -39), softmax collapses to [1.0, 0.0, 0.0, 0.0]
- Fix: Updated `shouldQuantize()` in quantizer.ts to check:
  1. Hard-coded `router` and `gate.weight` patterns
  2. HuggingFace `modules_to_not_convert` config from quantization_config
- Reconverted model: Router weights now BF16 (184KB vs 52KB Q4_K_M)
- **Result**: Router now produces distributed weights!
  ```
  [DEBUG MoE L0] Router logits (first 8 experts): -0.14, 0.59, 0.11, 0.88, -0.98, -1.73, 2.58, -0.54
  [DEBUG MoE L0] Expert weights: [0.5896, 0.1668, 0.1359, 0.1078, ...]
  ```
  vs before: `[1.0, 0.0, 0.0, 0.0]`

#### Current Status: Expert Loading
- Router works correctly (distributed weights)
- Expert tensor loading attempted but using wrong naming convention
- GPT-OSS uses packed MXFP4 experts (`model.layers.X.mlp.experts.gate_up_proj_blocks`)
- Loader fallback exists but may need debugging

**Files modified**:
- `gpu/kernel-selector.ts` - Added explicit bind group layout for MoE
- `gpu/kernels/moe_gather.wgsl` - Cleaned up, added layout note
- `tools/quantizer.ts` - Added router check in `shouldQuantize()`
- `tools/convert-cli.ts` - Pass `modules_to_not_convert` to shouldQuantize

---

## Hardware Configurations Tested

| Date | GPU | VRAM | Browser | OS | Model | Status | Notes |
|------|-----|------|---------|----|----|-------|-------|
| 2025-12 | Apple M3 | Unified | Safari/Chrome | macOS | Gemma 3 1B | ✓ WORKING | Reference implementation |
| 2025-12-14 | AMD Strix Halo | Integrated | Chrome 142 | Linux | Gemma 3 1B | ⏳ TESTING | In progress |
| 2025-12-14 | Apple M3 | Unified | Chrome | macOS | GPT-OSS 20B | ⚠️ PARTIAL | MoE pipeline works, output quality poor |

## Test Protocol

### Standard Result Capture (Recommended)

For each performance session, record:

- Model: `modelId`, quantization, shard count, tensor count
- Environment: OS, browser version, GPU adapter info, WebGPU feature flags
- Workloads: prompt names and token counts
- Metrics: TTFT, prefill tok/s, decode tok/s, peak VRAM estimate, GPU submit counts

Preferred output:

- A JSON file per run matching `docs/spec/BENCHMARK_HARNESS.md`.
- A short narrative summary in this document for context and troubleshooting.

To avoid instruction drift, prefer linking to the canonical runner docs:

- Kernel tests and microbenchmarks: `doppler/kernel-tests/TODO.md` and `doppler/kernel-tests/BENCHMARKS.md`
- DOPPLER end-to-end tests: `doppler/reploid/doppler/tests/` and `doppler/reploid/doppler/tests/helpers/test-config.ts`

## Known Issues by Platform

### AMD GPUs
- **Driver requirements**: Mesa 23.0+ (Linux) or Adrenalin 23.0+ (Windows)
- **WebGPU status**: Generally good support in recent drivers
- **Strix Halo**: New integrated RDNA architecture, untested

### Apple Silicon
- **Unified memory advantage**: No PCIe overhead, can load larger models
- **Safari vs Chrome**: Both support WebGPU, Safari may have better integration
- **F16 support**: Excellent on M-series chips

### NVIDIA
- **Status**: Untested in DOPPLER
- **Expected**: Should work well with recent drivers
- **Driver**: 525+ required for WebGPU

## Debugging Common Issues

### Model loads but produces garbage tokens
**Symptom**: Output like `<unused16>`, random Unicode, or non-English text for English prompts

**Causes**:
1. Quantization format mismatch (Q4_K encoding issue)
2. BF16 conversion error
3. Gemma 3 norm offset not applied
4. GPU dequantization kernel bug

**Debug**:
- Check logs for "Prefill logits" top-5 distribution
- Look for negative hidden state values (should be present)
- Compare against known-working Mac M3 output

### WebGPU not available
**Symptom**: `navigator.gpu` is undefined

**Solutions**:
- Update browser (Chrome 113+, Edge 113+)
- Enable in Firefox: `about:config` → `dom.webgpu.enabled`
- Check GPU drivers are up to date

### Out of memory errors
**Symptom**: Buffer allocation fails, model won't load

**Solutions**:
- Try smaller model (Gemma 3 1B needs ~1.2GB)
- Close other GPU-intensive apps
- Check browser console for specific buffer size limits

## Contributing Results

After testing:
1. Update this file with your results
2. Update HARDWARE_COMPATIBILITY.md matrix
3. Commit and push changes
4. Share any issues or findings

---

*Last updated: December 2025*

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes (4-bit/9-bit), CLI flags (`--force-fused-q4k`, `--kernel-hints`), and the OPFS purge helper.

