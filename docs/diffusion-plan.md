# Diffusion Pipeline Plan (WebGPU, JS + WGSL)

Detailed plan for adding a diffusion image pipeline to DOPPLER while keeping the
architecture modular, manifest-first, and config-driven.

---

## Goals

- Add a diffusion (image) pipeline that reuses DOPPLER core systems (storage,
  GPU, buffer pool, config merge, rules, logging).
- Keep model-type isolation: transformer code remains unchanged aside from
  registry hooks and shared utilities.
- Make all tunables schema-backed (no runtime literals).
- Keep manifest-first behavior (no runtime inference from tensor names).
- Allow SD 3.5 Medium / SDXL through a swapper + quantization path once the
  foundation is validated.

## Non-goals

- No UI polish beyond minimal controls to exercise the pipeline.
- No multi-model routing or speculative decoding in phase 1.
- No P2P transport changes (RDRR already supports shard hashing).

---

## Architecture Overview

### 1) Pipeline Registry

Add a registry that maps modelType -> pipeline factory:

- src/inference/pipeline/registry.js
  - registerPipeline(modelType, factory)
  - createPipeline(manifest, contexts) routes by manifest.modelType

Transformer stays registered as "transformer". Diffusion becomes "diffusion".

### 2) Manifest Component Groups

Use RDRR component grouping for diffusion submodules:

- text_encoder
- unet
- vae
- optional: clip_projection, safety_checker

The manifest defines where weights live and what the pipeline is. The loader
stays generic and only exposes component weight maps.

### 3) Config Schema (new)

Add a dedicated diffusion schema slice under runtime + manifest inference:

- src/config/schema/diffusion.schema.js
- runtime.inference.diffusion (runtime tunables)
- manifest.inference.diffusion (model constraints)

Example fields:

```
inference: {
  diffusion: {
    scheduler: {
      type: "ddim" | "euler" | "euler_a" | "dpmpp_2m",
      numSteps: 20,
      guidanceScale: 7.5,
      eta: 0.0
    },
    latent: {
      width: 512,
      height: 512,
      channels: 4,
      dtype: "f16"
    },
    textEncoder: {
      maxLength: 77
    },
    decode: {
      outputDtype: "f16",
      tiling: {
        enabled: false,
        tileSize: 64,
        overlap: 8
      }
    },
    swapper: {
      enabled: false,
      strategy: "sequential",
      evictTextEncoder: true,
      evictUnet: true
    },
    quantization: {
      weightDtype: "none",
      dequantize: "shader"
    }
  }
}
```

### 4) Rules Maps

Any variant selection must be rules-driven:

- src/rules/diffusion/*.rules.json
  - kernel variants (fused vs unfused)
  - dtype paths (f16 vs f32)

Selection uses selectRuleValue() only.

---

## Diffusion Pipeline (Phase 1)

### High-level steps

1. Tokenize prompt
2. Text encoder -> conditioning embeddings
3. Scheduler setup (timesteps)
4. UNet loop (denoise in latent space)
5. VAE decode -> image output

### Core data flow

```
prompt -> text_encoder -> embeddings
noise -> unet(t, embeddings) -> denoise latents
latents -> vae.decode -> image
```

### Pipeline modules

```
src/inference/diffusion/
  pipeline.js
  init.js
  scheduler.js
  unet.js
  vae.js
  text-encoder.js
  types.d.ts
```

---

## Loader + Storage

No OPFS or downloader changes required. The loader already supports:

- manifest + shard streaming
- grouped components

Add diffusion component metadata to the manifest and keep weights keyed by
component in the loader output.

---

## Memory Strategy (Foundation + Unlock)

### Sequential Component Offloading (Swapper)

Time-multiplex GPU memory across components:

1) Text encoder -> embeddings (store on CPU)
2) UNet/MMDiT denoise loop (hot path)
3) VAE decode (tiled)

Only the largest component needs to fit in VRAM at once.

### Tiled VAE Decode

Decode latents in tiles (with overlap) to reduce peak memory. This is required
for 1024x1024+ targets on 4 GB budgets.

### Quantization (W8A16)

Weights stored as int8, dequantized to f16 in shader just before matmul.
This is the critical path for SD 3.5 Medium and SDXL on 4 GB devices.

---

## Kernels (Phase 1)

Required kernels:

- conv2d (standard + depthwise)
- groupnorm
- layernorm (if required by text encoder)
- attention (cross-attention in UNet)
- resblock building blocks (conv + norm + activation)
- upsample / downsample
- vae decode (conv heavy)

Phase 2 additions:

- int8 dequant + matmul kernels (W8A16)
- tiled VAE decode path

Kernel tests under tests/kernels/:

- conv2d
- groupnorm
- upsample
- unet-smoke
- vae-smoke

---

## GPU Memory + Performance

### Buffer strategy

- Use buffer pool + bucketing.
- Pre-allocate latent buffers based on width x height x channels.
- Avoid CPU readbacks until final image.

### Perf metrics

Extend stats for diffusion:

- per-step time
- total steps
- per-stage breakdown (text encoder / unet / vae)

---

## Demo UI (minimal)

Add diffusion mode controls:

- prompt
- steps
- guidance scale
- seed
- width / height
- render output image canvas

---

## Milestones

### Phase 0: Scaffolding (1-2 days)
- Pipeline registry
- Diffusion schema + defaults
- Manifest inference wiring
- Basic demo UI controls (no output)

### Phase 1: Foundation (1-2 weeks)
- Model: SD 1.5 (DreamShaper 8 LCM) for validation.
- Text encoder (CLIP) path.
- UNet loop with basic scheduler (DDIM or LCM).
- Tiled VAE decode (enabled) for early memory discipline.
- 512x512 output working.

### Phase 2: Unlock (2-4 weeks)
- Model: SD 3.5 Medium / SDXL target.
- W8A16 quantization kernels + shader dequant.
- Component swapper (sequential offload) to keep VRAM bounded.
- Kernel fusion options + profiling.

---

## Risks and mitigations

- Model size too large: start with small SD variants or low-res outputs.
- Kernel coverage gaps: reuse existing attention + norm kernels where possible.
- Performance: focus on kernel fusion in phase 2.
- Swapper complexity: keep it opt-in and enabled only for large models.

---

## Tests and validation

- Kernel-level tests for each new op
- End-to-end smoke pipeline test with a tiny model
- Memory usage regression checks (VRAM bound)

---

## Decisions (Resolved)

- First target (foundation):
  - Decision: Stable Diffusion 1.5 class (latent 512x512) as v1.
  - Why: predictable weight layout + huge ecosystem for validation; lowest
    hardware requirements; avoids distillation-specific complexity so early
    debugging isolates pipeline issues. SD 3.5 Medium / SDXL comes after the
    swapper + quantization path is proven, making it an optimization pass
    instead of a structural risk.
- Baseline memory budget:
  - Decision: 4 GB WebGPU buffer limit baseline, assume ~8 GB system RAM.
  - Why: aligns with Apple unified memory and mainstream laptop/mobile limits;
    forces disciplined buffer reuse from day one. Treat >4 GB as best-effort.
    This constraint also justifies early tiled VAE decode to stay within budget.
- Safety checker:
  - Decision: include a manifest hook, but keep it disabled by default in v1.
  - Why: safety checker adds weight/latency that obscures perf tuning during
    the first optimization phase. Post-process safety is acceptable for v1.
