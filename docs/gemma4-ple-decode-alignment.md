# Gemma 4 PLE Decode Alignment

## Goal

Capture the runtime gap between Doppler's current Gemma 4 text decode path and the model's intended PLE usage.

## External contract we are aligning to

- Google describes PLE as data that can live outside accelerator memory, be cached, and be added as each layer runs.
- Hugging Face exposes `per_layer_inputs` as an explicit Gemma 4 text-model input.
- For text decode, the prepared per-layer input payload is token-driven and should be treated as cacheable runtime data.

## Current Doppler gap

- `range_backed` PLE works end to end, but the fallback decode lane still serializes on sampled-token CPU readback.
- Doppler was treating PLE primarily as a row-load and per-step orchestration problem instead of a cacheable prepared-token runtime object.

## Required direction

1. Keep the source PLE tables off the critical GPU residency path.
2. Cache prepared text-token `per_layer_inputs` payloads by token ID.
3. Use explicit hit/miss metrics to distinguish architecture wins from orchestration noise.
4. Move the common decode path toward cache hits and away from per-token host recomputation.

## This change

- Adds `session.perLayerInputs.hotCache` as an explicit manifest/runtime policy surface.
- Implements the first prepared-token GPU hot cache in `preparePerLayerInputs()`.
- Caches fully prepared per-layer input GPU buffers by token ID for decode-path reuse.

## What it does not solve yet

- It does not remove sampled-token CPU readback on cache misses.
- It does not yet provide a GPU-only hit path driven directly from sampled token buffers.
- It is an alignment step, not the final 30-50 tok/s architecture.
