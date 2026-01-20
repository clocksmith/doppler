# Browser-Only Doppler Plan

## Goal
Provide a fully browser-native Doppler pipeline that performs model conversion, storage, loading, diagnostics, and inference without any server-side components. The system must ingest GGUF or SafeTensors, convert to RDRR in-browser, persist shards in OPFS (or IndexedDB fallback), load into RAM and GPU, and run CLI-equivalent diagnostics and benchmarks via UI-configured runtime presets. This enables offline, reproducible, zero-install workflows with deterministic artifact provenance and secure local execution.

## Decisions (Confirmed)
- OPFS fallback: browser conversion must fall back to IndexedDB when OPFS is unavailable or denied.
- Hashing: browser conversion must support BLAKE3 (not SHA-256 only). Conversion should fail fast if BLAKE3 is required but unavailable.
- Remote sources without Range: supported by fully downloading source files into storage first, then converting from local storage.
- Manifest tensor map: keep `manifest.tensors` inline for now. `tensors.json` is optional and deferred unless manifest size becomes an issue.
- Tokenizer coverage: require `tokenizer.json` for browser conversion. Do not support `tokenizer.model` in the browser path.
- MoE support: browser conversion must support MoE manifests and expert mappings.
- Quantization: browser conversion should quantize non-quantized models (Q4_K path) when requested, using node defaults.
- BLAKE3 implementation: vendor a small, self-contained pure JS blake3 implementation in-repo. No WASM.
- Download-first policy: delete temporary source files on successful conversion. Keep on failure for debugging.
- Chunk sizing: add a converter-specific chunk size config (do not reuse loading distribution settings).

## Non-Goals
- Server-side conversion or a Node-only CLI workflow.
- Changing core model formats or the RDRR schema.
- Building a full production UI. This plan focuses on core pipeline and a minimal harness.

## Constraints and Assumptions
- WebGPU is required for inference. Conversion can run without WebGPU.
- Browser storage APIs must be available. OPFS is preferred; IndexedDB is the fallback.
- Remote conversion requires CORS and either Range requests or a full download fallback.
- Content-Encoding on remote sources breaks byte-offset reads. When Range is required, the source must be served with identity encoding.
- All conversion is config-driven. Presets are required to generate valid manifest inference.

## Current State Summary (Existing Components)
- Browser conversion entry: `src/browser/browser-converter.js`
- Browser SafeTensors parsing: `src/browser/safetensors-parser-browser.js`
- Browser GGUF parsing: `src/browser/gguf-parser-browser.js`
- Shared shard packing: `src/converter/shard-packer.js`
- Storage backends: `src/storage/shard-manager.js`, `src/storage/backends/opfs-store.js`, `src/storage/backends/idb-store.js`
- Loader and shard cache: `src/loader/doppler-loader.js`
- Pipeline initialization: `src/inference/pipeline/init.js`
- Tokenizer loader: `src/inference/tokenizer.js`
- Browser test harness: `src/inference/test-harness.js`

## Known Gaps (Must Fix)
- Tokenizer persistence: browser conversion does not save `tokenizer.json` to storage. Runtime expects `manifest.tokenizer.file` to resolve in OPFS/IDB.
- Hash algorithm mismatch: browser conversion uses SHA-256 in `BrowserShardIO`, but manifest can declare blake3.
- Group hashes: browser conversion builds group maps but does not compute `manifest.groups[*].hash` like the node writer.
- Memory spikes: conversion reads full tensors into RAM, then builds full shard buffers.
- Remote conversion: no Range-based tensor source; no streaming fetch path; no download fallback.
- Quota handling: conversion does not request persistence or check space.
- UI integration: no browser-native harness for CLI-equivalent runs.
- In-browser quantization: no browser path for Q4_K quantization of F16/BF16 weights.

## Architecture Overview
High-level flow:
- UI -> Source selection -> Conversion -> Storage -> Verification -> Load -> Diagnostics

Component mapping:
- Source selection: new module for file/URL/OPFS source discovery.
- Conversion: browser converter + streaming shard packer.
- Storage: shard manager + OPFS/IDB backends.
- Manifest: converter core + preset inference.
- Loader: doppler-loader and shard cache.
- Diagnostics: browser harness using pipeline/test-harness.

## Detailed End-to-End Flow
1) Bootstrap + UI
   - Provide a minimal browser UI with actions: Select source, Convert, Verify, Run diagnostics.
   - UI stores options in runtime config and passes them to the converter.

2) Conversion to RDRR (Browser)
   - Preflight
     - `initStorage()` to choose backend.
     - `requestPersistence()` and `checkSpaceAvailable()`.
     - If backend is IDB, log backend choice in progress callbacks.
     - If blake3 required but unavailable, fail fast.
   - Detect format
     - GGUF: parse header and tensor list.
     - SafeTensors: parse header, index if sharded, and tensor list.
   - Metadata load
     - Parse `config.json` and tokenizer assets if present.
     - Resolve modelId and sanitize.
     - Detect preset and modelType.
   - Prepare conversion config
     - Use `createConverterConfig()` merged with UI overrides.
     - Resolve shard size, hash algorithm, quantization layout, chunk size.
   - Pack tensors
     - For each tensor, stream data into shard writer.
     - Compute shard hash and group hash incrementally.
     - Track spans for multi-shard tensors.
   - Write outputs
     - `manifest.json` to storage.
     - `tokenizer.json` to storage if present.
     - Optional: registry entry for model listing.

3) Persistence
   - OPFS is preferred, IndexedDB fallback is accepted.
   - The storage backend is set by runtime config and availability.
   - Conversion must not require OPFS; must succeed in IDB with the same API.

4) Loading for Inference
   - `openModelStore()` with modelId.
   - Load `manifest.json` and parse.
   - Initialize tokenizer via `Tokenizer.initialize()` which reads `tokenizer.json` from storage.
   - Loader reads shards via `loadShard()` and caches in RAM.
   - Pipeline uploads tensors to GPU and runs inference.

5) Diagnostics and Reports
   - Provide a browser harness that wraps `test-harness.js` and pipeline creation.
   - Implement suites: kernels, inference, bench, debug.
   - Store structured results as JSON in storage.

## Data Structures and Interfaces

TensorSource (new)
- Purpose: unify local files, OPFS files, and remote URLs for random access.
- Interface:
  - `name: string`
  - `size: number`
  - `readRange(offset, length): Promise<ArrayBuffer>`
  - `readAll(): Promise<ArrayBuffer>`
  - `close(): Promise<void>`
  - `getAuxFiles(): Promise<{ config, tokenizer, tokenizerConfig, index, ... }>`

StreamingTensor (new)
- Descriptor passed to shard packer:
  - `name, shape, dtype, size`
  - `getChunks(): AsyncIterable<Uint8Array>`

ShardWriter (existing API from shard-manager)
- `write(chunk: Uint8Array): Promise<void>`
- `close(): Promise<void>`
- `abort(): Promise<void>`

Model Registry (new)
- Stored as `models.json` in storage root.
- Structure:
  - `models: Array<{ modelId, totalSize, quantization, hashAlgorithm, backend, createdAt }>`
- Used for UI listing and quick selection.

Diagnostics Report (new)
- Stored under `reports/<modelId>/<timestamp>.json`.
- Structure:
  - `suite, modelId, runtimePreset, deviceInfo, results, durationMs, timestamp`

## Detailed Workstreams and Actions

A) Tokenizer Persistence and Manifest Fixes
1. Save tokenizer JSON during browser conversion.
   - Update `src/browser/browser-converter.js` to call `saveTokenizer()` with raw JSON text.
2. Ensure `manifest.tokenizer.file = "tokenizer.json"`.
   - Update `src/converter/core.js` or set in browser conversion before saving manifest.
3. Harden tokenizer load errors.
   - Update `src/inference/tokenizer.js` to include modelId and missing file in error messages.

B) Hash Algorithm Alignment and Group Hashes
1. Support blake3 in browser conversion.
   - Use a JS-only blake3 implementation and expose it as `globalThis.blake3`.
   - Do not rely on WASM for conversion or verification.
2. Replace SHA-256-only hash in `BrowserShardIO`.
   - Route hashing through `createStreamingHasher()` from `src/storage/shard-manager.js`.
3. Add group hash tracking in browser conversion.
   - Extend `src/converter/shard-packer.js` to maintain per-group hashers and populate `manifest.groups[*].hash`.

C) Streaming Shard Packer
1. Add streaming pack API.
   - Extend `ShardPacker` to accept `getChunks()`.
   - Maintain spans and offsets per chunk.
2. Introduce streaming shard writer.
   - Use `createShardWriter()` from `src/storage/shard-manager.js`.
3. Limit in-memory buffering to shard-size window.
   - Chunk size controlled by converter config.

D) Remote Source Support
1. Implement HTTP Range-based tensor source.
   - Add `src/browser/tensor-source-http.js`.
   - Validate `Accept-Ranges: bytes` and `Content-Length`.
2. Implement download-first fallback.
   - Add `src/browser/tensor-source-download.js`.
   - Download full source to temporary storage when Range is missing.
   - Delete temporary files on success.
3. Update parsers to accept TensorSource.
   - `src/browser/safetensors-parser-browser.js`.
   - `src/browser/gguf-parser-browser.js`.
4. Content-Encoding guardrails.
   - If Range is required, reject sources with `Content-Encoding` or force download-first fallback.

E) Storage Fallback and Quota Handling
1. Allow conversion without OPFS.
   - Ensure `initStorage()` chooses IndexedDB when OPFS is unavailable.
2. Preflight quota.
   - Before conversion, run `requestPersistence()` and `checkSpaceAvailable()`.
3. Report storage backend choice.
   - Emit a conversion progress event with `backendType`.

F) Diagnostics Harness in Browser
1. Create a browser harness that mirrors CLI suites.
   - New module `src/inference/browser-harness.js`.
2. Persist reports.
   - New storage helper `src/storage/reports.js`.
3. Provide runtime presets via UI.
   - Use `setRuntimeConfig()` before starting a suite.

G) Optional VFS Boot
1. VFS bootstrap in browser.
   - Add `src/bootstrap.js` and service worker loader.
2. Hydration manifest.
   - Generate `config/vfs-manifest.json` and hydrate `src/` into IndexedDB.
3. Keep model shards in OPFS or IDB.
   - Do not store large shards in VFS.

## Conversion Algorithm Detail (Streaming)
- For each tensor:
  - Initialize `tensorSpans = []` and `remaining = tensor.size`.
  - Iterate `for await (chunk of getChunks())`:
    - While chunk has data:
      - `space = shardSize - currentShardSize`.
      - `take = min(space, chunk.length)`.
      - Write `chunk.slice(0, take)` to shard writer.
      - Update shard hasher and group hasher.
      - Append span `{ shardIndex, offset, size: take }`.
      - If shard full, finalize hash, record shard info, open next shard.
- At end, write tensor location with spans or single shard.

## Quantization Path (Browser)
- If source tensors are not already quantized and conversion requests Q4_K:
  - Convert BF16 or F16 to F32 in a streaming-safe buffer.
  - Apply Q4_K quantization with `quantizeToQ4KM` (and layout-aware variants).
  - Respect quantization rules from `shouldQuantize()` and config.
- Quantization must not allocate full-model buffers. Operate per tensor chunk where possible.

## Error Handling Requirements
- Missing preset: fail with explicit guidance to add a model preset.
- Missing tokenizer: fail with a clear error and modelId.
- Missing hash algorithm: fail with explicit requirement.
- Unsupported storage: fail with message listing supported backends.
- Remote source lacking Range: fall back to download-first or fail if storage unavailable.

## Testing Plan
- Unit tests
  - Streaming shard packer span correctness.
  - Per-group hash correctness matches node writer.
  - Per-shard hash correctness.
- Browser tests
  - Convert small GGUF and safetensors locally.
  - `verifyIntegrity()` passes after conversion.
  - Tokenizer loads from storage.
- Remote tests
  - Range-supported source.
  - No-Range source triggers download-first fallback.
- Diagnostics tests
  - Inference test suite produces report saved to storage.

## Milestones
1) Correctness fixes
   - Tokenizer persistence
   - Manifest tokenizer file
   - Hash alignment and blake3 support
   - Group hashes

2) Streaming conversion
   - Streaming shard packer
   - Streaming shard writer
   - Memory-bounded conversion

3) Remote source support
   - TensorSource abstraction
   - Range path
   - Download-first fallback

4) Diagnostics harness
   - Browser harness integration
   - Report persistence

5) Optional VFS boot
   - VFS bootstrap for code and config assets

6) Browser quantization
   - Q4_K path for BF16/F16 weights
   - Layout-aware quantization (row/col)
   - Streaming-safe conversions

## Task Breakdown (Parallelizable)

Agent 1 - Hashing and Storage Integration
- [x] Implement a pure JS blake3 hasher.
  - [x] Add `src/storage/blake3.js` (or `src/storage/hashers/blake3.js`).
  - [x] Expose `hash(bytes: Uint8Array): Promise<Uint8Array>` and `createHasher(): { update(bytes), finalize(): Promise<Uint8Array> }`.
  - [x] No WASM usage. Keep it self-contained.
- [x] Wire blake3 into `src/storage/shard-manager.js`.
  - [x] Replace `initBlake3` to use the JS hasher.
  - [x] Ensure `createStreamingHasher()` and `computeHash()` pick blake3 when requested.
  - [x] Update error messages to note JS-only blake3 requirement.
- [x] Add verification tests.
  - [x] Create a small test file (location TBD) with known blake3 vectors.
  - [x] Validate `computeHash()` and streaming hasher output.
- [x] Deliverables
  - [x] Pure JS blake3 module and shard-manager integration.
  - [x] Tests for blake3 hash and streaming hasher.

Agent 2 - Streaming Shard Packer + Group Hashes
- [x] Extend shard packing to support streaming tensor chunks.
  - [x] Update `src/converter/shard-packer.js`.
  - [x] Accept a streaming tensor interface: `getChunks(): AsyncIterable<Uint8Array>`.
  - [x] Maintain per-tensor spans and per-shard offsets while streaming.
- [x] Add per-group hash tracking.
  - [x] Maintain per-group hashers and update with tensor chunk bytes.
  - [x] Populate `manifest.groups[*].hash` in pack output.
- [x] Add converter chunk size config.
  - [x] Extend `src/config/schema/converter.schema.js` and `.d.ts` to include `streaming.chunkSizeBytes`.
  - [x] Propagate in `createConverterConfig()` and schema exports.
- [x] Deliverables
  - [x] Streaming-capable shard packer, group hash output, config for chunk size.

Agent 3 - Browser Converter Orchestration
- [x] Update browser conversion flow.
  - [x] File: `src/browser/browser-converter.js`.
  - [x] Use streaming shard packer API from Agent 2.
  - [x] Read chunk size from `converter.streaming.chunkSizeBytes`.
- [x] Storage preflight.
  - [x] Call `requestPersistence()` and `checkSpaceAvailable()` before conversion.
  - [x] Report backend selection (OPFS vs IDB) in progress callbacks.
- [x] Tokenizer persistence and manifest updates.
  - [x] Save `tokenizer.json` using `saveTokenizer()`.
  - [x] Ensure `manifest.tokenizer.file = "tokenizer.json"`.
- [x] Temporary download cleanup.
  - [x] If conversion used download-first fallback, delete temporary source files after success.
- [x] Deliverables
  - [x] Browser conversion uses streaming packer, stores tokenizer, preflights storage, and cleans temp sources.

Agent 4 - TensorSource Abstraction + Remote Support
- [x] Implement TensorSource types.
  - [x] `src/browser/tensor-source-file.js` for File handles.
  - [x] `src/browser/tensor-source-http.js` for Range-capable URLs.
  - [x] `src/browser/tensor-source-download.js` for download-first fallback to storage.
- [x] Add Range detection and fallback.
  - [x] `HEAD` validation for `Accept-Ranges: bytes` and `Content-Length`.
  - [x] If Range missing, fallback to download-first path.
- [x] Update parsers to use TensorSource.
  - [x] `src/browser/safetensors-parser-browser.js`.
  - [x] `src/browser/gguf-parser-browser.js`.
- [x] Deliverables
  - [x] Unified TensorSource abstraction, Range support, download-first fallback.

Agent 5 - Browser Quantization Path
- [x] Implement Q4_K quantization in browser conversion.
  - [x] Use `src/converter/quantizer.js` exports.
  - [x] Follow node defaults for weights, embeddings, and lm_head per config.
- [x] Streaming-safe quantization.
  - [x] Quantize per block (QK_K) without full tensor buffers.
  - [x] Provide helpers for BF16/F16 to F32 conversion per chunk.
- [x] Integrate with browser converter.
  - [x] Expose a function that returns an async chunk generator for quantized output.
- [x] Deliverables
  - [x] Streaming-safe quantization path with layout awareness.

Agent 6 - Diagnostics Harness + Reports
- Build browser diagnostics harness.
  - New file: `src/inference/browser-harness.js`.
  - Wrap `test-harness.js` and pipeline creation.
- Persist reports.
  - New file: `src/storage/reports.js`.
  - Save report JSON under `reports/<modelId>/<timestamp>.json`.
- Provide runtime preset selection.
  - Expose config loader to UI and call `setRuntimeConfig()`.
- Deliverables
  - Browser harness for kernels/inference/bench/debug plus report persistence.

Agent 7 - Optional VFS Boot (Optional)
- Add VFS bootstrap.
  - New `src/bootstrap.js` and service worker loader.
  - Hydrate `config/vfs-manifest.json` into IndexedDB.
- Add VFS manifest generation path.
  - Tooling or script to generate the manifest.
- Deliverables
  - VFS-only code path for offline operation.

## Integration Notes
- Agent 2 should land before Agent 3 (streaming packer dependency).
- Agent 1 should land before Agent 2 and Agent 3 (hashing dependency).
- Agent 4 can proceed in parallel, but parser updates should land before Agent 3 integrates remote sources.
- Agent 5 can proceed in parallel with Agent 2, but browser converter integration depends on both.
- Agent 6 depends on conversion and storage paths but can stub until conversion is stable.

## Acceptance Criteria
- Convert GGUF and SafeTensors in browser with no server-side tooling.
- Store RDRR shards and manifest in OPFS or IndexedDB.
- Load and run inference from stored model.
- Run at least one diagnostic suite and persist results in browser storage.
- Conversion enforces model preset availability, tokenizer presence, and hash integrity.
