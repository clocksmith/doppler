# Kernel Layout Migration Plan: One Pipeline Layout per File

## Goal

Refactor WGSL organization so each pipeline layout (bindings + entrypoint requirements + workgroup size) maps to a single file and, when practical, a single entrypoint. Use overrides only for algorithmic toggles that do not change bindings or array sizes.

Benefits:
- Clear mapping: variant id -> one file.
- Consistent pipeline caching and benchmarking.
- Easier debugging and reproducibility.

## Non-goals

- Do not merge kernels that differ in binding element types.
- Do not use overrides for array sizes or workgroup storage lengths (see legacy exceptions).
- Do not force subgroup or shader-f16 requirements on fallback variants.

## Hard Constraints (Do Not Cross)

1) Binding element types are fixed at compile time.
   - Example: array<f16> vs array<f32> outputs or embeddings cannot be toggled by override.
   - If you want one code path, use a packed u32 representation and explicit unpacking. Otherwise keep separate files.

2) Override constants cannot be used for array lengths.
   - Workgroup arrays must use fixed sizes or a MAX size.
   - If you need multiple tile sizes, compile fixed-size variants (for example 16 and 32) and select at runtime.
   - Legacy exceptions: attention_small.wgsl and attention_small_f16kv.wgsl currently use override-sized arrays.
     Treat these as migration debt to refactor into fixed-size or MAX-sized arrays.

3) Workgroup size is part of the pipeline.
   - Different @workgroup_size values are separate pipelines. Prefer separate files or separate entrypoints with clear variant ids.

4) Capability requirements are compile-time.
   - If any code path uses "enable f16" or "enable subgroups", that pipeline requires the capability even if an override disables the path. Keep subgroup/f16 fallbacks separate when needed.

## Current Inventory (Keep Accurate)

- Currently 48 WGSL files under src/gpu/kernels (expect drift).
- Includes f16/conversion kernels such as:
  - bias_add_f16.wgsl
  - cast_f16_to_f32.wgsl
  - residual_f16.wgsl
  - rmsnorm_f16.wgsl
  - silu_f16.wgsl
  - gather_f16.wgsl
- check_stop.wgsl no longer exists.
- matmul_f16w_f32a_naive.wgsl is still referenced in the registry and utils. Do not delete until references are removed.

Suggested inventory check:
  rg --files -g "*.wgsl" src/gpu/kernels | wc -l

## Strategy Summary

1) Group by pipeline layout first (bindings + workgroup size + capability requirements).
2) Use overrides only for algorithmic toggles inside a layout.
3) Keep baseline kernels (f32, non-subgroup, non-fused) always available for debugging.
4) Standardize variant ids and log them for benchmarks and traces.
5) Use codegen or templating for shared logic instead of forcing unsafe merges.

## Layout Boundaries and Suggested Consolidation

### Attention

- Separate by KV dtype:
  - prefill f32 and prefill f16kv should remain separate files unless KV is packed into u32.
  - streaming f32 vs f16kv remain separate.
  - decode f32 vs decode f16kv remain separate.
- Subgroup variants should be separate pipelines (capability requirement).
- Small tile vs large tile can be an override if array sizes are fixed to MAX.

Example target layout split:
- attention_prefill_f32.wgsl (main, small tiles via override)
- attention_prefill_f16kv.wgsl (main, small tiles via override)
- attention_streaming_f32.wgsl
- attention_streaming_f16kv.wgsl
- attention_decode_f32.wgsl (optimized path)
- attention_decode_subgroup.wgsl (subgroup-specific)
- attention_decode_f16kv.wgsl (chunked or other f16kv decode)

### Matmul

- Keep separate files for binding types:
  - matmul_f32.wgsl
  - matmul_f16.wgsl
  - matmul_f16w_f32a.wgsl
- GEMV and subgroup GEMV should be separate pipelines if they require subgroups or different workgroup sizes.
- Multicol/batched variants can use overrides only if workgroup size and storage layout match.

### Dequant

- Keep f32 output and f16 output as separate layouts (bindings differ).
- Q6K, Q8_0, MXFP4 remain separate files.
- If a unified kernel is desired, use packed u32 inputs and outputs with explicit unpacking.

### Activations and Norms

- Keep f32 and f16 variants separate (bindings differ).
- Within a dtype-specific file, use overrides for:
  - gate vs bias vs layout modes
  - vec4 paths (if workgroup size is fixed and arrays are MAX-sized)

### Residual, Gather, Cast

- Keep f16 variants separate unless using packed u32 representations.
- If vec4 uses a different workgroup size, treat it as a separate pipeline and track it as a distinct variant id.

### Fused Kernels

- Fused FFN, fused matmul q4, fused matmul rmsnorm:
  - keep f16 vs f32 pipeline layouts separate if shader-f16 is required.
  - if subgroup requirements differ, keep separate pipelines.
  - consolidate only the variants that truly share bindings and capabilities.

### Sampling, TopK, MoE

- Multi-phase kernels with different bind group layouts should remain separate files.
- If phases share layout, they can be a single file with one entrypoint and a uniform "phase" value.

## Registry and Runtime Plumbing

- Update src/gpu/kernels/utils.ts and src/config/kernels/registry.json in lockstep with any file/variant changes.
- Only remove matmul_f16w_f32a_naive.wgsl after all registry references are removed and a fallback is confirmed.

## Enforcement (Avoid Drift)

- Add a registry validation step that can run locally and is enforced in CI (fails if WGSL references are missing).
- Add a lint check that flags any workgroup array sizes derived from overrides, except for whitelisted legacy files.
- Require variant ids to log their pipeline layout (file + entrypoint + requirements) in debug traces.

## Testing and Benchmarking

- Kernel tests must cover each layout group and at least one baseline variant.
- Debug presets should be able to pin "safe" variants (f32, non-fused, non-subgroup).
- Benchmarks should log:
  - operation
  - variant id
  - pipeline constants (override values)

## Migration Order

1) Inventory and cleanup:
   - Update this doc and registry counts.
   - Remove stale references (check_stop, matmul_f16w_f32a_naive once ready).
2) Low-risk consolidations within the same layout:
   - scale, residual, gather (within dtype boundary).
3) Matmul and activation families.
4) Attention families.
5) Sampling/topk/MoE.

## Summary

Use "one pipeline layout per file" as the organizing rule. This preserves correctness and debuggability while still allowing consolidation where it is safe and legal. Avoid unsafe override usage for bindings or array sizes, and keep baseline kernels available for debugging and validation.
