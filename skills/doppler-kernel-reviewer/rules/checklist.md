# Kernel Review Checklist

## WGSL (`*.wgsl`)

- [ ] **Data Alignment**: Structs used in uniforms/storage buffers MUST be 16-byte aligned.
    - Check for `_pad` fields if necessary (especially `vec3` or odd `u32` counts).
    - `vec3` is NOT allowed in buffers (use `vec4` or `array<f32; 3>` with padding).
- [ ] **Naming**:
    - Constants: `UPPER_SNAKE_CASE` (e.g., `WORKGROUP_SIZE`).
    - Functions/Vars: `snake_case`.
    - Structs: `PascalCase`.
- [ ] **Overrides vs Uniforms**:
    - Use `override` for values known at pipeline creation (dims, workgroup size, flags that change topology).
    - Use `uniform` ONLY for values that change per-dispatch (seq values, dynamic offsets).
- [ ] **Bounds Checks**: All global memory access must be bounds-checked or proven safe.
    - `if (idx >= u.seq_len * HIDDEN_SIZE) return;`
- [ ] **Workgroup Size**: Must be an `override` constant, NOT hardcoded (unless truly invariant).
    - `@workgroup_size(WORKGROUP_SIZE, 1, 1)`

## JavaScript (`*.js`)

- [ ] **No Types**: JS files must NOT contain JSDoc type annotations or `/** ... */` blocks describing types.
    - Note: `//` comments are fine.
- [ ] **Config as Code**:
    - No inline `if (isDecode) ...` for kernel variant strings. Use `src/rules/*.json`.
    - No hardcoded magic numbers for block sizes. Import from `quantization-constants.js`.
- [ ] **Pipeline Caching**:
    - Must cache pipelines using a key derived from all constants.
    - Do NOT call `createComputePipeline` on every frame.
- [ ] **Uniform Layout**:
    - JS `UNIFORM_LAYOUT` object must match WGSL struct layout exactly.
    - Explicit `_pad` fields must be written as 0.

## TypeScript Definitions (`*.d.ts`)

- [ ] **Completeness**: Every JS function exported must have a TS signature.
- [ ] **Interfaces**: define `KernelUniforms` and `KernelConstants` interfaces.
