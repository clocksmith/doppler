# DOPPLER Style Guides

Coding conventions for the DOPPLER codebase.

## Guides

| Guide | Scope |
|-------|-------|
| [General Style Guide](./GENERAL_STYLE_GUIDE.md) | Architecture, language policy, file organization, naming |
| [JavaScript Style Guide](./JAVASCRIPT_STYLE_GUIDE.md) | Kernel wrappers, config-as-code, JSDoc, rule maps |
| [WGSL Style Guide](./WGSL_STYLE_GUIDE.md) | Shader structure, constants vs uniforms, bindings |
| [Benchmark Style Guide](./BENCHMARK_STYLE_GUIDE.md) | Benchmark harnesses, output schema, baselines |
| [Config Style Guide](./CONFIG_STYLE_GUIDE.md) | Config ownership, merge order, runtime boundaries |

## Quick Reference

### Language Policy

Doppler uses **JavaScript** with optional **.d.ts** declaration files. See [Language Policy](./GENERAL_STYLE_GUIDE.md#language-policy-javascript--declaration-files) for rationale.

### Config Flow

```
manifest.json → ModelConfig → PipelineSpec → KernelSpec → GPU Dispatch
```

### Key Principles

1. **Config as Code** - Maps over if/else
2. **Layered Config** - Each layer transforms the previous
3. **Pure Functions** - Config transformations are pure
4. **Constants vs Uniforms** - Model dims are constants, runtime values are uniforms
5. **Tests as Type System** - Comprehensive tests catch type errors pre-production

### File Naming

| Pattern | Example | Use |
|---------|---------|-----|
| `kebab-case.js` | `model-config.js` | JavaScript source |
| `kebab-case.d.ts` | `model-config.d.ts` | Type declarations (optional) |
| `snake_case.wgsl` | `matmul_f16.wgsl` | WGSL shaders |
| `UPPER_CASE.md` | `GENERAL_STYLE_GUIDE.md` | Documentation |
