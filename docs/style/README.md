# DOPPLER Style Guides

Coding conventions for the DOPPLER codebase.

## Guides

| Guide | Scope |
|-------|-------|
| [General Style Guide](./GENERAL_STYLE_GUIDE.md) | Architecture, file organization, naming, error handling |
| [TypeScript Style Guide](./TYPESCRIPT_STYLE_GUIDE.md) | Kernel wrappers, config-as-code, rule maps |
| [WGSL Style Guide](./WGSL_STYLE_GUIDE.md) | Shader structure, constants vs uniforms, bindings |

## Quick Reference

### Config Flow

```
manifest.json → ModelConfig → PipelineSpec → KernelSpec → GPU Dispatch
```

### Key Principles

1. **Config as Code** - Maps over if/else
2. **Layered Config** - Each layer transforms the previous
3. **Pure Functions** - Config transformations are pure
4. **Constants vs Uniforms** - Model dims are constants, runtime values are uniforms

### File Naming

| Pattern | Example | Use |
|---------|---------|-----|
| `kebab-case.ts` | `model-config.ts` | TypeScript |
| `snake_case.wgsl` | `matmul_f16.wgsl` | WGSL shaders |
| `UPPER_CASE.md` | `GENERAL_STYLE_GUIDE.md` | Documentation |
