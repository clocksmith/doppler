# DOPPLER Style Guides

Coding conventions for the DOPPLER codebase.

## Guides

| Guide | Scope |
|-------|-------|
| [General Style Guide](./general-style-guide.md) | Architecture, language policy, file organization, naming |
| [JavaScript Style Guide](./javascript-style-guide.md) | Kernel wrappers, config-as-code, declaration-file workflow, rule maps |
| [WGSL Style Guide](./wgsl-style-guide.md) | Shader structure, constants vs uniforms, bindings |
| [Benchmark Style Guide](./benchmark-style-guide.md) | Benchmark harnesses, output schema, baselines |
| [Config Style Guide](./config-style-guide.md) | Config ownership, merge order, runtime boundaries |
| [Harness Interface Style Guide](./harness-style-guide.md) | Browser + CLI execution rules and config-only controls |
| [Command Interface Design Guide](./command-interface-design-guide.md) | Shared command schema, intent mapping, parity contract |

Shared rule ownership:
- Cross-language naming/logging/testing policy is canonical in [General Style Guide](./general-style-guide.md).
- Benchmark claim policy is canonical in [../benchmark-methodology.md](../benchmark-methodology.md).

## Quick Reference

### Language Policy

Doppler uses **JavaScript** with **.d.ts** declaration files for all source modules (`src/` and `demo/`). Tests and tools may omit `.d.ts` unless they export public types. See [Language Policy](./general-style-guide.md#language-policy-javascript-declaration-files) for rationale.

### Config Flow

```
manifest.json → ModelConfig → PipelineSpec → KernelSpec → GPU Dispatch
```

Interpretation:
1. JSON assets define policy and selection.
2. JS resolves and enforces config, then executes dispatch.
3. WGSL performs deterministic numeric execution only.

### Benchmark Flow (Config-First)

```
shared benchmark contract + engine overlay → runtime payload per engine → run + compare
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
| `kebab-case.d.ts` | `model-config.d.ts` | Type declarations (required for modules) |
| `snake_case.wgsl` | `matmul_f16.wgsl` | WGSL shaders |
