# Doppler Style Guides

Coding conventions for the DOPPLER codebase.

## Read-First Matrix

Pick by what your change touches. Each row lists the guides to read before the first non-trivial edit in that area.

| Change area | Mandatory reads | Also read when |
|---|---|---|
| Any non-trivial edit | General + JavaScript (the Invariants section at the top of each is the quick-ref) | — |
| Runtime tunables or config schema | [Config Style Guide](./config-style-guide.md) | changing merge order or category rules |
| New kernel or WGSL changes | [General](./general-style-guide.md), [JavaScript](./javascript-style-guide.md), [WGSL](./wgsl-style-guide.md) | rule selection / dtype policy: [Config](./config-style-guide.md) |
| Bench / debug / verify command behavior | [General](./general-style-guide.md), [JavaScript](./javascript-style-guide.md), [Config](./config-style-guide.md), [Command Interface](./command-interface-design-guide.md), [Harness](./harness-style-guide.md) | benchmark methodology: [Benchmark](./benchmark-style-guide.md) |
| Benchmark harness or claim | [General](./general-style-guide.md), [JavaScript](./javascript-style-guide.md), [Config](./config-style-guide.md), [Harness](./harness-style-guide.md), [Benchmark](./benchmark-style-guide.md) | command surface: [Command Interface](./command-interface-design-guide.md) |
| Converter / manifest field | [General](./general-style-guide.md), [JavaScript](./javascript-style-guide.md), [Config](./config-style-guide.md), [Command Interface](./command-interface-design-guide.md) | — |
| Docs / logs / status output | [Emoji Policy](./emoji.md) | anywhere Unicode symbols replace prose or status glyphs |

## Guides

| Guide | Scope |
|-------|-------|
| [General Style Guide](./general-style-guide.md) | Invariants quick-ref + architecture, language policy, file organization, naming |
| [JavaScript Style Guide](./javascript-style-guide.md) | Invariants quick-ref + kernel wrappers, config-as-code, declaration-file workflow, rule maps |
| [WGSL Style Guide](./wgsl-style-guide.md) | Shader structure, constants vs uniforms, bindings |
| [Config Style Guide](./config-style-guide.md) | Config ownership, merge order, runtime boundaries |
| [Command Interface Design Guide](./command-interface-design-guide.md) | Shared command schema, intent mapping, parity contract |
| [Harness Interface Style Guide](./harness-style-guide.md) | Browser + CLI execution rules and config-only controls |
| [Benchmark Style Guide](./benchmark-style-guide.md) | Benchmark harnesses, output schema, baselines |
| [Emoji Policy](./emoji.md) | Strict no-emoji rule and the approved Unicode symbol set for status/log/doc output |

Shared rule ownership:
- Cross-language naming/logging/testing policy is canonical in [General Style Guide](./general-style-guide.md).
- Benchmark claim policy is canonical in [../benchmark-methodology.md](../benchmark-methodology.md).
- Execution-plane invariants appear as the Invariants quick-ref at the top of General; detailed sections live further down in the same guide.

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
