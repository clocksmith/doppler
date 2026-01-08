# Doppler JavaScript-First Migration

## Decision: JavaScript as Source Language

**Doppler source code is JavaScript**, not TypeScript. This is a deliberate architectural decision, not a compromise.

See [Language Policy](style/GENERAL_STYLE_GUIDE.md#language-policy-javascript--declaration-files) for full rationale and research citations.

### Why JavaScript

| Reason | Explanation |
|--------|-------------|
| **Hot-swap architecture** | JS/WGSL/JSON swap at runtime without build step |
| **Agent-generated code** | Agents will generate nearly 100% of this code; no benchmark shows LLMs generate better TS than JS |
| **TypeScript compiles to JS anyway** | The compilation step adds friction without benefit for hot-swappable code |
| **Tests are the type system** | Comprehensive tests catch type errors pre-production |

### What About Types?

Every module has a corresponding `.d.ts` file with comprehensive type definitions.

| File | Contains |
|------|----------|
| `module.js` | Clean code, no type annotations |
| `module.d.ts` | All type definitions for that module |

Agents read `.d.ts` files directly for type context—no need to pollute JS with JSDoc.

---

## Migration Phases

### Phase 0: Migrate Tests to JS

**Gate: Must complete before any source migration.**

| Category | Files | Owner |
|----------|-------|-------|
| Kernel correctness specs | 14 | Agent A |
| Unit tests | 7 | Agent A |
| Test harness | 5 | Agent A |
| Reference implementations | 14 | Agent B |
| Benchmark files | 6 | Agent B |
| Config/setup | 6 | Agent B |
| **Total** | **52** | |

### Phase 1: Add Critical Missing Tests

**Gate: Tests must exist for a module before that module's source is migrated.**

| Module | Current Tests | Needed | Priority |
|--------|---------------|--------|----------|
| `inference/pipeline/` | 0 | Core pipeline flow | P0 |
| `loader/` | 1 | Weight loading, manifest parsing | P0 |
| `formats/` | 0 | GGUF/RDRR parsing | P1 |
| `debug/` | 0 | Log level, trace output | P2 |
| `memory/` | 0 | Heap allocation | P2 |

### Phase 2: Migrate Source to JS

**Gate: Phase 0 complete. Module tests exist (Phase 1) before module migration.**

Parallelized into 4 work streams based on dependencies:

```
Stream A (Independent)        Stream B (Independent)        Stream C (Depends on A)       Stream D (Depends on B, C)
─────────────────────        ─────────────────────        ──────────────────────        ─────────────────────────
config/ (5)                   formats/ (4)                  gpu/kernels/ (31)             inference/pipeline/ (22)
debug/ (12)                   browser/ (6)                  gpu/ core (17)                inference/ core (12)
types/ (4)                    bridge/ (3)                   loader/ (22)
memory/ (4)                   adapters/ (5)                 storage/ (7)
                              converter/ (9)
─────────────────────        ─────────────────────        ──────────────────────        ─────────────────────────
Total: 25 files               Total: 27 files               Total: 77 files               Total: 34 files
```

**Execution order:**
1. Streams A and B start immediately (parallel)
2. Stream C starts when Stream A completes
3. Stream D starts when Streams B and C complete

---

## Conversion Process

For each `.ts` file:

1. **Create** `module.d.ts` with all type definitions
2. **Create** `module.js` with clean code (no type annotations)
3. **Update imports** to `.js` extensions
4. **Run tests** to verify correctness

### Example Conversion

**Before (TypeScript):**
```typescript
// uniforms.ts
interface KernelUniforms {
  seqLen: number;
  startPos: number;
}

function writeUniforms(view: DataView, u: KernelUniforms): void {
  view.setUint32(0, u.seqLen, true);
  view.setUint32(4, u.startPos, true);
}
```

**After (JavaScript + .d.ts):**
```javascript
// uniforms.js - clean code
function writeUniforms(view, u) {
  view.setUint32(0, u.seqLen, true);
  view.setUint32(4, u.startPos, true);
}
```

```typescript
// uniforms.d.ts - type specs
interface KernelUniforms {
  seqLen: number;
  startPos: number;
}

declare function writeUniforms(view: DataView, u: KernelUniforms): void;
```

---

## Verification

After converting each module:

```bash
# Run tests for that module
npm test -- --filter <module>

# Type check declarations
npx tsc --noEmit src/**/*.d.ts

# Full test suite
npm test
```

---

## Declaration Files

Every module gets a hand-written `.d.ts` file alongside its `.js` file:

```
src/gpu/kernels/
  matmul.js           # Clean code
  matmul.d.ts         # Type specs
  attention.js
  attention.d.ts
  ...
```

This provides:
- Type context for agents (read `.d.ts` files directly)
- Full type safety for TypeScript consumers
- No build step required

---

## Signing and Trust Model

For distributed hot-swaps, artifacts require signatures:

| Context | Policy |
|---------|--------|
| **Local dev** | Per-device key signs artifacts; unsigned allowed with explicit "local-only" flag |
| **P2P distribution** | Accept only signatures from trusted signer allowlist |
| **Official releases** | Signed by shared signer service key; trusted by default |

**Manifest includes:** artifact hashes + signer ID + signature (auditable and reproducible)

---

## References

- [Language Policy](style/GENERAL_STYLE_GUIDE.md#language-policy-javascript--declaration-files) — Full rationale with citations
- [JavaScript Style Guide](style/JAVASCRIPT_STYLE_GUIDE.md) — Code and test conventions
- [README](../README.md#why-javascript-over-typescript) — Public-facing rationale
