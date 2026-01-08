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

## Migration Status

### Completed

- [x] Language policy documented in style guides
- [x] README updated with JS-first rationale
- [x] 3 pilot JS files proving hot-swap works (`src/config/kernels/`, `src/config/platforms/`)
- [x] tsconfig supports `allowJs` and `checkJs`

### Remaining Work

| Scope | Files | Lines | Priority |
|-------|-------|-------|----------|
| `src/` runtime | ~186 | ~57,000 | High |
| `cli/` | ~10 | ~2,000 | Medium |
| `app/` | ~5 | ~1,500 | Low |
| `kernel-tests/` | ~30 | ~5,000 | Low |

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

## Signing and Trust Model

For distributed hot-swaps, artifacts require signatures:

| Context | Policy |
|---------|--------|
| **Local dev** | Per-device key signs artifacts; unsigned allowed with explicit "local-only" flag |
| **P2P distribution** | Accept only signatures from trusted signer allowlist |
| **Official releases** | Signed by shared signer service key; trusted by default |

**Manifest includes:** artifact hashes + signer ID + signature (auditable and reproducible)

---

## Conversion Order

Recommended order minimizes risk and validates the approach incrementally:

1. **GPU kernels** (`src/gpu/kernels/`) — Self-contained, easy to test
2. **Config** (`src/config/`) — Already has pilot files
3. **Debug/logging** (`src/debug/`) — Simple utilities
4. **Formats** (`src/formats/`) — Standalone parsers
5. **Loader** (`src/loader/`) — Depends on formats
6. **Storage** (`src/storage/`) — Depends on loader
7. **Inference** (`src/inference/`) — Core pipeline, most complex
8. **Types** (`src/types/`) — Convert to `.d.ts` only (no runtime code)

---

## Verification

After converting each module:

```bash
# Type check with JSDoc
npx tsc --allowJs --checkJs --noEmit src/**/*.js

# Run tests
npm test

# Verify hot-swap works
npm run debug -- --config debug
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

## References

- [Language Policy](style/GENERAL_STYLE_GUIDE.md#language-policy-javascript--declaration-files) — Full rationale with citations
- [JavaScript Style Guide](style/JAVASCRIPT_STYLE_GUIDE.md) — JSDoc conventions
- [README](../README.md#why-javascript-over-typescript) — Public-facing rationale
