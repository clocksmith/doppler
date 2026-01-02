## DOPPLER Code Agent

Repository: https://github.com/clocksmith/doppler

**Prime Directive:** Write TypeScript for the WebGPU inference engine running in the browser.

**See also:** [REPLOID](https://github.com/clocksmith/reploid) for browser-native AI agent (separate repo).

### Directory Structure
```
doppler/
├── src/
│   ├── inference/        # Pipeline, attention, FFN, embeddings
│   ├── gpu/              # WebGPU device, buffer pools, kernels
│   ├── storage/          # OPFS shard manager, model loading
│   ├── loader/           # GGUF parsing, RDRR manifest
│   ├── config/
│   │   ├── schema/       # Runtime/model schemas + defaults
│   │   ├── presets/
│   │   │   ├── runtime/  # Runtime presets (default/debug/bench/etc)
│   │   │   └── models/   # Model family presets (gemma/llama/etc)
│   │   └── runtime.ts    # Runtime config registry (get/set)
│   ├── memory/           # Heap management, capability detection
│   └── debug/            # Logging and tracing
├── kernel-tests/         # GPU kernel validation
├── app/                  # Demo UI
├── cli/                  # CLI entry point
└── docs/                 # Documentation
```

### Before Starting
- Read `docs/ARCHITECTURE.md` for system overview
- Read `docs/spec/RDRR_FORMAT.md` for model format specification
- Review `src/inference/pipeline.ts` for inference flow
- Review `src/config/runtime.ts` and `cli/config/` for runtime config plumbing

### Style Guides
- [Coding Guide](docs/style/CODING_GUIDE.md) - Architecture, naming
- [TypeScript Guide](docs/style/TYPESCRIPT_STYLE_GUIDE.md) - Config-as-code, kernel wrappers
- [WGSL Guide](docs/style/WGSL_STYLE_GUIDE.md) - Shader structure

### CLI Commands

| Script | Purpose |
|--------|---------|
| `npm start` | Dev server at localhost:8080 |
| `npm test` | Kernel correctness tests |
| `npm run bench` | Performance benchmarks |
| `npm run debug` | Debug with tracing enabled |
| `npm run build` | Compile TypeScript |

```bash
npm test -- --filter matmul      # Filter to specific kernel
npm run bench -- --runs 3        # Multiple benchmark runs
npm run debug -- --trace kernels # Trace specific categories
```

### Common Flags

| Flag | Description |
|------|-------------|
| `--model, -m <name>` | Model to use (default: gemma-3-1b-it-q4) |
| `--config <preset>` | Load runtime preset or config file |
| `--trace [cats]` | Trace categories: kernels, attn, ffn, kv, sample |
| `--verbose, -v` | Verbose output |
| `--quiet, -q` | Suppress logs |
| `--headed` | Show browser window |

### Config System

```bash
npm run debug -- --config debug              # Use preset
npm run bench -- --config ./my-config.json   # Use file
```

**Preset Rules:**
- Runtime presets live in `src/config/presets/runtime`. CLI only loads these.
- Model presets live in `src/config/presets/models`. Loader uses these for model detection.
- Do not mix model presets with runtime presets.

**Runtime Config Plumbing:**
- CLI `--config` loads a runtime config (merged with defaults) and passes it via the `runtimeConfig` URL param to the browser harness.
- Test harnesses parse `runtimeConfig` and call `setRuntimeConfig()` before pipeline/loader init.
- For per-instance overrides, pass `PipelineContexts.runtimeConfig` to `createPipeline()`.
- Subsystems should read tunables via `getRuntimeConfig()`; avoid importing `DEFAULT_*` in runtime code.
- Canonical max tokens lives in `runtime.inference.batching.maxTokens`. `runtime.inference.sampling.maxTokens` is deprecated but mapped for back-compat in `src/config/runtime.ts`.

**Development Guide:**

Configs are the development interface. Change behavior without editing source files.

**Two tiers of constants:**
1. **Invariants** — Format specs, protocol constants. Hardcoded. Not configurable.
2. **Tunables** — Cache sizes, thresholds, dtypes. Live in `src/config/schema/`.

**Workflow:** Change configs, not code. When debugging, use debug preset. When benchmarking, use bench preset. Feedback loop stays tight because nothing rebuilds.

**When building features:** Ask "Will I tweak this while developing?" If yes, make it configurable. Import from schema, don't hardcode.

**Presets capture knowledge:** A working config is documentation. Save configs, share configs.

### Logging

Use the debug module, not raw `console.*`:

```typescript
import { log, trace } from '../debug/index.js';

log.info('Pipeline', 'Model loaded');
log.warn('Attention', 'Fallback to CPU');
trace.kernels(`matmul M=${M} N=${N}`);
```

Exceptions: `tools/`, `kernel-tests/`, and one-time startup messages in `src/gpu/device.ts`.

### Guardrails
- Handle GPU device loss gracefully
- Validate tensor shapes at kernel boundaries
- Use BF16 for weights, F32 for activations
- Test with multiple quantization levels (Q4, Q8, F16)

### Kernel Tests

| Kernel | Description |
|--------|-------------|
| `matmul` | Matrix multiplication |
| `matmul-q4k` | Q4_K quantized matmul |
| `attention` | Flash attention |
| `rmsnorm` | RMS normalization |
| `rope` | Rotary embeddings |
| `silu`, `swiglu` | Activations |
| `gather` | Embedding lookup |
| `scatter-add` | MoE output |
| `topk` | Top-K selection |
| `sample` | GPU sampling |

### CLI Tools

```bash
# Convert model to RDRR format
npx tsx src/converter/node-converter.ts model.gguf ./output

# Dev server
npx tsx serve.ts --port 3000
```

### Build Artifacts
- `dist/` is built output for browser usage. If runtime changes affect browser code, run `npm run build`.
- `tests/benchmark/*.js` are generated from TS. After editing `tests/benchmark/*.ts`, run `npm run build:benchmark`.

### Skills
- Project skills (from `.claude/skills/`):
  - `doppler-debug`: debug inference issues. Read `.claude/skills/doppler-debug/SKILL.md`.
  - `doppler-benchmark`: run performance benchmarks. Read `.claude/skills/doppler-benchmark/SKILL.md`.
  - `model-convert`: convert models to RDRR. Read `.claude/skills/model-convert/SKILL.md`.
- System skills:
  - `skill-creator`: create or update Codex skills. Read `/Users/xyz/.codex/skills/.system/skill-creator/SKILL.md`.
  - `skill-installer`: list or install skills. Read `/Users/xyz/.codex/skills/.system/skill-installer/SKILL.md`.
- When a skill is named or its description matches the task, open the SKILL.md first and follow it.

See `docs/KERNEL_COMPATIBILITY.md` for kernel overrides and runtime modes.
