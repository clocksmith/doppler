## DOPPLER Code Agent

Repository: https://github.com/clocksmith/doppler

**Prime Directive:** Write JavaScript (with .d.ts declarations) for the WebGPU inference engine running in the browser.

**See also:** [REPLOID](https://github.com/clocksmith/reploid) for browser-native AI agent (separate repo).

### ⚠️ MANDATORY: Read Style Guides First

**Before writing ANY code, you MUST read the relevant style guides:**

1. **[General Style Guide](docs/style/GENERAL_STYLE_GUIDE.md)** - Architecture patterns, naming conventions, file organization
2. **[JavaScript Guide](docs/style/JAVASCRIPT_STYLE_GUIDE.md)** - Config-as-code, kernel wrappers, .d.ts requirements
3. **[WGSL Guide](docs/style/WGSL_STYLE_GUIDE.md)** - Shader structure, optimization patterns, shared memory usage

These guides contain critical performance patterns and architectural decisions. Ignoring them leads to:
- Kernels with O(N²) complexity when O(N) is possible
- Missing shared memory optimizations
- Incorrect dtype handling (f16 vs f32 bindings)
- Performance regressions instead of improvements

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
│   │   └── runtime.js    # Runtime config registry (get/set)
│   ├── memory/           # Heap management, capability detection
│   └── debug/            # Logging and tracing
├── tests/kernels/        # GPU kernel validation
├── demo/                 # Demo UI
├── cli/                  # CLI entry point
└── docs/                 # Documentation
```

### Before Starting
- Read `docs/ARCHITECTURE.md` for system overview
- Read `docs/FORMATS.md` for model format specification
- Review `src/inference/pipeline.js` for inference flow
- Review `src/config/runtime.js` and `cli/config/` for runtime config plumbing

### CLI Commands

| Script | Purpose |
|--------|---------|
| `npm start` | Dev server at localhost:8080 |
| `npm test` | Unit tests (Vitest) |
| `npm run test:gpu` | GPU/browser tests (config-only) |
| `npm run bench` | Performance benchmarks (config-only) |
| `npm run debug` | Debug runs (config-only) |

```bash
npm test -- --filter matmul          # Filter to specific kernel
npm run test:gpu -- --config <ref>   # GPU/browser suite via config
npm run bench -- --config bench      # Benchmark preset
npm run debug -- --config debug      # Debug preset (trace/logs via config)
```

### CLI Interface

The DOPPLER CLI is config-only. The only supported CLI flags are `--config` and `--help`.
Model selection, headless/headed, command, and suite must live in the config.

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
- Canonical max tokens lives in `runtime.inference.batching.maxTokens`. `runtime.inference.sampling.maxTokens` is removed.
- `runtime.shared.tooling.intent` is required for CLI runs (verify/investigate/calibrate).

**Layer Pipeline Plans (experimental):**
- Model presets may define `inference.pipeline` to drive per-layer step order.
- Runtime overrides can supply `runtime.inference.pipeline` for ad-hoc experiments.
- When unset, DOPPLER uses the optimized hardcoded layer path.

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

Exceptions: `tools/`, `tests/kernels/`, and one-time startup messages in `src/gpu/device.js`.
Also acceptable: CLI entry points (`cli/`, `serve.js`, `src/converter/node-converter.js`) for direct terminal output.

### No Ad-Hoc Debug Logging

**Do not add temporary log statements to source files for debugging.** This is not allowed. All debugging must use:

1. Existing trace categories via `--config debug`
2. Config-driven probes for specific values
3. Extended trace categories (permanent additions only)

See `docs/style/GENERAL_STYLE_GUIDE.md` for details.

### Debug Probes

Use config-driven probes to inspect values—do not add ad-hoc log statements:

- `runtime.inference.debug.probes` targets specific tokens/dims without code edits.
- Use `logits_final` probes for post-softcap comparisons on Gemma 2/3.
- If visibility is missing, extend the trace system permanently—don't add throwaway logs.

### Quantization Naming

- Manifests can include `quantizationInfo` with per-group precision (weights vs embeddings).
- Converter default modelId suffix: `w<weights>-emb<embeddings>` (override with `--model-id`).

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
npx tsx src/converter/node-converter.js model.gguf ./output

# Dev server
npx tsx serve.js --port 3000
```


### Skills
- Project skills (from `.claude/skills/`):
  - `doppler-debug`: debug inference issues. Read `.claude/skills/doppler-debug/SKILL.md`.
  - `doppler-bench`: run performance benchmarks. Read `.claude/skills/doppler-bench/SKILL.md`.
  - `doppler-convert`: convert models to RDRR. Read `.claude/skills/doppler-convert/SKILL.md`.
- System skills:
  - `skill-creator`: create or update Codex skills. Read `/Users/xyz/.codex/skills/.system/skill-creator/SKILL.md`.
  - `skill-installer`: list or install skills. Read `/Users/xyz/.codex/skills/.system/skill-installer/SKILL.md`.
- When a skill is named or its description matches the task, open the SKILL.md first and follow it.

See `docs/KERNEL_COMPATIBILITY.md` for kernel overrides and runtime modes.
