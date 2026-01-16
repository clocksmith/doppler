---
name: doppler-benchmark
description: Run DOPPLER performance benchmarks to measure throughput, compare against baselines, and detect regressions across models or kernel settings. Use when validating speed changes or collecting JSON benchmark artifacts. (project)
---

# DOPPLER Benchmark Skill

Use this skill to measure DOPPLER inference performance.

## Critical: Inference vs Kernel Benchmarks

```bash
npm run bench                # Runs INFERENCE benchmark (tok/s) - DEFAULT
npm run bench -- --kernels   # Runs KERNEL microbenchmarks (matmul, attention, etc.)
```

**Inference benchmarks are the default.** Use `--kernels` only for microbench.

## Completion Signals

DOPPLER emits standardized signals for CLI/automation detection:

| Signal | Meaning |
|--------|---------|
| `[DOPPLER:DONE]` | Task completed (success or error) - always emitted at end |
| `[DOPPLER:RESULT]` | Full benchmark result JSON - emitted before DONE |
| `[DOPPLER:ERROR]` | Error occurred - emitted before DONE on failure |

Example output:
```
[DOPPLER:RESULT] {"schemaVersion":1,"metrics":{"decode_tokens_per_sec":42.5,...}}
[DOPPLER:DONE] {"status":"success","elapsed":1234}
```

## Key Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| `decode_tokens_per_sec` | Main throughput metric | Higher is better |
| `ttft_ms` | Time to first token | Lower is better, measures prefill latency |
| `prefill_tokens_per_sec` | Prompt processing speed | Higher is better for long prompts |
| `decode_ms_per_token_p99` | 99th percentile per-token decode latency (ms) | Flag if >100ms for interactive use |
| `estimated_vram_bytes_peak` | Peak GPU memory usage (bytes) | Lower means more headroom |

## Discovery Commands

Before benchmarking, discover available models and current configuration:

```bash
# List models in RDRR format (ready to use)
ls models/

# List model presets (shows supported model families and their configs)
ls src/config/presets/models/

# Dump resolved runtime config
npm run bench -- --config bench --dump-config

# Check GPU capabilities (shader-f16 determines if F16 activations are available)
npm run debug -- -m MODEL 2>&1 | grep -i "shader-f16\|features"
```

**Common Model Names:**
- `gemma-2-2b-it-wf16` - Gemma 2 2B (F16 weights)
- `gemma-3-1b-it-q4` - Gemma 3 1B (Q4_K quantized, ~500MB)

**First-Time Setup:** Models must be downloaded/converted before benchmarking. If you see "Not found" errors, the model may not exist in `models/` directory.

## Standard Benchmark Commands

```bash
# Quick inference benchmark - single run
npm run bench -- --config bench -m MODEL 2>&1 | grep -E "TTFT|Prefill|Decode|tok/s"

# Multiple runs for statistical confidence (recommended)
npm run bench -- --config ./bench-3runs.json -m MODEL

# Save results to JSON for later comparison
npm run bench -- --config ./bench-3runs.json -m MODEL -o results.json

# Compare against saved baseline
npm run bench -- --config bench -m MODEL --compare baseline.json

# Extract full result JSON
npm run bench -- --config bench -m MODEL 2>&1 | grep "DOPPLER:RESULT" | sed 's/.*DOPPLER:RESULT] //'
```

## Configuration via --config

Use config files or inline JSON. CLI flags must not override runtime tunables.

```bash
# Fix decode length for apples-to-apples comparisons (inline JSON config)
npm run bench -- --config '{"runtime":{"inference":{"batching":{"maxTokens":64}}}}' -m MODEL

# Combine with runs and warmup in config
npm run bench -- --config ./bench-3runs.json -m MODEL

# Use a preset
npm run bench -- --config bench -m MODEL

# Use a config file
npm run bench -- --config ./my-bench-config.json -m MODEL
```

Example `my-bench-config.json`:
```json
{
  "extends": "bench",
  "runtime": {
    "shared": {
      "benchmark": {
        "run": {
          "warmupRuns": 1,
          "timedRuns": 3
        }
      }
    },
    "inference": {
      "batching": { "maxTokens": 64 },
      "sampling": { "temperature": 0 }
    }
  }
}
```

## Kernel Configuration Testing

Test different kernel thresholds or variants via `--config`:

```bash
# Test with fused kernel disabled (set threshold below model's hidden size)
npm run bench -- --config '{"runtime":{"shared":{"kernelThresholds":{"fusedMatmul":{"maxMediumN":0}}}}}' -m MODEL

# Test with fused kernel enabled
npm run bench -- --config '{"runtime":{"shared":{"kernelThresholds":{"fusedMatmul":{"maxMediumN":4096}}}}}' -m MODEL

# Explicit kernel path (config-only)
npm run bench -- --config '{"runtime":{"inference":{"kernelPath":"gemma2-q4k-dequant-f16a"}}}' -m MODEL
```

## Fast Iteration Pattern

```bash
# Quick single run
npm run bench -- --config ./bench-1run.json -m MODEL 2>&1 | sed '/DOPPLER:DONE/q'

# After code changes: rebuild then benchmark
npm run build && npm run bench -- --config ./bench-1run.json -m MODEL 2>&1 | sed '/DOPPLER:DONE/q'
```

Use `sed '/DOPPLER:DONE/q'` to exit immediately after benchmark completes.

## CDP Browser Reuse (Best Performance)

For fastest iteration, start Chrome once and reuse it across benchmarks:

```bash
# Step 0: Start Chrome with CDP (once per session)
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222

# Step 1: Run benchmarks reusing the browser (avoids startup overhead)
node cli/index.js bench inference --config ./bench-3runs.json -m MODEL \
  --no-server --reuse-browser --cdp-endpoint http://localhost:9222
```

**CDP flags:**
| Flag | Description |
|------|-------------|
| `--no-server` | Skip dev server (faster startup) |
| `--reuse-browser` | Don't close browser after run |
| `--cdp-endpoint <url>` | CDP endpoint (default: http://localhost:9222) |

This pattern is especially useful when A/B testing kernel configurations.

## Regression Detection Protocol

When asked "Is this slower?" or investigating performance regressions:

1. **Establish baseline** (if not already saved; use a clean worktree):
   ```bash
   git checkout main
   npm run bench -- --config ./bench-3runs.json -m MODEL -o baseline.json
   ```

2. **Benchmark current code**:
   ```bash
   npm run bench -- --config ./bench-3runs.json -m MODEL --compare baseline.json
   ```

3. **Interpret results**:
   - **<5% difference**: Within noise, not significant
   - **5-10% difference**: Investigate, may be real
   - **>10% difference**: Significant regression, needs fix

## Interpretation Guidelines

| Scenario | What to Check |
|----------|---------------|
| Prefill slow | Memory bandwidth, embedding gather |
| Decode slow | Matmul kernels, attention implementation |
| High variance | Background processes, thermal throttling |
| Memory regression | Buffer pool usage, tensor allocation |

## Key Grep Patterns

| Pattern | Purpose |
|---------|---------|
| `"DOPPLER:DONE\|DOPPLER:ERROR"` | Check completion status |
| `"DOPPLER:RESULT"` | Extract full result JSON |
| `"tok/s\|tokens_per"` | Find throughput metrics |
| `"TTFT\|ttft"` | Find latency metrics |
| `"memory\|Memory"` | Find memory usage |
| `"regression\|slower"` | Find comparison results |

## Reference Files

For detailed information, consult these files:

- **Benchmark harness**: `cli/helpers/inference-benchmark.js`
- **CLI implementation**: `cli/index.js`
- **Config resolution**: `docs/style/CONFIG_STYLE_GUIDE.md`
- **Historical results**: `tests/results/*.json`

## Related Skills

- Use `doppler-debug` if benchmark fails or produces errors
- Use `doppler-convert` to test different quantization levels
