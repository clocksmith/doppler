---
name: doppler-benchmark
description: Run DOPPLER performance benchmarks to measure throughput, compare against baselines, and detect regressions across models or kernel settings. Use when validating speed changes or collecting JSON benchmark artifacts. (project)
---

# DOPPLER Benchmark Skill

Use this skill to measure DOPPLER inference performance.

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
# List quick-start models available for download
grep -A2 "modelId:" src/storage/quickstart-downloader.ts | grep modelId

# List model presets (shows supported model families and their configs)
ls src/config/presets/models/

# Dump resolved runtime config (includes activationDtype + kernelPlan)
npm run bench -- --config bench --dump-config

# Check GPU capabilities (shader-f16 determines if F16 activations are available)
npm run debug -- -m MODEL 2>&1 | grep -i "shader-f16\\|features"
```

**Common Model Names:**
- `gemma-3-1b-it-q4` - Gemma 3 1B (Q4_K quantized, ~500MB)
- `gemma-2-2b-it-q4` - Gemma 2 2B (Q4_K quantized, ~1.2GB)
- `llama-3.2-1b-q4` - Llama 3.2 1B (Q4_K quantized, ~700MB)

**First-Time Setup:** Models must be downloaded before benchmarking. On first run, the browser will fetch from HuggingFace and cache to OPFS. Subsequent runs use the cached model. If you see "Not found" errors, the model download may have failed - check network/CORS.

**Compute Precision** (see `src/config/schema/inference-defaults.schema.ts`):
- `activationDtype: "f32"` is the safe default
- `activationDtype: "f16"` is experimental, faster on shader-f16 GPUs

```bash
# Force F16 activations via preset
npm run bench -- -m MODEL --config f16-activations
```

## Standard Benchmark Commands

```bash
# Quick benchmark - single run
npm run bench -- -m MODEL 2>&1 | grep -E "TTFT|Prefill|Decode|GPU Submits"

# Fix decode length for apples-to-apples comparisons
npm run bench -- -m MODEL --max-tokens 64 --runs 3

# Multiple runs for statistical confidence (recommended)
npm run bench -- -m MODEL --runs 3 2>&1 | sed '/DOPPLER:DONE/q'

# Save results to JSON for later comparison
npm run bench -- -m MODEL --runs 3 -o results.json

# Compare against saved baseline
npm run bench -- -m MODEL --compare baseline.json

# Extract full result JSON
npm run bench -- -m MODEL 2>&1 | grep "DOPPLER:RESULT" | sed 's/.*DOPPLER:RESULT] //'
```

## Fast Iteration Pattern

```bash
# Quick single run (minimal warmup)
npm run bench -- -m MODEL --runs 1 --warmup 0 2>&1 | sed '/DOPPLER:DONE/q'

# After code changes: rebuild then benchmark
npm run build && npm run bench -- -m MODEL --runs 1 --warmup 0 2>&1 | sed '/DOPPLER:DONE/q'
```

Use `sed '/DOPPLER:DONE/q'` to exit immediately after benchmark completes.

## Regression Detection Protocol

When asked "Is this slower?" or investigating performance regressions:

1. **Establish baseline** (if not already saved; use a clean worktree):
   ```bash
   git checkout main
   npm run bench -- -m MODEL --runs 3 -o baseline.json
   ```

2. **Benchmark current code**:
   ```bash
   npm run bench -- -m MODEL --runs 3 --compare baseline.json
   ```

3. **Interpret results**:
   - **<5% difference**: Within noise, not significant
   - **5-10% difference**: Investigate, may be real
   - **>10% difference**: Significant regression, needs fix

## Benchmark Configuration

For consistent results, use the `bench` config preset:

```bash
# Uses silent output, deterministic sampling, longer timeout
npm run bench -- --config bench -m MODEL
```

Or create a custom config for CI:

```json
{
  "extends": "bench",
  "runtime": {
    "debug": {
      "logOutput": { "stdout": false, "file": "./ci-logs/bench.log" }
    }
  },
  "cli": { "timeout": 300000 }
}
```

## Kernel Configuration

Test different kernel variants via `--config` or CLI flags:

```bash
# Via config file
npm run bench -- -m MODEL --config kernel-config.json

# Via CLI flags
npm run bench -- -m MODEL --kernel-profile fused
npm run bench -- -m MODEL --kernel-plan '{"q4kStrategy":"fused_q4k","variants":{"attention":{"prefill":"tiled_small","decode":"streaming"}}}'
```

Example `kernel-config.json`:
```json
{
  "runtime": {
    "inference": {
      "kernelPlan": {
        "q4kStrategy": "auto | fused_q4k | dequant_f16 | dequant_f32",
        "variants": {
          "attention": {
            "prefill": "auto | tiled_large | tiled_small | streaming",
            "decode": "auto | tiled_large | tiled_small | streaming"
          },
          "matmul": {
            "roles": {
              "q_proj": "auto | gemv_subgroup | gemv_subgroup_multicol | f16w_f32a | f32"
            }
          }
        }
      }
    }
  }
}
```

CLI flags override config file values. Use this to A/B test kernel variants.

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

- **Benchmark harness**: `tests/benchmark/pipeline-benchmark.ts`
- **Result format**: `docs/design/BENCHMARK_HARNESS.md`
- **Historical results**: `tests/results/*.json`
- **Regression guide**: `docs/PERFORMANCE_REGRESSION_INVESTIGATION.md`

## Related Skills

- Use `doppler-debug` if benchmark fails or produces errors
- Use `doppler-convert` to test different quantization levels
