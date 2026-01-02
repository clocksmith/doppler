---
name: doppler-debug
description: Debug DOPPLER WebGPU inference issues. Use when investigating model output problems, kernel bugs, hidden state explosions, or decode failures in the browser-based LLM inference engine. (project)
---

# DOPPLER Debug

## Using Config Presets

```bash
# Use built-in debug preset (verbose logging, trace enabled, headed browser)
npm run debug -- --config debug -m MODEL

# Use custom config file
npm run debug -- --config ./my-debug-config.json -m MODEL

# List available presets
npx tsx cli/index.ts --list-presets

# Dump resolved config
npx tsx cli/index.ts --dump-config --config debug
```

## Fast Iteration (use --skip-load after first run!)

```bash
# First run - loads model (~30s), keeps browser open
npm run debug -- -m MODEL --warm 2>&1 | grep --line-buffered -E "Config|Token|Done|Output" | sed '/Done/q'

# Subsequent runs - reuses model, ~12s inference only
npm run debug -- -m MODEL --skip-load 2>&1 | grep --line-buffered -E "Config|Token|Done|Output" | sed '/Done/q'

# With trace
npm run debug -- -m MODEL --skip-load --trace attn 2>&1 | grep --line-buffered -E "TRACE|Token|Done" | sed '/Done/q'
```

## Key: `sed '/Done/q'` exits after Done line

## Trace Categories

`loader`, `kernels`, `logits`, `embed`, `attn`, `ffn`, `kv`, `sample`, `buffers`, `perf`, `all`

## After Code Changes

```bash
npm run build && npm run debug -- -m MODEL --skip-load 2>&1 | grep --line-buffered -E "Token|Done|Output" | sed '/Done/q'
```

## Grep Patterns

- `"Config|Token|Done|Output"` - Quick check
- `"TRACE|ANOMALY"` - Problems
- `"maxAbs"` - Hidden state magnitudes
- `"RoPE|hasLocal"` - RoPE config

## Log Output to File

Create a custom config to log to file:

```json
{
  "extends": "debug",
  "runtime": {
    "debug": {
      "logOutput": { "stdout": true, "file": "./logs/debug.log" },
      "trace": { "file": "./logs/trace.jsonl" }
    }
  }
}
```
