---
name: doppler-debug
description: Debug DOPPLER WebGPU inference issues. Use when investigating model output problems, kernel bugs, hidden state explosions, or decode failures in the browser-based LLM inference engine. (project)
---

# DOPPLER Debug

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
