---
name: doppler-debug
description: Debug DOPPLER WebGPU inference issues such as gibberish output, hidden state explosions, NaN/Inf traces, sampling anomalies, or mismatches vs HuggingFace reference in the browser-based LLM inference engine. Use when investigating correctness of pipeline, kernels, or sampling behavior. (project)
---

# DOPPLER Debug Skill

Use this skill to debug WebGPU inference issues in DOPPLER.

## Triage First (Do This Before Running Commands)

| Symptom | What to Look For | Likely Root Cause |
|---------|------------------|-------------------|
| Gibberish output | Logits diverge from HuggingFace | Wrong softcapping, scaling, or norm offset |
| Values exploding | `maxAbs` growing 10x+ per layer | Missing `1+weight` norm offset (Gemma models) |
| Single dimension spike | Same dim# exploding across all layers | Weight corruption or index misalignment |
| NaN/Inf values | `NaN` appearing in trace output | Numerical overflow in attention or FFN |
| Repeating tokens | Same token generated repeatedly | KV cache corruption or attention mask bug |
| Wrong token probabilities | Top-k tokens different from HF | Compare `logits_final` (post-softcap) vs HF |
| PAD token wins | `<pad>` selected or repeated | pad_token_id not masked in sampling |

## Completion Signals

DOPPLER emits standardized signals for CLI/automation detection:

| Signal | Meaning |
|--------|---------|
| `[DOPPLER:DONE]` | Task completed (success or error) - always emitted at end |
| `[DOPPLER:RESULT]` | Full result payload (JSON) - emitted before DONE |
| `[DOPPLER:ERROR]` | Error occurred - emitted before DONE on failure |

Example output:
```
[DOPPLER:RESULT] {"output":"blue","tokens":8,"elapsed":1234.5,"tokensPerSecond":6.5}
[DOPPLER:DONE] {"status":"success","elapsed":1234.5,"tokens":8,"tokensPerSecond":6.5}
```

## Required Config Fields

**`runtime.inference.prompt` is required for all debug runs.** The debug preset does not include a default prompt.

### Starter Debug Config Template

Create a file (e.g., `tmp-debug.json`):

```json
{
  "extends": "debug",
  "runtime": {
    "shared": {
      "debug": {
        "trace": {
          "enabled": true,
          "categories": ["ffn", "attn", "logits"],
          "layers": [0, 12, 25]
        },
        "probes": [
          { "id": "layer_out", "stage": "layer_out", "layers": [0, 12, 25], "tokens": [-1], "dims": [0, 1, 2, 334] },
          { "id": "logits_final", "stage": "logits_final", "tokens": [-1], "dims": [0, 1, 2, 3, 4] }
        ]
      }
    },
    "inference": {
      "prompt": "The color of the sky is",
      "batching": { "maxTokens": 16 },
      "sampling": { "temperature": 0 }
    }
  }
}
```

Then run:
```bash
npm run debug -- --config ./tmp-debug.json -m MODEL 2>&1 | sed '/DOPPLER:DONE/q'
```

## Quick Diagnostic Commands

```bash
# Step 1: Quick sanity check - does it produce any output?
npm run debug -- --config debug -m MODEL 2>&1 | grep -E "DOPPLER:DONE|DOPPLER:ERROR"

# Step 2: Check for obvious errors in logs
npm run debug -- --config debug -m MODEL 2>&1 | grep -E "Error|NaN|Inf|ANOMALY"

# Step 3: Trace specific subsystem based on triage
npm run debug -- --config ./trace-category.json -m MODEL 2>&1 | grep -E "TRACE|maxAbs"
```

Note: The Step 1 grep only shows harness completion signals. Loader/manifest errors often occur
before the harness starts, so they will be hidden by that filter. If you suspect a load failure,
rerun without grep or with a broader filter like "Error|ERROR|Manifest|loader".

Example (manifest validation error):
```bash
npm run debug -- --config debug -m MODEL 2>&1 | grep -E "Manifest|inference config|Error|ERROR"
```

## Trace Categories (Choose Based on Triage)

| Category | When to Use |
|----------|-------------|
| `attn` | Attention-related bugs, wrong scaling, softcapping issues |
| `ffn` | FFN/MLP bugs, activation function issues, hidden state explosions |
| `logits` | Final projection bugs, LM head issues, wrong token probabilities |
| `kernels` | Low-level GPU kernel bugs, shader compilation issues |
| `kv` | KV cache corruption, sequence length bugs |
| `sample` | Sampling bugs, temperature/top-k issues |
| `loader` | Model loading bugs, weight corruption, quantization issues |
| `all` | When unsure - captures everything (verbose) |

```bash
# Single category
npm run debug -- --config ./trace-ffn.json -m MODEL

# Multiple categories
npm run debug -- --config ./trace-ffn-kernels.json -m MODEL

# All except expensive ones
npm run debug -- --config ./trace-all.json -m MODEL
```

## Kernel Path Overrides

Use kernel path overrides to isolate kernel-specific issues without editing code.

```bash
# Override kernel path for A/B testing
npm run debug -- --config '{"runtime":{"inference":{"prompt":"The color of the sky is","kernelPath":"gemma2-q4k-dequant-f16a"}}}' -m MODEL
```

## Config-Driven Probes (Required—No Ad-Hoc Logs Allowed)

**Do not add temporary log statements to source files for debugging.** All debugging must use config-driven probes or existing trace categories. This is a strict rule, not a preference.

Use probes to read specific token/dimension values without editing code. Configure via `--config` with inline JSON:

```bash
# Probe specific dimensions at layer_out for layers 0, 17, 25
npm run debug -- -m MODEL --config '{
  "runtime": {
    "inference": { "prompt": "The color of the sky is" },
    "shared": {
      "debug": {
        "trace": { "enabled": true, "categories": ["ffn"] },
        "probes": [
          { "id": "dim334", "stage": "layer_out", "layers": [0, 17, 25], "tokens": [0, -1], "dims": [334] }
        ]
      }
    }
  }
}' 2>&1 | grep -E "PROBE"

# Post-softcap logits probe for HF parity (Gemma 2 finalLogitSoftcapping)
npm run debug -- -m MODEL --config '{
  "runtime": {
    "inference": { "prompt": "The color of the sky is" },
    "shared": {
      "debug": {
        "trace": { "enabled": true, "categories": ["logits"] },
        "probes": [
          { "id": "topk", "stage": "logits_final", "tokens": [-1], "dims": [476, 3868, 17022] }
        ]
      }
    }
  }
}' 2>&1 | grep -E "PROBE"
```

## Tracing Specific Layers

To trace only specific layers, use config:

```bash
# Trace only layers 0, 12, 25
npm run debug -- -m MODEL --config '{
  "runtime": {
    "inference": { "prompt": "The color of the sky is" },
    "shared": {
      "debug": {
        "trace": { "enabled": true, "categories": ["ffn"], "layers": [0, 12, 25] }
      }
    }
  }
}' 2>&1 | grep -E "TRACE"
```

## Fast Iteration Pattern

```bash
# First run: loads model into GPU memory (~30s)
npm run debug -- --config ./trace-ffn.json -m MODEL 2>&1 | sed '/DOPPLER:DONE/q'

# Subsequent runs: reuses model via CDP (much faster, ~5s)
npm run debug -- --config ./trace-ffn.json -m MODEL --skip-load 2>&1 | sed '/DOPPLER:DONE/q'

# After code changes: just re-run (no build step needed)
npm run debug -- --config debug -m MODEL --skip-load 2>&1 | sed '/DOPPLER:DONE/q'

# Keep browser open for multiple runs
npm run debug -- --config debug -m MODEL --warm
# Then in another terminal:
npm run debug -- --config ./trace-kernels.json -m MODEL --skip-load
```

Use `sed '/DOPPLER:DONE/q'` to exit immediately after generation completes.

## Comparing Against HuggingFace Reference

When output differs from expected, compare layer-by-layer:

```bash
# Get HuggingFace reference values for specific layers
python3 src/debug/reference/hf_layer_out.py --model HF_MODEL_NAME --layers 0,12,25

# Then trace DOPPLER at same layers via config
npm run debug -- -m MODEL --config '{
  "runtime": {
    "inference": { "prompt": "The color of the sky is" },
    "shared": {
      "debug": {
        "trace": { "enabled": true, "categories": ["ffn"], "layers": [0, 12, 25] }
      }
    }
  }
}' 2>&1 | grep "LAYER_OUT"
```

Use `logits_final` probes for post-softcap comparisons when models apply finalLogitSoftcapping.
See `src/debug/reference/README.md` for additional HF scripts (attn, rope, weights).

## Resolution Patterns

Once you identify the failure class, apply the appropriate fix:

| Failure Class | Resolution |
|---------------|------------|
| Norm offset missing | Check `rmsNormWeightOffset: true` in model preset |
| Attention scaling wrong | Verify `queryPreAttnScalar` matches HF config (e.g., 256 for Gemma 2) |
| Softcapping disabled | Check `attnLogitSoftcapping` and `finalLogitSoftcapping` in manifest |
| Quantization drift | Re-convert model with `-w f16` to isolate Q4K bugs |
| Sliding window wrong | Check `slidingWindow` and layer pattern in model preset |
| Manifest validation error | Ensure `eos_token_id` and required inference fields are present; re-convert if needed |

## Key Grep Patterns

| Pattern | Purpose |
|---------|---------|
| `"DOPPLER:DONE\|DOPPLER:ERROR"` | Check completion status |
| `"DOPPLER:RESULT"` | Extract full result JSON |
| `"maxAbs\|ANOMALY\|NaN\|Inf"` | Find numerical problems |
| `"TRACE:attn\|TRACE:ffn"` | Filter specific traces |
| `"Error\|error\|ERROR"` | Find error messages |
| `"LAYER_OUT"` | Track hidden state magnitudes per layer |
| `"PROBE"` | Find probe readbacks |

## Reference Files

For detailed information, consult these files:

- **Model configs**: `src/config/presets/models/*.json`
- **Kernel implementations**: `src/gpu/kernels/*.wgsl`
- **Layer processing**: `src/inference/pipeline/layer.js`
- **Attention**: `src/inference/pipeline/attention.js`
- **Logits**: `src/inference/pipeline/logits.js`
- **CLI implementation**: `cli/index.js`
- **Config resolution**: `docs/style/CONFIG_STYLE_GUIDE.md`

## Related Skills

- Use `doppler-benchmark` after fixing to verify no performance regression
- Use `doppler-convert` if you need to re-convert the model with different settings
