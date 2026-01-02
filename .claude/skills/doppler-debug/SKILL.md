---
name: doppler-debug
description: Debug DOPPLER WebGPU inference issues such as gibberish output, hidden state explosions, NaN/Inf traces, sampling anomalies, or mismatches vs HuggingFace reference in the browser-based LLM inference engine. Use when investigating correctness of pipeline, kernels, or sampling behavior. (project)
---

# DOPPLER Debug Skill

This skill guides systematic debugging of WebGPU inference issues in DOPPLER.

## Triage First (Do This Before Running Commands)

| Symptom | What to Look For | Likely Root Cause |
|---------|------------------|-------------------|
| Gibberish output | Logits diverge from HuggingFace | Wrong softcapping, scaling, or norm offset |
| Values exploding | `maxAbs` growing 10x+ per layer | Missing `1+weight` norm offset (Gemma models) |
| Single dimension spike | Same dim# exploding across all layers | Weight corruption or index misalignment |
| NaN/Inf values | `NaN` appearing in trace output | Numerical overflow in attention or FFN |
| Repeating tokens | Same token generated repeatedly | KV cache corruption or attention mask bug |
| Wrong token probabilities | Top-k tokens different from HF | Final logit softcapping not applied |
| PAD token wins | `<pad>` selected or repeated | pad_token_id not masked in sampling |

## Quick Diagnostic Commands

```bash
# Step 1: Quick sanity check - does it produce any output?
npm run debug -- -m MODEL 2>&1 | grep -E "Done|Output|Error"

# Step 2: Check for obvious errors in logs
npm run debug -- -m MODEL 2>&1 | grep -E "Error|NaN|Inf|ANOMALY"

# Step 3: Trace specific subsystem based on triage
npm run debug -- -m MODEL --trace CATEGORY 2>&1 | grep -E "TRACE|maxAbs"
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

## Config-Driven Probes (Preferred Over Ad-hoc Logs)

Use probes to read specific token/dimension values without editing code.

```bash
# Example: probe dim 334 at layer_out for layers 0/17/25
npm run debug -- -m MODEL --config '{
  "runtime": {
    "debug": {
      "trace": { "enabled": true, "categories": ["ffn"] },
      "probes": [
        { "id": "dim334", "stage": "layer_out", "layers": [0, 17, 25], "tokens": [0, -1], "dims": [334] }
      ]
    }
  }
}' 2>&1 | grep -E "PROBE"

# Post-softcap logits probe for HF parity (Gemma 2 finalLogitSoftcapping)
npm run debug -- -m MODEL --config '{
  "runtime": {
    "debug": {
      "trace": { "enabled": true, "categories": ["logits"] },
      "probes": [
        { "id": "topk", "stage": "logits_final", "tokens": [-1], "dims": [476, 3868, 17022] }
      ]
    }
  }
}' 2>&1 | grep -E "PROBE"

# Use the preset when iterating on Gemma 2
npm run debug -- --config gemma2-debug -m MODEL 2>&1 | grep -E "PROBE|Output"
```

## Fast Iteration Pattern

```bash
# First run: loads model into GPU memory (~30s)
npm run debug -- -m MODEL --trace ffn 2>&1 | sed '/Done/q'

# Subsequent runs: reuses model, much faster (~5s)
npm run debug -- -m MODEL --skip-load --trace ffn 2>&1 | sed '/Done/q'

# After code changes: rebuild then run
npm run build && npm run debug -- -m MODEL --skip-load 2>&1 | sed '/Done/q'
```

The `sed '/Done/q'` pattern exits immediately after generation completes.

## Comparing Against HuggingFace Reference

When output differs from expected, compare layer-by-layer:

```bash
# Get HuggingFace reference values for specific layers
python3 src/debug/reference/hf_layer_out.py --model HF_MODEL_NAME --layers 0,12,25

# Then trace DOPPLER at same layers
npm run debug -- -m MODEL --trace ffn --trace-layers 0,12,25 2>&1 | grep "LAYER_OUT"
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
| Quantization drift | Re-convert model with `--quantize f16` to isolate Q4K bugs |
| Sliding window wrong | Check `slidingWindow` and layer pattern in model preset |

## Key Grep Patterns

| Pattern | Purpose |
|---------|---------|
| `"maxAbs\|ANOMALY\|NaN\|Inf"` | Find numerical problems |
| `"Config\|Done\|Output"` | Quick status check |
| `"TRACE:attn\|TRACE:ffn"` | Filter specific traces |
| `"Error\|error\|ERROR"` | Find error messages |
| `"LAYER_OUT"` | Track hidden state magnitudes per layer |
| `"PROBE"` | Find probe readbacks |

## Reference Files

For detailed information, consult these files:

- **Model configs**: `src/config/presets/models/*.json`
- **Kernel implementations**: `src/gpu/kernels/*.wgsl`
- **Layer processing**: `src/inference/pipeline/layer.ts`
- **Attention**: `src/inference/pipeline/attention.ts`
- **Logits**: `src/inference/pipeline/logits.ts`
- **Troubleshooting guide**: `docs/DOPPLER-TROUBLESHOOTING.md`

## Related Skills

- Use `doppler-benchmark` after fixing to verify no performance regression
- Use `doppler-convert` if you need to re-convert the model with different settings
