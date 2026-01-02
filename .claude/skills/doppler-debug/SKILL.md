---
name: doppler-debug
description: Debug DOPPLER WebGPU inference issues. Use when investigating model output problems, kernel bugs, hidden state explosions, or decode failures in the browser-based LLM inference engine. (project)
---

# Triage First

| Symptom | Log Pattern | Likely Cause |
|---------|-------------|--------------|
| Gibberish output | Logits diverge from HF | Wrong softcapping, scaling, or norm offset |
| Values exploding | `maxAbs` growing 10x+ per layer | Missing `1+weight` norm offset for Gemma |
| Single dimension spike | Same dim across all layers | Weight corruption or index misalignment |
| NaN/Inf values | `NaN` in trace output | Numerical overflow, check attention scaling |

# Investigation Commands

```bash
# Quick check - exits after generation completes
npm run debug -- -m MODEL 2>&1 | grep -E "Done|Output|Error"

# Trace specific subsystem (attn, ffn, logits, kernels, kv, sample)
npm run debug -- -m MODEL --trace attn 2>&1 | grep -E "TRACE|maxAbs"

# Compare with HuggingFace reference
python3 src/debug/reference/hf_layer_out.py --model HF_MODEL --layers 0,12,25
```

# Fast Iteration

```bash
# First run loads model, subsequent runs reuse GPU memory
npm run debug -- -m MODEL --skip-load --trace ffn 2>&1 | sed '/Done/q'

# After code changes
npm run build && npm run debug -- -m MODEL --skip-load 2>&1 | sed '/Done/q'
```

# Resolution Patterns

- **Norm offset missing**: Check `rmsNormWeightOffset` in model preset
- **Attention scaling wrong**: Verify `queryPreAttnScalar` matches HF config
- **Softcapping disabled**: Check `attnLogitSoftcapping` and `finalLogitSoftcapping`
- **Quantization drift**: Test with `--quantize f16` to isolate Q4K bugs

# Key Grep Patterns

`"maxAbs|ANOMALY|NaN|Inf"` - Problems. `"Config|Done|Output"` - Quick status.
