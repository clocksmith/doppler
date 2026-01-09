# Target Models for DOPPLER

Priority models to benchmark against WebLLM and establish DOPPLER's competitive position.

## Target Models

| Model | Size | Strategic Value | Architecture |
|-------|------|-----------------|--------------|
| **Llama-3.1-8B-Q4** | ~4.5GB | WebLLM's benchmark star - beat them here | LlamaForCausalLM |
| **Gemma-2-9B-Q4** | ~5GB | Gemma focus, builds on Gemma 3 work | Gemma2ForCausalLM |
| **Phi-3.5-mini-Q4** | ~2GB | Speed king (WebLLM claims 71 tok/s) | Phi3ForCausalLM |

## Architecture Notes

### Llama-3.1-8B
- Standard LLaMA architecture (well-supported)
- GQA (Grouped Query Attention): 8 KV heads for 32 attention heads
- RoPE with theta=500000
- SwiGLU FFN
- 32 layers, hidden_size=4096, intermediate=14336
- **Advantage**: Most widely tested architecture, good baseline

### Gemma-2-9B
- Similar to Gemma 3 but without sliding window
- GQA: 8 KV heads for 16 attention heads
- Logit soft-capping (30.0 for attention, 30.0 for final)
- Pre/post layernorm sandwich structure
- 42 layers, hidden_size=3584, intermediate=14336
- **Advantage**: Builds on our Gemma 3 work, shared norm offset handling

### Phi-3.5-mini
- Microsoft's efficient architecture
- Long context (128K) with rope scaling
- 32 layers, hidden_size=3072, intermediate=8192
- SwiGLU activation
- **Advantage**: Smallest, fastest - good for iteration speed

## Recommended Order

1. **Phi-3.5-mini-Q4** (Start here)
   - Smallest model = fastest iteration
   - Quick feedback loop for Q4K kernel tuning
   - Clear benchmark target (71 tok/s)
   - If we can't beat WebLLM here, we have work to do

2. **Llama-3.1-8B-Q4** (Second)
   - Most standard architecture
   - WebLLM's flagship benchmark
   - Direct competitive comparison

3. **Gemma-2-9B-Q4** (Third)
   - Leverages our Gemma expertise
   - Differentiator from WebLLM (they focus on Llama)

## HuggingFace Sources

```bash
# Phi-3.5-mini (already Q4 available)
huggingface-cli download microsoft/Phi-3.5-mini-instruct

# Llama-3.1-8B (need Q4 quantized or quantize ourselves)
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct

# Gemma-2-9B
huggingface-cli download google/gemma-2-9b-it
```

## Success Metrics

| Model | WebLLM Baseline | Target | Stretch |
|-------|-----------------|--------|---------|
| Phi-3.5-mini-Q4 | 71 tok/s | 75 tok/s | 90 tok/s |
| Llama-3.1-8B-Q4 | ~35 tok/s | 40 tok/s | 50 tok/s |
| Gemma-2-9B-Q4 | ~30 tok/s | 35 tok/s | 45 tok/s |

*Targets assume M1/M2 Mac with 16GB RAM*

## Conversion Commands

```bash
# After downloading, convert to RDRR format:
npx tsx src/converter/node-converter.js \
  ~/.cache/huggingface/hub/models--microsoft--Phi-3.5-mini-instruct/snapshots/<hash>/ \
  models/phi-3.5-mini \
  --quantize q4_k_m

npx tsx src/converter/node-converter.js \
  ~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/<hash>/ \
  models/llama-3.1-8b \
  --quantize q4_k_m
```
