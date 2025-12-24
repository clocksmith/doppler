# The Gemma 3 1B WebGPU Execution Pipeline

This document details the lifecycle of a token in Gemma 3 1B, moving from raw integer IDs to probability distributions. It covers the specific WGSL kernels used, tensor shapes at each step, memory hierarchy, and kernel fusion strategies for running LLMs efficiently in a browser environment.

**Related:** [ARCHITECTURE.md](ARCHITECTURE.md) for module structure, [GLOSSARY.md](GLOSSARY.md) for terminology.

---

## Model Parameters (Gemma 3 1B)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `numLayers` | 26 | Transformer blocks |
| `hiddenSize` | 1152 | Embedding dimension |
| `numHeads` | 4 | Query heads |
| `numKVHeads` | 1 | Key/Value heads (GQA) |
| `headDim` | 256 | `query_pre_attn_scalar` |
| `intermediateSize` | 6912 | FFN hidden (6x) |
| `vocabSize` | 262144 | Token vocabulary |
| `maxSeqLen` | 32768 | Context window |
| `slidingWindow` | 512 | Local attention window |
| `ropeTheta` | 1000000 | RoPE base frequency |
| `rmsNormEps` | 1e-6 | Normalization epsilon |
| `hiddenActivation` | gelu | GELU activation (not SiLU) |

---

## Part I: The Execution Pipeline

Gemma 3 1B utilizes a **mixed-precision architecture**. To maximize memory bandwidth (the primary bottleneck in LLM inference), weights are stored and loaded in `f16`. However, to prevent numerical underflow, all accumulations (dot products, reductions) occur in `f32`.

---

### Phase 1: Embedding

**Kernel:** `gather.wgsl`

The pipeline begins by converting discrete token IDs into dense vectors.

**What happens:**

```
Input:  token_ids[seq_len]           e.g., [1, 15043, 2845, ...] for 512 tokens
        embedding_table[262144, 1152]

For each token position i in [0, seq_len):
    token_id = token_ids[i]
    For each hidden dimension d in [0, 1151]:
        output[i, d] = embedding_table[token_id * 1152 + d]

Output: embeddings[seq_len, 1152]
```

The embedding table is `[262144, 1152]` - 262K vocabulary entries, each a 1152-dimensional vector. The kernel performs a parallel lookup, with each thread handling one output element.

**Gemma-specific scaling:** Unlike Llama, Gemma stabilizes signal propagation at the start by scaling embeddings:

```
scaled_output[i, d] = embeddings[i, d] * sqrt(1152)
```

The constant is approximately 33.94. This prevents the variance of embedding vectors from shrinking as they pass through deep layers.

**Dispatch:** `ceil((seq_len * 1152) / 256)` workgroups of 256 threads.

---

### Phase 2: Transformer Layers (x26)

The core computation happens in 26 identical stacked layers. Each layer transforms the hidden state `[seq_len, 1152]` while preserving its shape.

---

#### 2.1 Input RMSNorm

**Kernel:** `rmsnorm.wgsl`

RMSNorm normalizes each token's hidden state independently. Unlike LayerNorm, it doesn't subtract the mean - just divides by root-mean-square. This is computationally cheaper.

**What happens:**

```
Input:  hidden[seq_len, 1152]
        norm_weight[1152]

For each token position i in [0, seq_len):

    # Compute RMS of this token's hidden state
    sum_of_squares = 0
    For d in [0, 1151]:
        sum_of_squares += hidden[i, d] * hidden[i, d]

    rms = sqrt(sum_of_squares / 1152 + 1e-6)  # epsilon for stability

    # Normalize and apply learned weight
    For d in [0, 1151]:
        output[i, d] = (hidden[i, d] / rms) * norm_weight[d]

Output: normed[seq_len, 1152]
```

**Workgroup strategy:** 256 threads collaborate via shared memory to compute `sum_of_squares` using parallel reduction. Each workgroup handles one token entirely.

**Dispatch:** `seq_len` workgroups (one per token).

---

#### 2.2 Q/K/V Projections

**Kernel:** `matmul_f16w_f32a.wgsl` (three separate dispatches)

Three matrix multiplications project the normalized hidden states into query, key, and value spaces. Gemma 3 uses **Grouped Query Attention (GQA)** where 4 query heads share 1 KV head.

**What happens:**

```
Input:  normed[seq_len, 1152]
        q_weight[1024, 1152]  (F16)  # 4 heads × 256 headDim
        k_weight[256, 1152]   (F16)  # 1 KV head × 256 headDim
        v_weight[256, 1152]   (F16)  # 1 KV head × 256 headDim

# Q Projection: [seq_len, 1152] @ [1024, 1152]^T -> [seq_len, 1024]
For each output position (row i, col j):
    accumulator = 0.0  (F32 for precision)
    For k in [0, 1151]:
        w = f16_to_f32(q_weight[j, k])
        a = normed[i, k]
        accumulator += w * a
    Q[i, j] = accumulator

# K Projection: [seq_len, 1152] @ [256, 1152]^T -> [seq_len, 256]
# V Projection: [seq_len, 1152] @ [256, 1152]^T -> [seq_len, 256]
# (Same algorithm, smaller output dimension due to GQA)

Output: Q[seq_len, 1024], K[seq_len, 256], V[seq_len, 256]
```

**Why GQA?** The 4:1 ratio (4 Q heads, 1 KV head) reduces KV cache size by 4x, allowing longer context windows on consumer hardware.

**Tiling strategy:** The kernel uses 8x128 output tiles. Workgroups cooperatively load F16 weight tiles into shared memory to maximize data reuse before computing F32 dot products.

**Dispatch:** `ceil(N/128) x ceil(M/8)` workgroups for each projection.

---

#### 2.2.1 Q/K Normalization (Gemma 3 Specific)

**Kernel:** `rmsnorm.wgsl` (two dispatches)

Gemma 3 applies RMSNorm to Q and K tensors **before** RoPE. This stabilizes attention scores and improves training convergence.

**What happens:**

```
Input:  Q[seq_len, 1024]
        K[seq_len, 256]
        q_norm_weight[256]  # Per-head normalization
        k_norm_weight[256]

# For each query head h in [0, 3]:
For each token i:
    head_start = h * 256
    rms_q = sqrt(mean(Q[i, head_start:head_start+256]^2) + eps)
    Q[i, head_start:head_start+256] /= rms_q
    Q[i, head_start:head_start+256] *= q_norm_weight[:]

# For K (single KV head):
For each token i:
    rms_k = sqrt(mean(K[i, :]^2) + eps)
    K[i, :] /= rms_k
    K[i, :] *= k_norm_weight[:]

Output: Q[seq_len, 1024] (normalized), K[seq_len, 256] (normalized)
```

**Why Q/K Norm?** Without normalization, Q·K dot products can explode as sequence length grows. This is especially important for Gemma 3's large context window (32K tokens).

---

#### 2.3 Rotary Position Embeddings (RoPE)

**Kernel:** `rope.wgsl` (two dispatches: Q and K)

Since Transformers process all tokens simultaneously, they have no inherent concept of order. RoPE encodes position by rotating pairs of dimensions in complex space.

**What happens:**

```
Input:  Q[seq_len, 1024], K[seq_len, 256]
        rope_cos[max_seq_len, 128]  (precomputed)
        rope_sin[max_seq_len, 128]  (precomputed)

For each token position pos in [0, seq_len):
    For each head h in [0, num_heads):  # Q: 4 heads, K: 1 head
        For each dimension pair d in [0, 127]:  # 256 dims = 128 pairs

            # Lookup precomputed frequencies for this position
            cos_theta = rope_cos[pos, d]
            sin_theta = rope_sin[pos, d]

            # Get the two values to rotate
            idx = h * 256 + d * 2
            x0 = Q[pos, idx]
            x1 = Q[pos, idx + 1]

            # Apply 2D rotation matrix (in-place)
            Q[pos, idx]     = x0 * cos_theta - x1 * sin_theta
            Q[pos, idx + 1] = x0 * sin_theta + x1 * cos_theta

Output: Q[seq_len, 1024] (rotated), K[seq_len, 256] (rotated)
```

**The math:** Pairs of dimensions `(x, y)` are rotated by angle theta. Low dimensions rotate slowly (low frequency), high dimensions rotate quickly (high frequency). This frequency differential allows attention to measure relative distances between tokens through dot product patterns.

**Precomputation:** Frequencies are computed once at initialization:
```
For position m, dimension pair d:
    theta = rope_theta^(-2d/head_dim)  # rope_theta = 1,000,000 for Gemma 3
    rope_cos[m, d] = cos(m * theta)
    rope_sin[m, d] = sin(m * theta)
```

**Gemma 3 RoPE:** Uses a very high base frequency (1,000,000 vs typical 10,000) which enables better extrapolation to longer sequences.

---

#### 2.4 KV Cache Update

**Operation:** GPU buffer copy (+ optional F16 cast kernel)

Before attention, the rotated K and V vectors are appended to the KV cache for this layer.

```
# Prefill: Write all positions at once
kv_cache.keys[layer_idx, 0:seq_len, :] = K_rotated[0:seq_len, :]  # [seq_len, 256]
kv_cache.vals[layer_idx, 0:seq_len, :] = V_rotated[0:seq_len, :]  # [seq_len, 256]

# Decode: Append one position
kv_cache.keys[layer_idx, current_pos, :] = K_rotated[0, :]  # [256]
kv_cache.vals[layer_idx, current_pos, :] = V_rotated[0, :]  # [256]
```

**F16 casting:** If using F16 KV cache to save VRAM, `cast_f32_to_f16.wgsl` runs first. This halves KV cache memory, critical for Gemma 3's 32K context.

**Sliding window:** Gemma 3 uses a sliding window pattern (`slidingWindow: 512`) for most layers. Only every 6th layer attends to the full context.

---

#### 2.5 Attention

**Kernel:** `attention.wgsl` (tiled) or `attention_streaming.wgsl` (for large headDim)

This is the most complex kernel in the pipeline. It computes scaled dot-product attention: `Softmax(QK^T / sqrt(d)) @ V` with causal masking.

**Gemma 3 headDim (256):** With headDim=256, Gemma 3 can use efficient **tiled attention** (unlike older Gemma 1B which needed streaming due to headDim=384).

**What happens:**

```
Input:  Q[seq_len, 1024]          # 4 heads x 256 dims
        K_cache[kv_len, 256]      # 1 KV head x 256 dims
        V_cache[kv_len, 256]
        scale = 1/sqrt(256) = 0.0625

For each query position q_pos in [0, seq_len):
    For each query head h in [0, 3]:

        # Initialize online softmax tracking
        max_score = -infinity
        sum_exp = 0.0
        accumulator[256] = zeros

        # Determine attention window (sliding or full)
        if layer_idx % 6 == 0:
            window_start = 0  # Full attention every 6th layer
        else:
            window_start = max(0, q_pos - 512)  # Sliding window

        # Stream through key positions in window (causal: only up to q_pos)
        For kv_pos in [window_start, q_pos]:  # Causal mask + sliding window

            # Compute attention score: Q dot K / sqrt(d)
            score = 0.0
            For d in [0, 255]:
                score += Q[q_pos, h*256 + d] * K_cache[kv_pos, d]
            score = score * scale

            # Online softmax update (Flash Attention style)
            new_max = max(max_score, score)

            # Rescale previous accumulator for new max
            scale_old = exp(max_score - new_max)
            accumulator = accumulator * scale_old
            sum_exp = sum_exp * scale_old

            # Add this position's contribution
            weight = exp(score - new_max)
            sum_exp += weight
            For d in [0, 255]:
                accumulator[d] += weight * V_cache[kv_pos, d]

            max_score = new_max

        # Normalize by sum of exponentials
        For d in [0, 255]:
            output[q_pos, h*256 + d] = accumulator[d] / sum_exp

Output: attn_output[seq_len, 1024]
```

**Sliding window attention:** Most layers only attend to the last 512 tokens. Every 6th layer (0, 6, 12, 18, 24) attends to the full context. This reduces memory and compute while preserving long-range information.

**GQA handling:** All 4 query heads attend to the same single KV head. The kernel broadcasts K/V across query heads.

**Dispatch:** `seq_len * num_heads` workgroups for prefill (e.g., 512 * 4 = 2048).

---

#### 2.6 Output Projection

**Kernel:** `matmul_f16w_f32a.wgsl`

Projects attention output back to hidden dimension.

**What happens:**

```
Input:  attn_output[seq_len, 1024]
        o_proj_weight[1152, 1024]  (F16)

# [seq_len, 1024] @ [1152, 1024]^T -> [seq_len, 1152]
For each output position (i, j):
    output[i, j] = sum over k: attn_output[i, k] * o_proj_weight[j, k]

Output: proj_output[seq_len, 1152]
```

Same tiled matmul as Q/K/V projections.

---

#### 2.7 Residual Connection (Attention)

**Kernel:** `residual.wgsl`

The skip connection that makes deep networks trainable.

**What happens:**

```
Input:  attn_output[seq_len, 1152]
        layer_input[seq_len, 1152]  (before input norm)

For each element (i, j):
    output[i, j] = attn_output[i, j] + layer_input[i, j]

Output: post_attn[seq_len, 1152]
```

This creates the residual stream - information flows through additions rather than transformations, preserving gradient flow.

---

#### 2.8 Post-Attention RMSNorm (Sandwich Norm)

**Kernel:** `rmsnorm.wgsl`

**Gemma 3 specific:** Uses "sandwich norm" architecture with a **separate** post-attention norm (not shared with input norm). This improves training stability.

```
Input:  post_attn[seq_len, 1152]
        post_attn_norm_weight[1152]  # Separate learned weights

For each token i:
    rms = sqrt(mean(post_attn[i, :]^2) + eps)
    output[i, :] = (post_attn[i, :] / rms) * post_attn_norm_weight[:]

Output: normed_post_attn[seq_len, 1152]
```

---

#### 2.9 FFN Gate+Up Projection (Fused)

**Kernel:** `matmul_f16w_f32a.wgsl`

**Gemma 3 uses fused gate_up weights** - a single matmul produces both gate and up projections concatenated together. This is more efficient than two separate matmuls.

```
Input:  ffn_input[seq_len, 1152]
        gate_up_weight[13824, 1152]  # 6912 * 2 = 13824

# [seq_len, 1152] @ [13824, 1152]^T -> [seq_len, 13824]
# Output contains [gate, up] concatenated: first 6912 dims are gate, last 6912 are up
gate_up_output[i, j] = sum over k: ffn_input[i, k] * gate_up_weight[j, k]

Output: gate_up[seq_len, 13824]
```

**Why fused?** One matmul instead of two reduces kernel launch overhead and memory traffic.

---

#### 2.10 GELU Activation + Row Split

**Kernel:** `silu.wgsl` → `runSiLURowSplit` (with GELU mode)

**Gemma 3 uses GELU activation** (not SiLU like LLaMA). The fused kernel splits the gate_up output, applies GELU to the gate portion, and multiplies with up.

**What happens:**

```
Input:  gate_up[seq_len, 13824]  # [gate, up] concatenated

For each token i:
    # Split the row in half
    gate = gate_up[i, 0:6912]
    up = gate_up[i, 6912:13824]

    For each element j in [0, 6911]:
        x = gate[j]

        # GELU approximation (tanh variant used by Gemma 3)
        gelu_x = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

        # Multiply by up (gated linear unit)
        output[i, j] = gelu_x * up[j]

Output: activated[seq_len, 6912]
```

**Why GELU?** Gemma 3 uses `gelu_pytorch_tanh` activation which provides smoother gradients than SiLU, improving training stability.

---

#### 2.11 FFN Down Projection

**Kernel:** `matmul_f16w_f32a.wgsl`

Projects back from intermediate size to hidden size.

```
Input:  activated[seq_len, 6912]
        down_weight[1152, 6912]

# [seq_len, 6912] @ [1152, 6912]^T -> [seq_len, 1152]
output[i, j] = sum over k: activated[i, k] * down_weight[j, k]

Output: ffn_output[seq_len, 1152]
```

---

#### 2.12 Residual Connection (FFN)

**Kernel:** `residual.wgsl`

Add FFN output to residual stream.

```
For each element (i, j):
    output[i, j] = ffn_output[i, j] + normed_post_attn[i, j]

Output: layer_output[seq_len, 1152]
```

This output becomes the input to the next layer (or final norm if this is layer 25).

---

### Phase 3: Final Norm

**Kernel:** `rmsnorm.wgsl`

After all 26 layers complete:

```
Input:  hidden[seq_len, 1152]  (output of layer 25)
        final_norm_weight[1152]

For each token i:
    rms = sqrt(mean(hidden[i, :]^2) + eps)
    normed[i, :] = (hidden[i, :] / rms) * final_norm_weight[:]

Output: final_normed[seq_len, 1152]
```

---

### Phase 4: LM Head (Vocabulary Projection)

**Kernel:** `matmul_f16w_f32a.wgsl`

The largest single matmul - projects to vocabulary size.

```
Input:  final_normed[seq_len, 1152]
        lm_head_weight[262144, 1152]

# [seq_len, 1152] @ [262144, 1152]^T -> [seq_len, 262144]
For each position i, vocab index v:
    logits[i, v] = sum over k: final_normed[i, k] * lm_head_weight[v, k]

Output: logits[seq_len, 262144]
```

This is often the most bandwidth-intensive single operation due to the massive 262K vocabulary.

---

### Phase 5: Sampling

**CPU operation** (or optional `sample.wgsl` for GPU sampling)

```
# Extract last position's logits (for next token prediction)
last_logits = logits[seq_len - 1, :]  # [262144]

# Apply temperature scaling
last_logits = last_logits / temperature

# Apply top-k filtering
top_k_indices = argsort(last_logits, descending=True)[:k]
mask all other indices to -infinity

# Apply top-p (nucleus) filtering
sorted_probs = softmax(last_logits[top_k_indices])
cumulative = cumsum(sorted_probs)
cutoff = first index where cumulative > p
mask indices after cutoff

# Convert to probabilities
probs = softmax(last_logits)

# Sample from categorical distribution
next_token_id = multinomial_sample(probs)
```

---

## Decode Phase (Token-by-Token Generation)

After prefill, each new token uses the same kernels but optimized for single-token processing.

### Key Differences from Prefill

| Aspect | Prefill | Decode |
|--------|---------|--------|
| Input shape | `[seq_len, ...]` | `[1, ...]` |
| Matmul kernel | `matmul_f16w_f32a.wgsl` | `matmul_gemv.wgsl` |
| Attention | Process all positions | Read KV cache (sliding window), add one |
| RoPE position | 0 to seq_len-1 | current_position |
| Parallelism | High (many tokens) | Low (one token) |

### GEMV Optimization

**Kernel:** `matmul_gemv.wgsl` (or `matmul_gemv_subgroup.wgsl`)

For single-token decode, matrix-vector multiply (GEMV) is 5-10x faster than batched matmul:

```
# GEMV: [1, K] @ [K, N] -> [1, N]
# Instead of tiling, use full vector parallelism

For each output dimension j (in parallel):
    accumulator = 0.0
    For k in [0, K):
        accumulator += input[k] * weight[j, k]
    output[j] = accumulator
```

A single workgroup of 256 threads can handle the entire operation, with each thread computing one or more output elements.

### Decode Attention (with Sliding Window)

During decode, attention reads from the growing KV cache, but most layers only attend to the last 512 positions:

```
# Prefill filled positions [0, prefill_len)
# Decode step N reads from cache and appends one K/V

For decode step:
    kv_len = prefill_len + decode_step

    # Determine window (most layers use sliding window)
    if layer_idx % 6 == 0:
        window_start = 0  # Full attention
    else:
        window_start = max(0, kv_len - 512)  # Sliding window

    # Attention over window
    For kv_pos in [window_start, kv_len):
        score = Q[0] dot K_cache[kv_pos]
        # ... accumulate weighted V

    # Append new K/V at position kv_len
    K_cache[kv_len] = K_new
    V_cache[kv_len] = V_new
```

**Sliding window benefit:** Decode attention is O(1) for most layers (fixed 512 window) instead of O(N).

---

## Kernel Summary Tables

### Prefill Kernels Per Layer (seq_len=512, Gemma 3 1B)

| # | Operation | Kernel | Input Shape | Output Shape |
|---|-----------|--------|-------------|--------------|
| 1 | Input Norm | `rmsnorm.wgsl` | [512, 1152] | [512, 1152] |
| 2 | Q Projection | `matmul_f16w_f32a.wgsl` | [512, 1152] | [512, 1024] |
| 3 | K Projection | `matmul_f16w_f32a.wgsl` | [512, 1152] | [512, 256] |
| 4 | V Projection | `matmul_f16w_f32a.wgsl` | [512, 1152] | [512, 256] |
| 5 | Q Norm | `rmsnorm.wgsl` | [512, 1024] | [512, 1024] |
| 6 | K Norm | `rmsnorm.wgsl` | [512, 256] | [512, 256] |
| 7 | RoPE (Q) | `rope.wgsl` | [512, 1024] | [512, 1024] |
| 8 | RoPE (K) | `rope.wgsl` | [512, 256] | [512, 256] |
| 9 | KV F16 Cast | `cast_f32_to_f16.wgsl` | K, V | K_f16, V_f16 |
| 10 | Attention | `attention.wgsl` | Q, K, V | [512, 1024] |
| 11 | O Projection | `matmul_f16w_f32a.wgsl` | [512, 1024] | [512, 1152] |
| 12 | Residual | `residual.wgsl` | [512, 1152] x2 | [512, 1152] |
| 13 | Post-Attn Norm | `rmsnorm.wgsl` | [512, 1152] | [512, 1152] |
| 14 | Gate+Up Proj | `matmul_f16w_f32a.wgsl` | [512, 1152] | [512, 13824] |
| 15 | GELU + Split | `silu.wgsl` (GELU mode) | [512, 13824] | [512, 6912] |
| 16 | Down Projection | `matmul_f16w_f32a.wgsl` | [512, 6912] | [512, 1152] |
| 17 | Residual | `residual.wgsl` | [512, 1152] x2 | [512, 1152] |

**Total:** 17 kernels/layer × 26 layers = **442 dispatches** (+ embed + final norm + LM head)

### Decode Kernels Per Layer

| # | Operation | Kernel | Input Shape | Output Shape |
|---|-----------|--------|-------------|--------------|
| 1 | Input Norm | `rmsnorm.wgsl` | [1, 1152] | [1, 1152] |
| 2 | Q Projection | `matmul_gemv.wgsl` | [1, 1152] | [1, 1024] |
| 3 | K Projection | `matmul_gemv.wgsl` | [1, 1152] | [1, 256] |
| 4 | V Projection | `matmul_gemv.wgsl` | [1, 1152] | [1, 256] |
| 5 | Q Norm | `rmsnorm.wgsl` | [1, 1024] | [1, 1024] |
| 6 | K Norm | `rmsnorm.wgsl` | [1, 256] | [1, 256] |
| 7 | RoPE | `rope.wgsl` | Q, K | Q, K |
| 8 | KV F16 Cast | `cast_f32_to_f16.wgsl` | K, V | K_f16, V_f16 |
| 9 | Attention | `attention.wgsl` | Q + cache | [1, 1024] |
| 10 | O Projection | `matmul_gemv.wgsl` | [1, 1024] | [1, 1152] |
| 11 | Residual | `residual.wgsl` | [1, 1152] x2 | [1, 1152] |
| 12 | Post-Attn Norm | `rmsnorm.wgsl` | [1, 1152] | [1, 1152] |
| 13 | Gate+Up Proj | `matmul_gemv.wgsl` | [1, 1152] | [1, 13824] |
| 14 | GELU + Split | `silu.wgsl` | [1, 13824] | [1, 6912] |
| 15 | Down Projection | `matmul_gemv.wgsl` | [1, 6912] | [1, 1152] |
| 16 | Residual | `residual.wgsl` | [1, 1152] x2 | [1, 1152] |

---

## Part II: Kernel Fusion Analysis

Kernel fusion combines multiple kernels into one to reduce **memory bandwidth pressure**. In WebGPU, launching a kernel has overhead, and reading/writing to global memory (VRAM) is slow. Fusion keeps data in fast L1 cache or registers.

### Deterministic Fusion (The Easy Wins)

Fusion is "deterministic" (mechanically automatable) when two kernels are **element-wise** and share the same **iteration pattern**.

| Component A | Component B | Fused Kernel | Bandwidth Saved |
|-------------|-------------|--------------|-----------------|
| `gather` | `scale` | `gather_scaled` | 1 write eliminated |
| `silu` | `multiply` | `silu_gate` | 1 read + 1 write eliminated |
| `add` | `rmsnorm` | `rmsnorm_residual` | 1 read + 1 write eliminated |

**The transformation:**

```wgsl
// UNFUSED: 3 memory operations
temp[i] = silu(input[i]);        // Read input, write temp
output[i] = temp[i] * gate[i];   // Read temp + gate, write output

// FUSED: 2 memory operations
output[i] = silu(input[i]) * gate[i];  // Read input + gate, write output
```

The compiler (or human) inlines Kernel A into Kernel B. DOPPLER already ships fused variants like `silu_gate.wgsl`.

### The Limits of Fusion

Fusion breaks down when mathematical structure conflicts.

#### Problem 1: Iteration Mismatch (Reduction vs. Tiling)

**RMSNorm -> Matmul cannot fuse:**

- **RMSNorm** operates row-by-row. It must read the *entire* row (1152 floats for Gemma 3) to compute the mean before writing any output.
- **Matmul** operates on tiles. It calculates small blocks (e.g., 8x8) of the output.

**The conflict:** To fuse, matmul would wait for RMSNorm to finish the whole row, destroying the pipelining and cache strategies that make matmul fast. The fused version would be slower.

#### Problem 2: Synchronization Boundaries

**Attention -> O Projection cannot fuse:**

- Attention output is a reduction (weighted sum over all KV positions). The final value of `attention[i]` isn't known until all KV positions are processed.
- O Projection cannot start until attention finishes.

This "hard barrier" forces a write to global memory, preventing fusion.

#### Problem 3: Hardware Constraints

Even when logic permits, hardware may forbid fusion.

**Shared memory limits:**
```
Kernel A: needs 16KB shared memory for tiling
Kernel B: needs 24KB shared memory for tiling
Fused:    needs 40KB... but GPU limit is 32KB

Result: Fused kernel crashes or runs in "slow mode"
```

**Register pressure:**
```
Kernel A: uses 32 registers per thread
Kernel B: uses 48 registers per thread
Fused:    needs 80 registers per thread

If GPU limit is 64 registers: occupancy drops, performance tanks
```

### The DOPPLER Strategy

DOPPLER takes a pragmatic approach:

1. **Fused FFN:** Gemma 3's `gate_up_proj` weights enable a 2-matmul FFN path with fused `runSiLURowSplit` (or GELU mode), reducing kernel launches and memory traffic.

2. **Streaming Attention:** Instead of fusing attention with adjacent layers, DOPPLER fuses *within* attention. The **Matmul (QK)**, **Softmax**, and **Matmul (V)** are combined into a single streaming pass. This avoids materializing the `[N, N]` attention score matrix - the biggest optimization in modern LLM inference.

3. **Command batching:** All ~442 kernel dispatches (26 layers × 17 kernels) are batched into 1-2 GPU command buffer submissions via `CommandRecorder`, reducing driver overhead.

### Fusion Viability Matrix

| Kernel Pair | Fusable? | Reason |
|-------------|----------|--------|
| **Element -> Element** | Yes | Deterministic; saves bandwidth |
| **Element -> Reduction** | Partial | Can inline element-op into reduction's load phase |
| **Reduction -> Element** | No | Reduction output is a sync barrier |
| **Reduction -> Reduction** | No | Dependency mismatch |
| **Tiled -> Tiled** | Rarely | Index math complexity makes fusion slower |

### Theoretical Framework

Automatic fusion is studied in **polyhedral compilation** and **Halide-style scheduling**:

1. Express each kernel as a pure function over index space
2. Analyze data dependencies between kernels
3. Search for a fused schedule respecting dependencies while maximizing locality

```
# Halide-like representation (Gemma 3 1B: hiddenSize=1152)
embed(i, d) = embedding[tokens[i], d]
scaled(i, d) = embed(i, d) * sqrt(1152)
normed(i, d) = scaled(i, d) / rms(i) * weight[d]
  where rms(i) = sqrt(sum_d(scaled(i, d)^2) / 1152)

# Compiler analysis:
# - embed -> scaled: FUSABLE (element-wise)
# - scaled -> normed: NOT FUSABLE (rms depends on ALL d values)
```

**TVM**, **Triton**, and **XLA** implement variants of this. However, they often produce suboptimal code compared to hand-tuned kernels because:

1. Search space is exponential
2. Hardware details (cache sizes, warp scheduling) aren't fully modeled
3. Some optimizations require algorithmic insight (Flash Attention)

---

## Kernel Variant Selection

DOPPLER dynamically selects kernel variants based on GPU capabilities.

### Device Capability Detection

```typescript
{
  hasF16: boolean,           // shader-f16 extension
  hasSubgroups: boolean,     // subgroups extension
  maxComputeWorkgroupStorageSize: number,  // shared memory limit
  maxComputeInvocationsPerWorkgroup: number,
}
```

### Matmul Variants

| Condition | Kernel | Notes |
|-----------|--------|-------|
| F32 acts + F16 weights + hasF16 | `matmul_f16w_f32a.wgsl` | Default for Gemma |
| Both F16 + hasF16 | `matmul_f16.wgsl` | Full F16 path |
| All F32 (fallback) | `matmul_f32.wgsl` | Compatibility mode |
| M=1 (decode) + hasSubgroups | `matmul_gemv_subgroup.wgsl` | Fastest decode |
| M=1 (decode) | `matmul_gemv.wgsl` | Decode fallback |
| Q4_K quantized | `matmul_q4_fused.wgsl` | Quantized weights |

### Attention Variants

| Condition | Kernel | Notes |
|-----------|--------|-------|
| headDim <= 64, shared >= 49KB | `attention.wgsl` | Tiled (fastest) |
| headDim <= 256, shared >= 4KB | `attention_small.wgsl` | Small tiled |
| headDim = 256 (Gemma 3) | `attention_small.wgsl` | Small tiled (Gemma 3 default) |
| + F16 KV cache | `*_f16kv` variants | Half memory |

---

## Memory Footprint

### Prefill (seq_len=512, Gemma 3 1B)

| Buffer | Size | Notes |
|--------|------|-------|
| Hidden states | 512 × 1152 × 4 = 2.3 MB | Reused per layer |
| Q | 512 × 1024 × 4 = 2.0 MB | Temporary |
| K/V | 512 × 256 × 4 × 2 = 1.0 MB | Temporary → F16 for cache |
| Attention output | 512 × 1024 × 4 = 2.0 MB | Temporary |
| FFN gate_up | 512 × 13824 × 4 = 27.6 MB | Largest activation |
| FFN activated | 512 × 6912 × 4 = 13.8 MB | Post-split |
| KV cache (26 layers) | 26 × 512 × 256 × 2 × 2 = 13.6 MB (F16) | Persistent |

### Decode (per step)

| Buffer | Size | Notes |
|--------|------|-------|
| Hidden states | 1 × 1152 × 4 = 4.5 KB | Minimal |
| Q/K/V | ~6 KB | Single vectors |
| FFN intermediate | 1 × 13824 × 4 = 54 KB | Single token (largest) |
| KV cache append | 256 × 2 × 2 = 1 KB/layer | Growing (F16) |

---

## Performance Characteristics

### Bottlenecks by Phase

| Phase | Bottleneck | Why |
|-------|------------|-----|
| Prefill | Compute | Parallel token processing saturates ALUs |
| Decode | Memory bandwidth | Single token can't saturate compute |
| LM Head | Memory bandwidth | Reading 262K × 1152 weight matrix |
| Attention (long ctx) | Memory bandwidth | KV cache grows with context |

### Optimization Impact

| Optimization | Speedup | Implemented |
|--------------|---------|-------------|
| Mixed precision (F16 weights) | ~2x | Yes |
| GEMV for decode | 5-10x vs batched matmul | Yes |
| Streaming attention | Enables large headDim | Yes |
| F16 KV cache | 2x cache capacity | Yes |
| Command batching | ~20% latency reduction | Yes |
| Fused GELU+Gate | ~15% FFN speedup | Yes (Gemma 3 uses GELU) |

---

## Part III: Capability-Based Kernel Selection

DOPPLER dynamically selects kernel variants at runtime based on GPU capabilities and model configuration. This section documents the complete selection pipeline.

### GPU Capability Detection

At initialization (`gpu/device.ts`), DOPPLER probes the WebGPU adapter for available features:

```typescript
interface KernelCapabilities {
  hasF16: boolean;              // shader-f16 extension
  hasSubgroups: boolean;        // subgroups extension (shuffle ops)
  hasSubgroupsF16: boolean;     // subgroups-f16 (combined)
  hasTimestampQuery: boolean;   // GPU profiling
  maxBufferSize: number;        // Max storage buffer (bytes)
  maxWorkgroupSize: number;     // Max threads per workgroup
  maxWorkgroupStorageSize: number;  // Shared memory limit (bytes)
  adapterInfo: {
    vendor: string;             // "apple", "nvidia", "amd", etc.
    architecture: string;       // "common-3", "ampere", etc.
    device: string;             // GPU model name
  };
}
```

**Feature Detection Flow:**

```
navigator.gpu.requestAdapter()
       │
       ▼
Probe adapter.features for:
  - 'shader-f16'      → enables F16 matmul, F16 KV cache
  - 'subgroups'       → enables subgroup shuffle reductions
  - 'timestamp-query' → enables GPU-side profiling
       │
       ▼
Request device with detected features
       │
       ▼
Cache capabilities in kernelCapabilities global
```

### Kernel Configuration Schema

All kernel variants are defined in `gpu/kernels/utils.ts` as `KERNEL_CONFIGS`:

```typescript
interface KernelConfig {
  shaderFile: string;                        // WGSL file name
  entryPoint: string;                        // Function to call
  workgroupSize: [number, number, number];   // Default workgroup dims
  requires: string[];                        // Required GPU features
  validate?: (seqLen, numHeads, headDim) => void;  // Optional limits check
}
```

**Example - Matmul Variants:**

| Variant | Shader File | Requirements | Use Case |
|---------|-------------|--------------|----------|
| `f32` | `matmul_f32.wgsl` | none | Fallback for all GPUs |
| `f16` | `matmul_f16.wgsl` | `shader-f16` | Both inputs F16 |
| `f16w_f32a` | `matmul_f16w_f32a.wgsl` | `shader-f16` | F16 weights, F32 activations |
| `gemv` | `matmul_gemv.wgsl` | `shader-f16` | M=1 decode, basic |
| `gemv_subgroup` | `matmul_gemv_subgroup.wgsl` | `shader-f16`, `subgroups` | M=1 decode, optimized |
| `q4_fused` | `matmul_q4_fused.wgsl` | `shader-f16`, `subgroups` | Fused Q4_K dequant+matmul |

### Selection Decision Trees

#### Matmul Kernel Selection

```
selectMatmulKernel(aDtype, bDtype, M, outputDtype)
       │
       ├── bDtype == 'q4k'?
       │       │
       │       ├── M == 1 → 'q4_fused' (GEMV, fused dequant)
       │       └── M > 1  → 'q4_fused_batched' (tiled, fused dequant)
       │
       ├── M == 1 && bDtype == 'f16' && aDtype == 'f32'?
       │       │
       │       ├── hasSubgroups → 'gemv_subgroup' (1.5x faster)
       │       └── else         → 'gemv'
       │
       ├── aDtype == 'f16' && bDtype == 'f16' && hasF16?
       │       │
       │       └── outputDtype == 'f16' → 'f16'
       │
       ├── bDtype == 'f16' && aDtype == 'f32' && hasF16?
       │       │
       │       └── → 'f16w_f32a' (mixed precision)
       │
       └── else → 'f32' (universal fallback)
```

#### Attention Kernel Selection

```
selectAttentionKernel(headDim, kvDtype, phase)
       │
       ├── headDim <= 64 && sharedMem >= 49KB?
       │       │
       │       └── Tiled attention (fastest, fits in shared memory)
       │           ├── kvDtype == 'f16' → 'prefill_f16kv' / 'decode_f16kv'
       │           └── else             → 'prefill' / 'decode'
       │
       ├── headDim <= 256 && sharedMem >= 4KB?
       │       │
       │       └── Small tiled attention
       │           ├── kvDtype == 'f16' → 'prefill_small_f16kv' / 'decode_small_f16kv'
       │           └── else             → 'prefill_small' / 'decode_small'
       │
       └── else (headDim > 256)
               │
               └── Streaming attention (processes KV in blocks)
                   ├── kvDtype == 'f16' → 'prefill_streaming_f16kv' / 'decode_streaming_f16kv'
                   └── else             → 'prefill_streaming' / 'decode_streaming'

Note: Gemma 3 1B has headDim=256, so it uses the small tiled attention path.
```

#### Dequantization Kernel Selection

```
selectDequantKernel(outputDtype)
       │
       ├── hasSubgroups && outputDtype == 'f16' && hasF16?
       │       │
       │       └── 'subgroup_f16out' (fastest)
       │
       ├── hasSubgroups?
       │       │
       │       └── 'subgroup' (uses shuffle for reduction)
       │
       ├── outputDtype == 'f16' && hasF16?
       │       │
       │       └── 'shared_f16out'
       │
       └── else → 'shared' (universal fallback)
```

### Model Config → Kernel Mapping

Model architecture parameters directly influence kernel selection:

| Model Parameter | Kernel Impact |
|-----------------|---------------|
| `headDim` | Determines attention tier (tiled vs streaming) |
| `numKVHeads` | Affects KV cache size, F16 cache viability |
| `intermediateSize` | FFN matmul dimensions |
| `vocabSize` | LM head matmul size (often largest operation) |
| `quantization` | Selects dequant kernel, fused Q4K matmul |

**Example - Gemma 3 1B:**

```
headDim = 256  →  Uses small tiled attention (fits in 4KB shared memory)
                  Selects 'attention_small.wgsl' / 'attention_small_f16kv.wgsl'

quantization = 'Q4_K_M'  →  Uses 'q4_fused' for decode GEMV
                            Uses 'q4_fused_batched' for prefill

numKVHeads = 1 (GQA)  →  Small KV cache, enables F16 KV
                        Only 256 floats per token per layer
```

### RDRR Runtime Hints

The RDRR manifest can include `runtimeOptimizations` to hint kernel selection:

```json
{
  "runtimeOptimizations": {
    "preferF16KV": true,
    "preferFusedDequant": true,
    "attentionTier": "streaming",
    "matmulTile": [16, 16],
    "forceKernels": {
      "matmul": "gemv_subgroup",
      "attention": "decode_streaming_f16kv"
    }
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `preferF16KV` | boolean | Use F16 KV cache if GPU supports it |
| `preferFusedDequant` | boolean | Use fused Q4K matmul when available |
| `attentionTier` | string | Force attention tier: `tiled`, `small`, `streaming` |
| `matmulTile` | [number, number] | Override default tile size |
| `forceKernels` | object | Force specific kernel variants (debugging) |

### Auto-Tuning System

DOPPLER includes an auto-tuning system (`gpu/kernel-tuner.ts`) that benchmarks kernel variants at runtime:

```typescript
// Tune kernels for specific model config (Gemma 3 1B example)
const results = await autoTuneKernels({
  hiddenSize: 1152,
  intermediateSize: 6912,
  numHeads: 4,
  headDim: 256,
  vocabSize: 262144,
});

// Results cached in IndexedDB for future sessions
// Format: { operation: { optimalWorkgroupSize, variantTimings } }
```

**Tuning Flow:**

```
1. For each kernel operation (matmul, attention, rmsnorm, etc.):
   │
   ├── Generate test inputs matching model config
   │
   ├── Run each compatible variant N times (default: 10)
   │
   ├── Measure median execution time
   │
   └── Cache optimal variant and workgroup size

2. On subsequent runs:
   │
   └── Load cached results, skip benchmarking
```

**Manual Tuning:**

```typescript
import { getTunedWorkgroupSize } from './gpu/kernels/index.js';

// Get optimal workgroup size for matmul with specific dimensions (Gemma 3 example)
const [wgX, wgY, wgZ] = await getTunedWorkgroupSize('matmul', {
  M: 1,
  N: 4096,
  K: 1152,
});
```

### Kernel Prewarm

To avoid shader compilation stalls during inference, DOPPLER can prewarm all compatible kernels at startup:

```typescript
import { prewarmKernels } from './gpu/kernels/index.js';

// Compile all kernels that the current GPU supports
await prewarmKernels();

// Output: "[KernelSelector] Prewarmed 47 kernel pipelines"
```

This is especially important for:
- First inference after page load
- Mobile GPUs with slow shader compilation
- WebGPU implementations with synchronous compile

### Capability Tiers

DOPPLER defines capability tiers for common GPU classes:

| Tier | Example GPUs | Features | Typical Kernels |
|------|--------------|----------|-----------------|
| **Tier 1** | Apple M1+, RTX 30+ | F16, subgroups | `gemv_subgroup`, `q4_fused`, streaming F16KV |
| **Tier 2** | Intel Xe, AMD RDNA2+ | F16, no subgroups | `gemv`, `f16w_f32a`, streaming F16KV |
| **Tier 3** | Older Intel, mobile | No F16 | `f32`, shared dequant, F32 KV |

**Detection:**

```typescript
function getCapabilityTier(caps: KernelCapabilities): 1 | 2 | 3 {
  if (caps.hasF16 && caps.hasSubgroups) return 1;
  if (caps.hasF16) return 2;
  return 3;
}
```

### Debugging Kernel Selection

Enable verbose logging to see kernel selection decisions:

```typescript
// In browser console
localStorage.setItem('DOPPLER_DEBUG_KERNELS', 'true');

// Output during inference (Gemma 3 1B example):
// [Pipeline] MATMUL: M=1, N=1152, K=1152, variant=gemv_subgroup, aDtype=f32, bDtype=f16
// [Pipeline] ATTENTION: tier=small, variant=decode_small_f16kv, headDim=256
```

**Force Specific Kernels (Testing):**

```typescript
// Override kernel selection for debugging
window.DOPPLER_FORCE_KERNELS = {
  matmul: 'f32',           // Disable F16
  attention: 'prefill',    // Force tiled (may fail on large headDim)
};
```

---

*Last updated: December 2025*

<!-- DOPPLER_KERNEL_OVERRIDES -->
## Kernel Overrides & Compatibility
See `docs/KERNEL_COMPATIBILITY.md` for runtime kernel modes (4-bit/9-bit), CLI flags (`--force-fused-q4k`, `--kernel-hints`), and the OPFS purge helper.

