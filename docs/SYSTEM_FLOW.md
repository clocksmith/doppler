# DOPPLER System Flow: Model to Execution

This document visualizes DOPPLER's architecture as a series of bipartite graphs showing how models flow through conversions, configurations, and kernel selections to final execution.

**See also:**
- [ARCHITECTURE.md](ARCHITECTURE.md) - Detailed component documentation
- [EXECUTION_PIPELINE.md](EXECUTION_PIPELINE.md) - Line-by-line kernel execution trace

---

## Overview: 5-Stage Bipartite Graph

```mermaid
graph LR
    M[Models<br/>M1, M2, M3] -->|Conversion| R[RDRR Variants<br/>R1..R20]
    R -->|Manifest| C[Configs<br/>C1, C2, C3]
    C -->|Architecture| P[Pipelines<br/>P1, P2, P3]
    P -->|Operations| K[Kernels<br/>K1..K20]
    K -->|Hardware| W[WGSL Files<br/>W1..W33]

    style M fill:#e1f5ff
    style R fill:#fff4e1
    style C fill:#f0e1ff
    style P fill:#e1ffe1
    style K fill:#ffe1e1
    style W fill:#f5f5f5
```

**Key Insight:** DOPPLER is configuration-driven. Model architecture determines pipeline structure; hardware capabilities determine kernel variants.

---

## Graph 1: Model → RDRR Conversion

```mermaid
graph TD
    subgraph Models
        M1[Gemma 3 1B<br/>SafeTensors BF16<br/>2.6GB]
        M2[Mixtral 8x7B<br/>SafeTensors<br/>90GB]
        M3[GPT-OSS 20B<br/>SafeTensors<br/>40GB]
    end

    subgraph ConversionOptions["Conversion Options (convert-cli.ts)"]
        O1[quantize: q4_k_m<br/>q4kLayout: row_wise<br/>computePrecision: auto]
        O2[quantize: q4_k_m<br/>q4kLayout: column_wise<br/>fuseGateUp: true]
        O3[quantize: f16<br/>computePrecision: f16]
        O4[quantize: mxfp4<br/>MoE experts]
    end

    subgraph RDRRVariants["RDRR Variants (1.2GB each)"]
        R1[gemma-1b-q4-row<br/>manifest.json<br/>shards: 18×64MB]
        R2[gemma-1b-q4-col<br/>14% faster load]
        R3[gemma-1b-f16<br/>2.3GB]
        R4[gemma-1b-q4-fuse<br/>2→1 FFN passes]
        R5[mixtral-q4-moe]
        R6[mixtral-mxfp4]
        R7[gptoss-mxfp4<br/>expert quant]
    end

    M1 -->|convert-cli| O1
    M1 --> O2
    M1 --> O3
    M1 --> O2
    M2 --> O1
    M2 --> O4
    M3 --> O4

    O1 --> R1
    O2 --> R2
    O3 --> R3
    O2 --> R4
    O1 --> R5
    O4 --> R6
    O4 --> R7

    style M1 fill:#bbdefb
    style M2 fill:#bbdefb
    style M3 fill:#bbdefb
    style R1 fill:#fff9c4
    style R2 fill:#fff9c4
    style R3 fill:#fff9c4
```

**Dimensions:** 1 source model → ~10-20 RDRR variants
- Quantization: {q4_k_m, q8_0, f16, bf16, f32, mxfp4}
- Q4K layout: {row_wise, column_wise, flat}
- Compute precision: {auto, f16, f32}
- FFN fusion: {true, false}

---

## Graph 2: RDRR → Manifest Config

```mermaid
graph LR
    subgraph RDRRFiles["RDRR Files"]
        R1["gemma-1b-q4-row<br/>───────────<br/>manifest.json<br/>shard_000.bin<br/>shard_001.bin<br/>..."]
        R5["mixtral-q4-moe<br/>───────────<br/>manifest.json<br/>moeConfig<br/>expertShardMap"]
        R7["gptoss-mxfp4<br/>───────────<br/>manifest.json<br/>32 experts<br/>MXFP4 quant"]
    end

    subgraph ManifestFields["Manifest Fields"]
        F1["architecture: GemmaForCausalLM<br/>quantization: Q4_K_M<br/>q4kLayout: row_wise<br/>kernelHints:<br/>&nbsp;&nbsp;q4kMatmul: fused_q4k"]
        F2["architecture: MixtralForCausalLM<br/>moeConfig:<br/>&nbsp;&nbsp;numExperts: 8<br/>&nbsp;&nbsp;numExpertsPerTok: 2<br/>&nbsp;&nbsp;expertShardMap"]
        F3["architecture: GPTOSSForCausalLM<br/>moeConfig:<br/>&nbsp;&nbsp;numExperts: 32<br/>&nbsp;&nbsp;numExpertsPerTok: 4<br/>expertQuant: MXFP4"]
    end

    subgraph Configs["Runtime Configs"]
        C1["Gemma Config<br/>───────────<br/>hiddenSize: 1152<br/>numLayers: 26<br/>headDim: 256<br/>useSandwichNorm: true<br/>qkNorm: true<br/>activation: gelu"]
        C2["Mixtral Config<br/>───────────<br/>hiddenSize: 4096<br/>numExperts: 8<br/>topK: 2<br/>headDim: 128<br/>activation: silu"]
        C3["GPT-OSS Config<br/>───────────<br/>numExperts: 32<br/>topK: 4<br/>slidingWindow: 4096<br/>expertQuant: MXFP4"]
    end

    R1 --> F1 --> C1
    R5 --> F2 --> C2
    R7 --> F3 --> C3

    style R1 fill:#fff9c4
    style R5 fill:#fff9c4
    style R7 fill:#fff9c4
    style C1 fill:#e1bee7
    style C2 fill:#e1bee7
    style C3 fill:#e1bee7
```

**Key Branching:**
- Dense models → single pipeline config
- MoE models → pipeline + expert routing config
- Sliding window → KV cache policy config

---

## Graph 3: Config → Pipeline Architecture

```mermaid
graph TD
    subgraph Configs["Configs"]
        C1["Gemma Config<br/>useSandwichNorm: true<br/>qkNorm: true<br/>activation: gelu<br/>FFN: dense"]
        C2["Mixtral Config<br/>useSandwichNorm: false<br/>numExperts: 8<br/>topK: 2<br/>activation: silu"]
        C3["GPT-OSS Config<br/>numExperts: 32<br/>topK: 4<br/>slidingWindow: 4096<br/>expertQuant: MXFP4"]
    end

    subgraph Features["Architecture Features"]
        F1["4 norms per layer<br/>Q/K normalization<br/>GELU activation<br/>Dense FFN"]
        F2["2 norms per layer<br/>MoE router<br/>8 experts, top-2<br/>SiLU activation"]
        F3["2 norms per layer<br/>MoE router<br/>32 experts, top-4<br/>MXFP4 dequant<br/>Sliding window KV"]
    end

    subgraph Pipelines["Pipeline Structures"]
        P1["P1: Gemma Pipeline<br/>═══════════════<br/>RMSNorm input<br/>Q/K/V matmul<br/>RMSNorm Q<br/>RMSNorm K<br/>RoPE<br/>Attention<br/>O matmul<br/>RMSNorm post-attn<br/>Residual<br/>RMSNorm pre-FFN<br/>gate/up matmul ×2<br/>SiLU<br/>down matmul<br/>RMSNorm post-FFN<br/>Residual"]
        P2["P2: Mixtral Pipeline<br/>═══════════════<br/>RMSNorm input<br/>Q/K/V matmul<br/>RoPE<br/>Attention<br/>O matmul<br/>Residual<br/>RMSNorm<br/>MoE Router<br/>├─ router matmul<br/>├─ softmax<br/>└─ topk k=2<br/>FOR expert in top-2:<br/>├─ gather tokens<br/>├─ expert FFN<br/>└─ scatter-add<br/>Residual"]
        P3["P3: GPT-OSS Pipeline<br/>═══════════════<br/>Similar to Mixtral<br/>but:<br/>topk k=4<br/>32 experts<br/>MXFP4 dequant"]
    end

    C1 --> F1 --> P1
    C2 --> F2 --> P2
    C3 --> F3 --> P3

    style C1 fill:#e1bee7
    style C2 fill:#e1bee7
    style C3 fill:#e1bee7
    style P1 fill:#c8e6c9
    style P2 fill:#c8e6c9
    style P3 fill:#c8e6c9
```

**Pipeline Dimensions:**
- Normalization: {sandwich (4 norms), standard (2 norms)}
- FFN type: {dense, moe}
- Activation: {gelu, silu, swiglu}
- KV cache: {full, sliding_window}

---

## Graph 4: Pipeline → Kernel Sequence (Per Token)

### Gemma 3 Pipeline Kernel Sequence

```mermaid
graph TD
    Start([Token Input]) --> Embed["gather.wgsl<br/>scale *= sqrt(1152)"]
    Embed --> Layer0[Layer 0]

    subgraph Layer["FOR layer 0..25 (×26)"]
        L1["rmsnorm.wgsl<br/>input norm"] --> L2["matmul variant<br/>Q/K/V projections ×3"]
        L2 --> L3["rmsnorm.wgsl<br/>Q norm (Gemma only)"]
        L3 --> L4["rmsnorm.wgsl<br/>K norm (Gemma only)"]
        L4 --> L5["rope.wgsl<br/>positional encoding"]
        L5 --> L6["attention_small_f16kv.wgsl<br/>headDim=256"]
        L6 --> L7["matmul variant<br/>O projection"]
        L7 --> L8["rmsnorm.wgsl<br/>post-attn norm (Gemma only)"]
        L8 --> L9["residual.wgsl"]
        L9 --> L10["rmsnorm.wgsl<br/>pre-FFN norm (Gemma only)"]
        L10 --> L11["matmul variant ×2<br/>gate, up projections"]
        L11 --> L12["silu.wgsl<br/>gate activation"]
        L12 --> L13["matmul variant<br/>down projection"]
        L13 --> L14["rmsnorm.wgsl<br/>post-FFN norm (Gemma only)"]
        L14 --> L15["residual.wgsl"]
    end

    Layer0 --> Layer
    Layer --> FinalNorm["rmsnorm.wgsl<br/>final norm"]
    FinalNorm --> LMHead["matmul_gemv_subgroup.wgsl<br/>1×1152 @ 1152×262144"]
    LMHead --> Sample["argmax (GPU)<br/>sample.wgsl"]
    Sample --> End([Token Output])

    style Embed fill:#fff9c4
    style Layer fill:#e3f2fd
    style Sample fill:#c8e6c9
```

**Kernel count:** ~156 kernel dispatches per token
- 8 operations × 26 layers = 208 calls (many are rmsnorm due to sandwich)
- Gemma uses 4× more rmsnorm than standard architectures

### Mixtral Pipeline Kernel Sequence

```mermaid
graph TD
    Start([Token Input]) --> Embed["gather.wgsl"]
    Embed --> Layer0[Layer 0]

    subgraph Layer["FOR layer 0..31 (×32)"]
        L1["rmsnorm.wgsl"] --> L2["matmul ×3<br/>Q/K/V"]
        L2 --> L3["rope.wgsl"]
        L3 --> L4["attention_f16kv.wgsl<br/>headDim=128 → large tier"]
        L4 --> L5["matmul<br/>O projection"]
        L5 --> L6["residual.wgsl"]
        L6 --> L7["rmsnorm.wgsl"]

        L7 --> Router["MoE Router"]
        subgraph Router["MoE Router Block"]
            R1["matmul<br/>1×4096 @ 4096×8"] --> R2["softmax.wgsl"]
            R2 --> R3["topk.wgsl<br/>k=2"]
        end

        Router --> Expert["Expert Loop"]
        subgraph Expert["FOR expert in top-2"]
            E1["moe_gather.wgsl<br/>route tokens"] --> E2["Expert FFN<br/>gate/up/down matmul<br/>silu.wgsl"]
            E2 --> E3["scatter_add.wgsl<br/>weighted combine"]
        end

        Expert --> L8["residual.wgsl"]
    end

    Layer0 --> Layer
    Layer --> FinalNorm["rmsnorm.wgsl"]
    FinalNorm --> LMHead["matmul_gemv_subgroup.wgsl"]
    LMHead --> Sample["argmax"]
    Sample --> End([Token Output])

    style Router fill:#ffecb3
    style Expert fill:#c5e1a5
```

**Additional kernels vs Gemma:**
- `softmax.wgsl` - router logits
- `topk.wgsl` - expert selection
- `moe_gather.wgsl` - token routing
- `scatter_add.wgsl` - expert combination

**Removed kernels vs Gemma:**
- 2× fewer `rmsnorm.wgsl` (no sandwich norms)
- No Q/K normalization

---

## Graph 5: Kernel → WGSL Variant Selection

### MatMul Kernel Selection

```mermaid
graph TD
    MatMul["matmul operation<br/>A[M×K] @ B[K×N]"] --> CheckQuant{B dtype?}

    CheckQuant -->|Q4K| CheckLayout{Q4K layout?}
    CheckQuant -->|F16| CheckF16Support{hasF16?}
    CheckQuant -->|F32| Fallback["matmul_f32.wgsl<br/>baseline"]

    CheckLayout -->|row_wise| CheckSubgroups{hasSubgroups?}
    CheckLayout -->|column_wise| Dequant["dequant_subgroup.wgsl<br/>then matmul_gemv_subgroup"]
    CheckLayout -->|flat| DequantShared["dequant_shared.wgsl<br/>then matmul"]

    CheckSubgroups -->|true| FusedQ4K["matmul_q4_fused.wgsl<br/>4-bit direct compute<br/>FASTEST"]
    CheckSubgroups -->|false| DequantNormal["dequant path<br/>slower"]

    CheckF16Support -->|true| CheckMixed{A dtype?}
    CheckF16Support -->|false| Fallback

    CheckMixed -->|F16| F16Full["matmul_f16.wgsl<br/>full F16 path"]
    CheckMixed -->|F32| F16Mixed["matmul_f16w_f32a.wgsl<br/>mixed precision"]

    CheckGEMV{M = 1?<br/>GEMV} --> |yes| GEMV["matmul_gemv_subgroup.wgsl<br/>optimized for decode"]
    CheckGEMV -->|no| General["matmul_f32.wgsl<br/>tiled for prefill"]

    style FusedQ4K fill:#c8e6c9
    style F16Full fill:#c8e6c9
    style GEMV fill:#c8e6c9
    style Fallback fill:#ffcdd2
```

**Hardware dimensions:**
- `hasF16`: {true, false} → shader-f16 extension
- `hasSubgroups`: {true, false} → subgroup operations
- `M=1`: decode (GEMV) vs prefill (general matmul)

### Attention Kernel Selection

```mermaid
graph TD
    Attention["attention operation"] --> CheckPhase{seqLen?}

    CheckPhase -->|1| Decode["Decode phase<br/>M=1"]
    CheckPhase -->|>1| Prefill["Prefill phase<br/>M>1"]

    Decode --> CheckHeadDim{headDim?}
    Prefill --> CheckHeadDim

    CheckHeadDim -->|≤64| CheckShared1{sharedMem?}
    CheckHeadDim -->|≤256| CheckShared2{sharedMem?}
    CheckHeadDim -->|>256| Streaming["attention_streaming.wgsl<br/>no shared mem required"]

    CheckShared1 -->|≥49KB| Large["attention.wgsl<br/>tiled_large<br/>blockSize=64"]
    CheckShared1 -->|<49KB| CheckShared2

    CheckShared2 -->|≥8KB| Small["attention_small.wgsl<br/>tiled_small<br/>blockSize=32"]
    CheckShared2 -->|<8KB| Streaming

    Large --> CheckKV1{KV dtype?}
    Small --> CheckKV2{KV dtype?}
    Streaming --> CheckKV3{KV dtype?}

    CheckKV1 -->|F16| LargeF16["attention_f16kv.wgsl"]
    CheckKV1 -->|F32| Large
    CheckKV2 -->|F16| SmallF16["attention_small_f16kv.wgsl"]
    CheckKV2 -->|F32| Small
    CheckKV3 -->|F16| StreamF16["attention_streaming_f16kv.wgsl"]
    CheckKV3 -->|F32| Streaming

    style LargeF16 fill:#c8e6c9
    style SmallF16 fill:#c8e6c9
    style Streaming fill:#fff9c4
```

**Examples:**
- Gemma 1B (headDim=256, M3 32KB shared) → `attention_small_f16kv.wgsl`
- Mixtral (headDim=128, RTX 4090 48KB shared) → `attention_f16kv.wgsl`
- Low-memory device (<8KB shared) → `attention_streaming.wgsl`

---

## Complete Flow Example: Gemma 3 1B on Apple M3

```mermaid
graph TD
    subgraph Stage1["1. Source Model"]
        Source["google/gemma-3-1b-it<br/>SafeTensors BF16<br/>2.6GB"]
    end

    subgraph Stage2["2. Conversion"]
        Conv["convert-cli.ts<br/>--quantize q4_k_m<br/>--q4k-layout column_wise<br/>--compute-precision auto"]
    end

    subgraph Stage3["3. RDRR Output"]
        RDRR["gemma-1b-q4-col<br/>1.2GB<br/>18 shards × 64MB<br/>manifest.json"]
    end

    subgraph Stage4["4. Manifest Config"]
        Manifest["architecture: GemmaForCausalLM<br/>quantization: Q4_K_M<br/>q4kLayout: column_wise<br/>config:<br/>&nbsp;&nbsp;hiddenSize: 1152<br/>&nbsp;&nbsp;numLayers: 26<br/>&nbsp;&nbsp;headDim: 256<br/>kernelHints:<br/>&nbsp;&nbsp;q4kMatmul: dequant_f16<br/>&nbsp;&nbsp;computePrecision: f16"]
    end

    subgraph Stage5["5. Hardware Detection (M3)"]
        HW["hasF16: true<br/>hasSubgroups: true<br/>sharedMem: 32KB<br/>isUnifiedMemory: true"]
    end

    subgraph Stage6["6. Pipeline Selection"]
        Pipeline["Gemma Pipeline<br/>useSandwichNorm: true<br/>qkNorm: true<br/>activation: gelu<br/>4 norms per layer"]
    end

    subgraph Stage7["7. Kernel Execution (per token)"]
        Kernels["Embed: gather_f16.wgsl<br/>───────────────────<br/>FOR layer 0..25:<br/>&nbsp;&nbsp;Input norm: rmsnorm.wgsl<br/>&nbsp;&nbsp;Q/K/V: matmul_gemv_subgroup.wgsl ×3<br/>&nbsp;&nbsp;Q/K norm: rmsnorm.wgsl ×2<br/>&nbsp;&nbsp;RoPE: rope.wgsl<br/>&nbsp;&nbsp;Attention: attention_small_f16kv.wgsl<br/>&nbsp;&nbsp;O proj: matmul_gemv_subgroup.wgsl<br/>&nbsp;&nbsp;Post-attn norm: rmsnorm.wgsl<br/>&nbsp;&nbsp;Residual: residual.wgsl<br/>&nbsp;&nbsp;Pre-FFN norm: rmsnorm.wgsl<br/>&nbsp;&nbsp;gate/up: matmul_gemv_subgroup.wgsl ×2<br/>&nbsp;&nbsp;SiLU: silu.wgsl<br/>&nbsp;&nbsp;down: matmul_gemv_subgroup.wgsl<br/>&nbsp;&nbsp;Post-FFN norm: rmsnorm.wgsl<br/>&nbsp;&nbsp;Residual: residual.wgsl<br/>───────────────────<br/>Final: rmsnorm.wgsl<br/>LM head: matmul_gemv_subgroup.wgsl<br/>Sample: argmax (GPU)"]
    end

    subgraph Stage8["8. Performance"]
        Perf["~156 kernel dispatches/token<br/>~140ms/token<br/>~7 tok/s"]
    end

    Source --> Conv --> RDRR --> Manifest
    Manifest --> HW
    Manifest --> Pipeline
    HW --> Pipeline
    Pipeline --> Kernels --> Perf

    style Source fill:#bbdefb
    style RDRR fill:#fff9c4
    style Manifest fill:#e1bee7
    style HW fill:#ffecb3
    style Pipeline fill:#c8e6c9
    style Kernels fill:#f5f5f5
    style Perf fill:#c8e6c9
```

---

## Architecture Comparison: Gemma 3 vs GPT-OSS

```mermaid
graph LR
    subgraph Gemma["Gemma 3 1B"]
        G1["4 norms/layer<br/>Q/K normalization<br/>Dense FFN<br/>GELU<br/>1024 sliding window"]
        G2["Kernels Used:<br/>✓ gather<br/>✓ rmsnorm ×8<br/>✓ matmul ×7<br/>✓ rope<br/>✓ attention_small<br/>✓ residual ×2<br/>✓ silu"]
    end

    subgraph GPTOSS["GPT-OSS 20B"]
        O1["2 norms/layer<br/>No Q/K norm<br/>MoE FFN (32 experts)<br/>SiLU<br/>4096 sliding window"]
        O2["Kernels Used:<br/>✓ gather<br/>✓ rmsnorm ×2<br/>✓ matmul ×4<br/>✓ rope<br/>✓ attention<br/>✓ residual ×2<br/>✓ silu<br/>✓ softmax (router)<br/>✓ topk (router)<br/>✓ moe_gather<br/>✓ scatter_add<br/>✓ dequant_mxfp4"]
    end

    G1 --> G2
    O1 --> O2

    style Gemma fill:#c8e6c9
    style GPTOSS fill:#ffecb3
```

**Key Differences:**
- Gemma: More normalization layers (+6 rmsnorm/layer)
- GPT-OSS: MoE routing kernels (+5 unique kernels)
- Same core: matmul, attention, rope, residual (reused!)

---

## Kernel Reuse Matrix

```mermaid
graph TD
    subgraph Core["Core Kernels (ALL models)"]
        C1["gather.wgsl<br/>rmsnorm.wgsl<br/>rope.wgsl<br/>residual.wgsl<br/>matmul variants<br/>attention variants"]
    end

    subgraph Dense["Dense-Specific"]
        D1["silu.wgsl<br/>swiglu.wgsl<br/>gelu.wgsl"]
    end

    subgraph MoE["MoE-Specific"]
        M1["softmax.wgsl<br/>topk.wgsl<br/>moe_gather.wgsl<br/>scatter_add.wgsl"]
    end

    subgraph Quant["Quantization-Specific"]
        Q1["dequant_mxfp4.wgsl (GPT-OSS)<br/>dequant_subgroup.wgsl (Q4K)<br/>dequant_shared.wgsl (Q4K)<br/>bf16_to_f16.wgsl"]
    end

    Gemma[Gemma 3] --> Core
    Gemma --> Dense

    Mixtral[Mixtral 8x7B] --> Core
    Mixtral --> Dense
    Mixtral --> MoE

    GPTOSS[GPT-OSS 20B] --> Core
    GPTOSS --> Dense
    GPTOSS --> MoE
    GPTOSS --> Q1

    style Core fill:#c8e6c9
    style Dense fill:#fff9c4
    style MoE fill:#ffecb3
    style Quant fill:#e1bee7
```

**Total:** 33 WGSL files
**Used per model:** ~10-15 (subset based on architecture)

---

## Key Insights

### 1. Configuration-Driven System

DOPPLER is **not** hard-coded per model. The system adapts through configuration:

```
Model Architecture → Pipeline Structure
     ↓
Pipeline Structure → Kernel Sequence
     ↓
Hardware + dtypes → WGSL Variant
```

### 2. Variant Explosion

```
1 source model
  → ~10-20 RDRR variants (quantization × layout × options)
    → 1 config per RDRR
      → 1 pipeline type
        → N kernel dispatches (architecture-dependent)
          → M WGSL files (hardware-dependent)
```

### 3. Kernel Selection Factors

| Factor | Examples | Impact |
|--------|----------|--------|
| **Architecture** | useSandwichNorm, qkNorm, MoE | Determines WHICH operations |
| **Quantization** | Q4K layout, MXFP4 | Determines dequant path |
| **Hardware** | hasF16, hasSubgroups, sharedMem | Determines WGSL variant |
| **Tensor shape** | M=1 (GEMV), headDim (attention tier) | Determines kernel specialization |

### 4. Same Operations, Different Counts

Gemma 3 and Mixtral both use `matmul`, but:
- **Gemma:** 7 matmul calls per layer (Q/K/V/O + gate/up/down)
- **Mixtral:** 4 matmul calls + MoE router matmul + expert matmuls

Same kernel, different usage patterns based on architecture.

---

## Performance Analysis

### Kernel Dispatch Count (Decode, 1 token)

| Model | Layers | Kernels/Layer | Total/Token | Notes |
|-------|--------|---------------|-------------|-------|
| Gemma 3 1B | 26 | ~6 ops | ~156 | 4 norms = heavy |
| Mixtral 8x7B | 32 | ~5 ops + router | ~180 | +MoE overhead |
| Standard LLaMA | 32 | ~4 ops | ~128 | 2 norms only |

### Latency Breakdown (Gemma 3 1B, M3)

```
Total: 140ms/token
├─ MatMul: ~100ms (71%) - 7×26 = 182 matmuls
├─ Attention: ~25ms (18%) - 26 attention ops
├─ RMSNorm: ~10ms (7%) - 8×26 = 208 norms
├─ Other: ~5ms (4%) - rope, residual, silu
```

**Bottleneck:** MatMul (column-wise Q4K dequant + F16 compute)

---

*Last updated: December 2025*
