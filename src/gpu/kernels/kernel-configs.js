

// ============================================================================
// Kernel Configurations
// ============================================================================


export const KERNEL_CONFIGS = {
  matmul: {
    f16: {
      shaderFile: 'matmul_f16.wgsl',
      entryPoint: 'main',
      workgroupSize: [16, 16, 1],
      requires: ['shader-f16'],
      outputDtype: 'f16',
    },
    f16_vec4: {
      shaderFile: 'matmul_f16.wgsl',
      entryPoint: 'main_vec4',
      workgroupSize: [16, 16, 1],
      requires: ['shader-f16'],
      outputDtype: 'f16',
    },
    f16w_f32a: {
      shaderFile: 'matmul_f16w_f32a.wgsl',
      entryPoint: 'main',
      workgroupSize: [16, 16, 1],
      requires: ['shader-f16'],
    },
    // Optimized GEMV for M=1 decode: uses shared memory for A vector
    gemv: {
      shaderFile: 'matmul_gemv.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: ['shader-f16'],
    },
    // Subgroup-optimized GEMV - 1.5x faster using subgroupAdd
    gemv_subgroup: {
      shaderFile: 'matmul_gemv_subgroup.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: ['shader-f16', 'subgroups'],
    },
    gemv_subgroup_vec4: {
      shaderFile: 'matmul_gemv_subgroup.wgsl',
      entryPoint: 'main_vec4',
      workgroupSize: [256, 1, 1],
      requires: ['shader-f16', 'subgroups'],
    },
    // Multi-column GEMV for large vocab (LM head F16) - 32 columns per workgroup
    // Reduces workgroups from 65K to 8K for vocab=262144
    gemv_subgroup_multicol: {
      shaderFile: 'matmul_gemv_subgroup.wgsl',
      entryPoint: 'main_multicol',
      workgroupSize: [256, 1, 1],
      requires: ['shader-f16', 'subgroups'],
      variantMetadata: { colsPerWg: 32 },
    },
    // Fused Q4_K dequant + matmul - 2-3x faster (no separate dequant pass)
    q4_fused: {
      shaderFile: 'fused_matmul_q4.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: ['subgroups'],
    },
    q4_fused_batched: {
      shaderFile: 'fused_matmul_q4_batched.wgsl',
      entryPoint: 'main_batched',
      workgroupSize: [64, 4, 1],
      requires: ['subgroups'],
      variantMetadata: { tileM: 4 },
    },
    // Multi-column GEMV for large vocab (LM head) - 32 columns per workgroup
    q4_fused_multicol: {
      shaderFile: 'fused_matmul_q4.wgsl',
      entryPoint: 'main_multicol',
      workgroupSize: [256, 1, 1],
      requires: ['subgroups'],
      variantMetadata: { colsPerWg: 32 },
    },
    // F16 output variants - same as above but output to f16 buffer
    q4_fused_multicol_f16: {
      shaderFile: 'fused_matmul_q4_multicol_f16.wgsl',
      entryPoint: 'main_multicol_f16',
      workgroupSize: [256, 1, 1],
      requires: ['shader-f16', 'subgroups'],
      outputDtype: 'f16',
      variantMetadata: { colsPerWg: 32 },
    },
    q4_fused_batched_f16: {
      shaderFile: 'fused_matmul_q4_batched_f16.wgsl',
      entryPoint: 'main_batched_f16',
      workgroupSize: [64, 4, 1],
      requires: ['shader-f16', 'subgroups'],
      outputDtype: 'f16',
      variantMetadata: { tileM: 4 },
    },
    f32: {
      shaderFile: 'matmul_f32.wgsl',
      entryPoint: 'main',
      workgroupSize: [16, 16, 1],
      requires: [],
    },
  },
  // Fused FFN kernels (Tier 2 P0)
  fused_ffn: {
    default: {
      shaderFile: 'fused_ffn.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: ['subgroups'],
    },
    multi: {
      shaderFile: 'fused_ffn.wgsl',
      entryPoint: 'main_multi',
      workgroupSize: [256, 1, 1],
      requires: ['subgroups'],
    },
    f16: {
      shaderFile: 'fused_ffn.wgsl',
      entryPoint: 'main_f16',
      workgroupSize: [256, 1, 1],
      requires: ['subgroups'],
    },
    batched: {
      shaderFile: 'fused_ffn.wgsl',
      entryPoint: 'main_batched',
      workgroupSize: [256, 1, 1],
      requires: ['subgroups'],
    },
    q4k: {
      shaderFile: 'fused_ffn_q4k.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: ['subgroups'],
    },
    q4k_batched: {
      shaderFile: 'fused_ffn_q4k.wgsl',
      entryPoint: 'main_batched',
      workgroupSize: [256, 1, 1],
      requires: ['subgroups'],
    },
  },
  // Optimized attention decode (Tier 2 P0)
  attention_decode_optimized: {
    default: {
      shaderFile: 'attention_decode_optimized.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: ['subgroups'],
    },
    multihead: {
      shaderFile: 'attention_decode_optimized.wgsl',
      entryPoint: 'main_multihead',
      workgroupSize: [256, 1, 1],
      requires: ['subgroups'],
    },
    f16kv: {
      shaderFile: 'attention_decode_optimized.wgsl',
      entryPoint: 'main_f16kv',
      workgroupSize: [256, 1, 1],
      requires: ['subgroups'],
    },
  },
  dequant: {
    subgroup: {
      shaderFile: 'dequant_subgroup.wgsl',
      entryPoint: 'main',
      workgroupSize: [64, 1, 1],
      requires: ['subgroups'],
    },
    subgroup_vec4: {
      shaderFile: 'dequant_subgroup.wgsl',
      entryPoint: 'main_vec4',
      workgroupSize: [64, 1, 1],
      requires: ['subgroups'],
    },
    subgroup_f16out: {
      shaderFile: 'dequant_f16_out.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: ['shader-f16'],
    },
    subgroup_vec4_f16out: {
      shaderFile: 'dequant_f16_out_vec4.wgsl',
      entryPoint: 'main_vec4',
      workgroupSize: [64, 1, 1],
      requires: ['shader-f16'],
    },
    shared: {
      shaderFile: 'dequant_shared.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    shared_vec4: {
      shaderFile: 'dequant_shared_vec4.wgsl',
      entryPoint: 'main_vec4',
      workgroupSize: [64, 1, 1],
      requires: [],
    },
    shared_f16out: {
      shaderFile: 'dequant_f16_out.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: ['shader-f16'],
    },
    shared_vec4_f16out: {
      shaderFile: 'dequant_f16_out_vec4.wgsl',
      entryPoint: 'main_vec4',
      workgroupSize: [64, 1, 1],
      requires: ['shader-f16'],
    },
    // MXFP4 dequantization (GPT-OSS)
    mxfp4: {
      shaderFile: 'dequant_mxfp4.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    mxfp4_vec4: {
      shaderFile: 'dequant_mxfp4_vec4.wgsl',
      entryPoint: 'main_vec4',
      workgroupSize: [64, 1, 1],
      requires: [],
    },
    mxfp4_expert: {
      shaderFile: 'dequant_mxfp4_expert.wgsl',
      entryPoint: 'main_expert',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    // Q6_K dequantization (GGUF 6-bit quantization)
    q6k_f16out: {
      shaderFile: 'dequant_q6k.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: ['shader-f16'],
    },
    // Q8_0 dequantization (GGUF 8-bit quantization)
    q8_0_f16out: {
      shaderFile: 'dequant_q8_0.wgsl',
      entryPoint: 'main',
      workgroupSize: [32, 1, 1],
      requires: ['shader-f16'],
    },
  },
  attention: {
    prefill: {
      shaderFile: 'attention.wgsl',
      entryPoint: 'main',
      workgroupSize: [32, 1, 1],
      requires: [],
      // validate is set dynamically after import
    },
    decode: {
      shaderFile: 'attention_decode.wgsl',
      entryPoint: 'attention_decode',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    prefill_small: {
      shaderFile: 'attention_small.wgsl',
      entryPoint: 'main',
      workgroupSize: [32, 1, 1],
      requires: [],
    },
    decode_small: {
      shaderFile: 'attention_small.wgsl',
      entryPoint: 'main',
      workgroupSize: [32, 1, 1],
      requires: [],
    },
    prefill_streaming: {
      shaderFile: 'attention_streaming.wgsl',
      entryPoint: 'main',
      workgroupSize: [1, 1, 1],
      requires: [],
    },
    decode_streaming: {
      shaderFile: 'attention_streaming.wgsl',
      entryPoint: 'main',
      workgroupSize: [1, 1, 1],
      requires: [],
    },
    prefill_f16kv: {
      shaderFile: 'attention_f16kv.wgsl',
      entryPoint: 'main',
      workgroupSize: [64, 1, 1],
      requires: ['shader-f16'],
    },
    decode_f16kv: {
      shaderFile: 'attention_decode_f16kv.wgsl',
      entryPoint: 'attention_decode',
      workgroupSize: [256, 1, 1],
      requires: ['shader-f16'],
    },
    prefill_small_f16kv: {
      shaderFile: 'attention_small_f16kv.wgsl',
      entryPoint: 'main',
      workgroupSize: [32, 1, 1],
      requires: ['shader-f16'],
    },
    decode_small_f16kv: {
      shaderFile: 'attention_small_f16kv.wgsl',
      entryPoint: 'main',
      workgroupSize: [32, 1, 1],
      requires: ['shader-f16'],
    },
    prefill_streaming_f16kv: {
      shaderFile: 'attention_streaming_f16kv.wgsl',
      entryPoint: 'main',
      workgroupSize: [1, 1, 1],
      requires: ['shader-f16'],
    },
    decode_streaming_f16kv: {
      shaderFile: 'attention_streaming_f16kv.wgsl',
      entryPoint: 'main',
      workgroupSize: [1, 1, 1],
      requires: ['shader-f16'],
    },
    // Chunked decode kernel - parallelizes headDim for models with few heads (e.g., Gemma 3)
    decode_chunked_f16kv: {
      shaderFile: 'attention_decode_chunked_f16kv.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: ['shader-f16'],
      variantMetadata: { maxKVLen: 2048 },
    },
    // Subgroup-optimized decode kernel - 4 barriers (vs 80), 100% thread utilization
    decode_subgroup: {
      shaderFile: 'attention_decode_subgroup.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],  // headDim threads per workgroup
      requires: ['subgroups'],
    },
  },
  rmsnorm: {
    default: {
      shaderFile: 'rmsnorm.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    small: {
      shaderFile: 'rmsnorm.wgsl',
      entryPoint: 'main_small',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    cached: {
      shaderFile: 'rmsnorm.wgsl',
      entryPoint: 'main_cached',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    // Legacy alias for residual (now uses main_cached)
    residual: {
      shaderFile: 'rmsnorm.wgsl',
      entryPoint: 'main_cached',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    // Subgroup-accelerated variants (3-5x faster reductions)
    subgroup: {
      shaderFile: 'rmsnorm.wgsl',
      entryPoint: 'main_subgroup',
      workgroupSize: [256, 1, 1],
      requires: ['subgroups'],
    },
    small_subgroup: {
      shaderFile: 'rmsnorm.wgsl',
      entryPoint: 'main_small_subgroup',
      workgroupSize: [256, 1, 1],
      requires: ['subgroups'],
    },
    // F16 variants for reduced memory bandwidth
    default_f16: {
      shaderFile: 'rmsnorm_f16.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: ['shader-f16'],
    },
    small_f16: {
      shaderFile: 'rmsnorm_f16.wgsl',
      entryPoint: 'rmsnorm_small_f16',
      workgroupSize: [256, 1, 1],
      requires: ['shader-f16'],
    },
  },
  // Fused GEMV + RMSNorm for decode (M=1)
  // Combines down projection matmul with RMSNorm in single kernel
  fused_matmul_rmsnorm: {
    default: {
      shaderFile: 'fused_matmul_rmsnorm.wgsl',
      entryPoint: 'gemv_rmsnorm_medium',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    small: {
      shaderFile: 'fused_matmul_rmsnorm.wgsl',
      entryPoint: 'gemv_rmsnorm_small',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    // Medium variant for N up to ~4096 (covers Gemma 3's hiddenSize=1152)
    medium: {
      shaderFile: 'fused_matmul_rmsnorm.wgsl',
      entryPoint: 'gemv_rmsnorm_medium',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    phase1: {
      shaderFile: 'fused_matmul_rmsnorm.wgsl',
      entryPoint: 'gemv_rmsnorm_phase1',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
  },
  // Fused GEMV + Residual for decode (M=1)
  // Combines output projection matmul with residual add in single kernel
  fused_matmul_residual: {
    default: {
      shaderFile: 'matmul_gemv_residual.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
  },
  softmax: {
    default: {
      shaderFile: 'softmax.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    small: {
      shaderFile: 'softmax.wgsl',
      entryPoint: 'softmax_small',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    online: {
      shaderFile: 'softmax.wgsl',
      entryPoint: 'softmax_online',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    // Subgroup-accelerated variants (3-5x faster reductions)
    subgroup: {
      shaderFile: 'softmax.wgsl',
      entryPoint: 'main_subgroup',
      workgroupSize: [256, 1, 1],
      requires: ['subgroups'],
    },
    small_subgroup: {
      shaderFile: 'softmax.wgsl',
      entryPoint: 'softmax_small_subgroup',
      workgroupSize: [256, 1, 1],
      requires: ['subgroups'],
    },
  },
  rope: {
    default: {
      shaderFile: 'rope.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    compute_freqs: {
      shaderFile: 'rope.wgsl',
      entryPoint: 'rope_compute_freqs',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    qk: {
      shaderFile: 'rope.wgsl',
      entryPoint: 'rope_qk',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    ntk: {
      shaderFile: 'rope.wgsl',
      entryPoint: 'rope_ntk_scaled',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    yarn: {
      shaderFile: 'rope.wgsl',
      entryPoint: 'rope_yarn',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
  },
  silu: {
    default: {
      shaderFile: 'silu.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    gate: {
      shaderFile: 'silu.wgsl',
      entryPoint: 'silu_gate',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    gate_split: {
      shaderFile: 'silu.wgsl',
      entryPoint: 'silu_gate_split',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    vec4: {
      shaderFile: 'silu.wgsl',
      entryPoint: 'silu_vec4',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    gate_rowsplit: {
      shaderFile: 'silu.wgsl',
      entryPoint: 'silu_gate_rowsplit',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    // F16 variants for reduced memory bandwidth
    default_f16: {
      shaderFile: 'silu_f16.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: ['shader-f16'],
    },
    gate_f16: {
      shaderFile: 'silu_f16.wgsl',
      entryPoint: 'silu_gate_f16',
      workgroupSize: [256, 1, 1],
      requires: ['shader-f16'],
    },
    vec4_f16: {
      shaderFile: 'silu_f16.wgsl',
      entryPoint: 'silu_vec4_f16',
      workgroupSize: [256, 1, 1],
      requires: ['shader-f16'],
    },
    gate_rowsplit_f16: {
      shaderFile: 'silu_f16.wgsl',
      entryPoint: 'silu_gate_rowsplit_f16',
      workgroupSize: [256, 1, 1],
      requires: ['shader-f16'],
    },
  },
  gelu: {
    gelu: {
      shaderFile: 'gelu.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    geglu: {
      shaderFile: 'gelu.wgsl',
      entryPoint: 'geglu',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    geglu_rowsplit: {
      shaderFile: 'gelu.wgsl',
      entryPoint: 'geglu_rowsplit',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    gelu_f16: {
      shaderFile: 'gelu_f16.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: ['shader-f16'],
    },
    geglu_f16: {
      shaderFile: 'gelu_f16.wgsl',
      entryPoint: 'geglu_f16',
      workgroupSize: [256, 1, 1],
      requires: ['shader-f16'],
    },
    geglu_rowsplit_f16: {
      shaderFile: 'gelu_f16.wgsl',
      entryPoint: 'geglu_rowsplit_f16',
      workgroupSize: [256, 1, 1],
      requires: ['shader-f16'],
    },
  },
  scale: {
    default: {
      shaderFile: 'scale.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    inplace: {
      shaderFile: 'scale.wgsl',
      entryPoint: 'main_inplace',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
  },
  gather: {
    default: {
      shaderFile: 'gather.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    vec4: {
      shaderFile: 'gather_vec4.wgsl',
      entryPoint: 'gather_vec4',
      workgroupSize: [64, 1, 1],
      requires: [],
    },
    // F16 embeddings → F32 output (for weight-tied lm_head optimization)
    f16: {
      shaderFile: 'gather_f16.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: ['shader-f16'],
    },
    f16_vec4: {
      shaderFile: 'gather_f16_vec4.wgsl',
      entryPoint: 'gather_vec4',
      workgroupSize: [64, 1, 1],
      requires: ['shader-f16'],
    },
    // F32 embeddings → F16 output (for F16 activation mode)
    f16_out: {
      shaderFile: 'gather_f16_out.wgsl',
      entryPoint: 'gather_f16_out',
      workgroupSize: [256, 1, 1],
      requires: ['shader-f16'],
      outputDtype: 'f16',
      variantMetadata: { outputBinding: 4 },
    },
    vec4_f16_out: {
      shaderFile: 'gather_vec4_f16_out.wgsl',
      entryPoint: 'gather_vec4_f16_out',
      workgroupSize: [64, 1, 1],
      requires: ['shader-f16'],
      outputDtype: 'f16',
      variantMetadata: { outputBinding: 4 },
    },
    // F16 embeddings → F16 output (for F16 activation mode with F16 embeddings)
    f16_f16_out: {
      shaderFile: 'gather_f16_f16_out.wgsl',
      entryPoint: 'gather_f16_out',
      workgroupSize: [256, 1, 1],
      requires: ['shader-f16'],
      outputDtype: 'f16',
      variantMetadata: { outputBinding: 4 },
    },
    f16_vec4_f16_out: {
      shaderFile: 'gather_f16_vec4_f16_out.wgsl',
      entryPoint: 'gather_vec4_f16_out',
      workgroupSize: [64, 1, 1],
      requires: ['shader-f16'],
      outputDtype: 'f16',
      variantMetadata: { outputBinding: 4 },
    },
  },
  residual: {
    default: {
      shaderFile: 'residual.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    vec4: {
      shaderFile: 'residual_vec4.wgsl',
      entryPoint: 'add_vec4',
      workgroupSize: [64, 1, 1],
      requires: [],
    },
    default_f16: {
      shaderFile: 'residual_f16.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: ['shader-f16'],
      outputDtype: 'f16',
    },
    vec4_f16: {
      shaderFile: 'residual_f16_vec4.wgsl',
      entryPoint: 'add_vec4',
      workgroupSize: [64, 1, 1],
      requires: ['shader-f16'],
      outputDtype: 'f16',
    },
  },
  topk: {
    default: {
      shaderFile: 'topk.wgsl',
      entryPoint: 'main',
      workgroupSize: [32, 1, 1],
      requires: [],
    },
    small: {
      shaderFile: 'topk.wgsl',
      entryPoint: 'topk_2_small',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    fused: {
      shaderFile: 'topk.wgsl',
      entryPoint: 'softmax_topk',
      workgroupSize: [32, 1, 1],
      requires: [],
    },
  },
  scatter_add: {
    default: {
      shaderFile: 'scatter_add.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    vec4: {
      shaderFile: 'scatter_add_vec4.wgsl',
      entryPoint: 'scatter_add_vec4',
      workgroupSize: [64, 1, 1],
      requires: [],
    },
    dynamic: {
      shaderFile: 'scatter_add_dynamic.wgsl',
      entryPoint: 'scatter_add_dynamic',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    accumulate: {
      shaderFile: 'scatter_add.wgsl',
      entryPoint: 'scatter_add_accumulate',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
  },
  moe_gather: {
    count: {
      shaderFile: 'moe_gather.wgsl',
      entryPoint: 'count_and_map',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    gather: {
      shaderFile: 'moe_gather.wgsl',
      entryPoint: 'gather_tokens',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    gather_vec4: {
      shaderFile: 'moe_gather_vec4.wgsl',
      entryPoint: 'gather_tokens_vec4',
      workgroupSize: [64, 1, 1],
      requires: [],
    },
    single_pass: {
      shaderFile: 'moe_gather.wgsl',
      entryPoint: 'gather_single_pass',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    sparse: {
      shaderFile: 'moe_gather.wgsl',
      entryPoint: 'count_and_map',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
  },
  swiglu: {
    rowsplit_bias: {
      shaderFile: 'fused_swiglu.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
  },
  bias_add: {
    default: {
      shaderFile: 'bias_add.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    f16: {
      shaderFile: 'bias_add_f16.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: ['shader-f16'],
    },
  },
  cast: {
    f32_to_f16: {
      shaderFile: 'cast_f32_to_f16.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: ['shader-f16'],
    },
    f16_to_f32: {
      shaderFile: 'cast_f16_to_f32.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: ['shader-f16'],
    },
  },
  // Split fused QKV output into separate Q, K, V buffers
  split_qkv: {
    default: {
      shaderFile: 'split_qkv.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
  },
  sample: {
    argmax: {
      shaderFile: 'sample.wgsl',
      entryPoint: 'argmax',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    argmax_reduce: {
      shaderFile: 'sample.wgsl',
      entryPoint: 'argmax_reduce',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    find_topk_phase1: {
      shaderFile: 'sample.wgsl',
      entryPoint: 'find_topk_phase1',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    find_topk_phase2: {
      shaderFile: 'sample.wgsl',
      entryPoint: 'find_topk_phase2',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    softmax_and_sample: {
      shaderFile: 'sample.wgsl',
      entryPoint: 'softmax_and_sample',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
    single_pass: {
      shaderFile: 'sample.wgsl',
      entryPoint: 'sample_single_pass',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
  },
  bf16_to_f32: {
    default: {
      shaderFile: 'bf16_to_f32.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: [],
    },
  },
  bf16_to_f16: {
    default: {
      shaderFile: 'bf16_to_f16.wgsl',
      entryPoint: 'main',
      workgroupSize: [256, 1, 1],
      requires: ['shader-f16'],
    },
  },
};

// ============================================================================
// Config Helpers
// ============================================================================


export function getKernelConfig(operation, variant) {
  const config = KERNEL_CONFIGS[operation]?.[variant];
  if (!config) {
    throw new Error(`Unknown kernel: ${operation}/${variant}`);
  }
  return config;
}


export function setKernelValidator(
  operation,
  variant,
  validator
) {
  const config = KERNEL_CONFIGS[operation]?.[variant];
  if (config) {
    config.validate = validator;
  }
}
