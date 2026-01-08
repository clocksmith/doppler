/**
 * Kernel Selection - All operation selection logic in one place
 *
 * This is the single source of truth for kernel variant selection.
 * Hotswappable for debugging/tuning - modify this file and reload.
 *
 * @module config/kernels/selection
 */

/**
 * Context for matmul kernel selection.
 */
export interface MatmulContext {
  /** Batch/sequence dimension */
  M: number;
  /** Output dimension */
  N: number;
  /** Hidden/input dimension */
  K: number;
  /** Input A dtype ('f16', 'f32', 'q4k', etc.) */
  aDtype?: string;
  /** Input B (weights) dtype */
  bDtype?: string;
  /** Desired output dtype ('f16' or 'f32') */
  outputDtype?: string;
  /** Whether B is transposed */
  transposeB?: boolean;
  /** Prefer F16 kernels if available */
  preferF16?: boolean;
  /** Prefer vec4 variants */
  useVec4?: boolean;
}

/**
 * Select matmul kernel variant.
 */
export function selectMatmul(context: MatmulContext): string;

/**
 * Context for attention kernel selection.
 */
export interface AttentionContext {
  /** Query sequence length */
  seqLen: number;
  /** KV cache sequence length */
  kvSeqLen: number;
  /** Number of attention heads */
  numHeads: number;
  /** Dimension per head */
  headDim: number;
  /** Whether KV cache is F16 */
  useF16KV?: boolean;
  /** Device shared memory limit */
  sharedMemoryLimit?: number;
}

/**
 * Select attention kernel variant.
 */
export function selectAttention(context: AttentionContext): string;

/**
 * Context for dequant kernel selection.
 */
export interface DequantContext {
  /** Quantization type ('q4k', 'q6k', 'q8_0', 'mxfp4') */
  quantType?: string;
  /** Output dtype ('f16' or 'f32') */
  outputDtype?: string;
  /** Prefer vec4 variants */
  useVec4?: boolean;
  /** Whether dequanting MoE expert weights */
  isExpert?: boolean;
}

/**
 * Select dequant kernel variant.
 */
export function selectDequant(context: DequantContext): string;

/**
 * Context for RMSNorm kernel selection.
 */
export interface RMSNormContext {
  /** Hidden dimension size */
  hiddenSize?: number | null;
  /** Whether to fuse residual addition */
  hasResidual?: boolean;
}

/**
 * Select RMSNorm kernel variant.
 */
export function selectRMSNorm(context: RMSNormContext): string;

/**
 * Context for fused matmul+RMSNorm selection.
 */
export interface FusedMatmulRMSNormContext {
  /** Output dimension */
  N: number;
}

/**
 * Select fused matmul+RMSNorm kernel variant.
 */
export function selectFusedMatmulRMSNorm(context: FusedMatmulRMSNormContext): string;

/**
 * Context for fused FFN kernel selection.
 */
export interface FFNContext {
  /** Batch size */
  batchSize: number;
  /** FFN intermediate dimension */
  intermediateSize: number;
  /** Weight dtype ('f16' or 'f32') */
  weightDtype?: string;
}

/**
 * Select fused FFN kernel variant.
 */
export function selectFFN(context: FFNContext): string;

/**
 * Context for softmax kernel selection.
 */
export interface SoftmaxContext {
  /** Size of softmax dimension */
  innerSize: number;
}

/**
 * Select softmax kernel variant.
 */
export function selectSoftmax(context: SoftmaxContext): string;

/**
 * Context for gather kernel selection.
 */
export interface GatherContext {
  /** Embedding table dtype */
  embeddingDtype?: string;
  /** Prefer vec4 variants */
  useVec4?: boolean;
}

/**
 * Select gather kernel variant.
 */
export function selectGather(context: GatherContext): string;

/**
 * Context for sample kernel selection.
 */
export interface SampleContext {
  /** Vocabulary size */
  vocabSize: number;
  /** Sampling temperature */
  temperature?: number;
  /** Top-K sampling parameter */
  topK?: number;
}

/**
 * Select sample kernel variant.
 */
export function selectSample(context: SampleContext): string;

/**
 * Context for RoPE kernel selection.
 */
export interface RopeContext {
  /** RoPE variant ('default', 'ntk', 'yarn') */
  ropeType?: string;
  /** Whether to compute frequencies */
  computeFreqs?: boolean;
  /** Apply to both Q and K */
  applyToQK?: boolean;
}

/**
 * Select RoPE kernel variant.
 */
export function selectRope(context: RopeContext): string;

/**
 * Context for activation kernel selection.
 */
export interface ActivationContext {
  /** Activation type ('silu', 'gelu', 'geglu') */
  activation?: string;
  /** Whether gated activation */
  hasGate?: boolean;
  /** Gate/up split by rows */
  rowSplit?: boolean;
  /** Prefer vec4 variants */
  useVec4?: boolean;
}

/**
 * Select activation kernel variant.
 */
export function selectActivation(context: ActivationContext): string;

/**
 * Context for residual kernel selection.
 */
export interface ResidualContext {
  /** Prefer vec4 variants */
  useVec4?: boolean;
}

/**
 * Select residual kernel variant.
 */
export function selectResidual(context: ResidualContext): string;

/**
 * Context for scatter add kernel selection.
 */
export interface ScatterAddContext {
  /** Prefer vec4 variants */
  useVec4?: boolean;
  /** Accumulate mode */
  accumulate?: boolean;
  /** Dynamic indices */
  dynamic?: boolean;
}

/**
 * Select scatter add kernel variant for MoE.
 */
export function selectScatterAdd(context: ScatterAddContext): string;
