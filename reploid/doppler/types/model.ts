/**
 * Model Configuration Types
 */

/** Supported model architectures */
export type ModelArchitecture =
  | 'llama'
  | 'mistral'
  | 'gemma'
  | 'gemma2'
  | 'gemma3'
  | 'qwen2'
  | 'phi3'
  | 'gptoss'
  | 'mixtral';

/** Attention type variants */
export type AttentionType = 'mha' | 'gqa' | 'mqa';

/** Activation function types */
export type ActivationType = 'silu' | 'gelu' | 'relu' | 'swiglu';

/** Normalization types */
export type NormType = 'rmsnorm' | 'layernorm';

/** Position embedding types */
export type RopeType = 'standard' | 'neox' | 'yarn' | 'dynamic';

/** Core model configuration */
export interface ModelConfig {
  architecture: ModelArchitecture;
  modelId: string;
  vocabSize: number;
  hiddenSize: number;
  intermediateSize: number;
  numHiddenLayers: number;
  numAttentionHeads: number;
  numKeyValueHeads: number;
  headDim: number;
  maxPositionEmbeddings: number;
  rmsNormEps: number;
  ropeTheta: number;
  ropeType?: RopeType;
  tieWordEmbeddings: boolean;
  attentionType: AttentionType;
  activation: ActivationType;
  normType: NormType;

  // MoE specific
  numExperts?: number;
  numExpertsPerToken?: number;
  routerAux?: boolean;

  // Sliding window
  slidingWindow?: number;

  // Quantization
  quantization?: QuantizationConfig;
}

/** Quantization configuration */
export interface QuantizationConfig {
  format: QuantFormat;
  bitsPerWeight: number;
  groupSize?: number;
  modulesToNotConvert?: string[];
}

/** Supported quantization formats */
export type QuantFormat =
  | 'f32'
  | 'f16'
  | 'bf16'
  | 'q8_0'
  | 'q4_0'
  | 'q4_1'
  | 'q4_k'
  | 'q4_k_m'
  | 'q5_k'
  | 'q5_k_m'
  | 'q6_k'
  | 'q8_k';

/** Model family detection result */
export interface ModelFamily {
  family: ModelArchitecture;
  variant?: string;
  version?: number;
}

/** Model adapter interface for architecture-specific logic */
export interface ModelAdapter {
  readonly family: ModelFamily;
  readonly config: ModelConfig;

  /** Get tensor name for a logical tensor */
  getTensorName(logical: string, layer?: number): string;

  /** Get layer count */
  getLayerCount(): number;

  /** Check if model uses MoE */
  isMoE(): boolean;

  /** Get attention configuration */
  getAttentionConfig(): AttentionConfig;
}

/** Attention-specific configuration */
export interface AttentionConfig {
  type: AttentionType;
  numHeads: number;
  numKVHeads: number;
  headDim: number;
  scale: number;
  slidingWindow?: number;
  causalMask: boolean;
}
