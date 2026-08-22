export const MODEL_IR_SCHEMA_ID = 'doppler.model-ir/v1';
export const MODEL_IR_SCHEMA_VERSION = 1;

export interface ModelIRLayer {
  index: number;
  type: 'transformer' | 'linear-attention' | 'hybrid' | 'moe' | 'encoder-only';
  attention?: Record<string, unknown>;
  ffn?: Record<string, unknown>;
  norm?: Record<string, unknown>;
}

export interface ModelIRAttentionGeometry {
  numHeads: number;
  numKvHeads: number;
  headDim: number;
  slidingWindow?: number | null;
  qkNorm?: boolean;
}

export interface ModelIRNormalization {
  type: 'rmsnorm' | 'layernorm' | 'gemma-rmsnorm';
  eps: number;
}

export interface ModelIRRoPE {
  dimension: number;
  baseFreq: number;
  scaling?: Record<string, unknown> | null;
}

export interface ModelIRFFN {
  type: 'swiglu' | 'gelu' | 'silu' | 'moe-swiglu';
  intermediateSize: number;
  numExperts?: number | null;
  numExpertsPerToken?: number | null;
}

export interface ModelIROutputTopology {
  headType: 'causal-lm' | 'sequence-pool' | 'embedding' | 'cross-encoder-score';
  tieWeights: boolean;
  poolingMode?: 'last' | 'mean' | 'cls' | null;
}

export interface ModelIR {
  schema: 'doppler.model-ir/v1';
  schemaVersion: 1;
  modelId: string;
  architecture: string;
  vocabSize: number;
  hiddenSize: number;
  numLayers: number;
  tensorRoles: Record<string, { role: string; shape: number[]; semanticDtype?: string }>;
  layers: ModelIRLayer[];
  attentionGeometry: ModelIRAttentionGeometry;
  normalization: ModelIRNormalization;
  rope: ModelIRRoPE | null;
  ffn: ModelIRFFN;
  outputTopology: ModelIROutputTopology;
  phases: string[];
}

export declare function validateModelIR(ir: unknown): { ok: boolean; errors: string[] };
export declare function hashModelIR(ir: unknown): `sha256:${string}`;
export declare function createModelIR(params: Partial<ModelIR>): ModelIR;
