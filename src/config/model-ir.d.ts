export const MODEL_IR_SCHEMA_ID: 'doppler.model-ir/v1';
export const MODEL_IR_SCHEMA_VERSION: 1;
export {
  MODEL_IR_V2_SCHEMA_ID,
  MODEL_IR_V2_SCHEMA_VERSION,
  createModelIRV2,
  validateModelIRV2,
} from './model-ir-v2.js';
export type { ModelIRV2, ModelIRV2Fact, ModelIRV2Node } from './model-ir-v2.js';

export interface ModelIRLayer {
  index: number;
  type: string;
  attention?: Record<string, unknown>;
  ffn?: Record<string, unknown>;
  norm?: Record<string, unknown>;
}

export interface ModelIR {
  schema: 'doppler.model-ir/v1';
  schemaVersion: 1;
  modelId: string;
  architecture: string;
  vocabSize: number;
  hiddenSize: number;
  numLayers: number;
  sourceIdentity: {
    manifestArtifactId: string;
    manifestHash: `sha256:${string}`;
    sourceCheckpointId?: string;
  };
  tensorRoles: Record<string, { role: string; shape: number[]; semanticDtype: string }>;
  layers: ModelIRLayer[];
  attentionGeometry: {
    numHeads: number;
    numKvHeads: number;
    headDim: number;
    qkNorm?: boolean;
  };
  normalization: { type: string; eps: number };
  rope: Record<string, unknown> | null;
  ffn: { type: string; intermediateSize: number };
  outputTopology: { headType: string; tieWeights: boolean; sequence?: Record<string, unknown> };
  phases: string[];
}

export declare function validateModelIR(ir: unknown): { ok: boolean; errors: string[] };
export declare function hashModelIR(ir: unknown): `sha256:${string}`;
export declare function createModelIR(params: Omit<ModelIR, 'schema' | 'schemaVersion'>): ModelIR;
