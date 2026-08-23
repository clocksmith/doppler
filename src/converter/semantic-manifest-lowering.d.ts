import type { ModelIRV2 } from '../config/model-ir-v2.js';

export const SEMANTIC_MANIFEST_LOWERING_SCHEMA_ID: 'doppler.semantic-manifest-lowering/v1';

export interface SemanticManifestLoweringReceipt {
  schema: 'doppler.semantic-manifest-lowering-receipt/v1';
  modelId: string;
  requestedModelId: string;
  entryPointId: string;
  sourceModelIRHash: `sha256:${string}`;
  modelIRHash: `sha256:${string}`;
  modelIR: ModelIRV2;
  author: { kind: 'human' | 'ai' | 'tool'; actor: string; proposalId?: string };
  template: string;
  generatedCandidates: number;
  rejectedCandidates: Array<Record<string, unknown>>;
  acceptedCandidateId: string;
  dispositions: Array<Record<string, unknown>>;
  unresolvedFacts: [];
  conversionConfigDigest: `sha256:${string}`;
  conversionConfig: Record<string, unknown>;
}

export declare function materializeSemanticManifestCandidate(options: {
  modelIR: ModelIRV2;
  template: Record<string, unknown>;
  recipe: Record<string, unknown>;
}): SemanticManifestLoweringReceipt;
