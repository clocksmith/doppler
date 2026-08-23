import type { ModelIRV2 } from '../config/model-ir-v2.js';

export const LINEAGE_LOWERING_FORGE_SCHEMA_ID: 'doppler.lineage-lowering-forge/v1';

export interface LineageLoweringReceipt {
  schema: 'doppler.lineage-lowering-receipt/v1';
  modelId: string;
  sourceModelIRHash: `sha256:${string}`;
  modelIRHash: `sha256:${string}`;
  modelIR: ModelIRV2;
  template: string;
  author: { kind: 'human' | 'ai' | 'tool'; actor: string; proposalId?: string };
  generatedCandidates: number;
  rejectedCandidates: Array<Record<string, unknown>>;
  acceptedCandidateId: string;
  dispositions: Array<Record<string, unknown>>;
  unresolvedFacts: [];
  conversionConfigDigest: `sha256:${string}`;
  conversionConfig: Record<string, unknown>;
}

export declare function materializeLineageConversionCandidate(options: {
  modelIR: ModelIRV2;
  template: Record<string, unknown>;
  recipe: Record<string, unknown>;
}): LineageLoweringReceipt;
