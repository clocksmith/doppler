import type { ModelIRV2 } from '../config/model-ir-v2.js';

export const SOURCE_TRUTH_FORGE_SCHEMA_ID: 'doppler.source-truth-forge/v2';
export const SOURCE_TRUTH_FORGE_VERSION: '2.0.0';

export interface SourceTruthForgeReceipt {
  schema: 'doppler.source-truth-forge-receipt/v2';
  modelIR: ModelIRV2;
  intakeDigest: `sha256:${string}`;
  unresolvedFacts: [];
  generatedCandidates: number;
  rejectedCandidates: number;
  acceptedCandidates: 1;
  acceptedProposalId: string;
}

export declare function forgeModelIRV2(
  packet: Record<string, unknown>,
  sources: Record<string, unknown>
): SourceTruthForgeReceipt;
export declare function createSourceTruthPacket(
  spec: Record<string, unknown>,
  sources: Record<string, unknown>
): Record<string, unknown>;
