import type { ModelIRV2 } from '../config/model-ir-v2.js';
import type { SourceTruthForgeReceipt } from './source-truth-forge.js';
import type { auditEntryPointLowerability } from './execution-candidate-forge.js';

export interface ModelSupportAssessment {
  schema: 'doppler.model-support-assessment/v1';
  sourceIdentity: ModelIRV2['sourceIdentity'];
  modelIRHash: string;
  vocabularyDigest: string;
  implementationClass: 'missing-behavior-or-evidence' | 'known-operations-require-recipe';
  audits: ReturnType<typeof auditEntryPointLowerability>[];
  tasks: Array<Record<string, unknown>>;
  qualified: false;
  nextRequirement: string;
  digest: string;
}
export declare function assessModelSupport(options: {
  modelIR: ModelIRV2;
  entryPointIds: string[];
  vocabulary: Record<string, unknown>;
  unresolvedFacts: SourceTruthForgeReceipt['unresolvedFacts'];
}): ModelSupportAssessment;
