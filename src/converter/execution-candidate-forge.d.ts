import type { ModelIRV2 } from '../config/model-ir-v2.js';
import type { TargetPlanV2 } from '../config/target-plan.js';

export const EXECUTION_CANDIDATE_FORGE_SCHEMA_ID: 'doppler.execution-candidate-forge/v1';

export declare function searchExecutionCandidates(options: {
  modelIR: ModelIRV2;
  entryPointId: string;
  vocabulary: Record<string, unknown>;
  proposals: Array<Record<string, unknown>>;
}): Record<string, unknown>;
export declare function promoteExecutionCandidate(
  candidate: Record<string, unknown>,
  evidence: { qualification: TargetPlanV2['qualification']; initialExecutionIdentity: TargetPlanV2['initialExecutionIdentity'] }
): TargetPlanV2;
