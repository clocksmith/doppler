import type { RDRRManifest } from '../formats/rdrr/index.js';
import type { AnyLayerPartitionPlan } from '../inference/pipelines/text/layer-partition-contract.js';
import type { InitialExecutionIdentity } from '../config/initial-execution-identity.js';
export interface ResidentQualificationReport {
  schema: string; operation: string; status: string; modelId: string; manifestIdentity: string;
  executionGraphHash: string; surface: string; initialExecutionIdentity: InitialExecutionIdentity;
  partitionPlan: AnyLayerPartitionPlan; partitionPlanHash: string; partitionIndex: number;
  providerIdentity?: Record<string, unknown>;
  comparison: { contractHash: string; referenceHash: string; captureHash: string; evaluatorHash: string; tolerance: number;
    complete: boolean; steps: Array<{index: number; maxAbsoluteError: number}> };
}
export declare function assertResidentQualification(report: ResidentQualificationReport, context: {
  manifest: RDRRManifest; manifestIdentity: string; executionGraphHash: string;
  initialExecutionIdentity: InitialExecutionIdentity | null;
  contract: { schema: string; referenceHash: string; evaluatorHash: string; tolerance: number; comparedSteps: number } | import('../config/ordered-resident-qualification.js').OrderedResidentQualificationContract | null;
}): { operation: 'residentPartition'; surface: string; status: 'passed'; partitionPlanHash: string;
  orderedContract?: import('../config/ordered-resident-qualification.js').OrderedResidentQualificationContract; providerIdentityHash?: string;
  partitionIndex: number; comparedSteps: number; transcriptHash: string };
