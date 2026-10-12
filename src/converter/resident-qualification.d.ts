import type { RDRRManifest } from '../formats/rdrr/index.js';
import type { LayerPartitionPlan } from '../inference/pipelines/text/layer-partition-contract.js';
import type { InitialExecutionIdentity } from '../config/initial-execution-identity.js';
export interface ResidentQualificationReport {
  schema: string; operation: string; status: string; modelId: string; manifestIdentity: string;
  executionGraphHash: string; surface: string; initialExecutionIdentity: InitialExecutionIdentity;
  partitionPlan: LayerPartitionPlan; partitionPlanHash: string; partitionIndex: 0 | 1;
  comparison: { contractHash: string; referenceHash: string; captureHash: string; evaluatorHash: string; tolerance: number;
    complete: boolean; steps: Array<{index: number; maxAbsoluteError: number}> };
}
export declare function assertResidentQualification(report: ResidentQualificationReport, context: {
  manifest: RDRRManifest; manifestIdentity: string; executionGraphHash: string;
  initialExecutionIdentity: InitialExecutionIdentity | null;
  contract: { schema: string; referenceHash: string; evaluatorHash: string; tolerance: number; comparedSteps: number } | null;
}): { operation: 'residentPartition'; surface: string; status: 'passed'; partitionPlanHash: string;
  partitionIndex: 0 | 1; comparedSteps: number; transcriptHash: string };
