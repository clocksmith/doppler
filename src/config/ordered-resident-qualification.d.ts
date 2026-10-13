import type { OrderedLayerPartitionPlan } from './layer-partition-plan.js';
export interface OrderedResidentQualificationContract {
  schema: 'doppler.resident-qualification-contract/v2';
  manifestIdentity: string; executionGraphHash: string;
  partitionPlan: OrderedLayerPartitionPlan; partitionPlanHash: string;
  referenceHash: string; evaluatorHash: string; tolerance: number; comparedSteps: number;
  stages: readonly { index: number; surface: 'browser-webgpu' | 'node-webgpu'; providerIdentityHash: string }[];
}
export interface OrderedResidentQualificationRecord {
  orderedContract: OrderedResidentQualificationContract;
  operation: 'residentPartition'; status: 'passed'; surface: string;
  partitionIndex: number; partitionPlanHash: string; providerIdentityHash: string; comparedSteps: number;
}
export declare function assertOrderedResidentContract(contract: unknown): OrderedResidentQualificationContract;
export declare function assertOrderedResidentRecord(record: OrderedResidentQualificationRecord): OrderedResidentQualificationContract;
export declare function assertOrderedResidentQualificationSet(records: readonly unknown[]): void;
