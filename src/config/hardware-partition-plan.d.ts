import type { OrderedLayerPartitionRequest, OrderedLayerPartitionPlan, createOrderedLayerPartitionPlanFromManifest } from './layer-partition-plan.js';
export interface PartitionHardwareProfile {
  hardwareId: string;
  manifestHash: string;
  workloadIdentity: string;
  sourceReceipt: string;
  measuredAtMs: number;
  memoryBudgetBytes: number;
  /** Include runtime reserves, embedding/head as appropriate for this ordered device. */
  fixedBytes: number;
  fixedMs: number;
  /** Full workload cost on this device, indexed by global layer. No weights-only estimates. */
  layers: readonly { residentBytes: number; scratchBytes: number; elapsedMs: number }[];
}
export interface HardwarePartitionRequest {
  workloadIdentity: string;
  nowMs: number;
  maxProfileAgeMs: number;
  maxLayers: number;
  activation: Omit<OrderedLayerPartitionRequest, 'splitLayers'>;
  profiles: readonly [PartitionHardwareProfile, PartitionHardwareProfile];
  /** Entire workload transfer time per interior boundary; index 0 means split after layer 0. */
  transferMs: readonly number[];
}
/** Metadata only: minimizes the supplied additive single-request estimate, not measured runtime or throughput.
 * Profiles are caller-supplied evidence, not authenticated by this helper. No fallback or live migration.
 */
export declare function proposeHardwarePartitionPlan(
  manifest: Parameters<typeof createOrderedLayerPartitionPlanFromManifest>[0], request: HardwarePartitionRequest
): Readonly<{ plan: OrderedLayerPartitionPlan; manifestHash: string; workloadIdentity: string;
  estimatedMs: number; memoryBytes: readonly number[]; hardwareIds: readonly string[];
  sourceReceipts: readonly string[]; evaluatedSplits: number; feasibleSplits: number }>;
