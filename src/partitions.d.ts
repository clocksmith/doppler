export * from './inference/pipelines/text/layer-partition-contract.js';
export { createResidentPartitionFactory, createManifestResidentPartitionFactory,
  configureDeviceMemoryBudget, inspectDeviceMemory } from './client/resident-partitions.js';
export { createVerifiedPieceStorage } from './storage/verified-piece-storage.js';
export type { VerifiedPiece } from './storage/verified-piece-storage.js';
export type { ResidentPartitionFactory, ResidentPartitionOpenOptions } from './client/resident-partitions.js';
export type { ResidentPartitionAllocation, ResidentPartitionDescriptor, ResidentPartitionIdentity,
  ResidentPartitionLimits, ResidentPartitionSession, ResidentPartitionStep, ResidentRecoveryCapabilities,
  ResidentPartitionTokenizationRequest, ResidentPartitionTokenizationResult,
  ResidentPartitionARequest, ResidentPartitionAResult, ResidentPartitionBRequest,
  ResidentPartitionBResult, ResidentPartitionMetrics, PartitionTiming } from './inference/pipelines/text/resident-partition-contract.js';
