export * from './inference/pipelines/text/layer-partition-contract.js';
export { createResidentPartitionFactory, createManifestResidentPartitionFactory,
  configureDeviceMemoryBudget, inspectDeviceMemory } from './client/resident-partitions.js';
export { createVerifiedPieceStorage } from './storage/verified-piece-storage.js';
