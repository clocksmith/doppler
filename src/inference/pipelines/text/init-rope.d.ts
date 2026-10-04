import type { RoPEConfig } from './init.js';
import type { ParsedModelConfig } from './config.js';
import type { RuntimeConfigSchema } from '../../../config/schema/index.js';
import type { LayerPartitionPlan } from './layer-partition-contract.js';

export interface GPURoPEBuffers {
  cos: GPUBuffer;
  sin: GPUBuffer;
  localCos: GPUBuffer | null;
  localSin: GPUBuffer | null;
}
export function initRoPEFrequencies(config: RoPEConfig, useGPU: boolean): Promise<GPURoPEBuffers>;
/** Release one acquired lease; repeated release is harmless. Other leases stay live. */
export function releaseRoPEFrequencies(lease: GPURoPEBuffers | null | undefined): void;
export function isGPURoPEBuffers(buffers: unknown): buffers is GPURoPEBuffers;
export function _initRoPE(this: {
  modelConfig: ParsedModelConfig;
  modelPartition?: { plan: LayerPartitionPlan; index: 0 | 1 } | null;
  runtimeConfig?: RuntimeConfigSchema;
  useGPU: boolean;
  ropeFrequencyLease: GPURoPEBuffers | null;
  ropeFreqsCos: GPUBuffer | null;
  ropeFreqsSin: GPUBuffer | null;
  ropeLocalCos: GPUBuffer | null;
  ropeLocalSin: GPUBuffer | null;
}): Promise<void>;
