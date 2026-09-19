import type { RoPEConfig } from './init.js';
import type { ParsedModelConfig } from './config.js';

export interface GPURoPEBuffers {
  cos: GPUBuffer;
  sin: GPUBuffer;
  localCos: GPUBuffer | null;
  localSin: GPUBuffer | null;
}
export function initRoPEFrequencies(config: RoPEConfig, useGPU: boolean): Promise<GPURoPEBuffers>;
export function isGPURoPEBuffers(buffers: unknown): buffers is GPURoPEBuffers;
export function _initRoPE(this: {
  modelConfig: ParsedModelConfig;
  useGPU: boolean;
  ropeFreqsCos: GPUBuffer | null;
  ropeFreqsSin: GPUBuffer | null;
  ropeLocalCos: GPUBuffer | null;
  ropeLocalSin: GPUBuffer | null;
}): Promise<void>;
