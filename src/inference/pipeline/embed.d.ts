/**
 * Token embedding lookup with optional Gemma scaling.
 */

import type { CommandRecorder } from '../../gpu/command-recorder.js';
import type { ProbeConfigSchema } from '../../config/schema/index.js';
import type { Tensor } from '../../gpu/tensor.js';
import type { CpuWeightBuffer } from '../../gpu/weight-buffer.js';

export interface EmbedConfig {
  hiddenSize: number;
  vocabSize: number;
  scaleEmbeddings: boolean;
  debug?: boolean;
  recorder?: CommandRecorder;
  debugProbes?: ProbeConfigSchema[];
  outputBuffer?: GPUBuffer;
  numTokens?: number;
  transpose?: boolean;
  activationDtype?: 'f16' | 'f32';
  embeddingDtype?: 'f16' | 'f32';
}

export interface ValidationResult {
  min: number;
  max: number;
  mean: number;
  zeros: number;
  nanCount: number;
  infCount: number;
}

/**
 * Record scale operation (batched, no submit)
 */
export function recordScale(
  recorder: CommandRecorder,
  inputBuffer: GPUBuffer,
  scale: number,
  count: number,
  useF16?: boolean
): GPUBuffer;

/**
 * Scale GPU buffer (standalone, with submit)
 * @deprecated Use recordScale with CommandRecorder instead
 */
export function scaleGPUBuffer(
  inputBuffer: GPUBuffer,
  scale: number,
  count: number,
  useF16?: boolean
): Promise<GPUBuffer>;

export function embed(
  tokenIds: number[] | Uint32Array | GPUBuffer,
  embedBuffer: GPUBuffer | Float32Array | CpuWeightBuffer,
  config: EmbedConfig
): Promise<Tensor>;

export function validateEmbedding(
  buffer: GPUBuffer,
  label: string,
  numTokens: number,
  hiddenSize: number
): Promise<ValidationResult | null>;
