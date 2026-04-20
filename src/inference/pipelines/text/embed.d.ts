/**
 * Token embedding lookup with optional Gemma scaling.
 */

import type { CommandRecorder } from '../../../gpu/command-recorder.js';
import type { ProbeConfigSchema } from '../../../config/schema/index.js';
import type { ExecutionV1PoliciesSchema } from '../../../config/schema/execution-v1.schema.js';
import type { Tensor } from '../../../gpu/tensor.js';
import type { CpuWeightBuffer } from '../../../gpu/weight-buffer.js';

export interface EmbedConfig {
  hiddenSize: number;
  vocabSize: number;
  scaleEmbeddings: boolean;
  debug?: boolean;
  recorder?: CommandRecorder;
  debugProbes?: ProbeConfigSchema[];
  outputBuffer?: GPUBuffer;
  numTokens?: number;
  indexOffset?: number;
  transpose?: boolean;
  activationDtype?: 'f16' | 'f32';
  embeddingDtype?: 'f16' | 'f32';
  executionPolicies?: ExecutionV1PoliciesSchema | null;
  probeStage?: string;
  inputHiddenSize?: number;
  hiddenOffset?: number;
}

export interface ValidationResult {
  min: number;
  max: number;
  mean: number;
  zeros: number;
  nanCount: number;
  infCount: number;
}

export function embed(
  tokenIds: number[] | Uint32Array | GPUBuffer,
  embedBuffer: GPUBuffer | Float32Array | CpuWeightBuffer,
  config: EmbedConfig
): Promise<Tensor>;

/**
 * True when `value` looks like a range-backed CPU embedding source
 * (has a typed byte range that the loader can pull from).
 */
export declare function isRangeBackedCpuEmbeddingSource(value: unknown): boolean;

/**
 * Normalize a range bytes payload (typed array or buffer view) into a
 * Uint8Array, throwing a labeled error when the shape is not usable.
 */
export declare function normalizeRangeBytes(value: unknown, label: string): Uint8Array;

/**
 * Decode a range-backed chunk into `output` at `dstOffset` using the
 * declared `sourceDtype` and per-row `hiddenSize`.
 */
export declare function decodeRangeChunkIntoOutput(
  bytes: Uint8Array,
  sourceDtype: string,
  output: Float32Array,
  dstOffset: number,
  hiddenSize: number
): void;
