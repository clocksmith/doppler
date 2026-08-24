import type {
  CpuTensorRangeSource,
  CpuWeightBuffer,
} from '../../../../gpu/weight-buffer.js';
import type { LargeWeightConfigSchema } from '../../../../config/schema/index.js';

export declare function isRangeBackedCpuWeightSource(
  value: unknown
): value is CpuTensorRangeSource;
export declare function normalizeRangeBytes(value: unknown, label: string): Uint8Array;
export declare function resolveCpuWeightDims(
  lmHead: CpuWeightBuffer
): { vocabSize: number; hiddenSize: number };
export declare function extractLmHeadChunk(
  data: Float32Array | Uint16Array | CpuTensorRangeSource,
  layout: 'row' | 'column',
  hiddenSize: number,
  vocabSize: number,
  rowOffset: number,
  rowCount: number,
  sourceDtype: 'f16' | 'f32' | 'bf16'
): Promise<Float32Array>;
export declare function writeChunkLogits(
  target: Float32Array,
  chunk: Float32Array,
  numTokens: number,
  vocabSize: number,
  rowOffset: number,
  rowCount: number
): void;
export declare function shouldMaterializeSplitLmHeadGPU(
  lmHead: CpuWeightBuffer,
  largeWeightConfig: LargeWeightConfigSchema
): boolean;

