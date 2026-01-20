/**
 * quantization.ts - Browser quantization helpers
 *
 * @module browser/quantization
 */

import type { QuantizationInfoSchema, ConverterConfigSchema } from '../config/index.js';
import type { QuantizeResult } from '../converter/quantizer.js';

export declare function normalizeQuantTag(value: string | null | undefined): string;

export declare function resolveManifestQuantization(quantize: string | null, fallback: string): string;

export declare function buildQuantizationInfo(
  opts: ConverterConfigSchema,
  originalDtype: string,
  embedDtype: string | null,
  lmHeadDtype: string | null,
  hasVision?: boolean,
  hasAudio?: boolean,
  hasProjector?: boolean,
  modelConfig?: Record<string, unknown> | null
): QuantizationInfoSchema;

export declare function resolveModelId(
  modelId: string | null,
  baseName: string,
  variantTag: string | undefined
): string;

export declare function toWebGPUDtype(dtype: string): string;

export declare function isMatmulWeight(name: string, shape: number[]): boolean;

export declare function resolveTensorDtype(
  name: string,
  shape: number[],
  origDtype: string,
  quantizationInfo: QuantizationInfoSchema | null
): string;

export declare function resolveQ4KLayout(
  name: string,
  shape: number[],
  quantizationInfo: QuantizationInfoSchema | null
): string | null;

export declare function getQ4KOutputSize(shape: number[], layout: string | null): number;

export declare function decodeTensorToFloat32(buffer: ArrayBuffer, sourceDtype: string): Float32Array;

export declare function createQ4KChunkStream(
  chunks: AsyncIterable<Uint8Array>,
  sourceDtype: string,
  shape: number[],
  layout: string | null,
  chunkSizeBytes: number
): AsyncGenerator<Uint8Array>;

export declare function createF16ChunkStream(
  chunks: AsyncIterable<Uint8Array>,
  sourceDtype: string
): AsyncGenerator<Uint8Array>;

export declare function quantizeToQ4KColumnWise(
  data: Float32Array,
  shape: [number, number]
): QuantizeResult & { transposedShape: [number, number] };
