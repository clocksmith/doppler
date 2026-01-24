/**
 * Quantization helpers shared across converter paths.
 *
 * @module converter/quantization-info
 */

import type { ConvertOptions } from './core.js';
import type { ConverterConfigSchema, QuantizationInfoSchema } from '../config/index.js';

export declare function normalizeQuantTag(value: string | null | undefined): string;

export declare function validateQuantType(value: string | null, flagName: string): void;

export declare function resolveManifestQuantization(quantize: string | null, fallback: string): string;

export declare function buildVariantTag(info: QuantizationInfoSchema): string;

export declare function normalizeQ4KLayout(value: string | null | undefined): string | null;

export declare function buildQuantizationInfo(
  opts: ConvertOptions | ConverterConfigSchema,
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
