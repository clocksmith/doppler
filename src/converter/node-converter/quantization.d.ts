/**
 * Quantization utilities for the Node.js Model Converter.
 *
 * @module converter/node-converter/quantization
 */

import type { ConvertOptions } from './types.js';
import type { QuantizationInfoSchema } from '../../config/index.js';

export declare function normalizeQuantTag(value: string | null | undefined): string;

export declare function validateQuantType(value: string | null, flagName: string): void;

export declare function resolveManifestQuantization(quantize: string | null, fallback: string): string;

export declare function buildVariantTag(info: QuantizationInfoSchema): string;

export declare function buildQuantizationInfo(
  opts: ConvertOptions,
  originalDtype: string,
  embedDtype: string | null,
  lmHeadDtype: string | null,
  hasVision?: boolean,
  hasAudio?: boolean,
  hasProjector?: boolean
): QuantizationInfoSchema;

export declare function resolveModelId(
  modelId: string | null,
  baseName: string,
  variantTag: string | undefined
): string;

export declare function toWebGPUDtype(dtype: string): string;
