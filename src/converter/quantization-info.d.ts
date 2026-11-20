/**
 * Quantization helpers shared across converter paths.
 *
 * @module converter/quantization-info
 */

import type { ConvertOptions } from './core.js';
import type { ConverterConfigSchema, QuantizationInfoSchema } from '../config/schema/index.js';

/** Source model types whose experts use Doppler's qwen3_5 expert format. */
export declare const QWEN3_5_EXPERT_SOURCE_TYPES: ReadonlySet<string>;

export declare function normalizeQuantTag(value: string | null | undefined): string;

/**
 * The form of a quantization name used inside model IDs and variant tags: lowercase letters and digits only.
 * The plain GGUF block formats lose their underscore (`q4_0` becomes `q40`); manifest fields keep the GGUF spelling.
 */
export declare function quantNameToken(value: string | null | undefined): string;

/**
 * The variant tag a manifest stores: the weight format, then one group per type that differs from it, with `e`
 * for embeddings, `h` for the output head and `a` for activations. It is manifest data and keeps the GGUF
 * spelling of each format, so that converting an existing model again reproduces its identity.
 */
export declare function buildVariantTag(info: QuantizationInfoSchema): string;

export declare function resolveManifestQuantization(quantize: string | null, fallback: string): string;

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

export declare function resolveEffectiveQuantizationInfo(
  baseInfo: QuantizationInfoSchema,
  tensors: Record<string, unknown>
): QuantizationInfoSchema;
