/**
 * Quantization utilities for the Node.js Model Converter.
 *
 * Handles quantization tag normalization, validation, and variant tag building.
 *
 * @module converter/node-converter/quantization
 */

import type { ConvertOptions } from './types.js';
import {
  DEFAULT_QUANTIZATION_DEFAULTS,
  type QuantizationInfoSchema,
} from '../../config/index.js';

/**
 * Normalize quantization tag to canonical short form.
 *
 * DOPPLER naming uses concise storage-only tags:
 * - q4k = Q4_K_M block quantization (the only Q4 we support)
 * - q6k = Q6_K block quantization
 * - q8_0 = Q8_0 quantization
 * - f16/bf16/f32 = Float formats
 * - fp8e4/fp8e5 = Float8 formats
 * - i4/i8 = Integer formats
 *
 * @param value - Input quantization string
 * @returns Normalized quantization tag
 */
export function normalizeQuantTag(value: string | null | undefined): string {
  if (!value) return 'f16';
  const lower = value.toLowerCase();

  // Q4_K_M variants -> q4k (canonical short form)
  if (lower === 'q4_k_m' || lower === 'q4k' || lower === 'q4' || lower === 'q4km') return 'q4k';
  // Q6_K variants -> q6k
  if (lower === 'q6_k' || lower === 'q6k' || lower === 'q6') return 'q6k';
  // Q8_0 (keep as-is, common format)
  if (lower === 'q8_0' || lower === 'q8') return 'q8_0';
  // Float formats
  if (lower === 'f16' || lower === 'fp16' || lower === 'float16') return 'f16';
  if (lower === 'bf16' || lower === 'bfloat16') return 'bf16';
  if (lower === 'f32' || lower === 'fp32' || lower === 'float32') return 'f32';
  // Float8 formats
  if (lower === 'fp8e4' || lower === 'fp8e4m3' || lower === 'e4m3') return 'fp8e4';
  if (lower === 'fp8e5' || lower === 'fp8e5m2' || lower === 'e5m2') return 'fp8e5';
  // Integer formats
  if (lower === 'i8' || lower === 'int8') return 'i8';
  if (lower === 'i4' || lower === 'int4') return 'i4';

  return lower;
}

/**
 * Validate that a quantization type is supported for conversion.
 *
 * Currently implemented:
 * - q4k: Q4_K_M block quantization
 * - f16, f32: Float formats (and bf16 auto-converted to f16)
 *
 * Not yet implemented (will error):
 * - q6k, q8_0: Other block quantization
 * - fp8e4, fp8e5: Float8 formats
 * - i4, i8: Integer formats
 *
 * @param value - Quantization type to validate
 * @param flagName - Name of CLI flag for error messages
 * @throws Error if quantization type is not supported
 */
export function validateQuantType(value: string | null, flagName: string): void {
  if (!value) return;
  const normalized = normalizeQuantTag(value);

  // Supported types
  const supported = ['q4k', 'f16', 'bf16', 'f32'];
  if (supported.includes(normalized)) return;

  // Not yet implemented
  const planned = ['q6k', 'q8_0', 'fp8e4', 'fp8e5', 'i4', 'i8'];
  if (planned.includes(normalized)) {
    throw new Error(
      `Quantization type "${normalized}" is not yet implemented.\n` +
      `Supported types: ${supported.join(', ')}\n` +
      `Planned types: ${planned.join(', ')}`
    );
  }

  throw new Error(`Unknown quantization type: "${value}" (flag: ${flagName})`);
}

/**
 * Resolve quantization for manifest field (display format).
 *
 * @param quantize - Quantization option
 * @param fallback - Fallback dtype if no quantization specified
 * @returns Manifest-format quantization string
 */
export function resolveManifestQuantization(quantize: string | null, fallback: string): string {
  if (!quantize) return fallback;
  const normalized = normalizeQuantTag(quantize);
  // Return uppercase for manifest field (display format)
  if (normalized === 'q4k') return 'Q4_K_M';
  if (normalized === 'q6k') return 'Q6_K';
  if (normalized === 'q8_0') return 'Q8_0';
  return normalized.toUpperCase();
}

/**
 * Build variant tag for model naming.
 *
 * Format: w{weights}[-e{embeddings}][-h{head}][-v{vision}][-a{audio}][-t{tts}][-p{projector}]
 *
 * Components are only included if they differ from defaults:
 * - embeddings defaults to weights
 * - head defaults to embeddings
 * - multimodal components only included if present
 *
 * Examples:
 * - "wq4k" (weights Q4K, embeddings same)
 * - "wq4k-ef16" (weights Q4K, embeddings F16)
 * - "wq4k-ef16-hf16" (with explicit head)
 * - "wq4k-vf16-pf16" (multimodal with vision + projector)
 *
 * @param info - Quantization info object
 * @returns Variant tag string
 */
export function buildVariantTag(info: QuantizationInfoSchema): string {
  const weights = info.weights;
  const embeddings = info.embeddings ?? weights;
  const lmHead = info.lmHead ?? embeddings;

  // Start with weights (always present)
  const parts = [`w${weights}`];

  // Add embeddings only if different from weights
  if (embeddings !== weights) {
    parts.push(`e${embeddings}`);
  }

  // Add head only if different from embeddings
  if (lmHead !== embeddings) {
    parts.push(`h${lmHead}`);
  }

  // Multimodal components (only if present)
  if (info.vision) {
    parts.push(`v${info.vision}`);
  }
  if (info.audio) {
    parts.push(`a${info.audio}`);
  }
  if (info.tts) {
    parts.push(`t${info.tts}`);
  }
  if (info.projector) {
    parts.push(`p${info.projector}`);
  }

  return parts.join('-');
}

/**
 * Build quantization info from conversion options.
 *
 * Handles all component quantization with proper defaults:
 * - weights: from --weight-quant or original dtype (WebGPU-safe)
 * - embeddings: from --embed-quant or defaults to original dtype (WebGPU-safe)
 * - lmHead: from --head-quant or defaults to embeddings
 * - vision/audio/tts/projector: from explicit flags only
 *
 * All float formats are normalized to WebGPU-safe types (bf16 -> f16).
 *
 * @param opts - Conversion options
 * @param originalDtype - Original dtype from source tensors
 * @param embedDtype - Detected embedding dtype (or null)
 * @param lmHeadDtype - Detected LM head dtype (or null)
 * @param hasVision - Whether model has vision encoder
 * @param hasAudio - Whether model has audio encoder
 * @param hasProjector - Whether model has multimodal projector
 * @returns Quantization info schema
 */
export function buildQuantizationInfo(
  opts: ConvertOptions,
  originalDtype: string,
  embedDtype: string | null,
  lmHeadDtype: string | null,
  hasVision = false,
  hasAudio = false,
  hasProjector = false
): QuantizationInfoSchema {
  // Validate all explicit quantization flags
  validateQuantType(opts.weightQuant, '--weight-quant');
  validateQuantType(opts.embedQuant, '--embed-quant');
  validateQuantType(opts.headQuant, '--head-quant');
  validateQuantType(opts.visionQuant, '--vision-quant');
  validateQuantType(opts.audioQuant, '--audio-quant');
  validateQuantType(opts.projectorQuant, '--projector-quant');

  // WebGPU only supports F16/F32 for floats - BF16 must convert to F16
  // This must be applied to ALL stored dtypes so naming matches storage
  const webgpuSafe = (dtype: string): string => {
    const normalized = normalizeQuantTag(dtype);
    // BF16 not supported in WebGPU, convert to F16
    if (normalized === 'bf16') return 'f16';
    return normalized;
  };

  // Weights: explicit flag > original dtype, always WebGPU-safe
  const weights = webgpuSafe(opts.weightQuant ?? originalDtype);

  // Embeddings: explicit flag > original dtype (WebGPU-safe)
  let embeddings: string;
  if (opts.embedQuant) {
    embeddings = webgpuSafe(opts.embedQuant);
  } else {
    embeddings = webgpuSafe(embedDtype || originalDtype);
  }

  // Head: explicit flag > original head dtype > embeddings
  // Only add explicit lmHead if it differs from embeddings
  let lmHead: string;
  if (opts.headQuant) {
    lmHead = webgpuSafe(opts.headQuant);
  } else if (lmHeadDtype) {
    // Model has explicit lm_head tensor with known dtype
    lmHead = webgpuSafe(lmHeadDtype);
  } else {
    // No explicit lm_head, will use embeddings (tied weights)
    lmHead = embeddings;
  }

  const info: QuantizationInfoSchema = {
    weights,
    embeddings,
    lmHead: lmHead !== embeddings ? lmHead : undefined,
  };

  // Multimodal components (only if present in model and explicitly set)
  if (hasVision && opts.visionQuant) {
    info.vision = normalizeQuantTag(opts.visionQuant);
  } else if (hasVision && !opts.textOnly) {
    info.vision = DEFAULT_QUANTIZATION_DEFAULTS.visionDtype;
  }

  if (hasAudio && opts.audioQuant) {
    info.audio = normalizeQuantTag(opts.audioQuant);
  } else if (hasAudio) {
    info.audio = DEFAULT_QUANTIZATION_DEFAULTS.audioDtype;
  }

  if (hasProjector && opts.projectorQuant) {
    info.projector = normalizeQuantTag(opts.projectorQuant);
  } else if (hasProjector && !opts.textOnly) {
    info.projector = DEFAULT_QUANTIZATION_DEFAULTS.projectorDtype;
  }

  // Runtime hints (not included in variantTag)
  if (opts.computePrecision) {
    info.compute = opts.computePrecision;
  }

  info.variantTag = buildVariantTag(info);
  return info;
}

/**
 * Resolve model ID with variant tag.
 *
 * @param modelId - Explicit model ID or null
 * @param baseName - Base name from input path
 * @param variantTag - Quantization variant tag
 * @returns Resolved model ID
 */
export function resolveModelId(
  modelId: string | null,
  baseName: string,
  variantTag: string | undefined
): string {
  // Import sanitizeModelId dynamically to avoid circular deps
  const sanitize = (id: string): string => {
    return id.toLowerCase().replace(/[^a-z0-9-]/g, '-').replace(/-+/g, '-').replace(/^-|-$/g, '');
  };

  // Use provided modelId or derive from baseName
  const base = modelId ? sanitize(modelId) : sanitize(baseName);
  // Always append variant tag unless already present (per RDRR naming convention)
  if (!variantTag) return base;
  return base.endsWith(variantTag) ? base : `${base}-${variantTag}`;
}

/**
 * Convert normalized dtype to WebGPU-compatible dtype string.
 *
 * @param dtype - Normalized dtype (e.g., 'q4k', 'bf16')
 * @returns WebGPU-compatible dtype (e.g., 'Q4_K_M', 'F16')
 */
export function toWebGPUDtype(dtype: string): string {
  if (dtype === 'q4k') return 'Q4_K_M';
  if (dtype === 'bf16') return 'F16'; // WebGPU doesn't support BF16
  return dtype.toUpperCase();
}
