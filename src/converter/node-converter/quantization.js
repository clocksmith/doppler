/**
 * Quantization utilities for the Node.js Model Converter.
 *
 * Handles quantization tag normalization, validation, and variant tag building.
 *
 * @module converter/node-converter/quantization
 */

import { DEFAULT_QUANTIZATION_DEFAULTS } from '../../config/index.js';

/**
 * Normalize quantization tag to canonical short form.
 */
export function normalizeQuantTag(value) {
  if (!value) return 'f16';
  const lower = value.toLowerCase();

  if (lower === 'q4_k_m' || lower === 'q4k' || lower === 'q4' || lower === 'q4km') return 'q4k';
  if (lower === 'q6_k' || lower === 'q6k' || lower === 'q6') return 'q6k';
  if (lower === 'q8_0' || lower === 'q8') return 'q8_0';
  if (lower === 'f16' || lower === 'fp16' || lower === 'float16') return 'f16';
  if (lower === 'bf16' || lower === 'bfloat16') return 'bf16';
  if (lower === 'f32' || lower === 'fp32' || lower === 'float32') return 'f32';
  if (lower === 'fp8e4' || lower === 'fp8e4m3' || lower === 'e4m3') return 'fp8e4';
  if (lower === 'fp8e5' || lower === 'fp8e5m2' || lower === 'e5m2') return 'fp8e5';
  if (lower === 'i8' || lower === 'int8') return 'i8';
  if (lower === 'i4' || lower === 'int4') return 'i4';

  return lower;
}

/**
 * Validate that a quantization type is supported for conversion.
 */
export function validateQuantType(value, flagName) {
  if (!value) return;
  const normalized = normalizeQuantTag(value);

  const supported = ['q4k', 'f16', 'bf16', 'f32'];
  if (supported.includes(normalized)) return;

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
 */
export function resolveManifestQuantization(quantize, fallback) {
  if (!quantize) return fallback;
  const normalized = normalizeQuantTag(quantize);
  if (normalized === 'q4k') return 'Q4_K_M';
  if (normalized === 'q6k') return 'Q6_K';
  if (normalized === 'q8_0') return 'Q8_0';
  return normalized.toUpperCase();
}

/**
 * Build variant tag for model naming.
 */
export function buildVariantTag(info) {
  const weights = info.weights;
  const embeddings = info.embeddings ?? weights;
  const lmHead = info.lmHead ?? embeddings;

  const parts = [`w${weights}`];

  if (embeddings !== weights) {
    parts.push(`e${embeddings}`);
  }

  if (lmHead !== embeddings) {
    parts.push(`h${lmHead}`);
  }

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
 */
export function buildQuantizationInfo(
  opts,
  originalDtype,
  embedDtype,
  lmHeadDtype,
  hasVision = false,
  hasAudio = false,
  hasProjector = false
) {
  validateQuantType(opts.weightQuant, '--weight-quant');
  validateQuantType(opts.embedQuant, '--embed-quant');
  validateQuantType(opts.headQuant, '--head-quant');
  validateQuantType(opts.visionQuant, '--vision-quant');
  validateQuantType(opts.audioQuant, '--audio-quant');
  validateQuantType(opts.projectorQuant, '--projector-quant');

  const webgpuSafe = (dtype) => {
    const normalized = normalizeQuantTag(dtype);
    if (normalized === 'bf16') return 'f16';
    return normalized;
  };

  const weights = webgpuSafe(opts.weightQuant ?? originalDtype);

  let embeddings;
  if (opts.embedQuant) {
    embeddings = webgpuSafe(opts.embedQuant);
  } else {
    embeddings = webgpuSafe(embedDtype || originalDtype);
  }

  let lmHead;
  if (opts.headQuant) {
    lmHead = webgpuSafe(opts.headQuant);
  } else if (lmHeadDtype) {
    lmHead = webgpuSafe(lmHeadDtype);
  } else {
    lmHead = embeddings;
  }

  const info = {
    weights,
    embeddings,
    lmHead: lmHead !== embeddings ? lmHead : undefined,
  };

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

  if (opts.computePrecision) {
    info.compute = opts.computePrecision;
  }

  info.variantTag = buildVariantTag(info);
  return info;
}

/**
 * Resolve model ID with variant tag.
 */
export function resolveModelId(modelId, baseName, variantTag) {
  const sanitize = (id) => {
    return id.toLowerCase().replace(/[^a-z0-9-]/g, '-').replace(/-+/g, '-').replace(/^-|-$/g, '');
  };

  const base = modelId ? sanitize(modelId) : sanitize(baseName);
  if (!variantTag) return base;
  return base.endsWith(variantTag) ? base : `${base}-${variantTag}`;
}

/**
 * Convert normalized dtype to WebGPU-compatible dtype string.
 */
export function toWebGPUDtype(dtype) {
  if (dtype === 'q4k') return 'Q4_K_M';
  if (dtype === 'bf16') return 'F16';
  return dtype.toUpperCase();
}
