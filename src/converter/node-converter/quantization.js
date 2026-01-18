

import { DEFAULT_QUANTIZATION_DEFAULTS } from '../../config/index.js';


export function normalizeQuantTag(value) {
  if (!value) return 'f16';
  const lower = value.toLowerCase();

  if (lower === 'q4_k_m' || lower === 'q4k' || lower === 'q4' || lower === 'q4km') return 'q4k';
  if (lower === 'q6_k' || lower === 'q6k' || lower === 'q6') return 'q6k';
  if (lower === 'q8_0' || lower === 'q8') return 'q8_0';
  if (lower === 'mxfp4' || lower === 'mxp4') return 'mxfp4';
  if (lower === 'f16' || lower === 'fp16' || lower === 'float16') return 'f16';
  if (lower === 'bf16' || lower === 'bfloat16') return 'bf16';
  if (lower === 'f32' || lower === 'fp32' || lower === 'float32') return 'f32';
  if (lower === 'fp8e4' || lower === 'fp8e4m3' || lower === 'e4m3') return 'fp8e4';
  if (lower === 'fp8e5' || lower === 'fp8e5m2' || lower === 'e5m2') return 'fp8e5';
  if (lower === 'i8' || lower === 'int8') return 'i8';
  if (lower === 'i4' || lower === 'int4') return 'i4';

  return lower;
}


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


export function resolveManifestQuantization(quantize, fallback) {
  if (!quantize) return fallback;
  const normalized = normalizeQuantTag(quantize);
  if (normalized === 'q4k') return 'Q4_K_M';
  if (normalized === 'q6k') return 'Q6_K';
  if (normalized === 'q8_0') return 'Q8_0';
  return normalized.toUpperCase();
}


export function buildVariantTag(info) {
  const weights = info.weights;
  const embeddings = info.embeddings ?? weights;
  const lmHead = info.lmHead ?? embeddings;
  const experts = info.experts ?? null;

  const parts = [`w${weights}`];

  if (embeddings !== weights) {
    parts.push(`e${embeddings}`);
  }

  if (lmHead !== embeddings) {
    parts.push(`h${lmHead}`);
  }

  if (experts && experts !== weights) {
    parts.push(`x${experts}`);
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

function resolveExpertQuantization(modelConfig) {
  if (!modelConfig) return null;
  const quantMethod = modelConfig.quantization_config?.quant_method;
  if (!quantMethod) return null;
  return normalizeQuantTag(quantMethod);
}

function resolveExpertFormat(modelConfig, expertQuant) {
  if (!modelConfig) return null;
  if (expertQuant === 'mxfp4') return 'gpt-oss';
  const rawType = (
    modelConfig.model_type ??
    modelConfig.text_config?.model_type ??
    ''
  ).toLowerCase();
  if (rawType.includes('gpt_oss') || rawType.includes('gpt-oss') || rawType.includes('gptoss')) {
    return 'gpt-oss';
  }
  const hasExperts = Boolean(
    modelConfig.num_local_experts ||
    modelConfig.num_experts ||
    modelConfig.expertCount
  );
  return hasExperts ? 'mixtral' : null;
}


export function buildQuantizationInfo(
  opts,
  originalDtype,
  embedDtype,
  lmHeadDtype,
  hasVision = false,
  hasAudio = false,
  hasProjector = false,
  modelConfig = null
) {
  const config = opts?.converterConfig ?? opts ?? {};
  const quantization = { ...(config.quantization ?? {}) };

  if (opts?.weightQuant !== undefined) quantization.weights = opts.weightQuant;
  if (opts?.embedQuant !== undefined) quantization.embeddings = opts.embedQuant;
  if (opts?.headQuant !== undefined) quantization.lmHead = opts.headQuant;
  if (opts?.visionQuant !== undefined) quantization.vision = opts.visionQuant;
  if (opts?.audioQuant !== undefined) quantization.audio = opts.audioQuant;
  if (opts?.projectorQuant !== undefined) quantization.projector = opts.projectorQuant;
  if (opts?.computePrecision !== undefined) quantization.computePrecision = opts.computePrecision;

  const textOnly = opts?.textOnly !== undefined
    ? opts.textOnly
    : config.output?.textOnly ?? false;
  const allowMultimodal = !textOnly;

  const weightQuant = quantization.weights ?? null;
  const embedQuant = quantization.embeddings ?? null;
  const headQuant = quantization.lmHead ?? null;
  const visionQuant = quantization.vision ?? null;
  const audioQuant = quantization.audio ?? null;
  const projectorQuant = quantization.projector ?? null;
  const computePrecision = quantization.computePrecision ?? null;

  validateQuantType(weightQuant, '--weight-quant');
  validateQuantType(embedQuant, '--embed-quant');
  validateQuantType(headQuant, '--head-quant');
  validateQuantType(visionQuant, '--vision-quant');
  validateQuantType(audioQuant, '--audio-quant');
  validateQuantType(projectorQuant, '--projector-quant');

  const webgpuSafe = (dtype) => {
    const normalized = normalizeQuantTag(dtype);
    if (normalized === 'bf16') return 'f16';
    return normalized;
  };

  const weights = webgpuSafe(weightQuant ?? originalDtype);

  let embeddings;
  if (embedQuant) {
    embeddings = webgpuSafe(embedQuant);
  } else {
    embeddings = webgpuSafe(embedDtype || originalDtype);
  }

  let lmHead;
  if (headQuant) {
    lmHead = webgpuSafe(headQuant);
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

  const hasExperts = Boolean(
    modelConfig?.num_local_experts ||
    modelConfig?.num_experts ||
    modelConfig?.expertCount
  );
  if (hasExperts) {
    const expertQuant = resolveExpertQuantization(modelConfig) ?? weights;
    info.experts = expertQuant;
    const expertFormat = resolveExpertFormat(modelConfig, expertQuant);
    if (expertFormat) {
      info.expertsFormat = expertFormat;
    }
  }

  if (hasVision && allowMultimodal) {
    if (visionQuant) {
      info.vision = normalizeQuantTag(visionQuant);
    } else {
      info.vision = DEFAULT_QUANTIZATION_DEFAULTS.visionDtype;
    }
  }

  if (hasAudio && allowMultimodal) {
    if (audioQuant) {
      info.audio = normalizeQuantTag(audioQuant);
    } else {
      info.audio = DEFAULT_QUANTIZATION_DEFAULTS.audioDtype;
    }
  }

  if (hasProjector && allowMultimodal) {
    if (projectorQuant) {
      info.projector = normalizeQuantTag(projectorQuant);
    } else {
      info.projector = DEFAULT_QUANTIZATION_DEFAULTS.projectorDtype;
    }
  }

  if (computePrecision) {
    info.compute = computePrecision;
  }

  info.variantTag = buildVariantTag(info);
  return info;
}


export function resolveModelId(modelId, baseName, variantTag) {
  const sanitize = (id) => {
    return id.toLowerCase().replace(/[^a-z0-9-]/g, '-').replace(/-+/g, '-').replace(/^-|-$/g, '');
  };

  const base = modelId ? sanitize(modelId) : sanitize(baseName);
  if (!variantTag) return base;
  return base.endsWith(variantTag) ? base : `${base}-${variantTag}`;
}


export function toWebGPUDtype(dtype) {
  if (dtype === 'q4k') return 'Q4_K_M';
  if (dtype === 'bf16') return 'F16';
  return dtype.toUpperCase();
}
