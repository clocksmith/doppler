
import { DEFAULT_QUANTIZATION_DEFAULTS, DEFAULT_Q4K_LAYOUT } from '../config/index.js';
import { classifyTensorRole } from '../storage/rdrr-format.js';

// Quantization tag aliases mapped to canonical names.
// Add new aliases here rather than adding if/else branches.
const QUANT_TAG_ALIASES = {
  // Q4_K_M variants
  'q4_k_m': 'q4k',
  'q4k': 'q4k',
  'q4': 'q4k',
  'q4km': 'q4k',
  // Q6_K variants
  'q6_k': 'q6k',
  'q6k': 'q6k',
  'q6': 'q6k',
  // Q8_0 variants
  'q8_0': 'q8_0',
  'q8': 'q8_0',
  // MXFP4 variants
  'mxfp4': 'mxfp4',
  'mxp4': 'mxfp4',
  // F16 variants
  'f16': 'f16',
  'fp16': 'f16',
  'float16': 'f16',
  // BF16 variants
  'bf16': 'bf16',
  'bfloat16': 'bf16',
  // F32 variants
  'f32': 'f32',
  'fp32': 'f32',
  'float32': 'f32',
  // FP8 E4M3 variants
  'fp8e4': 'fp8e4',
  'fp8e4m3': 'fp8e4',
  'e4m3': 'fp8e4',
  // FP8 E5M2 variants
  'fp8e5': 'fp8e5',
  'fp8e5m2': 'fp8e5',
  'e5m2': 'fp8e5',
  // Integer variants
  'i8': 'i8',
  'int8': 'i8',
  'i4': 'i4',
  'int4': 'i4',
};

export function normalizeQuantTag(value) {
  if (!value) return 'f16';
  const lower = value.toLowerCase();
  return QUANT_TAG_ALIASES[lower] ?? lower;
}


export function validateQuantType(value, source) {
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

  throw new Error(`Unknown quantization type: "${value}" (source: ${source})`);
}


// Canonical dtype to manifest format mapping.
const MANIFEST_QUANT_NAMES = {
  'q4k': 'Q4_K_M',
  'q6k': 'Q6_K',
  'q8_0': 'Q8_0',
};

export function resolveManifestQuantization(quantize, fallback) {
  if (!quantize) return fallback;
  const normalized = normalizeQuantTag(quantize);
  return MANIFEST_QUANT_NAMES[normalized] ?? normalized.toUpperCase();
}


export function buildVariantTag(info) {
  const weights = info.weights;
  const embeddings = info.embeddings ?? weights;
  const lmHead = info.lmHead ?? embeddings;
  const experts = info.experts ?? null;
  const layout = info.layout ?? null;

  // For Q4K weights, include layout in tag
  // 'row' = fused kernel compatible (fast), 'col' = dequant fallback
  const weightTag = weights === 'q4k' && layout
    ? `${weights}${layout === 'row' ? '' : '-col'}`
    : weights;

  const parts = [`w${weightTag}`];

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


// Q4K layout aliases mapped to canonical names.
const Q4K_LAYOUT_ALIASES = {
  'row': 'row',
  'rowwise': 'row',
  'col': 'col',
  'column': 'col',
  'columnwise': 'col',
};

export function normalizeQ4KLayout(value) {
  if (!value) return null;
  const lower = String(value).toLowerCase().replace(/_/g, '');
  return Q4K_LAYOUT_ALIASES[lower] ?? null;
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

  validateQuantType(weightQuant, 'converter.quantization.weights');
  validateQuantType(embedQuant, 'converter.quantization.embeddings');
  validateQuantType(headQuant, 'converter.quantization.lmHead');
  validateQuantType(visionQuant, 'converter.quantization.vision');
  validateQuantType(audioQuant, 'converter.quantization.audio');
  validateQuantType(projectorQuant, 'converter.quantization.projector');

  // Preserve requested/storage dtypes in manifest + shard planning.
  // Kernel-path selection performs its own dtype normalization separately.
  const asStorageQuant = (dtype) => normalizeQuantTag(dtype);

  const weights = asStorageQuant(weightQuant ?? originalDtype);

  let embeddings;
  if (embedQuant) {
    embeddings = asStorageQuant(embedQuant);
  } else {
    embeddings = asStorageQuant(embedDtype || originalDtype);
  }

  let lmHead;
  if (headQuant) {
    lmHead = asStorageQuant(headQuant);
  } else if (lmHeadDtype) {
    lmHead = asStorageQuant(lmHeadDtype);
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

  // Q4K layout: 'row' (fused kernel compatible) or 'col' (dequant fallback)
  // Default to 'row' for Q4K weights since that's the performant path
  const q4kLayoutRaw = opts?.q4kLayout ?? quantization.q4kLayout ?? null;
  if (weights === 'q4k') {
    info.layout = normalizeQ4KLayout(q4kLayoutRaw) ?? DEFAULT_Q4K_LAYOUT;
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


// Canonical dtype to WebGPU dtype mapping.
const WEBGPU_DTYPE_NAMES = {
  'q4k': 'Q4_K_M',
  'bf16': 'F16',
};

export function toWebGPUDtype(dtype) {
  return WEBGPU_DTYPE_NAMES[dtype] ?? dtype.toUpperCase();
}

function normalizeStoredDtype(value) {
  if (!value) return null;
  const upper = String(value).trim().toUpperCase();
  if (!upper) return null;
  if (upper === 'Q4_K' || upper === 'Q4_K_M') return 'q4k';
  if (upper === 'Q6_K') return 'q6k';
  if (upper === 'Q8_0') return 'q8_0';
  if (upper === 'BF16') return 'bf16';
  if (upper === 'F16') return 'f16';
  if (upper === 'F32') return 'f32';
  return normalizeQuantTag(upper);
}

function normalizeRole(role, name) {
  if (role && typeof role === 'string') return role;
  if (typeof name !== 'string' || !name) return null;
  return classifyTensorRole(name);
}

export function resolveEffectiveQuantizationInfo(baseInfo, tensors) {
  const base = (baseInfo && typeof baseInfo === 'object') ? baseInfo : {};
  const entries = Array.isArray(tensors) ? tensors : [];

  let detectedWeights = null;
  let detectedEmbeddings = null;
  let detectedLmHead = null;
  let detectedLayout = null;

  for (const tensor of entries) {
    const dtype = normalizeStoredDtype(tensor?.dtype);
    if (!dtype) continue;
    const role = normalizeRole(tensor?.role, tensor?.name);
    if (!role) continue;

    if (role === 'embedding') {
      detectedEmbeddings = detectedEmbeddings ?? dtype;
      continue;
    }
    if (role === 'lm_head') {
      detectedLmHead = detectedLmHead ?? dtype;
      continue;
    }
    if (role === 'matmul' || role === 'expert' || role === 'router') {
      detectedWeights = detectedWeights ?? dtype;
      if (detectedLayout == null && (dtype === 'q4k') && tensor?.layout) {
        detectedLayout = normalizeQ4KLayout(tensor.layout);
      }
    }
  }

  const weights = detectedWeights
    ?? normalizeQuantTag(base.weights ?? 'f16');
  const embeddings = detectedEmbeddings
    ?? normalizeQuantTag(base.embeddings ?? weights);
  const lmHead = detectedLmHead
    ?? normalizeQuantTag(base.lmHead ?? embeddings);

  const resolved = {
    ...base,
    weights,
    embeddings,
  };

  if (lmHead !== embeddings) {
    resolved.lmHead = lmHead;
  } else {
    delete resolved.lmHead;
  }

  if (weights === 'q4k') {
    resolved.layout = detectedLayout
      ?? normalizeQ4KLayout(base.layout)
      ?? DEFAULT_Q4K_LAYOUT;
  } else {
    delete resolved.layout;
  }

  resolved.variantTag = buildVariantTag(resolved);
  return resolved;
}
