import {
  // Constants
  SHARD_SIZE as SCHEMA_SHARD_SIZE,
  RDRR_VERSION as SCHEMA_RDRR_VERSION,
  ConversionStage as SchemaConversionStage,
  DEFAULT_MANIFEST_INFERENCE,
  formatBytes,
} from '../config/schema/index.js';
import {
  classifyTensorRole,
  generateShardFilename,
  resolveTensorGroup,
  resolveTensorRole,
} from '../formats/rdrr/index.js';
import { log } from '../debug/index.js';
import {
  getInferenceLayerPatternContractArtifact,
  selectRuleValue,
} from '../rules/rule-registry.js';
import {
  createConverterConfig,
} from '../config/index.js';
import { buildExecutionContractArtifact } from '../config/execution-contract-check.js';
import { buildManifestRequiredInferenceFieldsArtifact } from '../config/required-inference-fields-contract-check.js';
import { resolveEosTokenId } from './tokenizer-special-tokens.js';
import { inferBundledTokenizerBehaviorFlags } from '../inference/tokenizers/behavior-flags.js';
import {
  normalizeQ4KLayout,
  resolveManifestQuantization,
  resolveEffectiveQuantizationInfo,
} from './quantization-info.js';
import {
  float16ToFloat32,
  float32ToFloat16,
  quantizeToQ4KM,
  quantizeToQ4KMRowWise,
  quantizeToQ4KMColumnWise,
  quantizeToInt4PerRowSymmetric,
} from './quantizer.js';
import { cloneJsonValue } from '../formats/clone-json.js';

export function normalizeStorageQuant(value) {
  if (value == null) return null;
  const lower = String(value).trim().toLowerCase();
  if (!lower) return null;
  if (lower === 'fp16' || lower === 'float16') return 'f16';
  if (lower === 'fp32' || lower === 'float32') return 'f32';
  if (lower === 'bfloat16') return 'bf16';
  if (lower === 'q4_k_m' || lower === 'q4km') return 'q4k';
  if (lower === 'q4_0' || lower === 'q4-0') return 'q4_0';
  if (
    lower === 'w4a16-ct'
    || lower === 'w4a16_ct'
    || lower === 'compressed-tensors-w4a16'
    || lower === 'compressed_tensors_w4a16'
  ) return 'w4a16';
  if (lower === 'wna8-o8' || lower === 'wna8_o8') return 'wna8o8';
  return lower;
}

export const SOURCE_PACKED_QUANT_DTYPES = new Set(['q4_0', 'w4a16', 'wna8o8']);

export const SOURCE_PACKED_MANIFEST_DTYPES = {
  q4_0: 'Q4_0',
  w4a16: 'W4A16',
  wna8o8: 'WNA8O8',
};

export const SOURCE_PACKED_STORAGE_DESCRIPTORS = {
  q4_0: {
    packing: 'q4_0',
    blockShape: [32],
    blockBytes: 18,
  },
  w4a16: {
    packing: 'w4a16',
    blockShape: [32],
    blockBytes: 16,
  },
};

export function cloneSourcePackedStorageDescriptor(targetQuant) {
  const descriptor = SOURCE_PACKED_STORAGE_DESCRIPTORS[targetQuant];
  if (!descriptor) return null;
  return {
    ...descriptor,
    blockShape: [...descriptor.blockShape],
    ...(Array.isArray(descriptor.companions)
      ? { companions: descriptor.companions.map((companion) => ({ ...companion })) }
      : {}),
  };
}

export function resolveExplicitRoleQuant(tensor, quantizationInfo) {
  if (!quantizationInfo || typeof quantizationInfo !== 'object') {
    return null;
  }
  const role = resolveTensorRole(tensor);
  if (role === 'embedding') {
    return normalizeStorageQuant(quantizationInfo.embeddings ?? null);
  }
  if (role === 'lm_head') {
    return normalizeStorageQuant(
      quantizationInfo.lmHead
        ?? quantizationInfo.embeddings
        ?? null
    );
  }
  if (role === 'matmul' || role === 'expert' || role === 'router') {
    return normalizeStorageQuant(quantizationInfo.weights ?? null);
  }
  return null;
}

export function resolveTensorTargetQuant(tensorOrName, fallbackQuant, quantizationInfo) {
  const fallback = normalizeStorageQuant(fallbackQuant);
  if (!quantizationInfo || typeof quantizationInfo !== 'object') {
    return fallback;
  }

  const role = resolveTensorRole(tensorOrName);
  if (role === 'embedding') {
    return normalizeStorageQuant(quantizationInfo.embeddings ?? fallback) ?? fallback;
  }
  if (role === 'lm_head') {
    const headQuant = quantizationInfo.lmHead ?? quantizationInfo.embeddings ?? fallback;
    return normalizeStorageQuant(headQuant) ?? fallback;
  }
  if (role === 'vision') {
    return normalizeStorageQuant(quantizationInfo.vision ?? fallback) ?? fallback;
  }
  if (role === 'projector') {
    return normalizeStorageQuant(quantizationInfo.projector ?? fallback) ?? fallback;
  }
  if (role === 'audio') {
    return normalizeStorageQuant(quantizationInfo.audio ?? fallback) ?? fallback;
  }
  return normalizeStorageQuant(quantizationInfo.weights ?? fallback) ?? fallback;
}

export function bf16ToFloat32(value) {
  const view = new DataView(new ArrayBuffer(4));
  view.setUint32(0, (value & 0xffff) << 16, true);
  return view.getFloat32(0, true);
}

export function isCompressedTensorsW4A16CompanionTensor(tensor) {
  return Boolean(
    tensor?.compressedTensorsW4A16Companion
    && typeof tensor.compressedTensorsW4A16Companion === 'object'
  );
}

export function toFloat32ForQ4K(tensorData, sourceDtype, tensorName) {
  const dtype = String(sourceDtype || '').toUpperCase();
  if (dtype === 'F32') {
    if (tensorData.byteLength % 4 !== 0) {
      throw new Error(`Invalid F32 tensor byte length for ${tensorName}: ${tensorData.byteLength}`);
    }
    return new Float32Array(
      tensorData.buffer,
      tensorData.byteOffset,
      tensorData.byteLength / 4
    );
  }
  if (dtype === 'F16') {
    if (tensorData.byteLength % 2 !== 0) {
      throw new Error(`Invalid F16 tensor byte length for ${tensorName}: ${tensorData.byteLength}`);
    }
    const f16 = new Uint16Array(
      tensorData.buffer,
      tensorData.byteOffset,
      tensorData.byteLength / 2
    );
    const f32 = new Float32Array(f16.length);
    for (let i = 0; i < f16.length; i++) {
      f32[i] = float16ToFloat32(f16[i]);
    }
    return f32;
  }
  if (dtype === 'BF16') {
    if (tensorData.byteLength % 2 !== 0) {
      throw new Error(`Invalid BF16 tensor byte length for ${tensorName}: ${tensorData.byteLength}`);
    }
    const bf16 = new Uint16Array(
      tensorData.buffer,
      tensorData.byteOffset,
      tensorData.byteLength / 2
    );
    const f32 = new Float32Array(bf16.length);
    for (let i = 0; i < bf16.length; i++) {
      f32[i] = bf16ToFloat32(bf16[i]);
    }
    return f32;
  }
  throw new Error(`Cannot quantize ${tensorName} from ${dtype} to Q4_K_M`);
}

export const BF16_ROUND_VIEW = new DataView(new ArrayBuffer(4));

export function float32ToBFloat16(value) {
  BF16_ROUND_VIEW.setFloat32(0, value, true);
  const bits = BF16_ROUND_VIEW.getUint32(0, true);
  const lsb = (bits >> 16) & 1;
  const roundingBias = 0x7fff + lsb;
  return ((bits + roundingBias) >> 16) & 0xffff;
}

export function resolveQuantizeEmbeddings(quantizationInfo, explicitValue = null) {
  if (typeof explicitValue === 'boolean') {
    return explicitValue;
  }
  return (
    normalizeStorageQuant(quantizationInfo?.embeddings ?? null) === 'q4k'
    || normalizeStorageQuant(quantizationInfo?.lmHead ?? null) === 'q4k'
  );
}

export function normalizeModulesToNotConvert(modulesToNotConvert) {
  if (!Array.isArray(modulesToNotConvert)) {
    return null;
  }
  const normalized = modulesToNotConvert
    .map((value) => (
      typeof value === 'string' ? value.trim() : ''
    ))
    .filter(Boolean);
  return normalized.length > 0 ? normalized : null;
}

export function shouldSkipModuleQuantization(tensorName, modulesToNotConvert) {
  const patterns = normalizeModulesToNotConvert(modulesToNotConvert);
  if (!patterns) {
    return false;
  }

  for (const pattern of patterns) {
    const regexPattern = pattern
      .replace(/\./g, '\\.')
      .replace(/\*/g, '\\d+');
    const matcher = new RegExp(regexPattern);
    if (matcher.test(tensorName)) {
      return true;
    }
  }
  return false;
}

export function shouldQuantize(tensorName, shape, options = {}) {
  const {
    quantizeEmbeddings = false,
    modulesToNotConvert = null,
    role: explicitRole = null,
  } = options;

  if (!shape || !Array.isArray(shape) || shape.length === 0) {
    log.warn('Convert', `Invalid shape for tensor "${tensorName}": ${JSON.stringify(shape)}`);
    return false;
  }
  const numElements = shape.reduce((a, b) => a * b, 1);
  const role = typeof explicitRole === 'string' && explicitRole.trim()
    ? explicitRole.trim()
    : classifyTensorRole(tensorName);
  const lower = tensorName.toLowerCase();
  const isBias = lower.endsWith('.bias') || lower.endsWith('_bias');

  const shouldQuantizeByRole = selectRuleValue('converter', 'tensorRoles', 'shouldQuantize', {
    numElements,
    role,
    isBias,
    quantizeEmbeddings,
  });

  if (!shouldQuantizeByRole) {
    return false;
  }

  if (shouldSkipModuleQuantization(tensorName, modulesToNotConvert)) {
    return false;
  }

  return true;
}

export const GEMMA4_PLE_TENSOR_SUFFIXES = [
  'embed_tokens_per_layer.weight',
  'per_layer_embeddings.weight',
  // GGUF (unsloth / ggml-org) naming for Gemma 4's PLE table.
  'per_layer_token_embd.weight',
];

export function isGemma4PerLayerEmbedTensor(tensorName) {
  if (typeof tensorName !== 'string') return false;
  const normalized = tensorName.trim().toLowerCase();
  return GEMMA4_PLE_TENSOR_SUFFIXES.some((suffix) => normalized.endsWith(suffix));
}

export function resolveOriginalTensorShape(options) {
  const shape = options?.originalTensorShape;
  if (!Array.isArray(shape) || shape.length !== 2) {
    return null;
  }
  const rows = Number(shape[0]);
  const cols = Number(shape[1]);
  if (!Number.isInteger(rows) || rows <= 0 || !Number.isInteger(cols) || cols <= 0) {
    return null;
  }
  return [rows, cols];
}

export function resolvePerLayerEmbeddingQuant(options) {
  const value = (
    options?.perLayerEmbeddings
    ?? options?.quantizationInfo?.perLayerEmbeddings
    ?? null
  );
  if (value == null) return null;
  return String(value).trim().toLowerCase().replace(/\s+/g, '_') || null;
}

export function canInt4QuantizePerRow(tensor, options) {
  if (options?.skipInt4PlePerRow === true) return false;
  if (resolvePerLayerEmbeddingQuant(options) !== 'int4_per_row') return false;
  if (!Array.isArray(tensor.shape) || tensor.shape.length !== 2) return false;
  const [rows, cols] = tensor.shape;
  if (!Number.isInteger(rows) || rows <= 0) return false;
  if (!Number.isInteger(cols) || cols <= 0) return false;
  if ((cols & 1) !== 0) return false;
  const originalShape = resolveOriginalTensorShape(options);
  if (originalShape && originalShape[1] !== cols) return false;
  // Per-row symmetric INT4 produces one scale per row. PLE semantics require
  // one scale per VOCAB token (typically 262144 for Gemma 4). HF stores PLE
  // as [vocab, hidden] (rows = vocab ≫ cols). GGUF stores it transposed as
  // [hidden, vocab] (rows = hidden ≪ cols). Quantizing the GGUF layout with
  // per-row would wrongly share a single scale across all vocab tokens.
  // Require the tensor to already be in [vocab, hidden] layout. Callers that
  // have a GGUF PLE must transpose before reaching this function.
  if (rows <= cols && (!originalShape || originalShape[0] <= originalShape[1])) return false;
  const srcDtype = String(tensor.dtype || '').toUpperCase();
  return srcDtype === 'F32' || srcDtype === 'F16' || srcDtype === 'BF16';
}

export function toFloat32FromTensor(bytes, sourceDtype, tensorName) {
  const src = String(sourceDtype || '').toUpperCase();
  if (src === 'F32') {
    if (bytes.byteLength % 4 !== 0) {
      throw new Error(`Invalid F32 tensor byte length for ${tensorName}: ${bytes.byteLength}`);
    }
    return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  }
  if (src === 'F16') {
    if (bytes.byteLength % 2 !== 0) {
      throw new Error(`Invalid F16 tensor byte length for ${tensorName}: ${bytes.byteLength}`);
    }
    const src16 = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
    const out = new Float32Array(src16.length);
    for (let i = 0; i < src16.length; i++) out[i] = float16ToFloat32(src16[i]);
    return out;
  }
  if (src === 'BF16') {
    if (bytes.byteLength % 2 !== 0) {
      throw new Error(`Invalid BF16 tensor byte length for ${tensorName}: ${bytes.byteLength}`);
    }
    const src16 = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
    const out = new Float32Array(src16.length);
    for (let i = 0; i < src16.length; i++) out[i] = bf16ToFloat32(src16[i]);
    return out;
  }
  throw new Error(`Unsupported source dtype "${sourceDtype}" for PLE INT4 quantization of ${tensorName}`);
}

export function buildInt4PerRowPleTransform(tensor, bytes, sourceDtype, options = {}) {
  const f32 = toFloat32FromTensor(bytes, sourceDtype, tensor.name);
  const { quantized, scales } = quantizeToInt4PerRowSymmetric(f32, tensor.shape);
  const scalesBytes = new Uint8Array(scales.buffer, scales.byteOffset, scales.byteLength);
  const [rows, cols] = resolveOriginalTensorShape(options) ?? tensor.shape;
  return {
    tensorData: quantized,
    companionData: scalesBytes,
    // outDtype is the LOGICAL dtype the tensor resolves to after dequant at
    // load time. Storage dtype (INT4) is carried in sourceTransform.sourceDtype.
    outDtype: 'F16',
    outLayout: 'row',
    sourceDtype: String(sourceDtype || '').toUpperCase(),
    tensorTargetQuant: 'int4_per_row_ple',
    sourceTransform: {
      kind: 'litert_axis_dequant',
      scheme: 'per_axis_affine',
      sourceDtype: 'INT4',
      targetDtype: 'F16',
      storageEncoding: 'offset_binary',
      scaleSemantics: 'step',
      storageShape: [rows, cols],
      quantAxis: 1,
      // scaleSource gets filled in by the writer with the scales companion's
      // shard/offset/size after appendTensorBytes().
    },
  };
}

export function transformTensorBytes(tensor, rawData, options = {}) {
  const tensorDataInput = rawData instanceof Uint8Array ? rawData : new Uint8Array(rawData);
  let tensorData = tensorDataInput;
  let outDtype = tensor.dtype;
  let outLayout = null;

  const sourceDtype = String(tensor.dtype).toUpperCase();
  const targetQuant = normalizeStorageQuant(options.targetQuant ?? options.quantization ?? null);
  const quantizationInfo = options.quantizationInfo ?? null;
  const tensorTargetQuant = resolveTensorTargetQuant(
    tensor,
    targetQuant,
    quantizationInfo
  );
  const q4kLayout = normalizeQ4KLayout(options.q4kLayout ?? quantizationInfo?.layout);
  const quantizeEmbeddings = resolveQuantizeEmbeddings(
    quantizationInfo,
    options.quantizeEmbeddings
  );
  const modulesToNotConvert = normalizeModulesToNotConvert(
    options.modulesToNotConvert ?? null
  );
  const forceQuantizeDecision = (
    typeof options.forceQuantizeDecision === 'boolean'
      ? options.forceQuantizeDecision
      : null
  );

  // Gemma 4 per-layer embeddings use MediaPipe's INT4 per-row symmetric
  // quantization (verified against gemma-4-E2B-it.litertlm composites:
  // quantizedDimension=0, one F32 scale per vocab row, zero_point=0). Saves
  // ~3.5 GB per model vs the default F16 path. Runtime reads via
  // sourceTransform.kind=litert_axis_dequant with scaleSemantics=step.
  if (isGemma4PerLayerEmbedTensor(tensor.name) && canInt4QuantizePerRow(tensor, options)) {
    return buildInt4PerRowPleTransform(tensor, tensorDataInput, sourceDtype, options);
  }

  if (isCompressedTensorsW4A16CompanionTensor(tensor)) {
    return {
      tensorData,
      outDtype: sourceDtype,
      outLayout: null,
      sourceDtype,
      tensorTargetQuant: null,
    };
  }

  if (SOURCE_PACKED_QUANT_DTYPES.has(tensorTargetQuant)) {
    const sourceQuant = normalizeStorageQuant(sourceDtype);
    if (sourceQuant === tensorTargetQuant) {
      const descriptor = cloneSourcePackedStorageDescriptor(tensorTargetQuant);
      const sourceCompanions = Array.isArray(tensor?.storage?.companions)
        ? tensor.storage.companions.map((companion) => ({ ...companion }))
        : null;
      return {
        tensorData,
        outDtype: SOURCE_PACKED_MANIFEST_DTYPES[tensorTargetQuant],
        outLayout: null,
        sourceDtype,
        tensorTargetQuant,
        storage: {
          ...descriptor,
          ...(sourceCompanions ? { companions: sourceCompanions } : {}),
        },
      };
    }
    const roleQuant = resolveExplicitRoleQuant(tensor, quantizationInfo);
    const requiresPackedSource = (
      forceQuantizeDecision
      ?? (
        roleQuant === tensorTargetQuant
        || shouldQuantize(tensor.name, tensor.shape, {
          quantizeEmbeddings,
          modulesToNotConvert,
          role: tensor.role ?? null,
        })
      )
    );
    if (!requiresPackedSource) {
      return {
        tensorData,
        outDtype: sourceDtype,
        outLayout: null,
        sourceDtype,
        tensorTargetQuant,
      };
    }
    throw new Error(
      `Cannot materialize ${tensorTargetQuant} for ${tensor.name}: ` +
      `native import requires source dtype ${SOURCE_PACKED_MANIFEST_DTYPES[tensorTargetQuant]}; ` +
      'the converter does not re-quantize tensors into this packed format.'
    );
  }

  if (tensorTargetQuant === 'q4k') {
    const sourceQuant = normalizeStorageQuant(sourceDtype);
    const tensorRole = resolveTensorRole(tensor);
    const isMatrixLikeShape = Array.isArray(tensor.shape) && tensor.shape.length >= 2;
    const is2DMatrixShape = Array.isArray(tensor.shape) && tensor.shape.length === 2;
    const useQ4KRowWise = isMatrixLikeShape
      && q4kLayout === 'row'
      && (is2DMatrixShape || tensorRole === 'expert');
    if (sourceQuant === 'q4k') {
      outDtype = 'Q4_K_M';
      if (is2DMatrixShape) {
        outLayout = q4kLayout;
      }
      return {
        tensorData,
        outDtype,
        outLayout,
        sourceDtype,
        tensorTargetQuant,
      };
    }

    const shouldQuantizeTensor = (
      forceQuantizeDecision ?? shouldQuantize(tensor.name, tensor.shape, {
        quantizeEmbeddings,
        modulesToNotConvert,
        role: tensor.role ?? null,
      })
    );
    if (shouldQuantizeTensor) {
      const f32Data = toFloat32ForQ4K(tensorData, sourceDtype, tensor.name);
      const quantized = (
        is2DMatrixShape
          ? (q4kLayout === 'col'
            ? quantizeToQ4KMColumnWise(f32Data, tensor.shape)
            : quantizeToQ4KMRowWise(f32Data, tensor.shape))
          : useQ4KRowWise
            ? quantizeToQ4KMRowWise(f32Data, tensor.shape)
          : quantizeToQ4KM(f32Data, tensor.shape)
      );
      tensorData = quantized.quantized;
      outDtype = 'Q4_K_M';
      if (is2DMatrixShape || useQ4KRowWise) {
        outLayout = q4kLayout;
      }
    } else if (sourceDtype === 'BF16') {
      // BF16 is not a native WebGPU dtype. When quantization is skipped
      // (e.g. via modulesToNotConvert), convert BF16→F16 so the runtime
      // can load the tensor without a BF16 dequant shader.
      const bf16 = new Uint16Array(
        tensorData.buffer,
        tensorData.byteOffset,
        tensorData.byteLength / 2
      );
      const f16 = new Uint16Array(bf16.length);
      for (let j = 0; j < bf16.length; j++) {
        f16[j] = float32ToFloat16(bf16ToFloat32(bf16[j]));
      }
      tensorData = new Uint8Array(f16.buffer, f16.byteOffset, f16.byteLength);
      outDtype = 'F16';
    }
  } else if (tensorTargetQuant === 'f16' && sourceDtype === 'F32') {
    if (tensorData.byteLength % 4 !== 0) {
      throw new Error(`Invalid F32 tensor byte length for ${tensor.name}: ${tensorData.byteLength}`);
    }
    const f32 = new Float32Array(
      tensorData.buffer,
      tensorData.byteOffset,
      tensorData.byteLength / 4
    );
    const f16 = new Uint16Array(f32.length);
    for (let j = 0; j < f32.length; j++) {
      f16[j] = float32ToFloat16(f32[j]);
    }
    tensorData = new Uint8Array(f16.buffer, f16.byteOffset, f16.byteLength);
    outDtype = 'F16';
  } else if (tensorTargetQuant === 'f16' && sourceDtype === 'BF16') {
    if (tensorData.byteLength % 2 !== 0) {
      throw new Error(`Invalid BF16 tensor byte length for ${tensor.name}: ${tensorData.byteLength}`);
    }
    const bf16 = new Uint16Array(
      tensorData.buffer,
      tensorData.byteOffset,
      tensorData.byteLength / 2
    );
    const f16 = new Uint16Array(bf16.length);
    for (let j = 0; j < bf16.length; j++) {
      f16[j] = float32ToFloat16(bf16ToFloat32(bf16[j]));
    }
    tensorData = new Uint8Array(f16.buffer, f16.byteOffset, f16.byteLength);
    outDtype = 'F16';
  } else if (tensorTargetQuant === 'bf16' && sourceDtype === 'F32') {
    if (tensorData.byteLength % 4 !== 0) {
      throw new Error(`Invalid F32 tensor byte length for ${tensor.name}: ${tensorData.byteLength}`);
    }
    const f32 = new Float32Array(
      tensorData.buffer,
      tensorData.byteOffset,
      tensorData.byteLength / 4
    );
    const bf16 = new Uint16Array(f32.length);
    for (let j = 0; j < f32.length; j++) {
      bf16[j] = float32ToBFloat16(f32[j]);
    }
    tensorData = new Uint8Array(bf16.buffer, bf16.byteOffset, bf16.byteLength);
    outDtype = 'BF16';
  } else if (tensorTargetQuant === 'f32' && sourceDtype === 'F16') {
    if (tensorData.byteLength % 2 !== 0) {
      throw new Error(`Invalid F16 tensor byte length for ${tensor.name}: ${tensorData.byteLength}`);
    }
    const f16 = new Uint16Array(
      tensorData.buffer,
      tensorData.byteOffset,
      tensorData.byteLength / 2
    );
    const f32 = new Float32Array(f16.length);
    for (let j = 0; j < f16.length; j++) {
      f32[j] = float16ToFloat32(f16[j]);
    }
    tensorData = new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
    outDtype = 'F32';
  } else if (tensorTargetQuant === 'f32' && sourceDtype === 'BF16') {
    if (tensorData.byteLength % 2 !== 0) {
      throw new Error(`Invalid BF16 tensor byte length for ${tensor.name}: ${tensorData.byteLength}`);
    }
    const bf16 = new Uint16Array(
      tensorData.buffer,
      tensorData.byteOffset,
      tensorData.byteLength / 2
    );
    const f32 = new Float32Array(bf16.length);
    for (let j = 0; j < bf16.length; j++) {
      f32[j] = bf16ToFloat32(bf16[j]);
    }
    tensorData = new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
    outDtype = 'F32';
  }

  return {
    tensorData,
    outDtype,
    outLayout,
    sourceDtype,
    tensorTargetQuant,
  };
}
