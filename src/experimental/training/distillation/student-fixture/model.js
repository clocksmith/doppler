import { getDevice } from '../../../../gpu/device.js';
import { createTrainingConfig } from '../../../../config/training-defaults.js';
import {
  runAttention,
  castF16ToF32,
  runGather,
  runMatmul,
  runResidualAdd,
  runRMSNorm,
  runRoPE,
  runScale,
  runSiLU,
  runSiLURowSplit,
} from '../../../../gpu/kernels/index.js';
import { createTensor } from '../../../../gpu/tensor.js';
import { acquireBuffer, uploadData, releaseBuffer } from '../../../../memory/buffer-pool.js';
import { getBufferDtype, getWeightDtype, isCpuWeightBuffer, isWeightBuffer } from '../../../../gpu/weight-buffer.js';
import { OpType } from '../../autograd.js';
import { LoraAdapter } from '../../lora.js';
import { normalizeOptionalString } from '../suite-data.js';
import { LORA_MODULE_ALIASES } from '../../../../inference/pipelines/text/lora.js';
import { resolveEmbeddingScale } from '../../../../inference/pipelines/text/embed.js';
import { isSlidingLayerType } from '../../../../inference/pipelines/text/attention/dispatch-params.js';

export const DISTILL_ADAPTER_TOP_K = 64;

export const TRANSFORMER_LORA_TARGET_MODULES = Object.freeze([
  'q_proj',
  'k_proj',
  'v_proj',
  'o_proj',
  'gate_proj',
  'up_proj',
  'gate_up_proj',
  'down_proj',
]);

export function makeTensorFromFloat32(values, shape, label) {
  const data = values instanceof Float32Array ? values : new Float32Array(values);
  const buffer = acquireBuffer(data.byteLength, undefined, label || 'train_tensor');
  uploadData(buffer, data);
  return createTensor(buffer, 'f32', shape, label || 'train_tensor');
}

export function makeTensorFromF16Bits(values, shape, label) {
  const data = values instanceof Uint16Array ? values : new Uint16Array(values);
  const buffer = acquireBuffer(data.byteLength, undefined, label || 'train_tensor_f16');
  uploadData(buffer, data);
  return createTensor(buffer, 'f16', shape, label || 'train_tensor_f16');
}

export function releaseTensor(tensor) {
  if (!tensor?.buffer) return;
  releaseBuffer(tensor.buffer);
}

export function toFloat32Array(values, label = 'values') {
  if (values instanceof Float32Array) return values;
  if (ArrayBuffer.isView(values)) {
    return new Float32Array(values.buffer.slice(values.byteOffset, values.byteOffset + values.byteLength));
  }
  if (values instanceof ArrayBuffer) {
    return new Float32Array(values.slice(0));
  }
  if (Array.isArray(values)) {
    return new Float32Array(values);
  }
  throw new Error(`Expected ${label} to be a Float32Array-compatible value.`);
}

export function disposePrefillSnapshot(result) {
  const cache = result?.cache;
  if (cache && typeof cache.clear === 'function') {
    cache.clear();
  }
}

export function toFiniteNumber(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export function clampDistillTopK(value) {
  const parsed = Math.floor(toFiniteNumber(value, DISTILL_ADAPTER_TOP_K));
  return Math.max(2, Math.min(256, parsed));
}

export function normalizeTransformerLoraConfig(value) {
  if (!value || typeof value !== 'object') {
    return null;
  }
  const rank = Math.floor(Number(value.rank));
  const alpha = Number(value.alpha);
  if (!Number.isInteger(rank) || rank < 1) {
    throw new Error('Transformer LoRA config requires rank >= 1.');
  }
  if (!Number.isFinite(alpha) || alpha <= 0) {
    throw new Error('Transformer LoRA config requires alpha > 0.');
  }
  const rawModules = Array.isArray(value.targetModules) ? value.targetModules : [];
  const targetModules = [];
  for (const rawModule of rawModules) {
    const normalized = String(rawModule || '').trim();
    if (!normalized) continue;
    const moduleName = LORA_MODULE_ALIASES[normalized] || normalized;
    if (!TRANSFORMER_LORA_TARGET_MODULES.includes(moduleName)) {
      throw new Error(`Transformer LoRA target module "${normalized}" is not supported.`);
    }
    if (!targetModules.includes(moduleName)) {
      targetModules.push(moduleName);
    }
  }
  if (targetModules.length === 0) {
    throw new Error('Transformer LoRA config requires targetModules.');
  }
  return {
    rank,
    alpha,
    targetModules,
  };
}

export function resolveTensorDtype(value, fallback = 'f32') {
  const dtype = isWeightBuffer(value)
    ? value.dtype
    : (value?.dtype || getWeightDtype(value) || null);
  const normalized = String(dtype || '').toLowerCase();
  return normalized === 'f16' ? 'f16' : (normalized === 'f32' ? 'f32' : fallback);
}

export async function ensureTrainableTensor(
  value,
  shape,
  label,
  ownedTrainables = null,
  options = {}
) {
  if (!value) {
    throw new Error(`Distill full-graph student missing required weight "${label}".`);
  }
  const registerOwned = (tensor) => {
    if (ownedTrainables instanceof Set && tensor?.buffer instanceof GPUBuffer) {
      ownedTrainables.add(tensor);
    }
    return tensor;
  };
  const preserveF16 = options.preserveF16 === true;
  if (isWeightBuffer(value)) {
    if (value.dtype === 'f32') {
      return value;
    }
    if (value.dtype === 'f16') {
      const sourceShape = Array.isArray(value.shape) && value.shape.length > 0 ? value.shape : [...shape];
      const source = createTensor(value.buffer, 'f16', sourceShape, `${label}_source_f16`);
      if (preserveF16) return source;
      const promoted = await castF16ToF32(source);
      return registerOwned(createTensor(promoted.buffer, 'f32', sourceShape, `${label}_trainable_f32`));
    }
    throw new Error(`Distill full-graph student weight "${label}" uses unsupported dtype "${value.dtype}".`);
  }
  if (value instanceof GPUBuffer) {
    const sourceShape = [...shape];
    const rawDtype = String(getBufferDtype(value) || 'f32').toLowerCase();
    const dtype = rawDtype === 'f16' ? 'f16' : 'f32';
    const tensor = createTensor(value, dtype, sourceShape, label);
    if (dtype === 'f16') {
      if (preserveF16) return tensor;
      const promoted = await castF16ToF32(tensor);
      return registerOwned(createTensor(promoted.buffer, 'f32', sourceShape, `${label}_trainable_f32`));
    }
    return tensor;
  }
  if (isCpuWeightBuffer(value)) {
    const sourceShape = Array.isArray(value.shape) && value.shape.length > 0 ? value.shape : [...shape];
    const dtype = resolveTensorDtype(value, 'f32');
    if (dtype === 'f32') {
      const tensor = makeTensorFromFloat32(value.data, sourceShape, `${label}_cpu_f32`);
      return registerOwned(tensor);
    }
    if (dtype === 'f16') {
      let raw = null;
      if (value.data instanceof Uint16Array) {
        raw = value.data;
      } else if (ArrayBuffer.isView(value.data)) {
        raw = new Uint16Array(
          value.data.buffer,
          value.data.byteOffset,
          Math.floor(value.data.byteLength / 2)
        );
      } else if (value.data instanceof ArrayBuffer) {
        raw = new Uint16Array(value.data);
      }
      if (!raw) {
        throw new Error(`Distill full-graph student weight "${label}" has non-typed f16 CPU data.`);
      }
      const source = makeTensorFromF16Bits(raw, sourceShape, `${label}_cpu_f16`);
      if (preserveF16) return registerOwned(source);
      const promoted = await castF16ToF32(source);
      releaseTensor(source);
      return registerOwned(createTensor(promoted.buffer, 'f32', sourceShape, `${label}_trainable_f32`));
    }
    throw new Error(`Distill full-graph student weight "${label}" has unsupported CPU dtype "${dtype}".`);
  }
  if (value.buffer instanceof GPUBuffer) {
    const resolvedShape = Array.isArray(value.shape) && value.shape.length > 0 ? value.shape : [...shape];
    const tensor = createTensor(
      value.buffer,
      resolveTensorDtype(value, 'f32'),
      resolvedShape,
      label
    );
    if (tensor.dtype === 'f16') {
      if (preserveF16) return tensor;
      const promoted = await castF16ToF32(tensor);
      return registerOwned(createTensor(promoted.buffer, 'f32', resolvedShape, `${label}_trainable_f32`));
    }
    return tensor;
  }
  throw new Error(`Distill full-graph student weight "${label}" is not GPU-resident.`);
}

export function resolvePhasePrompts(batch, phase) {
  const distill = batch?.distill || {};
  const prompts = phase === 'positive'
    ? distill.tripletPositivePrompts
    : (phase === 'negative' ? distill.tripletNegativePrompts : distill.prompts);
  if (!Array.isArray(prompts) || prompts.length === 0) {
    throw new Error(`Distill student fixture requires distill prompts for phase "${phase}".`);
  }
  return prompts;
}

export function createDistillStudentProjectionModelFixture(overrides = {}, options = {}) {
  const distillRuntime = options.distillRuntime && typeof options.distillRuntime === 'object'
    ? options.distillRuntime
    : null;
  if (!distillRuntime?.studentPipeline) {
    throw new Error('Distill student fixture requires distillRuntime.studentPipeline.');
  }
  const outputDim = clampDistillTopK(
    options.outputDim
    ?? options.inputDim
    ?? DISTILL_ADAPTER_TOP_K
  );
  const inferredEmbeddingDim = Math.floor(
    Number(distillRuntime.studentPipeline?.modelConfig?.hiddenSize)
  );
  const embeddingDim = Number.isInteger(options.embeddingDim) && options.embeddingDim > 0
    ? options.embeddingDim
    : (Number.isFinite(inferredEmbeddingDim) && inferredEmbeddingDim > 0
      ? inferredEmbeddingDim
      : outputDim);
  const config = createTrainingConfig({
    ...overrides,
    training: {
      enabled: true,
      lossScaling: { enabled: false },
      gradient: { maxNorm: 0 },
      ...(overrides.training || {}),
    },
  });

  const projectionWeights = new Float32Array(embeddingDim * outputDim);
  const projectionWeight = makeTensorFromFloat32(
    projectionWeights,
    [embeddingDim, outputDim],
    'distill_student_head_weight'
  );
  const temporaryInputs = new Set();

  async function projectEmbeddingInput(inputTensor, tape) {
    const rows = Number.isFinite(inputTensor?.shape?.[0]) ? inputTensor.shape[0] : 1;
    return tape.record(
      OpType.MATMUL,
      (a, b) => runMatmul(a, b, rows, outputDim, embeddingDim, { transposeB: false, outputDtype: 'f32' }),
      [inputTensor, projectionWeight],
      { M: rows, N: outputDim, K: embeddingDim, transposeB: false }
    );
  }

  async function buildStudentEmbeddingInput(batch, phase = 'anchor') {
    const prompts = resolvePhasePrompts(batch, phase);
    const rows = prompts.length;
    const features = new Float32Array(rows * embeddingDim);
    for (let row = 0; row < rows; row += 1) {
      const prompt = String(prompts[row] || '').trim();
      const studentResult = await distillRuntime.studentPipeline.prefillWithEmbedding(prompt, {
        useChatTemplate: false,
        embeddingMode: 'last',
      });
      try {
        const studentEmbedding = toFloat32Array(studentResult?.embedding, 'student embedding');
        const rowOffset = row * embeddingDim;
        const copyCount = Math.min(embeddingDim, studentEmbedding.length);
        features.set(studentEmbedding.subarray(0, copyCount), rowOffset);
      } finally {
        disposePrefillSnapshot(studentResult);
        distillRuntime.studentPipeline.reset();
      }
    }
    const inputTensor = makeTensorFromFloat32(
      features,
      [rows, embeddingDim],
      `distill_student_${phase}_embedding`
    );
    temporaryInputs.add(inputTensor);
    return inputTensor;
  }

  const model = {
    async forward(inputTensor, tape) {
      return projectEmbeddingInput(inputTensor, tape);
    },
    async forwardDistill(batch, tape, forwardOptions = {}) {
      const requestedPhase = String(forwardOptions?.phase || 'anchor').trim();
      const phase = requestedPhase === 'positive'
        ? 'positive'
        : (requestedPhase === 'negative' ? 'negative' : 'anchor');
      const inputTensor = await buildStudentEmbeddingInput(batch, phase);
      const logits = await projectEmbeddingInput(inputTensor, tape);
      return { logits };
    },
    cleanupDistillStep() {
      for (const tensor of temporaryInputs) {
        releaseTensor(tensor);
      }
      temporaryInputs.clear();
    },
    loraParams() {
      return [projectionWeight];
    },
    paramGroups() {
      return {
        encoder: [],
        prior: [],
        decoder: [],
        base: [projectionWeight],
        lora: [projectionWeight],
      };
    },
  };

  return {
    config,
    model,
    outputDim,
    embeddingDim,
    cleanup() {
      model.cleanupDistillStep();
      releaseTensor(projectionWeight);
    },
  };
}
