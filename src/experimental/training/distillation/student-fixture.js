import { getDevice } from '../../../gpu/device.js';
import { createTrainingConfig } from '../../../config/training-defaults.js';
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
} from '../../../gpu/kernels/index.js';
import { createTensor } from '../../../gpu/tensor.js';
import { acquireBuffer, uploadData, releaseBuffer } from '../../../memory/buffer-pool.js';
import { getBufferDtype, getWeightDtype, isCpuWeightBuffer, isWeightBuffer } from '../../../gpu/weight-buffer.js';
import { OpType } from '../autograd.js';
import { LoraAdapter } from '../lora.js';
import { normalizeOptionalString } from './suite-data.js';
import { LORA_MODULE_ALIASES } from '../../../inference/pipelines/text/lora.js';
import { resolveEmbeddingScale } from '../../../inference/pipelines/text/embed.js';
import { isSlidingLayerType } from '../../../inference/pipelines/text/attention/dispatch-params.js';

const DISTILL_ADAPTER_TOP_K = 64;
const DISTILL_STUDENT_GRAPH_PROJECTION = 'projection_head';
const DISTILL_STUDENT_GRAPH_FULL = 'transformer_full';
const TRANSFORMER_LORA_TARGET_MODULES = Object.freeze([
  'q_proj',
  'k_proj',
  'v_proj',
  'o_proj',
  'gate_proj',
  'up_proj',
  'gate_up_proj',
  'down_proj',
]);

function makeTensorFromFloat32(values, shape, label) {
  const data = values instanceof Float32Array ? values : new Float32Array(values);
  const buffer = acquireBuffer(data.byteLength, undefined, label || 'train_tensor');
  uploadData(buffer, data);
  return createTensor(buffer, 'f32', shape, label || 'train_tensor');
}

function makeTensorFromF16Bits(values, shape, label) {
  const data = values instanceof Uint16Array ? values : new Uint16Array(values);
  const buffer = acquireBuffer(data.byteLength, undefined, label || 'train_tensor_f16');
  uploadData(buffer, data);
  return createTensor(buffer, 'f16', shape, label || 'train_tensor_f16');
}

function makeTensorFromUint32(values, shape, label) {
  const data = values instanceof Uint32Array ? values : new Uint32Array(values);
  const buffer = acquireBuffer(data.byteLength, undefined, label || 'train_tokens');
  uploadData(buffer, data);
  return createTensor(buffer, 'f32', shape, label || 'train_tokens');
}

function releaseTensor(tensor) {
  if (!tensor?.buffer) return;
  releaseBuffer(tensor.buffer);
}

function tensorElementCount(shape) {
  return shape.reduce((product, value) => product * value, 1);
}

async function recordTensorView(tape, input, shape, label) {
  if (tensorElementCount(input.shape) !== tensorElementCount(shape)) {
    throw new Error(`${label} cannot change tensor element count.`);
  }
  return tape.record(
    OpType.RESHAPE,
    (value) => createTensor(value.buffer, value.dtype, [...shape], label),
    [input],
    { shape: [...shape] }
  );
}

function toFloat32Array(values, label = 'values') {
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

function disposePrefillSnapshot(result) {
  const cache = result?.cache;
  if (cache && typeof cache.clear === 'function') {
    cache.clear();
  }
}

function toFiniteNumber(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function clampDistillTopK(value) {
  const parsed = Math.floor(toFiniteNumber(value, DISTILL_ADAPTER_TOP_K));
  return Math.max(2, Math.min(256, parsed));
}

function normalizeDistillStudentGraphMode(value) {
  const normalized = normalizeOptionalString(value);
  if (!normalized) return DISTILL_STUDENT_GRAPH_FULL;
  const compact = normalized.toLowerCase().replace(/[-\s]/g, '_');
  if (compact === 'projection_head' || compact === 'projection') {
    return DISTILL_STUDENT_GRAPH_PROJECTION;
  }
  return DISTILL_STUDENT_GRAPH_FULL;
}

function normalizeTransformerLoraConfig(value) {
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

function resolveTransformerLoraShape(moduleName, dims) {
  if (moduleName === 'q_proj') {
    return { inDim: dims.hiddenSize, outDim: dims.numHeads * dims.headDim };
  }
  if (moduleName === 'k_proj' || moduleName === 'v_proj') {
    return { inDim: dims.hiddenSize, outDim: dims.numKVHeads * dims.headDim };
  }
  if (moduleName === 'o_proj') {
    return { inDim: dims.attentionSize, outDim: dims.hiddenSize };
  }
  if (moduleName === 'gate_up_proj') {
    return { inDim: dims.hiddenSize, outDim: dims.intermediateSize * 2 };
  }
  if (moduleName === 'gate_proj' || moduleName === 'up_proj') {
    return { inDim: dims.hiddenSize, outDim: dims.intermediateSize };
  }
  if (moduleName === 'down_proj') {
    return { inDim: dims.intermediateSize, outDim: dims.hiddenSize };
  }
  throw new Error(`Transformer LoRA target module "${moduleName}" is not supported.`);
}

function createTransformerLoraAdapters(config, dims) {
  if (!config) return {};
  const adapters = {};
  for (const moduleName of config.targetModules) {
    const shape = resolveTransformerLoraShape(moduleName, dims);
    adapters[moduleName] = new LoraAdapter({
      inDim: shape.inDim,
      outDim: shape.outDim,
      rank: config.rank,
      alpha: config.alpha,
    });
  }
  return adapters;
}

function disposeTransformerLoraAdapters(layers) {
  for (const layer of layers) {
    for (const adapter of Object.values(layer.lora || {})) {
      if (adapter && typeof adapter.dispose === 'function') {
        adapter.dispose();
      }
    }
  }
}

function resolveTensorDtype(value, fallback = 'f32') {
  const dtype = isWeightBuffer(value)
    ? value.dtype
    : (value?.dtype || getWeightDtype(value) || null);
  const normalized = String(dtype || '').toLowerCase();
  return normalized === 'f16' ? 'f16' : (normalized === 'f32' ? 'f32' : fallback);
}

async function ensureTrainableTensor(
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

async function ensureNormTensor(value, hiddenSize, label, ownedTrainables = null) {
  return ensureTrainableTensor(value, [hiddenSize], label, ownedTrainables);
}

function hasTensorPayload(value) {
  if (!value) return false;
  if (value instanceof GPUBuffer) return true;
  if (isWeightBuffer(value) || isCpuWeightBuffer(value)) return true;
  if (value?.buffer instanceof GPUBuffer) return true;
  if (ArrayBuffer.isView(value) || Array.isArray(value)) return true;
  return false;
}

function getTensorRows(value) {
  const shape = Array.isArray(value?.shape) ? value.shape : null;
  const rows = Number(shape?.[0]);
  return Number.isFinite(rows) && rows > 0 ? Math.floor(rows) : null;
}

function getTensorCols(value) {
  const shape = Array.isArray(value?.shape) ? value.shape : null;
  const cols = Number(shape?.[1]);
  return Number.isFinite(cols) && cols > 0 ? Math.floor(cols) : null;
}

function resolveLayerFfnIntermediateSize(layerIdx, weights, fallback) {
  const gateUpRows = getTensorRows(weights.gateUp || weights.ffnGateUp);
  if (gateUpRows !== null && gateUpRows % 2 === 0) {
    return gateUpRows / 2;
  }
  const gateRows = getTensorRows(weights.gate || weights.ffnGate);
  if (gateRows !== null) {
    return gateRows;
  }
  const upRows = getTensorRows(weights.up || weights.ffnUp);
  if (upRows !== null) {
    return upRows;
  }
  const downCols = getTensorCols(weights.down || weights.ffnDown);
  if (downCols !== null) {
    return downCols;
  }
  if (Number.isInteger(fallback) && fallback > 0) {
    return fallback;
  }
  throw new Error(`Distill full-graph student cannot resolve FFN size for layer ${layerIdx}.`);
}

function resolvePhasePrompts(batch, phase) {
  const distill = batch?.distill || {};
  const prompts = phase === 'positive'
    ? distill.tripletPositivePrompts
    : (phase === 'negative' ? distill.tripletNegativePrompts : distill.prompts);
  if (!Array.isArray(prompts) || prompts.length === 0) {
    throw new Error(`Distill student fixture requires distill prompts for phase "${phase}".`);
  }
  return prompts;
}

function createRowSliceTensor(inputTensor, rows, cols, rowIndex, label) {
  const device = getDevice();
  if (!device) {
    throw new Error('Distill full-graph student requires active GPU device.');
  }
  const dtype = inputTensor?.dtype === 'f16' ? 'f16' : 'f32';
  const bytesPerElement = dtype === 'f16' ? 2 : 4;
  const rowBytes = cols * bytesPerElement;
  const clampedRow = Math.max(0, Math.min(rows - 1, rowIndex));
  const outputBuffer = acquireBuffer(rowBytes, undefined, label);
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(
    inputTensor.buffer,
    clampedRow * rowBytes,
    outputBuffer,
    0,
    rowBytes
  );
  device.queue.submit([encoder.finish()]);
  return createTensor(outputBuffer, dtype, [1, cols], label);
}

function createDistillStudentProjectionModelFixture(overrides = {}, options = {}) {
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

async function createDistillStudentTransformerModelFixture(overrides = {}, options = {}) {
  const distillRuntime = options.distillRuntime && typeof options.distillRuntime === 'object'
    ? options.distillRuntime
    : null;
  const studentPipeline = distillRuntime?.studentPipeline || null;
  if (!studentPipeline?.modelConfig || !(studentPipeline.weights instanceof Map)) {
    throw new Error('Distill full-graph student fixture requires loaded student pipeline weights.');
  }
  const modelConfig = studentPipeline.modelConfig;
  const hiddenSize = Math.max(1, Math.floor(Number(modelConfig.hiddenSize) || 0));
  const intermediateSize = Math.max(1, Math.floor(Number(modelConfig.intermediateSize) || 0));
  const numLayers = Math.max(1, Math.floor(Number(modelConfig.numLayers) || 0));
  const numHeads = Math.max(1, Math.floor(Number(modelConfig.numHeads) || 0));
  const numKVHeads = Math.max(1, Math.floor(Number(modelConfig.numKVHeads || numHeads) || 0));
  const headDim = Math.max(1, Math.floor(Number(modelConfig.headDim) || 0));
  const vocabSize = Math.max(1, Math.floor(Number(modelConfig.vocabSize) || 0));
  const rmsNormEps = Number.isFinite(modelConfig.rmsNormEps) ? modelConfig.rmsNormEps : 1e-6;
  const hiddenActivation = String(modelConfig.hiddenActivation || 'silu').toLowerCase();
  const swigluLimit = Number.isFinite(modelConfig.swigluLimit) ? modelConfig.swigluLimit : 0;
  const useEmbeddingTranspose = modelConfig.embeddingTranspose === true;
  const tieWordEmbeddings = modelConfig.useTiedEmbeddings === true;
  const embeddingScale = resolveEmbeddingScale(modelConfig, hiddenSize);
  const loraConfig = normalizeTransformerLoraConfig(options.loraAdapter || null);
  const freezeBaseGrad = Boolean(loraConfig);
  const frozenWeightOptions = { preserveF16: freezeBaseGrad };
  const stopBaseWeight = freezeBaseGrad ? { stopGradInputs: [1] } : {};
  const stopRopeWeights = freezeBaseGrad ? { stopGradInputs: [1, 2] } : {};

  const config = createTrainingConfig({
    ...overrides,
    training: {
      enabled: true,
      lossScaling: { enabled: false },
      gradient: { maxNorm: 0 },
      ...(overrides.training || {}),
    },
  });

  const ownedTrainables = new Set();
  const embeddingWeight = await ensureTrainableTensor(
    studentPipeline.weights.get('embed'),
    [vocabSize, hiddenSize],
    'embed',
    ownedTrainables,
    frozenWeightOptions
  );
  const lmHeadWeight = tieWordEmbeddings
    ? embeddingWeight
    : await ensureTrainableTensor(
      studentPipeline.weights.get('lm_head'),
      [vocabSize, hiddenSize],
      'lm_head',
      ownedTrainables,
      frozenWeightOptions
    );
  const finalNormWeight = await ensureNormTensor(
    studentPipeline.weights.get('final_norm'),
    hiddenSize,
    'final_norm',
    ownedTrainables
  );

  const ropeDim = Math.max(1, Math.floor(headDim / 2));
  const ropeRows = Math.max(1, Math.floor(Number(modelConfig.maxSeqLen) || 1));
  const ropeCos = await ensureTrainableTensor(
    createTensor(studentPipeline.ropeFreqsCos, 'f32', [ropeRows, ropeDim], 'rope_cos'),
    [ropeRows, ropeDim],
    'rope_cos',
    ownedTrainables
  );
  const ropeSin = await ensureTrainableTensor(
    createTensor(studentPipeline.ropeFreqsSin, 'f32', [ropeRows, ropeDim], 'rope_sin'),
    [ropeRows, ropeDim],
    'rope_sin',
    ownedTrainables
  );
  const hasLocalAttention = Array.isArray(modelConfig.layerTypes)
    && modelConfig.layerTypes.some((layerType) => isSlidingLayerType(layerType));
  let ropeLocalCos = null;
  let ropeLocalSin = null;
  if (hasLocalAttention) {
    if (!(studentPipeline.ropeLocalCos instanceof GPUBuffer)
      || !(studentPipeline.ropeLocalSin instanceof GPUBuffer)) {
      throw new Error(
        'Distill full-graph student requires local RoPE tables for sliding-attention layers.'
      );
    }
    ropeLocalCos = await ensureTrainableTensor(
      createTensor(studentPipeline.ropeLocalCos, 'f32', [ropeRows, ropeDim], 'rope_local_cos'),
      [ropeRows, ropeDim],
      'rope_local_cos',
      ownedTrainables
    );
    ropeLocalSin = await ensureTrainableTensor(
      createTensor(studentPipeline.ropeLocalSin, 'f32', [ropeRows, ropeDim], 'rope_local_sin'),
      [ropeRows, ropeDim],
      'rope_local_sin',
      ownedTrainables
    );
  }

  const layerParams = [];
  const loraParams = [];
  const layers = [];
  const loraDims = {
    hiddenSize,
    intermediateSize,
    numHeads,
    numKVHeads,
    headDim,
    attentionSize: numHeads * headDim,
  };
  for (let layerIdx = 0; layerIdx < numLayers; layerIdx += 1) {
    const layerWeights = studentPipeline.weights.get(`layer_${layerIdx}`);
    if (!layerWeights) {
      throw new Error(`Distill full-graph student missing layer_${layerIdx} weights.`);
    }
    const layerIntermediateSize = resolveLayerFfnIntermediateSize(layerIdx, layerWeights, intermediateSize);
    const gateUpWeight = layerWeights.gateUp || layerWeights.ffnGateUp || null;
    let layerGateUp = null;
    let layerGate = null;
    let layerUp = null;
    if (hasTensorPayload(gateUpWeight)) {
      layerGateUp = await ensureTrainableTensor(
        gateUpWeight,
        [layerIntermediateSize * 2, hiddenSize],
        `layer_${layerIdx}.ffn_gate_up`,
        ownedTrainables,
        frozenWeightOptions
      );
    } else {
      const gateWeight = layerWeights.gate || layerWeights.ffnGate || null;
      const upWeight = layerWeights.up || layerWeights.ffnUp || null;
      if (!hasTensorPayload(gateWeight) || !hasTensorPayload(upWeight)) {
        throw new Error(
          `Distill full-graph student missing gate/up projections on layer ${layerIdx}.`
        );
      }
      layerGate = await ensureTrainableTensor(
        gateWeight,
        [layerIntermediateSize, hiddenSize],
        `layer_${layerIdx}.ffn_gate`,
        ownedTrainables,
        frozenWeightOptions
      );
      layerUp = await ensureTrainableTensor(
        upWeight,
        [layerIntermediateSize, hiddenSize],
        `layer_${layerIdx}.ffn_up`,
        ownedTrainables,
        frozenWeightOptions
      );
    }
    const layer = {
      inputNorm: await ensureNormTensor(
        layerWeights.inputNorm,
        hiddenSize,
        `layer_${layerIdx}.input_norm`,
        ownedTrainables
      ),
      queryKeyNorm: modelConfig.queryKeyNorm === true
        && (!Array.isArray(modelConfig.queryKeyNormLayers)
          || modelConfig.queryKeyNormLayers.includes(layerIdx)),
      qNorm: null,
      kNorm: null,
      qProj: await ensureTrainableTensor(
        layerWeights.qProj,
        [numHeads * headDim, hiddenSize],
        `layer_${layerIdx}.q_proj`,
        ownedTrainables,
        frozenWeightOptions
      ),
      kProj: await ensureTrainableTensor(
        layerWeights.kProj,
        [numKVHeads * headDim, hiddenSize],
        `layer_${layerIdx}.k_proj`,
        ownedTrainables,
        frozenWeightOptions
      ),
      vProj: await ensureTrainableTensor(
        layerWeights.vProj,
        [numKVHeads * headDim, hiddenSize],
        `layer_${layerIdx}.v_proj`,
        ownedTrainables,
        frozenWeightOptions
      ),
      oProj: await ensureTrainableTensor(
        layerWeights.oProj,
        [hiddenSize, numHeads * headDim],
        `layer_${layerIdx}.o_proj`,
        ownedTrainables,
        frozenWeightOptions
      ),
      postAttentionNorm: layerWeights.postAttentionNorm
        ? await ensureNormTensor(
          layerWeights.postAttentionNorm,
          hiddenSize,
          `layer_${layerIdx}.post_attention_norm`,
          ownedTrainables
        )
        : null,
      preFeedforwardNorm: layerWeights.preFeedforwardNorm
        ? await ensureNormTensor(
          layerWeights.preFeedforwardNorm,
          hiddenSize,
          `layer_${layerIdx}.pre_feedforward_norm`,
          ownedTrainables
        )
        : null,
      postFeedforwardNorm: layerWeights.postFeedforwardNorm
        ? await ensureNormTensor(
          layerWeights.postFeedforwardNorm,
          hiddenSize,
          `layer_${layerIdx}.post_feedforward_norm`,
          ownedTrainables
        )
        : null,
      gateUp: layerGateUp,
      gate: layerGate,
      up: layerUp,
      down: await ensureTrainableTensor(
        layerWeights.down || layerWeights.ffnDown,
        [hiddenSize, layerIntermediateSize],
        `layer_${layerIdx}.ffn_down`,
        ownedTrainables,
        frozenWeightOptions
      ),
      intermediateSize: layerIntermediateSize,
      lora: createTransformerLoraAdapters(loraConfig, {
        ...loraDims,
        intermediateSize: layerIntermediateSize,
      }),
    };
    if (layer.gateUp && (layer.lora.gate_proj || layer.lora.up_proj)) {
      throw new Error(
        `Layer ${layerIdx} has fused gate/up weights but separate gate_proj or up_proj LoRA targets.`
      );
    }
    if (!layer.gateUp && layer.lora.gate_up_proj) {
      throw new Error(
        `Layer ${layerIdx} has separate gate/up weights but a fused gate_up_proj LoRA target.`
      );
    }
    if (layer.queryKeyNorm) {
      const weightedLayers = modelConfig.queryKeyNormWeightLayers;
      const expectsWeightedNorm = !Array.isArray(weightedLayers) || weightedLayers.includes(layerIdx);
      if (!expectsWeightedNorm) {
        throw new Error(
          `Distill full-graph student does not support unit-weight Q/K norm on layer ${layerIdx}.`
        );
      }
      layer.qNorm = await ensureNormTensor(
        layerWeights.qNorm,
        headDim,
        `layer_${layerIdx}.q_norm`,
        ownedTrainables
      );
      layer.kNorm = await ensureNormTensor(
        layerWeights.kNorm,
        headDim,
        `layer_${layerIdx}.k_norm`,
        ownedTrainables
      );
    }
    if (modelConfig.postAttentionNorm === true && !layer.postAttentionNorm) {
      throw new Error(`Distill full-graph student missing post-attention norm on layer ${layerIdx}.`);
    }
    if (modelConfig.preFeedforwardNorm === true && !layer.preFeedforwardNorm) {
      throw new Error(`Distill full-graph student missing pre-feedforward norm on layer ${layerIdx}.`);
    }
    if (modelConfig.postFeedforwardNorm === true && !layer.postFeedforwardNorm) {
      throw new Error(`Distill full-graph student missing post-feedforward norm on layer ${layerIdx}.`);
    }
    layers.push(layer);
    layerParams.push(
      layer.inputNorm,
      layer.qProj,
      layer.kProj,
      layer.vProj,
      layer.oProj,
      ...(layer.gateUp ? [layer.gateUp] : [layer.gate, layer.up]),
      layer.down
    );
    for (const adapter of Object.values(layer.lora)) {
      loraParams.push(adapter.A, adapter.B);
    }
    if (layer.postAttentionNorm) {
      layerParams.push(layer.postAttentionNorm);
    }
    if (layer.preFeedforwardNorm) {
      layerParams.push(layer.preFeedforwardNorm);
    }
    if (layer.postFeedforwardNorm) {
      layerParams.push(layer.postFeedforwardNorm);
    }
    if (layer.qNorm) {
      layerParams.push(layer.qNorm, layer.kNorm);
    }
  }

  const encoderParams = [embeddingWeight, ...layerParams];
  const decoderParams = [finalNormWeight, lmHeadWeight];
  const baseParams = [...encoderParams, ...decoderParams];
  const temporaryInputs = new Set();

  async function buildPromptTokens(prompt) {
    const normalized = String(prompt || '').trim();
    if (!normalized) {
      throw new Error('Distill full-graph student prompt is empty.');
    }
    const tokenIds = studentPipeline.tokenizer.encode(normalized);
    if (!Array.isArray(tokenIds) || tokenIds.length === 0) {
      throw new Error('Distill full-graph student tokenizer produced no tokens.');
    }
    const tokenTensor = makeTensorFromUint32(
      tokenIds,
      [tokenIds.length],
      'distill_student_prompt_tokens'
    );
    temporaryInputs.add(tokenTensor);
    return { tokenTensor, seqLen: tokenIds.length };
  }

  async function runTransformerTokenTensor(tokenTensor, seqLen, tape, forwardOptions = {}) {
    const captureStage = async (stage, tensor, layerIdx = null) => {
      if (typeof forwardOptions.captureStage === 'function') {
        await forwardOptions.captureStage({
          stage,
          layerIdx,
          seqLen,
          tensor,
        });
      }
      return tensor;
    };
    let hidden = await tape.record(
      OpType.EMBED,
      (indices, embeddings) => runGather(
        indices,
        embeddings,
        seqLen,
        hiddenSize,
        vocabSize,
        {
          embeddingDtype: resolveTensorDtype(embeddingWeight, 'f32'),
          outputDtype: 'f32',
          transpose: useEmbeddingTranspose,
        }
      ),
      [tokenTensor, embeddingWeight],
      {
        numTokens: seqLen,
        hiddenSize,
        vocabSize,
        transpose: useEmbeddingTranspose,
        indexOffset: 0,
        ...stopBaseWeight,
      }
    );
    if (embeddingScale !== 1) {
      hidden = await tape.record(
        OpType.SCALE,
        (x) => runScale(x, embeddingScale, { count: seqLen * hiddenSize }),
        [hidden],
        { count: seqLen * hiddenSize, scale: embeddingScale }
      );
    }
    await captureStage('embed.out', hidden);

    for (let layerIdx = 0; layerIdx < layers.length; layerIdx += 1) {
      const layer = layers[layerIdx];
      const layerIntermediateSize = layer.intermediateSize || intermediateSize;
      const normed = await tape.record(
        OpType.RMSNORM,
        (x, gamma) => runRMSNorm(x, gamma, rmsNormEps, {
          batchSize: seqLen,
          hiddenSize,
          rmsNormWeightOffset: modelConfig.rmsNormWeightOffset === true,
        }),
        [hidden, layer.inputNorm],
        {
          numTokens: seqLen,
          hiddenSize,
          eps: rmsNormEps,
          rmsNormWeightOffset: modelConfig.rmsNormWeightOffset === true,
          ...stopBaseWeight,
        }
      );
      await captureStage('attn.post_input_norm', normed, layerIdx);

      let q2d = await tape.record(
        OpType.MATMUL,
        (x, w) => runMatmul(x, w, seqLen, numHeads * headDim, hiddenSize, {
          transposeB: 'auto',
          outputDtype: 'f32',
        }),
        [normed, layer.qProj],
        { M: seqLen, N: numHeads * headDim, K: hiddenSize, transposeB: 'auto', ...stopBaseWeight }
      );
      let k2d = await tape.record(
        OpType.MATMUL,
        (x, w) => runMatmul(x, w, seqLen, numKVHeads * headDim, hiddenSize, {
          transposeB: 'auto',
          outputDtype: 'f32',
        }),
        [normed, layer.kProj],
        { M: seqLen, N: numKVHeads * headDim, K: hiddenSize, transposeB: 'auto', ...stopBaseWeight }
      );
      let v2d = await tape.record(
        OpType.MATMUL,
        (x, w) => runMatmul(x, w, seqLen, numKVHeads * headDim, hiddenSize, {
          transposeB: 'auto',
          outputDtype: 'f32',
        }),
        [normed, layer.vProj],
        { M: seqLen, N: numKVHeads * headDim, K: hiddenSize, transposeB: 'auto', ...stopBaseWeight }
      );
      if (layer.lora.q_proj) {
        const delta = await layer.lora.q_proj.forward(normed, tape);
        q2d = await tape.record(
          OpType.RESIDUAL_ADD,
          (a, b) => runResidualAdd(a, b, seqLen * numHeads * headDim),
          [q2d, delta],
          { size: seqLen * numHeads * headDim }
        );
      }
      if (layer.lora.k_proj) {
        const delta = await layer.lora.k_proj.forward(normed, tape);
        k2d = await tape.record(
          OpType.RESIDUAL_ADD,
          (a, b) => runResidualAdd(a, b, seqLen * numKVHeads * headDim),
          [k2d, delta],
          { size: seqLen * numKVHeads * headDim }
        );
      }
      if (layer.lora.v_proj) {
        const delta = await layer.lora.v_proj.forward(normed, tape);
        v2d = await tape.record(
          OpType.RESIDUAL_ADD,
          (a, b) => runResidualAdd(a, b, seqLen * numKVHeads * headDim),
          [v2d, delta],
          { size: seqLen * numKVHeads * headDim }
        );
      }
      await captureStage('attn.q_proj', q2d, layerIdx);
      await captureStage('attn.k_proj', k2d, layerIdx);
      await captureStage('attn.v_proj', v2d, layerIdx);

      if (layer.queryKeyNorm) {
        const qNormInput = await recordTensorView(
          tape,
          q2d,
          [seqLen * numHeads, headDim],
          `layer_${layerIdx}_q_norm_input`
        );
        const qNormed = await tape.record(
          OpType.RMSNORM,
          (x, gamma) => runRMSNorm(x, gamma, rmsNormEps, {
            batchSize: seqLen * numHeads,
            hiddenSize: headDim,
            rmsNormWeightOffset: modelConfig.rmsNormWeightOffset === true,
          }),
          [qNormInput, layer.qNorm],
          {
            numTokens: seqLen * numHeads,
            hiddenSize: headDim,
            eps: rmsNormEps,
            rmsNormWeightOffset: modelConfig.rmsNormWeightOffset === true,
            ...stopBaseWeight,
          }
        );
        q2d = await recordTensorView(
          tape,
          qNormed,
          [seqLen, numHeads * headDim],
          `layer_${layerIdx}_q_normed`
        );
        const kNormInput = await recordTensorView(
          tape,
          k2d,
          [seqLen * numKVHeads, headDim],
          `layer_${layerIdx}_k_norm_input`
        );
        const kNormed = await tape.record(
          OpType.RMSNORM,
          (x, gamma) => runRMSNorm(x, gamma, rmsNormEps, {
            batchSize: seqLen * numKVHeads,
            hiddenSize: headDim,
            rmsNormWeightOffset: modelConfig.rmsNormWeightOffset === true,
          }),
          [kNormInput, layer.kNorm],
          {
            numTokens: seqLen * numKVHeads,
            hiddenSize: headDim,
            eps: rmsNormEps,
            rmsNormWeightOffset: modelConfig.rmsNormWeightOffset === true,
            ...stopBaseWeight,
          }
        );
        k2d = await recordTensorView(
          tape,
          kNormed,
          [seqLen, numKVHeads * headDim],
          `layer_${layerIdx}_k_normed`
        );
        await captureStage('attn.q_norm', q2d, layerIdx);
        await captureStage('attn.k_norm', k2d, layerIdx);
      }

      const q3d = await recordTensorView(
        tape,
        q2d,
        [seqLen, numHeads, headDim],
        `layer_${layerIdx}_q`
      );
      const k3d = await recordTensorView(
        tape,
        k2d,
        [seqLen, numKVHeads, headDim],
        `layer_${layerIdx}_k`
      );
      const v3d = await recordTensorView(
        tape,
        v2d,
        [seqLen, numKVHeads, headDim],
        `layer_${layerIdx}_v`
      );
      const useLocalRope = isSlidingLayerType(modelConfig.layerTypes?.[layerIdx]);
      const layerRopeCos = useLocalRope ? ropeLocalCos : ropeCos;
      const layerRopeSin = useLocalRope ? ropeLocalSin : ropeSin;
      if (!layerRopeCos || !layerRopeSin) {
        throw new Error(`Distill full-graph student missing RoPE tensors on layer ${layerIdx}.`);
      }

      const qRope = await tape.record(
        OpType.ROPE,
        (q, cos, sin) => runRoPE(q, cos, sin, seqLen, { numHeads, headDim, startPos: 0 }),
        [q3d, layerRopeCos, layerRopeSin],
        { seqLen, numHeads, headDim, startPos: 0, ...stopRopeWeights }
      );
      const kRope = await tape.record(
        OpType.ROPE,
        (k, cos, sin) => runRoPE(k, cos, sin, seqLen, { numHeads: numKVHeads, headDim, startPos: 0 }),
        [k3d, layerRopeCos, layerRopeSin],
        { seqLen, numHeads: numKVHeads, headDim, startPos: 0, ...stopRopeWeights }
      );
      await captureStage('attn.q_rope', qRope, layerIdx);
      await captureStage('attn.k_rope', kRope, layerIdx);

      const attention = await tape.record(
        OpType.ATTENTION,
        (q, k, v) => runAttention(q, k, v, null, numHeads, headDim, {
          seqLen,
          kvLen: seqLen,
          numKVHeads,
          causal: true,
          startPos: 0,
          scale: 1 / Math.sqrt(headDim),
        }),
        [qRope, kRope, v3d],
        {
          seqLen,
          numHeads,
          numKVHeads,
          headDim,
          scale: 1 / Math.sqrt(headDim),
          causal: true,
          recomputeForward: true,
        }
      );
      await captureStage('attn.core_out', attention, layerIdx);
      const attention2d = await recordTensorView(
        tape,
        attention,
        [seqLen, numHeads * headDim],
        `layer_${layerIdx}_attn_2d`
      );

      let attentionOutput = await tape.record(
        OpType.MATMUL,
        (x, w) => runMatmul(x, w, seqLen, hiddenSize, numHeads * headDim, {
          transposeB: 'auto',
          outputDtype: 'f32',
        }),
        [attention2d, layer.oProj],
        {
          M: seqLen,
          N: hiddenSize,
          K: numHeads * headDim,
          transposeB: 'auto',
          ...stopBaseWeight,
        }
      );
      if (layer.lora.o_proj) {
        const delta = await layer.lora.o_proj.forward(attention2d, tape);
        attentionOutput = await tape.record(
          OpType.RESIDUAL_ADD,
          (a, b) => runResidualAdd(a, b, seqLen * hiddenSize),
          [attentionOutput, delta],
          { size: seqLen * hiddenSize }
        );
      }
      await captureStage('attn.out', attentionOutput, layerIdx);
      const normalizedAttentionOutput = modelConfig.postAttentionNorm === true
        ? await tape.record(
          OpType.RMSNORM,
          (x, gamma) => runRMSNorm(x, gamma, rmsNormEps, {
            batchSize: seqLen,
            hiddenSize,
            rmsNormWeightOffset: modelConfig.rmsNormWeightOffset === true,
          }),
          [attentionOutput, layer.postAttentionNorm],
          {
            numTokens: seqLen,
            hiddenSize,
            eps: rmsNormEps,
            rmsNormWeightOffset: modelConfig.rmsNormWeightOffset === true,
            ...stopBaseWeight,
          }
        )
        : attentionOutput;
      const postAttention = await tape.record(
        OpType.RESIDUAL_ADD,
        (a, b) => runResidualAdd(a, b, seqLen * hiddenSize),
        [normalizedAttentionOutput, hidden],
        { size: seqLen * hiddenSize }
      );
      await captureStage('attn.post_attn', postAttention, layerIdx);

      const ffnInput = modelConfig.preFeedforwardNorm === true
        ? await tape.record(
          OpType.RMSNORM,
          (x, gamma) => runRMSNorm(x, gamma, rmsNormEps, {
            batchSize: seqLen,
            hiddenSize,
            rmsNormWeightOffset: modelConfig.rmsNormWeightOffset === true,
          }),
          [postAttention, layer.preFeedforwardNorm],
          {
            numTokens: seqLen,
            hiddenSize,
            eps: rmsNormEps,
            rmsNormWeightOffset: modelConfig.rmsNormWeightOffset === true,
            ...stopBaseWeight,
          }
        )
        : postAttention;
      await captureStage('ffn.in', ffnInput, layerIdx);
      let activated = null;
      if (layer.gateUp) {
        let gateUp = await tape.record(
          OpType.MATMUL,
          (x, w) => runMatmul(x, w, seqLen, layerIntermediateSize * 2, hiddenSize, {
            transposeB: 'auto',
            outputDtype: 'f32',
          }),
          [ffnInput, layer.gateUp],
          { M: seqLen, N: layerIntermediateSize * 2, K: hiddenSize, transposeB: 'auto', ...stopBaseWeight }
        );
        if (layer.lora.gate_up_proj) {
          const delta = await layer.lora.gate_up_proj.forward(ffnInput, tape);
          gateUp = await tape.record(
            OpType.RESIDUAL_ADD,
            (a, b) => runResidualAdd(a, b, seqLen * layerIntermediateSize * 2),
            [gateUp, delta],
            { size: seqLen * layerIntermediateSize * 2 }
          );
        }
        activated = await tape.record(
          OpType.SILU_ROWSPLIT,
          (x) => runSiLURowSplit(x, {
            numTokens: seqLen,
            dim: layerIntermediateSize,
            activation: hiddenActivation === 'gelu' ? 'gelu' : 'silu',
            swigluLimit: hiddenActivation === 'gelu' ? null : swigluLimit,
          }),
          [gateUp],
          {
            numTokens: seqLen,
            dim: layerIntermediateSize,
            activation: hiddenActivation === 'gelu' ? 'gelu' : 'silu',
            swigluLimit: hiddenActivation === 'gelu' ? 0 : swigluLimit,
          }
        );
      } else {
        if (hiddenActivation === 'gelu') {
          throw new Error('Split gate/up training currently requires SiLU activation.');
        }
        let gate = await tape.record(
          OpType.MATMUL,
          (x, w) => runMatmul(x, w, seqLen, layerIntermediateSize, hiddenSize, {
            transposeB: 'auto',
            outputDtype: 'f32',
          }),
          [ffnInput, layer.gate],
          { M: seqLen, N: layerIntermediateSize, K: hiddenSize, transposeB: 'auto', ...stopBaseWeight }
        );
        let up = await tape.record(
          OpType.MATMUL,
          (x, w) => runMatmul(x, w, seqLen, layerIntermediateSize, hiddenSize, {
            transposeB: 'auto',
            outputDtype: 'f32',
          }),
          [ffnInput, layer.up],
          { M: seqLen, N: layerIntermediateSize, K: hiddenSize, transposeB: 'auto', ...stopBaseWeight }
        );
        if (layer.lora.gate_proj) {
          const delta = await layer.lora.gate_proj.forward(ffnInput, tape);
          gate = await tape.record(
            OpType.RESIDUAL_ADD,
            (a, b) => runResidualAdd(a, b, seqLen * layerIntermediateSize),
            [gate, delta],
            { size: seqLen * layerIntermediateSize }
          );
        }
        if (layer.lora.up_proj) {
          const delta = await layer.lora.up_proj.forward(ffnInput, tape);
          up = await tape.record(
            OpType.RESIDUAL_ADD,
            (a, b) => runResidualAdd(a, b, seqLen * layerIntermediateSize),
            [up, delta],
            { size: seqLen * layerIntermediateSize }
          );
        }
        activated = await tape.record(
          OpType.SILU_GATED,
          (gateInput, upInput) => runSiLU(upInput, {
            size: seqLen * layerIntermediateSize,
            gate: gateInput,
            inputActivation: 'identity',
            swigluLimit,
          }),
          [gate, up],
          { count: seqLen * layerIntermediateSize, swigluLimit }
        );
      }
      await captureStage('ffn.act', activated, layerIdx);
      let ffnOutput = await tape.record(
        OpType.MATMUL,
        (x, w) => runMatmul(x, w, seqLen, hiddenSize, layerIntermediateSize, {
          transposeB: 'auto',
          outputDtype: 'f32',
        }),
        [activated, layer.down],
        { M: seqLen, N: hiddenSize, K: layerIntermediateSize, transposeB: 'auto', ...stopBaseWeight }
      );
      if (layer.lora.down_proj) {
        const delta = await layer.lora.down_proj.forward(activated, tape);
        ffnOutput = await tape.record(
          OpType.RESIDUAL_ADD,
          (a, b) => runResidualAdd(a, b, seqLen * hiddenSize),
          [ffnOutput, delta],
          { size: seqLen * hiddenSize }
        );
      }
      await captureStage('ffn.out', ffnOutput, layerIdx);
      const normalizedFfnOutput = modelConfig.postFeedforwardNorm === true
        ? await tape.record(
          OpType.RMSNORM,
          (x, gamma) => runRMSNorm(x, gamma, rmsNormEps, {
            batchSize: seqLen,
            hiddenSize,
            rmsNormWeightOffset: modelConfig.rmsNormWeightOffset === true,
          }),
          [ffnOutput, layer.postFeedforwardNorm],
          {
            numTokens: seqLen,
            hiddenSize,
            eps: rmsNormEps,
            rmsNormWeightOffset: modelConfig.rmsNormWeightOffset === true,
            ...stopBaseWeight,
          }
        )
        : ffnOutput;
      hidden = await tape.record(
        OpType.RESIDUAL_ADD,
        (a, b) => runResidualAdd(a, b, seqLen * hiddenSize),
        [normalizedFfnOutput, postAttention],
        { size: seqLen * hiddenSize }
      );
      await captureStage('layer.out', hidden, layerIdx);
    }

    await captureStage('final_norm.pre', hidden);
    const finalHidden = await tape.record(
      OpType.RMSNORM,
      (x, gamma) => runRMSNorm(x, gamma, rmsNormEps, {
        batchSize: seqLen,
        hiddenSize,
        rmsNormWeightOffset: modelConfig.rmsNormWeightOffset === true,
      }),
      [hidden, finalNormWeight],
      {
        numTokens: seqLen,
        hiddenSize,
        eps: rmsNormEps,
        rmsNormWeightOffset: modelConfig.rmsNormWeightOffset === true,
        ...stopBaseWeight,
      }
    );
    await captureStage('final_norm.out', finalHidden);
    if (forwardOptions.logitsMode === 'all') {
      const logits = await tape.record(
        OpType.MATMUL,
        (x, w) => runMatmul(x, w, seqLen, vocabSize, hiddenSize, {
          transposeB: 'auto',
          outputDtype: 'f32',
        }),
        [finalHidden, lmHeadWeight],
        { M: seqLen, N: vocabSize, K: hiddenSize, transposeB: 'auto', ...stopBaseWeight }
      );
      await captureStage('logits.out', logits);
      return logits;
    }
    const lastHidden = await tape.record(
      OpType.ROW_SLICE,
      (x) => createRowSliceTensor(x, seqLen, hiddenSize, seqLen - 1, 'distill_last_hidden'),
      [finalHidden],
      { rows: seqLen, cols: hiddenSize, rowIndex: seqLen - 1 }
    );
    const logits = await tape.record(
      OpType.MATMUL,
      (x, w) => runMatmul(x, w, 1, vocabSize, hiddenSize, {
        transposeB: 'auto',
        outputDtype: 'f32',
      }),
      [lastHidden, lmHeadWeight],
      { M: 1, N: vocabSize, K: hiddenSize, transposeB: 'auto', ...stopBaseWeight }
    );
    await captureStage('logits.out', logits);
    return logits;
  }

  async function runTransformerPrompt(prompt, tape, forwardOptions = {}) {
    const { tokenTensor, seqLen } = await buildPromptTokens(prompt);
    return runTransformerTokenTensor(tokenTensor, seqLen, tape, forwardOptions);
  }

  function collectLoraTensorEntries() {
    const entries = [];
    for (let layerIdx = 0; layerIdx < layers.length; layerIdx += 1) {
      const layer = layers[layerIdx];
      for (const moduleName of Object.keys(layer.lora).sort((left, right) => left.localeCompare(right))) {
        const adapter = layer.lora[moduleName];
        entries.push(
          { name: `layers.${layerIdx}.${moduleName}.lora_a`, tensor: adapter.A },
          { name: `layers.${layerIdx}.${moduleName}.lora_b`, tensor: adapter.B }
        );
      }
    }
    return entries;
  }

  const model = {
    async forward(inputTensor, tape) {
      if (Array.isArray(inputTensor?.shape) && inputTensor.shape.length === 1) {
        return runTransformerTokenTensor(inputTensor, inputTensor.shape[0], tape, { logitsMode: 'all' });
      }
      return tape.record(
        OpType.MATMUL,
        (x, w) => runMatmul(x, w, 1, vocabSize, hiddenSize, {
          transposeB: 'auto',
          outputDtype: 'f32',
        }),
        [inputTensor, lmHeadWeight],
        { M: 1, N: vocabSize, K: hiddenSize, transposeB: 'auto', ...stopBaseWeight }
      );
    },
    async forwardCausalLm(batch, tape) {
      const inputTensor = batch?.input || null;
      const seqLen = Array.isArray(inputTensor?.shape) && inputTensor.shape.length === 1
        ? inputTensor.shape[0]
        : 0;
      if (!Number.isInteger(seqLen) || seqLen < 1) {
        throw new Error('Transformer LoRA causal-LM batch requires input token tensor shape [seqLen].');
      }
      const logits = await runTransformerTokenTensor(inputTensor, seqLen, tape, { logitsMode: 'all' });
      return { logits };
    },
    async forwardDistill(batch, tape, forwardOptions = {}) {
      const requestedPhase = String(forwardOptions?.phase || 'anchor').trim();
      const phase = requestedPhase === 'positive'
        ? 'positive'
        : (requestedPhase === 'negative' ? 'negative' : 'anchor');
      const prompts = resolvePhasePrompts(batch, phase);
      if (prompts.length !== 1) {
        throw new Error(
          `Distill full-graph student currently requires batchSize=1, got ${prompts.length}.`
        );
      }
      const logits = await runTransformerPrompt(prompts[0], tape, forwardOptions);
      return { logits };
    },
    cleanupDistillStep() {
      for (const tensor of temporaryInputs) {
        releaseTensor(tensor);
      }
      temporaryInputs.clear();
    },
    loraParams() {
      return loraParams.length > 0 ? loraParams : decoderParams;
    },
    loraTensorEntries() {
      return collectLoraTensorEntries();
    },
    paramGroups() {
      if (loraParams.length > 0) {
        return {
          encoder: [],
          prior: [],
          decoder: [],
          base: baseParams,
          lora: loraParams,
        };
      }
      return {
        encoder: encoderParams,
        prior: [],
        decoder: decoderParams,
        base: baseParams,
        lora: loraParams,
      };
    },
  };

  return {
    config,
    model,
    outputDim: vocabSize,
    embeddingDim: hiddenSize,
    cleanup() {
      model.cleanupDistillStep();
      disposeTransformerLoraAdapters(layers);
      for (const tensor of ownedTrainables) {
        releaseTensor(tensor);
      }
      ownedTrainables.clear();
    },
  };
}

export async function createDistillStudentRuntimeModelFixture(overrides = {}, options = {}) {
  const distillRuntime = options.distillRuntime && typeof options.distillRuntime === 'object'
    ? options.distillRuntime
    : null;
  const graphMode = normalizeDistillStudentGraphMode(
    options.studentGraphMode
    ?? distillRuntime?.studentGraphMode
    ?? overrides?.training?.distill?.studentGraphMode
  );
  if (graphMode === DISTILL_STUDENT_GRAPH_PROJECTION) {
    return createDistillStudentProjectionModelFixture(overrides, options);
  }
  return createDistillStudentTransformerModelFixture(overrides, options);
}
