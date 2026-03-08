import { getDevice } from '../../gpu/device.js';
import { createTrainingConfig } from '../../config/training-defaults.js';
import {
  runAttention,
  castF16ToF32,
  runGather,
  runMatmul,
  runResidualAdd,
  runRMSNorm,
  runRoPE,
  runSiLURowSplit,
} from '../../gpu/kernels/index.js';
import { createTensor } from '../../gpu/tensor.js';
import { acquireBuffer, uploadData, releaseBuffer } from '../../memory/buffer-pool.js';
import { getBufferDtype, getWeightDtype, isCpuWeightBuffer, isWeightBuffer } from '../../gpu/weight-buffer.js';
import { OpType } from '../autograd.js';
import { normalizeOptionalString } from './suite-data.js';

const DISTILL_ADAPTER_TOP_K = 64;
const DISTILL_STUDENT_GRAPH_PROJECTION = 'projection_head';
const DISTILL_STUDENT_GRAPH_FULL = 'transformer_full';

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

function resolveTensorDtype(value, fallback = 'f32') {
  const dtype = isWeightBuffer(value)
    ? value.dtype
    : (value?.dtype || getWeightDtype(value) || null);
  const normalized = String(dtype || '').toLowerCase();
  return normalized === 'f16' ? 'f16' : (normalized === 'f32' ? 'f32' : fallback);
}

async function ensureTrainableTensor(value, shape, label, ownedTrainables = null) {
  if (!value) {
    throw new Error(`Distill full-graph student missing required weight "${label}".`);
  }
  const registerOwned = (tensor) => {
    if (ownedTrainables instanceof Set && tensor?.buffer instanceof GPUBuffer) {
      ownedTrainables.add(tensor);
    }
    return tensor;
  };
  if (isWeightBuffer(value)) {
    if (value.dtype === 'f32') {
      return value;
    }
    if (value.dtype === 'f16') {
      const sourceShape = Array.isArray(value.shape) && value.shape.length > 0 ? value.shape : [...shape];
      const source = createTensor(value.buffer, 'f16', sourceShape, `${label}_source_f16`);
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

async function fuseGateUpTensors(gateTensor, upTensor, intermediateSize, hiddenSize, label, ownedTrainables = null) {
  const device = getDevice();
  if (!device) {
    throw new Error('Distill full-graph student requires active GPU device.');
  }
  if (gateTensor?.dtype !== 'f32' || upTensor?.dtype !== 'f32') {
    throw new Error(`Distill fused gate_up expects f32 tensors for "${label}".`);
  }
  const expectedRows = intermediateSize;
  const expectedCols = hiddenSize;
  const gateRows = Number.isFinite(gateTensor?.shape?.[0]) ? gateTensor.shape[0] : 0;
  const gateCols = Number.isFinite(gateTensor?.shape?.[1]) ? gateTensor.shape[1] : 0;
  const upRows = Number.isFinite(upTensor?.shape?.[0]) ? upTensor.shape[0] : 0;
  const upCols = Number.isFinite(upTensor?.shape?.[1]) ? upTensor.shape[1] : 0;
  if (gateRows !== expectedRows || gateCols !== expectedCols || upRows !== expectedRows || upCols !== expectedCols) {
    throw new Error(
      `Distill gate/up shape mismatch for "${label}": gate=[${gateRows},${gateCols}] up=[${upRows},${upCols}] ` +
      `expected=[${expectedRows},${expectedCols}]`
    );
  }
  const rowBytes = expectedCols * 4;
  const blockBytes = expectedRows * rowBytes;
  const fusedBuffer = acquireBuffer(blockBytes * 2, undefined, `${label}_fused`);
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(gateTensor.buffer, 0, fusedBuffer, 0, blockBytes);
  encoder.copyBufferToBuffer(upTensor.buffer, 0, fusedBuffer, blockBytes, blockBytes);
  device.queue.submit([encoder.finish()]);
  const fused = createTensor(fusedBuffer, 'f32', [expectedRows * 2, expectedCols], `${label}_fused`);
  if (ownedTrainables instanceof Set) {
    ownedTrainables.add(fused);
  }
  return fused;
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
      (a, b) => runMatmul(a, b, rows, outputDim, embeddingDim, { transposeB: false }),
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
    ownedTrainables
  );
  const lmHeadWeight = tieWordEmbeddings
    ? embeddingWeight
    : await ensureTrainableTensor(
      studentPipeline.weights.get('lm_head'),
      [vocabSize, hiddenSize],
      'lm_head',
      ownedTrainables
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

  const layerParams = [];
  const layers = [];
  for (let layerIdx = 0; layerIdx < numLayers; layerIdx += 1) {
    const layerWeights = studentPipeline.weights.get(`layer_${layerIdx}`);
    if (!layerWeights) {
      throw new Error(`Distill full-graph student missing layer_${layerIdx} weights.`);
    }
    const gateUpWeight = layerWeights.gateUp || layerWeights.ffnGateUp || null;
    let layerGateUp = null;
    if (hasTensorPayload(gateUpWeight)) {
      layerGateUp = await ensureTrainableTensor(
        gateUpWeight,
        [intermediateSize * 2, hiddenSize],
        `layer_${layerIdx}.ffn_gate_up`,
        ownedTrainables
      );
    } else {
      const gateWeight = layerWeights.gate || layerWeights.ffnGate || null;
      const upWeight = layerWeights.up || layerWeights.ffnUp || null;
      if (!hasTensorPayload(gateWeight) || !hasTensorPayload(upWeight)) {
        throw new Error(
          `Distill full-graph student missing gate/up projections on layer ${layerIdx}.`
        );
      }
      const gateTensor = await ensureTrainableTensor(
        gateWeight,
        [intermediateSize, hiddenSize],
        `layer_${layerIdx}.ffn_gate`,
        ownedTrainables
      );
      const upTensor = await ensureTrainableTensor(
        upWeight,
        [intermediateSize, hiddenSize],
        `layer_${layerIdx}.ffn_up`,
        ownedTrainables
      );
      layerGateUp = await fuseGateUpTensors(
        gateTensor,
        upTensor,
        intermediateSize,
        hiddenSize,
        `layer_${layerIdx}.ffn_gate_up`,
        ownedTrainables
      );
    }
    const layer = {
      inputNorm: await ensureNormTensor(
        layerWeights.inputNorm,
        hiddenSize,
        `layer_${layerIdx}.input_norm`,
        ownedTrainables
      ),
      qProj: await ensureTrainableTensor(
        layerWeights.qProj,
        [numHeads * headDim, hiddenSize],
        `layer_${layerIdx}.q_proj`,
        ownedTrainables
      ),
      kProj: await ensureTrainableTensor(
        layerWeights.kProj,
        [numKVHeads * headDim, hiddenSize],
        `layer_${layerIdx}.k_proj`,
        ownedTrainables
      ),
      vProj: await ensureTrainableTensor(
        layerWeights.vProj,
        [numKVHeads * headDim, hiddenSize],
        `layer_${layerIdx}.v_proj`,
        ownedTrainables
      ),
      oProj: await ensureTrainableTensor(
        layerWeights.oProj,
        [hiddenSize, hiddenSize],
        `layer_${layerIdx}.o_proj`,
        ownedTrainables
      ),
      postAttentionNorm: layerWeights.postAttentionNorm
        ? await ensureNormTensor(
          layerWeights.postAttentionNorm,
          hiddenSize,
          `layer_${layerIdx}.post_attention_norm`,
          ownedTrainables
        )
        : null,
      gateUp: layerGateUp,
      down: await ensureTrainableTensor(
        layerWeights.down || layerWeights.ffnDown,
        [hiddenSize, intermediateSize],
        `layer_${layerIdx}.ffn_down`,
        ownedTrainables
      ),
    };
    layers.push(layer);
    layerParams.push(layer.inputNorm, layer.qProj, layer.kProj, layer.vProj, layer.oProj, layer.gateUp, layer.down);
    if (layer.postAttentionNorm) {
      layerParams.push(layer.postAttentionNorm);
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

  async function runTransformerPrompt(prompt, tape) {
    const { tokenTensor, seqLen } = await buildPromptTokens(prompt);
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
      }
    );

    for (let layerIdx = 0; layerIdx < layers.length; layerIdx += 1) {
      const layer = layers[layerIdx];
      const normed = await tape.record(
        OpType.RMSNORM,
        (x, gamma) => runRMSNorm(x, gamma, rmsNormEps, {
          batchSize: seqLen,
          hiddenSize,
          rmsNormWeightOffset: modelConfig.rmsNormWeightOffset === true,
        }),
        [hidden, layer.inputNorm],
        { numTokens: seqLen, hiddenSize, eps: rmsNormEps }
      );

      const q2d = await tape.record(
        OpType.MATMUL,
        (x, w) => runMatmul(x, w, seqLen, numHeads * headDim, hiddenSize, {
          transposeB: 'auto',
          outputDtype: 'f32',
        }),
        [normed, layer.qProj],
        { M: seqLen, N: numHeads * headDim, K: hiddenSize, transposeB: 'auto' }
      );
      const k2d = await tape.record(
        OpType.MATMUL,
        (x, w) => runMatmul(x, w, seqLen, numKVHeads * headDim, hiddenSize, {
          transposeB: 'auto',
          outputDtype: 'f32',
        }),
        [normed, layer.kProj],
        { M: seqLen, N: numKVHeads * headDim, K: hiddenSize, transposeB: 'auto' }
      );
      const v2d = await tape.record(
        OpType.MATMUL,
        (x, w) => runMatmul(x, w, seqLen, numKVHeads * headDim, hiddenSize, {
          transposeB: 'auto',
          outputDtype: 'f32',
        }),
        [normed, layer.vProj],
        { M: seqLen, N: numKVHeads * headDim, K: hiddenSize, transposeB: 'auto' }
      );

      const q3d = createTensor(q2d.buffer, q2d.dtype, [seqLen, numHeads, headDim], `layer_${layerIdx}_q`);
      const k3d = createTensor(k2d.buffer, k2d.dtype, [seqLen, numKVHeads, headDim], `layer_${layerIdx}_k`);
      const v3d = createTensor(v2d.buffer, v2d.dtype, [seqLen, numKVHeads, headDim], `layer_${layerIdx}_v`);

      const qRope = await tape.record(
        OpType.ROPE,
        (q, cos, sin) => runRoPE(q, cos, sin, seqLen, { numHeads, headDim, startPos: 0 }),
        [q3d, ropeCos, ropeSin],
        { seqLen, numHeads, headDim, startPos: 0 }
      );
      const kRope = await tape.record(
        OpType.ROPE,
        (k, cos, sin) => runRoPE(k, cos, sin, seqLen, { numHeads: numKVHeads, headDim, startPos: 0 }),
        [k3d, ropeCos, ropeSin],
        { seqLen, numHeads: numKVHeads, headDim, startPos: 0 }
      );

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
        { seqLen, numHeads, headDim, scale: 1 / Math.sqrt(headDim), causal: true, recomputeForward: true }
      );
      const attention2d = createTensor(
        attention.buffer,
        attention.dtype,
        [seqLen, hiddenSize],
        `layer_${layerIdx}_attn_2d`
      );

      const attentionOutput = await tape.record(
        OpType.MATMUL,
        (x, w) => runMatmul(x, w, seqLen, hiddenSize, hiddenSize, {
          transposeB: 'auto',
          outputDtype: 'f32',
        }),
        [attention2d, layer.oProj],
        { M: seqLen, N: hiddenSize, K: hiddenSize, transposeB: 'auto' }
      );
      const postAttention = await tape.record(
        OpType.RESIDUAL_ADD,
        (a, b) => runResidualAdd(a, b, seqLen * hiddenSize),
        [attentionOutput, hidden],
        { size: seqLen * hiddenSize }
      );

      const ffnInput = layer.postAttentionNorm
        ? await tape.record(
          OpType.RMSNORM,
          (x, gamma) => runRMSNorm(x, gamma, rmsNormEps, {
            batchSize: seqLen,
            hiddenSize,
            rmsNormWeightOffset: modelConfig.rmsNormWeightOffset === true,
          }),
          [postAttention, layer.postAttentionNorm],
          { numTokens: seqLen, hiddenSize, eps: rmsNormEps }
        )
        : postAttention;
      const gateUp = await tape.record(
        OpType.MATMUL,
        (x, w) => runMatmul(x, w, seqLen, intermediateSize * 2, hiddenSize, {
          transposeB: 'auto',
          outputDtype: 'f32',
        }),
        [ffnInput, layer.gateUp],
        { M: seqLen, N: intermediateSize * 2, K: hiddenSize, transposeB: 'auto' }
      );
      const activated = await tape.record(
        OpType.SILU_ROWSPLIT,
        (x) => runSiLURowSplit(x, {
          numTokens: seqLen,
          dim: intermediateSize,
          activation: hiddenActivation === 'gelu' ? 'gelu' : 'silu',
          swigluLimit: hiddenActivation === 'gelu' ? null : swigluLimit,
        }),
        [gateUp],
        {
          numTokens: seqLen,
          dim: intermediateSize,
          activation: hiddenActivation === 'gelu' ? 'gelu' : 'silu',
          swigluLimit: hiddenActivation === 'gelu' ? 0 : swigluLimit,
        }
      );
      const ffnOutput = await tape.record(
        OpType.MATMUL,
        (x, w) => runMatmul(x, w, seqLen, hiddenSize, intermediateSize, {
          transposeB: 'auto',
          outputDtype: 'f32',
        }),
        [activated, layer.down],
        { M: seqLen, N: hiddenSize, K: intermediateSize, transposeB: 'auto' }
      );
      hidden = await tape.record(
        OpType.RESIDUAL_ADD,
        (a, b) => runResidualAdd(a, b, seqLen * hiddenSize),
        [ffnOutput, postAttention],
        { size: seqLen * hiddenSize }
      );
    }

    const finalHidden = await tape.record(
      OpType.RMSNORM,
      (x, gamma) => runRMSNorm(x, gamma, rmsNormEps, {
        batchSize: seqLen,
        hiddenSize,
        rmsNormWeightOffset: modelConfig.rmsNormWeightOffset === true,
      }),
      [hidden, finalNormWeight],
      { numTokens: seqLen, hiddenSize, eps: rmsNormEps }
    );
    const lastHidden = await tape.record(
      OpType.ROW_SLICE,
      (x) => createRowSliceTensor(x, seqLen, hiddenSize, seqLen - 1, 'distill_last_hidden'),
      [finalHidden],
      { rows: seqLen, cols: hiddenSize, rowIndex: seqLen - 1 }
    );
    return tape.record(
      OpType.MATMUL,
      (x, w) => runMatmul(x, w, 1, vocabSize, hiddenSize, {
        transposeB: 'auto',
        outputDtype: 'f32',
      }),
      [lastHidden, lmHeadWeight],
      { M: 1, N: vocabSize, K: hiddenSize, transposeB: 'auto' }
    );
  }

  const model = {
    async forward(inputTensor, tape) {
      return tape.record(
        OpType.MATMUL,
        (x, w) => runMatmul(x, w, 1, vocabSize, hiddenSize, {
          transposeB: 'auto',
          outputDtype: 'f32',
        }),
        [inputTensor, lmHeadWeight],
        { M: 1, N: vocabSize, K: hiddenSize, transposeB: 'auto' }
      );
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
      const logits = await runTransformerPrompt(prompts[0], tape);
      return { logits };
    },
    cleanupDistillStep() {
      for (const tensor of temporaryInputs) {
        releaseTensor(tensor);
      }
      temporaryInputs.clear();
    },
    loraParams() {
      return decoderParams;
    },
    paramGroups() {
      return {
        encoder: encoderParams,
        prior: [],
        decoder: decoderParams,
        base: baseParams,
        lora: [],
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
