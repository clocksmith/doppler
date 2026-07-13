import { setRegistryUrl } from '../../../src/config/kernels/registry.js';
import { setPlatformsBaseUrl } from '../../../src/config/platforms/loader.js';
import {
  qwenLinearDecoderLayerBackward as referenceBackward,
  qwenLinearDecoderLayerForward as referenceForward,
} from '../../../src/experimental/training/qwen-linear-decoder-reference.js';
import {
  releaseQwenLinearDecoderLayerCache,
  runQwenLinearDecoderLayerBackward,
  runQwenLinearDecoderLayerForward,
} from '../../../src/experimental/training/qwen-linear-decoder-training-module.js';
import { getKernelCapabilities, initDevice } from '../../../src/gpu/device.js';
import { createTensor } from '../../../src/gpu/tensor.js';
import { f16ToF32Array, f32ToF16Array } from '../../../src/inference/kv-cache/types.js';
import { acquireBuffer, readBuffer, releaseBuffer, uploadData } from '../../../src/memory/buffer-pool.js';

function values(length, offset, scale) {
  return Float32Array.from(
    { length },
    (_, index) => Math.sin((index + offset) * 0.43) * scale
  );
}

function makeTypedTensor(data, dtype, shape, label) {
  const byteLength = Math.ceil(data.byteLength / 4) * 4;
  const upload = byteLength === data.byteLength
    ? data
    : (() => {
        const padded = new Uint8Array(byteLength);
        padded.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
        return padded;
      })();
  const buffer = acquireBuffer(byteLength, undefined, label);
  uploadData(buffer, upload);
  return createTensor(buffer, dtype, shape, label);
}

function makeTensor(data, shape, label) {
  return makeTypedTensor(data, 'f32', shape, label);
}

async function readF32(tensor) {
  const count = tensor.shape.reduce((product, value) => product * value, 1);
  return new Float32Array(await readBuffer(tensor.buffer, count * Float32Array.BYTES_PER_ELEMENT));
}

function compare(actual, expected) {
  let maxAbsError = 0;
  let squaredError = 0;
  let allFinite = true;
  for (let index = 0; index < expected.length; index += 1) {
    const error = Math.abs(actual[index] - expected[index]);
    maxAbsError = Math.max(maxAbsError, error);
    squaredError += error * error;
    allFinite = allFinite && Number.isFinite(actual[index]);
  }
  return {
    elementCount: expected.length,
    allFinite,
    maxAbsError,
    rmse: Math.sqrt(squaredError / expected.length),
  };
}

function maxAbs(valuesToCheck) {
  let maximum = 0;
  for (const value of valuesToCheck) maximum = Math.max(maximum, Math.abs(value));
  return maximum;
}

function makeAdapter(valuesForAdapter, inputSize, outputSize, label) {
  return {
    A: makeTensor(valuesForAdapter.A, [inputSize, valuesForAdapter.rank], `${label}_a`),
    B: makeTensor(valuesForAdapter.B, [valuesForAdapter.rank, outputSize], `${label}_b`),
    rank: valuesForAdapter.rank,
    alpha: valuesForAdapter.alpha,
  };
}

function releaseAdapter(adapter) {
  releaseBuffer(adapter.A.buffer);
  releaseBuffer(adapter.B.buffer);
}

async function executeCase(downAdapterPerturbation = 0) {
  const options = {
    numTokens: 3,
    hiddenSize: 4,
    intermediateSize: 6,
    numKeyHeads: 1,
    numValueHeads: 2,
    keyDim: 2,
    valueDim: 2,
    kernelSize: 2,
    checkpointInterval: 2,
    queryScale: 1 / Math.sqrt(2),
    l2Eps: 1e-6,
    rmsEps: 1e-6,
  };
  const convSize = (options.numKeyHeads * options.keyDim * 2)
    + (options.numValueHeads * options.valueDim);
  const valueSize = options.numValueHeads * options.valueDim;
  const rank = 2;
  const alpha = 4;
  const adapterValues = (inputSize, outputSize, offset) => ({
    A: values(inputSize * rank, offset, 0.07),
    B: values(rank * outputSize, offset + 17, 0.05),
    rank,
    alpha,
  });
  const weightBits = {
    inputNormWeight: f32ToF16Array(values(options.hiddenSize, 3, 0.04)),
    postAttentionNormWeight: f32ToF16Array(values(options.hiddenSize, 11, 0.04)),
    qkvWeight: f32ToF16Array(values(convSize * options.hiddenSize, 23, 0.11)),
    zWeight: f32ToF16Array(values(valueSize * options.hiddenSize, 61, 0.1)),
    aWeight: f32ToF16Array(values(options.numValueHeads * options.hiddenSize, 83, 0.09)),
    bWeight: f32ToF16Array(values(options.numValueHeads * options.hiddenSize, 97, 0.09)),
    outWeight: f32ToF16Array(values(options.hiddenSize * valueSize, 109, 0.11)),
    gateWeight: f32ToF16Array(
      values(options.intermediateSize * options.hiddenSize, 131, 0.12)
    ),
    upWeight: f32ToF16Array(
      values(options.intermediateSize * options.hiddenSize, 157, 0.11)
    ),
    downWeight: f32ToF16Array(
      values(options.hiddenSize * options.intermediateSize, 181, 0.1)
    ),
  };
  const loraValues = {
    gate: adapterValues(options.hiddenSize, options.intermediateSize, 211),
    up: adapterValues(options.hiddenSize, options.intermediateSize, 239),
    down: adapterValues(options.intermediateSize, options.hiddenSize, 269),
  };
  loraValues.down.B[1] += downAdapterPerturbation;
  const inputValues = {
    hidden: values(options.numTokens * options.hiddenSize, 293, 0.18),
    inputNormWeight: f16ToF32Array(weightBits.inputNormWeight),
    postAttentionNormWeight: f16ToF32Array(weightBits.postAttentionNormWeight),
    attention: {
      qkvWeight: f16ToF32Array(weightBits.qkvWeight),
      zWeight: f16ToF32Array(weightBits.zWeight),
      aWeight: f16ToF32Array(weightBits.aWeight),
      bWeight: f16ToF32Array(weightBits.bWeight),
      outWeight: f16ToF32Array(weightBits.outWeight),
      convWeight: values(convSize * options.kernelSize, 317, 0.09),
      aLog: values(options.numValueHeads, 347, 0.08),
      dtBias: values(options.numValueHeads, 353, 0.07),
      normWeight: values(options.valueDim, 359, 0.05),
      initialState: values(
        options.numValueHeads * options.keyDim * options.valueDim,
        367,
        0.06
      ),
    },
    mlp: {
      gateWeight: f16ToF32Array(weightBits.gateWeight),
      upWeight: f16ToF32Array(weightBits.upWeight),
      downWeight: f16ToF32Array(weightBits.downWeight),
      lora: loraValues,
    },
  };
  const mlpLora = {
    gate: makeAdapter(
      loraValues.gate,
      options.hiddenSize,
      options.intermediateSize,
      'linear_decoder_gate_lora'
    ),
    up: makeAdapter(
      loraValues.up,
      options.hiddenSize,
      options.intermediateSize,
      'linear_decoder_up_lora'
    ),
    down: makeAdapter(
      loraValues.down,
      options.intermediateSize,
      options.hiddenSize,
      'linear_decoder_down_lora'
    ),
  };
  const tensors = {
    hidden: makeTensor(
      inputValues.hidden,
      [options.numTokens, options.hiddenSize],
      'linear_decoder_hidden'
    ),
    inputNormWeight: makeTypedTensor(
      weightBits.inputNormWeight,
      'f16',
      [options.hiddenSize],
      'linear_decoder_input_norm'
    ),
    postAttentionNormWeight: makeTypedTensor(
      weightBits.postAttentionNormWeight,
      'f16',
      [options.hiddenSize],
      'linear_decoder_post_attention_norm'
    ),
    attention: {
      qkvWeight: makeTypedTensor(
        weightBits.qkvWeight,
        'f16',
        [convSize, options.hiddenSize],
        'linear_decoder_qkv_weight'
      ),
      zWeight: makeTypedTensor(
        weightBits.zWeight,
        'f16',
        [valueSize, options.hiddenSize],
        'linear_decoder_z_weight'
      ),
      aWeight: makeTypedTensor(
        weightBits.aWeight,
        'f16',
        [options.numValueHeads, options.hiddenSize],
        'linear_decoder_a_weight'
      ),
      bWeight: makeTypedTensor(
        weightBits.bWeight,
        'f16',
        [options.numValueHeads, options.hiddenSize],
        'linear_decoder_b_weight'
      ),
      outWeight: makeTypedTensor(
        weightBits.outWeight,
        'f16',
        [options.hiddenSize, valueSize],
        'linear_decoder_out_weight'
      ),
      convWeight: makeTensor(
        inputValues.attention.convWeight,
        [convSize, options.kernelSize],
        'linear_decoder_conv_weight'
      ),
      aLog: makeTensor(inputValues.attention.aLog, [options.numValueHeads], 'linear_decoder_a_log'),
      dtBias: makeTensor(
        inputValues.attention.dtBias,
        [options.numValueHeads],
        'linear_decoder_dt_bias'
      ),
      normWeight: makeTensor(
        inputValues.attention.normWeight,
        [options.valueDim],
        'linear_decoder_norm_weight'
      ),
      initialState: makeTensor(
        inputValues.attention.initialState,
        [options.numValueHeads, options.keyDim, options.valueDim],
        'linear_decoder_initial_state'
      ),
    },
    mlp: {
      gateWeight: makeTypedTensor(
        weightBits.gateWeight,
        'f16',
        [options.intermediateSize, options.hiddenSize],
        'linear_decoder_gate_weight'
      ),
      upWeight: makeTypedTensor(
        weightBits.upWeight,
        'f16',
        [options.intermediateSize, options.hiddenSize],
        'linear_decoder_up_weight'
      ),
      downWeight: makeTypedTensor(
        weightBits.downWeight,
        'f16',
        [options.hiddenSize, options.intermediateSize],
        'linear_decoder_down_weight'
      ),
      lora: mlpLora,
    },
  };
  const gradOutputValues = values(options.numTokens * options.hiddenSize, 389, 0.22);
  const gradOutput = makeTensor(
    gradOutputValues,
    [options.numTokens, options.hiddenSize],
    'linear_decoder_grad_output'
  );
  const expectedForward = referenceForward(inputValues, options);
  const expectedBackward = referenceBackward(
    inputValues,
    gradOutputValues,
    expectedForward.cache,
    options
  );
  let forward = null;
  let backward = null;
  try {
    forward = await runQwenLinearDecoderLayerForward(tensors, options);
    backward = await runQwenLinearDecoderLayerBackward(
      tensors,
      gradOutput,
      forward.cache,
      options
    );
    const actualOutput = await readF32(forward.output);
    const comparisons = {
      decoderForward: compare(actualOutput, expectedForward.output),
      decoderFinalState: compare(await readF32(forward.finalState), expectedForward.finalState),
      decoderBackwardHidden: compare(await readF32(backward.hidden), expectedBackward.hidden),
      decoderBackwardInitialState: compare(
        await readF32(backward.initialState),
        expectedBackward.initialState
      ),
    };
    for (const projection of ['gate', 'up', 'down']) {
      for (const matrix of ['A', 'B']) {
        const actualGradient = await readF32(backward.lora[projection][matrix]);
        comparisons[`decoderBackwardLora${projection.toUpperCase()}${matrix}`] = {
          ...compare(actualGradient, expectedBackward.lora[projection][matrix]),
          maxAbsValue: maxAbs(actualGradient),
        };
      }
    }
    return { actualOutput, comparisons };
  } finally {
    if (backward) {
      releaseBuffer(backward.hidden.buffer);
      releaseBuffer(backward.initialState.buffer);
      for (const gradients of Object.values(backward.lora)) {
        releaseBuffer(gradients.A.buffer);
        releaseBuffer(gradients.B.buffer);
      }
    }
    if (forward) {
      releaseBuffer(forward.output.buffer);
      releaseBuffer(forward.finalState.buffer);
      releaseQwenLinearDecoderLayerCache(forward.cache);
    }
    releaseBuffer(tensors.hidden.buffer);
    releaseBuffer(tensors.inputNormWeight.buffer);
    releaseBuffer(tensors.postAttentionNormWeight.buffer);
    for (const tensor of Object.values(tensors.attention)) releaseBuffer(tensor.buffer);
    for (const [name, tensor] of Object.entries(tensors.mlp)) {
      if (name !== 'lora') releaseBuffer(tensor.buffer);
    }
    for (const adapter of Object.values(mlpLora)) releaseAdapter(adapter);
    releaseBuffer(gradOutput.buffer);
  }
}

export async function runQwenLinearDecoderBackwardOracle() {
  const baseUrl = new URL('../../../src/config/', import.meta.url);
  setPlatformsBaseUrl(new URL('platforms/', baseUrl).toString());
  setRegistryUrl(new URL('kernels/registry.json', baseUrl).toString());
  await initDevice();

  const tolerance = 3e-5;
  const baseline = await executeCase();
  const perturbed = await executeCase(0.125);
  const perturbation = compare(perturbed.actualOutput, baseline.actualOutput);
  const passed = Object.values(baseline.comparisons).every(
    (entry) => entry.allFinite
      && entry.maxAbsError <= tolerance
      && (entry.maxAbsValue == null || entry.maxAbsValue > 1e-10)
  ) && perturbation.maxAbsError > 1e-5;
  const capabilities = getKernelCapabilities();
  return {
    artifactType: 'qwen_linear_decoder_layer_backward_oracle',
    schemaVersion: 1,
    passed,
    tolerance: { maxAbsError: tolerance },
    comparisons: baseline.comparisons,
    negativeControl: {
      perturbation: 'down_proj_lora_b_index_1_plus_0.125',
      decoderOutputDifference: perturbation,
      passed: perturbation.maxAbsError > 1e-5,
    },
    adapterInfo: capabilities.adapterInfo || null,
    claimBoundary: 'Tiny Qwen linear-attention decoder layer with checkpointed recurrence, input/post-attention offset RMSNorm, residuals, frozen F16 attention and MLP projections, and all three per-layer V12 MLP LoRA families; production geometry, loss, optimizer update, hybrid-layer composition, memory, and performance remain absent.',
  };
}
