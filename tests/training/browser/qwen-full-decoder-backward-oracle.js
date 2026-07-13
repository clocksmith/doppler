import { setRegistryUrl } from '../../../src/config/kernels/registry.js';
import { setPlatformsBaseUrl } from '../../../src/config/platforms/loader.js';
import {
  qwenFullDecoderLayerBackward as referenceBackward,
  qwenFullDecoderLayerForward as referenceForward,
} from '../../../src/experimental/training/qwen-full-decoder-reference.js';
import {
  releaseQwenFullDecoderLayerCache,
  runQwenFullDecoderLayerBackward,
  runQwenFullDecoderLayerForward,
} from '../../../src/experimental/training/qwen-full-decoder-training-module.js';
import { getKernelCapabilities, initDevice } from '../../../src/gpu/device.js';
import { createTensor } from '../../../src/gpu/tensor.js';
import { f16ToF32Array, f32ToF16Array } from '../../../src/inference/kv-cache/types.js';
import { acquireBuffer, readBuffer, releaseBuffer, uploadData } from '../../../src/memory/buffer-pool.js';

function values(length, offset, scale) {
  return Float32Array.from(
    { length },
    (_, index) => Math.sin((index + offset) * 0.41) * scale
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
    seqLen: 2,
    hiddenSize: 8,
    intermediateSize: 12,
    numHeads: 2,
    numKVHeads: 1,
    headDim: 256,
    rotaryDim: 64,
    pairSpanDim: 64,
    interleaved: false,
    startPos: 0,
    rmsEps: 1e-6,
  };
  const querySize = options.numHeads * options.headDim;
  const kvSize = options.numKVHeads * options.headDim;
  const rank = 2;
  const alpha = 4;
  const adapterValues = (inputSize, outputSize, offset) => ({
    A: values(inputSize * rank, offset, 0.07),
    B: values(rank * outputSize, offset + 19, 0.05),
    rank,
    alpha,
  });
  const weightBits = {
    inputNormWeight: f32ToF16Array(values(options.hiddenSize, 3, 0.04)),
    postAttentionNormWeight: f32ToF16Array(values(options.hiddenSize, 17, 0.04)),
    qWeight: f32ToF16Array(values(querySize * 2 * options.hiddenSize, 31, 0.07)),
    kWeight: f32ToF16Array(values(kvSize * options.hiddenSize, 8231, 0.07)),
    vWeight: f32ToF16Array(values(kvSize * options.hiddenSize, 10289, 0.07)),
    oWeight: f32ToF16Array(values(options.hiddenSize * querySize, 12347, 0.07)),
    qNormWeight: f32ToF16Array(values(options.headDim, 16451, 0.04)),
    kNormWeight: f32ToF16Array(values(options.headDim, 16711, 0.04)),
    gateWeight: f32ToF16Array(
      values(options.intermediateSize * options.hiddenSize, 16979, 0.08)
    ),
    upWeight: f32ToF16Array(
      values(options.intermediateSize * options.hiddenSize, 17077, 0.08)
    ),
    downWeight: f32ToF16Array(
      values(options.hiddenSize * options.intermediateSize, 17179, 0.08)
    ),
  };
  const cosValues = Float32Array.from(
    { length: options.seqLen * (options.rotaryDim / 2) },
    (_, index) => Math.cos(index * 0.011)
  );
  const sinValues = Float32Array.from(
    { length: cosValues.length },
    (_, index) => Math.sin(index * 0.011)
  );
  const loraValues = {
    q: adapterValues(options.hiddenSize, querySize * 2, 17291),
    k: adapterValues(options.hiddenSize, kvSize, 17317),
    v: adapterValues(options.hiddenSize, kvSize, 17341),
    o: adapterValues(querySize, options.hiddenSize, 17359),
    gate: adapterValues(options.hiddenSize, options.intermediateSize, 17377),
    up: adapterValues(options.hiddenSize, options.intermediateSize, 17401),
    down: adapterValues(options.intermediateSize, options.hiddenSize, 17431),
  };
  loraValues.down.B[1] += downAdapterPerturbation;
  const inputValues = {
    hidden: values(options.seqLen * options.hiddenSize, 17449, 0.18),
    inputNormWeight: f16ToF32Array(weightBits.inputNormWeight),
    postAttentionNormWeight: f16ToF32Array(weightBits.postAttentionNormWeight),
    attention: {
      qWeight: f16ToF32Array(weightBits.qWeight),
      kWeight: f16ToF32Array(weightBits.kWeight),
      vWeight: f16ToF32Array(weightBits.vWeight),
      oWeight: f16ToF32Array(weightBits.oWeight),
      qNormWeight: f16ToF32Array(weightBits.qNormWeight),
      kNormWeight: f16ToF32Array(weightBits.kNormWeight),
      cos: cosValues,
      sin: sinValues,
      lora: {
        q: loraValues.q,
        k: loraValues.k,
        v: loraValues.v,
        o: loraValues.o,
      },
    },
    mlp: {
      gateWeight: f16ToF32Array(weightBits.gateWeight),
      upWeight: f16ToF32Array(weightBits.upWeight),
      downWeight: f16ToF32Array(weightBits.downWeight),
      lora: {
        gate: loraValues.gate,
        up: loraValues.up,
        down: loraValues.down,
      },
    },
  };
  const attentionLora = {
    q: makeAdapter(loraValues.q, options.hiddenSize, querySize * 2, 'decoder_q_lora'),
    k: makeAdapter(loraValues.k, options.hiddenSize, kvSize, 'decoder_k_lora'),
    v: makeAdapter(loraValues.v, options.hiddenSize, kvSize, 'decoder_v_lora'),
    o: makeAdapter(loraValues.o, querySize, options.hiddenSize, 'decoder_o_lora'),
  };
  const mlpLora = {
    gate: makeAdapter(
      loraValues.gate,
      options.hiddenSize,
      options.intermediateSize,
      'decoder_gate_lora'
    ),
    up: makeAdapter(
      loraValues.up,
      options.hiddenSize,
      options.intermediateSize,
      'decoder_up_lora'
    ),
    down: makeAdapter(
      loraValues.down,
      options.intermediateSize,
      options.hiddenSize,
      'decoder_down_lora'
    ),
  };
  const tensors = {
    hidden: makeTensor(inputValues.hidden, [options.seqLen, options.hiddenSize], 'decoder_hidden'),
    inputNormWeight: makeTypedTensor(
      weightBits.inputNormWeight,
      'f16',
      [options.hiddenSize],
      'decoder_input_norm'
    ),
    postAttentionNormWeight: makeTypedTensor(
      weightBits.postAttentionNormWeight,
      'f16',
      [options.hiddenSize],
      'decoder_post_attention_norm'
    ),
    attention: {
      qWeight: makeTypedTensor(
        weightBits.qWeight,
        'f16',
        [querySize * 2, options.hiddenSize],
        'decoder_q_weight'
      ),
      kWeight: makeTypedTensor(
        weightBits.kWeight,
        'f16',
        [kvSize, options.hiddenSize],
        'decoder_k_weight'
      ),
      vWeight: makeTypedTensor(
        weightBits.vWeight,
        'f16',
        [kvSize, options.hiddenSize],
        'decoder_v_weight'
      ),
      oWeight: makeTypedTensor(
        weightBits.oWeight,
        'f16',
        [options.hiddenSize, querySize],
        'decoder_o_weight'
      ),
      qNormWeight: makeTypedTensor(
        weightBits.qNormWeight,
        'f16',
        [options.headDim],
        'decoder_q_norm'
      ),
      kNormWeight: makeTypedTensor(
        weightBits.kNormWeight,
        'f16',
        [options.headDim],
        'decoder_k_norm'
      ),
      cos: makeTensor(cosValues, [options.seqLen, options.rotaryDim / 2], 'decoder_cos'),
      sin: makeTensor(sinValues, [options.seqLen, options.rotaryDim / 2], 'decoder_sin'),
      lora: attentionLora,
    },
    mlp: {
      gateWeight: makeTypedTensor(
        weightBits.gateWeight,
        'f16',
        [options.intermediateSize, options.hiddenSize],
        'decoder_gate_weight'
      ),
      upWeight: makeTypedTensor(
        weightBits.upWeight,
        'f16',
        [options.intermediateSize, options.hiddenSize],
        'decoder_up_weight'
      ),
      downWeight: makeTypedTensor(
        weightBits.downWeight,
        'f16',
        [options.hiddenSize, options.intermediateSize],
        'decoder_down_weight'
      ),
      lora: mlpLora,
    },
  };
  const gradOutputValues = values(options.seqLen * options.hiddenSize, 17471, 0.23);
  const gradOutput = makeTensor(
    gradOutputValues,
    [options.seqLen, options.hiddenSize],
    'decoder_grad_output'
  );
  const referenceOptions = { ...options, numTokens: options.seqLen };
  const expectedForward = referenceForward(inputValues, referenceOptions);
  const expectedBackward = referenceBackward(
    inputValues,
    gradOutputValues,
    expectedForward.cache,
    referenceOptions
  );
  let forward = null;
  let backward = null;
  try {
    forward = await runQwenFullDecoderLayerForward(tensors, options);
    backward = await runQwenFullDecoderLayerBackward(
      tensors,
      gradOutput,
      forward.cache,
      options
    );
    const actualOutput = await readF32(forward.output);
    const comparisons = {
      decoderForward: compare(actualOutput, expectedForward.output),
      decoderBackwardHidden: compare(await readF32(backward.hidden), expectedBackward.hidden),
    };
    for (const projection of ['q', 'k', 'v', 'o', 'gate', 'up', 'down']) {
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
      for (const gradients of Object.values(backward.lora)) {
        releaseBuffer(gradients.A.buffer);
        releaseBuffer(gradients.B.buffer);
      }
    }
    if (forward) {
      releaseBuffer(forward.output.buffer);
      releaseQwenFullDecoderLayerCache(forward.cache);
    }
    releaseBuffer(tensors.hidden.buffer);
    releaseBuffer(tensors.inputNormWeight.buffer);
    releaseBuffer(tensors.postAttentionNormWeight.buffer);
    for (const [name, tensor] of Object.entries(tensors.attention)) {
      if (name !== 'lora') releaseBuffer(tensor.buffer);
    }
    for (const [name, tensor] of Object.entries(tensors.mlp)) {
      if (name !== 'lora') releaseBuffer(tensor.buffer);
    }
    for (const adapter of Object.values(attentionLora)) releaseAdapter(adapter);
    for (const adapter of Object.values(mlpLora)) releaseAdapter(adapter);
    releaseBuffer(gradOutput.buffer);
  }
}

export async function runQwenFullDecoderBackwardOracle() {
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
    artifactType: 'qwen_full_decoder_layer_backward_oracle',
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
    claimBoundary: 'Tiny Qwen full-attention decoder layer with exact head-width/split-half partial-rotary geometry, input norm, attention residual before post-attention offset RMSNorm, MLP residual, frozen F16 projections, and all seven V12 LoRA families; production head count/hidden width, loss, optimizer update, hybrid-layer composition, memory, and performance remain absent.',
  };
}
