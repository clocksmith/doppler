import { setRegistryUrl } from '../../../src/config/kernels/registry.js';
import { setPlatformsBaseUrl } from '../../../src/config/platforms/loader.js';
import {
  qwenFullDecoderLayerBackward as fullReferenceBackward,
  qwenFullDecoderLayerForward as fullReferenceForward,
} from '../../../src/experimental/training/qwen-full-decoder-reference.js';
import {
  releaseQwenHybridDecoderCache,
  runQwenHybridDecoderBackward,
  runQwenHybridDecoderForward,
} from '../../../src/experimental/training/qwen-hybrid-decoder-training-module.js';
import {
  qwenLinearDecoderLayerBackward as linearReferenceBackward,
  qwenLinearDecoderLayerForward as linearReferenceForward,
} from '../../../src/experimental/training/qwen-linear-decoder-reference.js';
import { getKernelCapabilities, initDevice } from '../../../src/gpu/device.js';
import { createTensor } from '../../../src/gpu/tensor.js';
import { f16ToF32Array, f32ToF16Array } from '../../../src/inference/kv-cache/types.js';
import { acquireBuffer, readBuffer, releaseBuffer, uploadData } from '../../../src/memory/buffer-pool.js';

function values(length, offset, scale) {
  return Float32Array.from(
    { length },
    (_, index) => Math.sin((index + offset) * 0.31) * scale
  );
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

function createTensorFactory(ownedTensors) {
  return (data, dtype, shape, label) => {
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
    const tensor = createTensor(buffer, dtype, shape, label);
    ownedTensors.push(tensor);
    return tensor;
  };
}

async function readF32(tensor) {
  const count = tensor.shape.reduce((product, value) => product * value, 1);
  return new Float32Array(await readBuffer(tensor.buffer, count * Float32Array.BYTES_PER_ELEMENT));
}

function adapterValues(inputSize, outputSize, offset) {
  const rank = 2;
  return {
    A: values(inputSize * rank, offset, 0.06),
    B: values(rank * outputSize, offset + 11, 0.045),
    rank,
    alpha: 4,
  };
}

function makeAdapter(makeTensor, source, inputSize, outputSize, label) {
  return {
    A: makeTensor(source.A, 'f32', [inputSize, source.rank], `${label}_a`),
    B: makeTensor(source.B, 'f32', [source.rank, outputSize], `${label}_b`),
    rank: source.rank,
    alpha: source.alpha,
  };
}

function makeF16Weight(makeTensor, length, offset, scale, shape, label) {
  const bits = f32ToF16Array(values(length, offset, scale));
  return {
    reference: f16ToF32Array(bits),
    tensor: makeTensor(bits, 'f16', shape, label),
  };
}

function buildMlp(makeTensor, hiddenSize, intermediateSize, offset, perturbDownB = 0) {
  const gateWeight = makeF16Weight(
    makeTensor,
    intermediateSize * hiddenSize,
    offset,
    0.1,
    [intermediateSize, hiddenSize],
    `hybrid_${offset}_gate_weight`
  );
  const upWeight = makeF16Weight(
    makeTensor,
    intermediateSize * hiddenSize,
    offset + 29,
    0.095,
    [intermediateSize, hiddenSize],
    `hybrid_${offset}_up_weight`
  );
  const downWeight = makeF16Weight(
    makeTensor,
    hiddenSize * intermediateSize,
    offset + 53,
    0.09,
    [hiddenSize, intermediateSize],
    `hybrid_${offset}_down_weight`
  );
  const lora = {
    gate: adapterValues(hiddenSize, intermediateSize, offset + 79),
    up: adapterValues(hiddenSize, intermediateSize, offset + 101),
    down: adapterValues(intermediateSize, hiddenSize, offset + 127),
  };
  lora.down.B[1] += perturbDownB;
  return {
    reference: {
      gateWeight: gateWeight.reference,
      upWeight: upWeight.reference,
      downWeight: downWeight.reference,
      lora,
    },
    tensors: {
      gateWeight: gateWeight.tensor,
      upWeight: upWeight.tensor,
      downWeight: downWeight.tensor,
      lora: {
        gate: makeAdapter(makeTensor, lora.gate, hiddenSize, intermediateSize, `hybrid_${offset}_gate_lora`),
        up: makeAdapter(makeTensor, lora.up, hiddenSize, intermediateSize, `hybrid_${offset}_up_lora`),
        down: makeAdapter(makeTensor, lora.down, intermediateSize, hiddenSize, `hybrid_${offset}_down_lora`),
      },
    },
  };
}

function buildLinearLayer(makeTensor, layerIndex, common) {
  const offset = 1000 + (layerIndex * 500);
  const options = {
    numTokens: common.numTokens,
    hiddenSize: common.hiddenSize,
    intermediateSize: common.intermediateSize,
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
  const inputNorm = makeF16Weight(
    makeTensor,
    common.hiddenSize,
    offset,
    0.04,
    [common.hiddenSize],
    `hybrid_l${layerIndex}_input_norm`
  );
  const postNorm = makeF16Weight(
    makeTensor,
    common.hiddenSize,
    offset + 13,
    0.04,
    [common.hiddenSize],
    `hybrid_l${layerIndex}_post_norm`
  );
  const qkv = makeF16Weight(
    makeTensor,
    convSize * common.hiddenSize,
    offset + 31,
    0.1,
    [convSize, common.hiddenSize],
    `hybrid_l${layerIndex}_qkv`
  );
  const z = makeF16Weight(
    makeTensor,
    valueSize * common.hiddenSize,
    offset + 67,
    0.09,
    [valueSize, common.hiddenSize],
    `hybrid_l${layerIndex}_z`
  );
  const a = makeF16Weight(
    makeTensor,
    options.numValueHeads * common.hiddenSize,
    offset + 89,
    0.08,
    [options.numValueHeads, common.hiddenSize],
    `hybrid_l${layerIndex}_a`
  );
  const b = makeF16Weight(
    makeTensor,
    options.numValueHeads * common.hiddenSize,
    offset + 103,
    0.08,
    [options.numValueHeads, common.hiddenSize],
    `hybrid_l${layerIndex}_b`
  );
  const out = makeF16Weight(
    makeTensor,
    common.hiddenSize * valueSize,
    offset + 127,
    0.1,
    [common.hiddenSize, valueSize],
    `hybrid_l${layerIndex}_out`
  );
  const attentionValues = {
    qkvWeight: qkv.reference,
    zWeight: z.reference,
    aWeight: a.reference,
    bWeight: b.reference,
    outWeight: out.reference,
    convWeight: values(convSize * options.kernelSize, offset + 151, 0.08),
    aLog: values(options.numValueHeads, offset + 179, 0.07),
    dtBias: values(options.numValueHeads, offset + 191, 0.06),
    normWeight: values(options.valueDim, offset + 199, 0.04),
    initialState: values(
      options.numValueHeads * options.keyDim * options.valueDim,
      offset + 211,
      0.05
    ),
  };
  const attentionTensors = {
    qkvWeight: qkv.tensor,
    zWeight: z.tensor,
    aWeight: a.tensor,
    bWeight: b.tensor,
    outWeight: out.tensor,
    convWeight: makeTensor(
      attentionValues.convWeight,
      'f32',
      [convSize, options.kernelSize],
      `hybrid_l${layerIndex}_conv`
    ),
    aLog: makeTensor(attentionValues.aLog, 'f32', [options.numValueHeads], `hybrid_l${layerIndex}_alog`),
    dtBias: makeTensor(attentionValues.dtBias, 'f32', [options.numValueHeads], `hybrid_l${layerIndex}_dt`),
    normWeight: makeTensor(attentionValues.normWeight, 'f32', [options.valueDim], `hybrid_l${layerIndex}_norm`),
    initialState: makeTensor(
      attentionValues.initialState,
      'f32',
      [options.numValueHeads, options.keyDim, options.valueDim],
      `hybrid_l${layerIndex}_state`
    ),
  };
  const mlp = buildMlp(makeTensor, common.hiddenSize, common.intermediateSize, offset + 251);
  return {
    type: 'linear_attention',
    options,
    referenceInputs: {
      inputNormWeight: inputNorm.reference,
      postAttentionNormWeight: postNorm.reference,
      attention: attentionValues,
      mlp: mlp.reference,
    },
    tensorInputs: {
      inputNormWeight: inputNorm.tensor,
      postAttentionNormWeight: postNorm.tensor,
      attention: attentionTensors,
      mlp: mlp.tensors,
    },
  };
}

function buildFullLayer(makeTensor, layerIndex, common, perturbDownB) {
  const offset = 3000;
  const options = {
    seqLen: common.numTokens,
    hiddenSize: common.hiddenSize,
    intermediateSize: common.intermediateSize,
    numHeads: 2,
    numKVHeads: 1,
    headDim: 4,
    rotaryDim: 2,
    pairSpanDim: 2,
    interleaved: true,
    startPos: 0,
    rmsEps: 1e-6,
  };
  const querySize = options.numHeads * options.headDim;
  const kvSize = options.numKVHeads * options.headDim;
  const inputNorm = makeF16Weight(
    makeTensor,
    common.hiddenSize,
    offset,
    0.04,
    [common.hiddenSize],
    `hybrid_l${layerIndex}_input_norm`
  );
  const postNorm = makeF16Weight(
    makeTensor,
    common.hiddenSize,
    offset + 13,
    0.04,
    [common.hiddenSize],
    `hybrid_l${layerIndex}_post_norm`
  );
  const q = makeF16Weight(
    makeTensor,
    querySize * 2 * common.hiddenSize,
    offset + 31,
    0.1,
    [querySize * 2, common.hiddenSize],
    `hybrid_l${layerIndex}_q`
  );
  const k = makeF16Weight(
    makeTensor,
    kvSize * common.hiddenSize,
    offset + 101,
    0.09,
    [kvSize, common.hiddenSize],
    `hybrid_l${layerIndex}_k`
  );
  const v = makeF16Weight(
    makeTensor,
    kvSize * common.hiddenSize,
    offset + 127,
    0.09,
    [kvSize, common.hiddenSize],
    `hybrid_l${layerIndex}_v`
  );
  const o = makeF16Weight(
    makeTensor,
    common.hiddenSize * querySize,
    offset + 151,
    0.1,
    [common.hiddenSize, querySize],
    `hybrid_l${layerIndex}_o`
  );
  const qNorm = makeF16Weight(
    makeTensor,
    options.headDim,
    offset + 181,
    0.04,
    [options.headDim],
    `hybrid_l${layerIndex}_q_norm`
  );
  const kNorm = makeF16Weight(
    makeTensor,
    options.headDim,
    offset + 193,
    0.04,
    [options.headDim],
    `hybrid_l${layerIndex}_k_norm`
  );
  const cos = Float32Array.from(
    { length: common.numTokens * (options.rotaryDim / 2) },
    (_, index) => Math.cos(index * 0.17)
  );
  const sin = Float32Array.from({ length: cos.length }, (_, index) => Math.sin(index * 0.17));
  const attentionLora = {
    q: adapterValues(common.hiddenSize, querySize * 2, offset + 211),
    k: adapterValues(common.hiddenSize, kvSize, offset + 241),
    v: adapterValues(common.hiddenSize, kvSize, offset + 269),
    o: adapterValues(querySize, common.hiddenSize, offset + 293),
  };
  const attentionValues = {
    qWeight: q.reference,
    kWeight: k.reference,
    vWeight: v.reference,
    oWeight: o.reference,
    qNormWeight: qNorm.reference,
    kNormWeight: kNorm.reference,
    cos,
    sin,
    lora: attentionLora,
  };
  const attentionTensors = {
    qWeight: q.tensor,
    kWeight: k.tensor,
    vWeight: v.tensor,
    oWeight: o.tensor,
    qNormWeight: qNorm.tensor,
    kNormWeight: kNorm.tensor,
    cos: makeTensor(cos, 'f32', [common.numTokens, options.rotaryDim / 2], `hybrid_l${layerIndex}_cos`),
    sin: makeTensor(sin, 'f32', [common.numTokens, options.rotaryDim / 2], `hybrid_l${layerIndex}_sin`),
    lora: {
      q: makeAdapter(makeTensor, attentionLora.q, common.hiddenSize, querySize * 2, `hybrid_l${layerIndex}_q_lora`),
      k: makeAdapter(makeTensor, attentionLora.k, common.hiddenSize, kvSize, `hybrid_l${layerIndex}_k_lora`),
      v: makeAdapter(makeTensor, attentionLora.v, common.hiddenSize, kvSize, `hybrid_l${layerIndex}_v_lora`),
      o: makeAdapter(makeTensor, attentionLora.o, querySize, common.hiddenSize, `hybrid_l${layerIndex}_o_lora`),
    },
  };
  const mlp = buildMlp(
    makeTensor,
    common.hiddenSize,
    common.intermediateSize,
    offset + 331,
    perturbDownB
  );
  return {
    type: 'full_attention',
    options,
    referenceInputs: {
      inputNormWeight: inputNorm.reference,
      postAttentionNormWeight: postNorm.reference,
      attention: attentionValues,
      mlp: mlp.reference,
    },
    tensorInputs: {
      inputNormWeight: inputNorm.tensor,
      postAttentionNormWeight: postNorm.tensor,
      attention: attentionTensors,
      mlp: mlp.tensors,
    },
  };
}

function runReference(layers, hidden, gradOutput) {
  const entries = [];
  const finalStates = [];
  let current = hidden;
  for (let index = 0; index < layers.length; index += 1) {
    const layer = layers[index];
    const inputs = { ...layer.referenceInputs, hidden: current };
    const forward = layer.type === 'linear_attention'
      ? linearReferenceForward(inputs, layer.options)
      : fullReferenceForward(inputs, { ...layer.options, numTokens: layer.options.seqLen });
    entries.push({ ...layer, inputs, forward });
    current = forward.output;
    if (layer.type === 'linear_attention') finalStates.push(forward.finalState);
  }
  const gradients = new Array(layers.length);
  let currentGradient = gradOutput;
  for (let index = entries.length - 1; index >= 0; index -= 1) {
    const entry = entries[index];
    const backward = entry.type === 'linear_attention'
      ? linearReferenceBackward(entry.inputs, currentGradient, entry.forward.cache, entry.options)
      : fullReferenceBackward(
          entry.inputs,
          currentGradient,
          entry.forward.cache,
          { ...entry.options, numTokens: entry.options.seqLen }
        );
    gradients[index] = backward;
    currentGradient = backward.hidden;
  }
  return { output: current, hiddenGradient: currentGradient, finalStates, gradients };
}

async function executeCase(perturbDownB = 0) {
  const common = { numTokens: 2, hiddenSize: 4, intermediateSize: 6 };
  const ownedTensors = [];
  const makeTensor = createTensorFactory(ownedTensors);
  const layers = [
    buildLinearLayer(makeTensor, 0, common),
    buildLinearLayer(makeTensor, 1, common),
    buildLinearLayer(makeTensor, 2, common),
    buildFullLayer(makeTensor, 3, common, perturbDownB),
  ];
  const hiddenValues = values(common.numTokens * common.hiddenSize, 503, 0.16);
  const gradOutputValues = values(common.numTokens * common.hiddenSize, 541, 0.21);
  const reference = runReference(layers, hiddenValues, gradOutputValues);
  const hidden = makeTensor(
    hiddenValues,
    'f32',
    [common.numTokens, common.hiddenSize],
    'hybrid_hidden'
  );
  const gradOutput = makeTensor(
    gradOutputValues,
    'f32',
    [common.numTokens, common.hiddenSize],
    'hybrid_grad_output'
  );
  let forward = null;
  let backward = null;
  try {
    forward = await runQwenHybridDecoderForward({
      hidden,
      layers: layers.map((layer) => ({
        type: layer.type,
        inputs: layer.tensorInputs,
        options: layer.options,
      })),
    });
    backward = await runQwenHybridDecoderBackward(gradOutput, forward.cache);
    const actualOutput = await readF32(forward.output);
    const comparisons = {
      hybridForward: compare(actualOutput, reference.output),
      hybridBackwardHidden: compare(await readF32(backward.hidden), reference.hiddenGradient),
    };
    for (let index = 0; index < forward.finalStates.length; index += 1) {
      comparisons[`layer${index}FinalState`] = compare(
        await readF32(forward.finalStates[index].state),
        reference.finalStates[index]
      );
    }
    for (let index = 0; index < layers.length; index += 1) {
      const expected = reference.gradients[index];
      const actual = backward.layers[index];
      if (actual.initialState) {
        comparisons[`layer${index}InitialStateGradient`] = compare(
          await readF32(actual.initialState),
          expected.initialState
        );
      }
      for (const projection of Object.keys(actual.lora)) {
        for (const matrix of ['A', 'B']) {
          const actualGradient = await readF32(actual.lora[projection][matrix]);
          comparisons[`layer${index}Lora${projection.toUpperCase()}${matrix}`] = {
            ...compare(actualGradient, expected.lora[projection][matrix]),
            maxAbsValue: maxAbs(actualGradient),
          };
        }
      }
    }
    return {
      actualOutput,
      comparisons,
      layerTypes: forward.cache.layerTypes,
    };
  } finally {
    if (backward) {
      releaseBuffer(backward.hidden.buffer);
      for (const layer of backward.layers) {
        if (layer.initialState) releaseBuffer(layer.initialState.buffer);
        for (const gradients of Object.values(layer.lora)) {
          releaseBuffer(gradients.A.buffer);
          releaseBuffer(gradients.B.buffer);
        }
      }
    }
    if (forward) {
      releaseBuffer(forward.output.buffer);
      for (const item of forward.finalStates) releaseBuffer(item.state.buffer);
      releaseQwenHybridDecoderCache(forward.cache);
    }
    for (const tensor of ownedTensors) releaseBuffer(tensor.buffer);
  }
}

export async function runQwenHybridDecoderBackwardOracle() {
  const baseUrl = new URL('../../../src/config/', import.meta.url);
  setPlatformsBaseUrl(new URL('platforms/', baseUrl).toString());
  setRegistryUrl(new URL('kernels/registry.json', baseUrl).toString());
  await initDevice();

  const tolerance = 5e-5;
  const baseline = await executeCase();
  const perturbed = await executeCase(0.125);
  const perturbation = compare(perturbed.actualOutput, baseline.actualOutput);
  const expectedLayerTypes = [
    'linear_attention',
    'linear_attention',
    'linear_attention',
    'full_attention',
  ];
  const layerPatternPassed = JSON.stringify(baseline.layerTypes) === JSON.stringify(expectedLayerTypes);
  const passed = layerPatternPassed && Object.values(baseline.comparisons).every(
    (entry) => entry.allFinite
      && entry.maxAbsError <= tolerance
      && (entry.maxAbsValue == null || entry.maxAbsValue > 1e-10)
  ) && perturbation.maxAbsError > 1e-5;
  const capabilities = getKernelCapabilities();
  return {
    artifactType: 'qwen_hybrid_decoder_backward_oracle',
    schemaVersion: 1,
    passed,
    layerTypes: baseline.layerTypes,
    expectedLayerTypes,
    layerPatternPassed,
    tolerance: { maxAbsError: tolerance },
    comparisons: baseline.comparisons,
    negativeControl: {
      perturbation: 'layer3_down_proj_lora_b_index_1_plus_0.125',
      hybridOutputDifference: perturbation,
      passed: perturbation.maxAbsError > 1e-5,
    },
    adapterInfo: capabilities.adapterInfo || null,
    claimBoundary: 'Tiny four-layer Qwen hybrid graph matching the three-linear/one-full layer pattern with cross-layer backward, recurrent-state gradients, and every layer-local V12 LoRA family; loss, optimizer update, production geometry, activation checkpointing across layers, memory, and performance remain absent.',
  };
}
