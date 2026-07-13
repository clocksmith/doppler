import { setRegistryUrl } from '../../../src/config/kernels/registry.js';
import { setPlatformsBaseUrl } from '../../../src/config/platforms/loader.js';
import {
  qwenRmsNormOffsetBackward,
  qwenRmsNormOffsetForward,
} from '../../../src/experimental/training/qwen-full-attention-reference.js';
import {
  qwenFullDecoderLayerBackward,
  qwenFullDecoderLayerForward,
} from '../../../src/experimental/training/qwen-full-decoder-reference.js';
import { runQwenHybridSftMicrostep } from '../../../src/experimental/training/qwen-hybrid-sft-microstep.js';
import { AdamOptimizer } from '../../../src/experimental/training/optimizer.js';
import { getKernelCapabilities, initDevice } from '../../../src/gpu/device.js';
import { createTensor } from '../../../src/gpu/tensor.js';
import { f16ToF32Array, f32ToF16Array } from '../../../src/inference/kv-cache/types.js';
import { acquireBuffer, readBuffer, releaseBuffer, uploadData } from '../../../src/memory/buffer-pool.js';

const MASKED_TARGET = 0xffffffff;

function values(length, offset, scale) {
  return Float32Array.from(
    { length },
    (_, index) => Math.sin((index + offset) * 0.37) * scale
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

function maxAbsDifference(left, right) {
  return compare(left, right).maxAbsError;
}

function makeTensorFactory(ownedTensors) {
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
  return new Float32Array(await readBuffer(tensor.buffer, count * 4));
}

function matmulRightTransposed(input, weight, rows, inputSize, outputSize) {
  const output = new Float32Array(rows * outputSize);
  for (let row = 0; row < rows; row += 1) {
    for (let outputIndex = 0; outputIndex < outputSize; outputIndex += 1) {
      for (let inputIndex = 0; inputIndex < inputSize; inputIndex += 1) {
        output[(row * outputSize) + outputIndex] += input[(row * inputSize) + inputIndex]
          * weight[(outputIndex * inputSize) + inputIndex];
      }
    }
  }
  return output;
}

function transposedWeightInputGradient(gradOutput, weight, rows, inputSize, outputSize) {
  const output = new Float32Array(rows * inputSize);
  for (let row = 0; row < rows; row += 1) {
    for (let inputIndex = 0; inputIndex < inputSize; inputIndex += 1) {
      for (let outputIndex = 0; outputIndex < outputSize; outputIndex += 1) {
        output[(row * inputSize) + inputIndex] += gradOutput[(row * outputSize) + outputIndex]
          * weight[(outputIndex * inputSize) + inputIndex];
      }
    }
  }
  return output;
}

function softmaxRows(logits, rows, columns) {
  const output = new Float32Array(logits.length);
  for (let row = 0; row < rows; row += 1) {
    const base = row * columns;
    let maximum = -Infinity;
    for (let column = 0; column < columns; column += 1) {
      maximum = Math.max(maximum, logits[base + column]);
    }
    let sum = 0;
    for (let column = 0; column < columns; column += 1) {
      const value = Math.exp(logits[base + column] - maximum);
      output[base + column] = value;
      sum += value;
    }
    for (let column = 0; column < columns; column += 1) output[base + column] /= sum;
  }
  return output;
}

function maskedCrossEntropy(softmax, targets, rows, columns) {
  const losses = new Float32Array(rows);
  for (let row = 0; row < rows; row += 1) {
    if (targets[row] >= columns) continue;
    losses[row] = -Math.log(Math.max(softmax[(row * columns) + targets[row]], 1e-9));
  }
  return losses;
}

function maskedCrossEntropyGradient(softmax, targets, rows, columns, activeTokenCount) {
  const output = new Float32Array(softmax.length);
  for (let row = 0; row < rows; row += 1) {
    if (targets[row] >= columns) continue;
    const base = row * columns;
    for (let column = 0; column < columns; column += 1) {
      output[base + column] = (
        softmax[base + column] - (column === targets[row] ? 1 : 0)
      ) / activeTokenCount;
    }
  }
  return output;
}

function scalarAdamw(parameter, gradient, options) {
  const updated = new Float32Array(parameter.length);
  const moment1 = new Float32Array(parameter.length);
  const moment2 = new Float32Array(parameter.length);
  for (let index = 0; index < parameter.length; index += 1) {
    moment1[index] = (1 - options.beta1) * gradient[index];
    moment2[index] = (1 - options.beta2) * gradient[index] * gradient[index];
    const first = moment1[index] / (1 - options.beta1);
    const second = moment2[index] / (1 - options.beta2);
    updated[index] = parameter[index] - options.lr * (
      first / (Math.sqrt(second) + options.eps) + options.weightDecay * parameter[index]
    );
  }
  return { updated, moment1, moment2 };
}

function adapterValues(inputSize, outputSize, offset) {
  const rank = 2;
  return {
    A: values(inputSize * rank, offset, 0.065),
    B: values(rank * outputSize, offset + 17, 0.05),
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

function addReferenceAdapterEntries(entries, prefix, adapter, gradients) {
  entries.push(
    { name: `${prefix}.lora_A`, parameter: adapter.A, gradient: gradients.A },
    { name: `${prefix}.lora_B`, parameter: adapter.B, gradient: gradients.B }
  );
}

function buildFixture(makeTensor) {
  const options = {
    numTokens: 3,
    hiddenSize: 4,
    intermediateSize: 6,
    vocabSize: 11,
    activeTokenCount: 2,
    rmsEps: 1e-6,
  };
  const layerOptions = {
    seqLen: options.numTokens,
    hiddenSize: options.hiddenSize,
    intermediateSize: options.intermediateSize,
    numHeads: 2,
    numKVHeads: 1,
    headDim: 16,
    rotaryDim: 4,
    pairSpanDim: 4,
    interleaved: false,
    startPos: 0,
    rmsEps: options.rmsEps,
  };
  const querySize = layerOptions.numHeads * layerOptions.headDim;
  const kvSize = layerOptions.numKVHeads * layerOptions.headDim;
  const f16Weight = (length, offset, scale, shape, label) => {
    const bits = f32ToF16Array(values(length, offset, scale));
    return {
      reference: f16ToF32Array(bits),
      tensor: makeTensor(bits, 'f16', shape, label),
    };
  };
  const embedding = f16Weight(
    options.vocabSize * options.hiddenSize,
    3,
    0.12,
    [options.vocabSize, options.hiddenSize],
    'microstep_embedding'
  );
  const inputNorm = f16Weight(options.hiddenSize, 59, 0.04, [options.hiddenSize], 'microstep_input_norm');
  const postNorm = f16Weight(options.hiddenSize, 71, 0.04, [options.hiddenSize], 'microstep_post_norm');
  const q = f16Weight(querySize * 2 * options.hiddenSize, 83, 0.1, [querySize * 2, options.hiddenSize], 'microstep_q');
  const k = f16Weight(kvSize * options.hiddenSize, 151, 0.09, [kvSize, options.hiddenSize], 'microstep_k');
  const v = f16Weight(kvSize * options.hiddenSize, 179, 0.09, [kvSize, options.hiddenSize], 'microstep_v');
  const o = f16Weight(options.hiddenSize * querySize, 211, 0.1, [options.hiddenSize, querySize], 'microstep_o');
  const qNorm = f16Weight(layerOptions.headDim, 251, 0.04, [layerOptions.headDim], 'microstep_q_norm');
  const kNorm = f16Weight(layerOptions.headDim, 263, 0.04, [layerOptions.headDim], 'microstep_k_norm');
  const gate = f16Weight(
    options.intermediateSize * options.hiddenSize,
    277,
    0.11,
    [options.intermediateSize, options.hiddenSize],
    'microstep_gate'
  );
  const up = f16Weight(
    options.intermediateSize * options.hiddenSize,
    307,
    0.1,
    [options.intermediateSize, options.hiddenSize],
    'microstep_up'
  );
  const down = f16Weight(
    options.hiddenSize * options.intermediateSize,
    337,
    0.1,
    [options.hiddenSize, options.intermediateSize],
    'microstep_down'
  );
  const finalNorm = f16Weight(options.hiddenSize, 367, 0.04, [options.hiddenSize], 'microstep_final_norm');
  const lmHead = f16Weight(
    options.vocabSize * options.hiddenSize,
    379,
    0.11,
    [options.vocabSize, options.hiddenSize],
    'microstep_lm_head'
  );
  const cos = Float32Array.from(
    { length: options.numTokens * (layerOptions.rotaryDim / 2) },
    (_, index) => Math.cos(index * 0.19)
  );
  const sin = Float32Array.from({ length: cos.length }, (_, index) => Math.sin(index * 0.19));
  const loraValues = {
    q: adapterValues(options.hiddenSize, querySize * 2, 431),
    k: adapterValues(options.hiddenSize, kvSize, 461),
    v: adapterValues(options.hiddenSize, kvSize, 491),
    o: adapterValues(querySize, options.hiddenSize, 521),
    gate: adapterValues(options.hiddenSize, options.intermediateSize, 557),
    up: adapterValues(options.hiddenSize, options.intermediateSize, 593),
    down: adapterValues(options.intermediateSize, options.hiddenSize, 631),
  };
  const referenceLayer = {
    inputNormWeight: inputNorm.reference,
    postAttentionNormWeight: postNorm.reference,
    attention: {
      qWeight: q.reference,
      kWeight: k.reference,
      vWeight: v.reference,
      oWeight: o.reference,
      qNormWeight: qNorm.reference,
      kNormWeight: kNorm.reference,
      cos,
      sin,
      lora: { q: loraValues.q, k: loraValues.k, v: loraValues.v, o: loraValues.o },
    },
    mlp: {
      gateWeight: gate.reference,
      upWeight: up.reference,
      downWeight: down.reference,
      lora: { gate: loraValues.gate, up: loraValues.up, down: loraValues.down },
    },
  };
  const tensorLora = {
    q: makeAdapter(makeTensor, loraValues.q, options.hiddenSize, querySize * 2, 'microstep_q_lora'),
    k: makeAdapter(makeTensor, loraValues.k, options.hiddenSize, kvSize, 'microstep_k_lora'),
    v: makeAdapter(makeTensor, loraValues.v, options.hiddenSize, kvSize, 'microstep_v_lora'),
    o: makeAdapter(makeTensor, loraValues.o, querySize, options.hiddenSize, 'microstep_o_lora'),
    gate: makeAdapter(makeTensor, loraValues.gate, options.hiddenSize, options.intermediateSize, 'microstep_gate_lora'),
    up: makeAdapter(makeTensor, loraValues.up, options.hiddenSize, options.intermediateSize, 'microstep_up_lora'),
    down: makeAdapter(makeTensor, loraValues.down, options.intermediateSize, options.hiddenSize, 'microstep_down_lora'),
  };
  const tensorLayer = {
    type: 'full_attention',
    options: layerOptions,
    inputs: {
      inputNormWeight: inputNorm.tensor,
      postAttentionNormWeight: postNorm.tensor,
      attention: {
        qWeight: q.tensor,
        kWeight: k.tensor,
        vWeight: v.tensor,
        oWeight: o.tensor,
        qNormWeight: qNorm.tensor,
        kNormWeight: kNorm.tensor,
        cos: makeTensor(cos, 'f32', [options.numTokens, layerOptions.rotaryDim / 2], 'microstep_cos'),
        sin: makeTensor(sin, 'f32', [options.numTokens, layerOptions.rotaryDim / 2], 'microstep_sin'),
        lora: { q: tensorLora.q, k: tensorLora.k, v: tensorLora.v, o: tensorLora.o },
      },
      mlp: {
        gateWeight: gate.tensor,
        upWeight: up.tensor,
        downWeight: down.tensor,
        lora: { gate: tensorLora.gate, up: tensorLora.up, down: tensorLora.down },
      },
    },
  };
  const tokenIds = new Uint32Array([1, 2, 3]);
  const targets = new Uint32Array([MASKED_TARGET, 4, 5]);
  return {
    options,
    layerOptions,
    tokenIds,
    targets,
    reference: {
      embedding: embedding.reference,
      layer: referenceLayer,
      finalNorm: finalNorm.reference,
      lmHead: lmHead.reference,
      loraValues,
    },
    tensors: {
      tokenIds: makeTensor(tokenIds, 'u32', [options.numTokens], 'microstep_tokens'),
      targets: makeTensor(targets, 'u32', [options.numTokens], 'microstep_targets'),
      embedding: embedding.tensor,
      layer: tensorLayer,
      finalNorm: finalNorm.tensor,
      lmHead: lmHead.tensor,
      lora: tensorLora,
    },
  };
}

function runScalarMicrostep(fixture, optimizerOptions) {
  const { options, layerOptions, tokenIds, targets, reference } = fixture;
  const hidden = new Float32Array(options.numTokens * options.hiddenSize);
  for (let token = 0; token < options.numTokens; token += 1) {
    const source = tokenIds[token] * options.hiddenSize;
    hidden.set(reference.embedding.subarray(source, source + options.hiddenSize), token * options.hiddenSize);
  }
  const layerInputs = { ...reference.layer, hidden };
  const forward = qwenFullDecoderLayerForward(
    layerInputs,
    { ...layerOptions, numTokens: options.numTokens }
  );
  const finalNorm = qwenRmsNormOffsetForward(
    forward.output,
    reference.finalNorm,
    options.numTokens,
    options.hiddenSize,
    options.rmsEps
  );
  const logits = matmulRightTransposed(
    finalNorm.output,
    reference.lmHead,
    options.numTokens,
    options.hiddenSize,
    options.vocabSize
  );
  const softmax = softmaxRows(logits, options.numTokens, options.vocabSize);
  const losses = maskedCrossEntropy(softmax, targets, options.numTokens, options.vocabSize);
  const meanLoss = losses.reduce((sum, value) => sum + value, 0) / options.activeTokenCount;
  const gradLogits = maskedCrossEntropyGradient(
    softmax,
    targets,
    options.numTokens,
    options.vocabSize,
    options.activeTokenCount
  );
  const gradFinalNorm = transposedWeightInputGradient(
    gradLogits,
    reference.lmHead,
    options.numTokens,
    options.hiddenSize,
    options.vocabSize
  );
  const gradLayerOutput = qwenRmsNormOffsetBackward(
    forward.output,
    reference.finalNorm,
    gradFinalNorm,
    finalNorm.cache,
    options.numTokens,
    options.hiddenSize
  );
  const backward = qwenFullDecoderLayerBackward(
    layerInputs,
    gradLayerOutput,
    forward.cache,
    { ...layerOptions, numTokens: options.numTokens }
  );
  const entries = [];
  for (const projection of ['q', 'k', 'v', 'o']) {
    addReferenceAdapterEntries(
      entries,
      `layers.0.self_attn.${projection}_proj`,
      reference.loraValues[projection],
      backward.lora[projection]
    );
  }
  for (const projection of ['gate', 'up', 'down']) {
    addReferenceAdapterEntries(
      entries,
      `layers.0.mlp.${projection}_proj`,
      reference.loraValues[projection],
      backward.lora[projection]
    );
  }
  const expected = {};
  for (const entry of entries) {
    expected[entry.name] = {
      gradient: entry.gradient,
      ...scalarAdamw(entry.parameter, entry.gradient, optimizerOptions),
    };
  }
  const unmaskedTargets = new Uint32Array([6, 4, 5]);
  const unmaskedLosses = maskedCrossEntropy(
    softmax,
    unmaskedTargets,
    options.numTokens,
    options.vocabSize
  );
  const unmaskedMeanLoss = unmaskedLosses.reduce((sum, value) => sum + value, 0) / options.numTokens;
  return { meanLoss, expected, unmaskedMeanLoss };
}

export async function runQwenHybridSftMicrostepOracle() {
  const baseUrl = new URL('../../../src/config/', import.meta.url);
  setPlatformsBaseUrl(new URL('platforms/', baseUrl).toString());
  setRegistryUrl(new URL('kernels/registry.json', baseUrl).toString());
  await initDevice();

  const ownedTensors = [];
  const makeTensor = makeTensorFactory(ownedTensors);
  const fixture = buildFixture(makeTensor);
  const optimizerOptions = {
    type: 'adamw',
    lr: 0.001,
    beta1: 0.9,
    beta2: 0.999,
    eps: 1e-8,
    weightDecay: 0.01,
    scheduler: { enabled: false },
  };
  const trainingConfig = { training: { optimizer: optimizerOptions } };
  const optimizer = new AdamOptimizer(trainingConfig);
  const scalar = runScalarMicrostep(fixture, optimizerOptions);
  const initialParameters = Object.fromEntries(
    Object.entries(fixture.tensors.lora).flatMap(([name, adapter]) => [
      [`${name}.A`, new Float32Array(fixture.reference.loraValues[name].A)],
      [`${name}.B`, new Float32Array(fixture.reference.loraValues[name].B)],
    ])
  );
  try {
    const actual = await runQwenHybridSftMicrostep({
      tokenIds: fixture.tensors.tokenIds,
      embeddingWeight: fixture.tensors.embedding,
      layers: [fixture.tensors.layer],
      finalNormWeight: fixture.tensors.finalNorm,
      lmHeadWeight: fixture.tensors.lmHead,
      targets: fixture.tensors.targets,
    }, {
      ...fixture.options,
      optimizer,
      trainingConfig,
      captureGradients: true,
    });
    const comparisons = {
      meanLoss: {
        elementCount: 1,
        allFinite: Number.isFinite(actual.meanLoss),
        maxAbsError: Math.abs(actual.meanLoss - scalar.meanLoss),
        rmse: Math.abs(actual.meanLoss - scalar.meanLoss),
      },
    };
    let everyParameterChanged = true;
    for (const name of actual.parameterNames) {
      const parts = name.split('.');
      const projection = parts[parts.length - 2].replace('_proj', '');
      const matrix = parts[parts.length - 1] === 'lora_A' ? 'A' : 'B';
      const tensor = fixture.tensors.lora[projection][matrix];
      const parameter = await readF32(tensor);
      const optimizerState = optimizer.getState(tensor);
      comparisons[`${name}.gradient`] = compare(
        actual.gradientSnapshots[name],
        scalar.expected[name].gradient
      );
      comparisons[`${name}.parameter`] = compare(parameter, scalar.expected[name].updated);
      comparisons[`${name}.moment1`] = compare(
        await readF32(optimizerState.m),
        scalar.expected[name].moment1
      );
      comparisons[`${name}.moment2`] = compare(
        await readF32(optimizerState.v),
        scalar.expected[name].moment2
      );
      everyParameterChanged = everyParameterChanged
        && maxAbsDifference(parameter, initialParameters[`${projection}.${matrix}`]) > 1e-7;
    }
    const tolerance = 5e-5;
    const maskDifference = Math.abs(actual.meanLoss - scalar.unmaskedMeanLoss);
    const allGradientsNonzero = Object.values(actual.gradientSnapshots).every(
      (gradient) => gradient.some((value) => value !== 0)
    );
    const passed = optimizer.stepCount === 1
      && actual.activeTokenCount === fixture.options.activeTokenCount
      && actual.parameterNames.length === 14
      && allGradientsNonzero
      && everyParameterChanged
      && Object.values(comparisons).every(
        (entry) => entry.allFinite && entry.maxAbsError <= tolerance
      )
      && maskDifference > 1e-4;
    const capabilities = getKernelCapabilities();
    return {
      artifactType: 'qwen_hybrid_sft_microstep_oracle',
      schemaVersion: 1,
      passed,
      activeTokenCount: actual.activeTokenCount,
      parameterCount: actual.parameterNames.length,
      parameterNames: actual.parameterNames,
      optimizerStepCount: optimizer.stepCount,
      allGradientsNonzero,
      everyParameterChanged,
      tolerance: { maxAbsError: tolerance },
      comparisons,
      negativeControl: {
        control: 'same_logits_with_prompt_token_included_in_loss_mean',
        maskedVsUnmaskedMeanLossDifference: maskDifference,
        passed: maskDifference > 1e-4,
      },
      adapterInfo: capabilities.adapterInfo || null,
      claimBoundary: 'Tiny one-full-layer completion-masked SFT microstep from frozen F16 embedding through logits using the pinned Qwen residual-before-post-attention-norm and split-half partial-RoPE contracts, all seven rank-two LoRA families, and one decoupled AdamW update; not rank-32 Gamma parity, production Qwen geometry, accumulation, resume, or capability evidence.',
    };
  } finally {
    for (const state of optimizer.state.values()) {
      releaseBuffer(state.m.buffer);
      releaseBuffer(state.v.buffer);
    }
    for (const tensor of ownedTensors) releaseBuffer(tensor.buffer);
  }
}
