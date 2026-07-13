import { setRegistryUrl } from '../../../src/config/kernels/registry.js';
import { setPlatformsBaseUrl } from '../../../src/config/platforms/loader.js';
import {
  qwenAttentionSplitQGateBackward,
  qwenAttentionSplitQGateForward,
  partialRopeBackward,
  partialRopeForward,
  qwenFullAttentionModuleBackward,
  qwenFullAttentionModuleForward,
  sigmoidGateBackward,
  sigmoidGateForward,
} from '../../../src/experimental/training/qwen-full-attention-reference.js';
import {
  releaseQwenFullAttentionTrainingModuleCache,
  runQwenFullAttentionTrainingModuleBackward,
  runQwenFullAttentionTrainingModuleForward,
} from '../../../src/experimental/training/qwen-full-attention-training-module.js';
import {
  computeAttentionBackwardData,
  computeAttentionSoftmaxData,
} from '../../../src/experimental/training/attention-backward.js';
import { getKernelCapabilities, initDevice } from '../../../src/gpu/device.js';
import {
  runQwenAttentionSplitQGate,
  runRoPE,
  runSiLU,
} from '../../../src/gpu/kernels/index.js';
import {
  runQwenAttentionSplitQGateBackward,
  runQwenGqaAttentionBackward,
  runSigmoidGatedBackward,
  runRoPEBackward,
} from '../../../src/gpu/kernels/backward/index.js';
import { createTensor } from '../../../src/gpu/tensor.js';
import { f16ToF32Array, f32ToF16Array } from '../../../src/inference/kv-cache/types.js';
import { acquireBuffer, readBuffer, releaseBuffer, uploadData } from '../../../src/memory/buffer-pool.js';

function values(length, offset, scale) {
  return Float32Array.from(
    { length },
    (_, index) => Math.sin((index + offset) * 0.47) * scale
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

async function executeCase(gatePerturbation) {
  const options = { numTokens: 3, numHeads: 2, headDim: 4 };
  const elementCount = options.numTokens * options.numHeads * options.headDim;
  const qGateValues = values(elementCount * 2, 3, 0.4);
  qGateValues[options.headDim + 1] += gatePerturbation;
  const attentionValues = values(elementCount, 41, 0.35);
  const gradOutputValues = values(elementCount, 67, 0.3);
  const gradQueryValues = values(elementCount, 89, 0.25);
  const gradGateValues = values(elementCount, 101, 0.2);
  const qGate = makeTensor(
    qGateValues,
    [options.numTokens, options.numHeads, options.headDim * 2],
    'full_attention_q_gate'
  );
  const attention = makeTensor(
    attentionValues,
    [options.numTokens, options.numHeads, options.headDim],
    'full_attention_output'
  );
  const gradOutput = makeTensor(
    gradOutputValues,
    [options.numTokens, options.numHeads, options.headDim],
    'full_attention_grad_output'
  );
  const gradQuery = makeTensor(
    gradQueryValues,
    [options.numTokens, options.numHeads, options.headDim],
    'full_attention_grad_query'
  );
  const splitGradGate = makeTensor(
    gradGateValues,
    [options.numTokens, options.numHeads, options.headDim],
    'full_attention_split_grad_gate'
  );
  let split = null;
  let splitBackward = null;
  let gated = null;
  let gatedBackward = null;
  try {
    split = await runQwenAttentionSplitQGate(qGate, options);
    splitBackward = await runQwenAttentionSplitQGateBackward(
      gradQuery,
      splitGradGate,
      options
    );
    gated = await runSiLU(attention, {
      size: elementCount,
      gate: split.gate,
      gateActivation: 'sigmoid',
      inputActivation: 'identity',
      swigluLimit: null,
    });
    gatedBackward = await runSigmoidGatedBackward(
      attention,
      split.gate,
      gradOutput,
      { count: elementCount }
    );
    const expectedSplit = qwenAttentionSplitQGateForward(qGateValues, options);
    const expectedSplitBackward = qwenAttentionSplitQGateBackward(
      gradQueryValues,
      gradGateValues,
      options
    );
    const expectedGated = sigmoidGateForward(attentionValues, expectedSplit.gate);
    const expectedGatedBackward = sigmoidGateBackward(
      attentionValues,
      expectedSplit.gate,
      gradOutputValues
    );
    return {
      actualGated: await readF32(gated),
      comparisons: {
        splitQuery: compare(await readF32(split.query), expectedSplit.query),
        splitGate: compare(await readF32(split.gate), expectedSplit.gate),
        splitBackward: compare(await readF32(splitBackward), expectedSplitBackward),
        sigmoidGatedForward: compare(await readF32(gated), expectedGated),
        sigmoidGatedBackwardInput: compare(
          await readF32(gatedBackward.input),
          expectedGatedBackward.input
        ),
        sigmoidGatedBackwardGate: compare(
          await readF32(gatedBackward.gate),
          expectedGatedBackward.gate
        ),
      },
    };
  } finally {
    if (split) {
      releaseBuffer(split.query.buffer);
      releaseBuffer(split.gate.buffer);
    }
    if (splitBackward) releaseBuffer(splitBackward.buffer);
    if (gated) releaseBuffer(gated.buffer);
    if (gatedBackward) {
      releaseBuffer(gatedBackward.input.buffer);
      releaseBuffer(gatedBackward.gate.buffer);
    }
    releaseBuffer(qGate.buffer);
    releaseBuffer(attention.buffer);
    releaseBuffer(gradOutput.buffer);
    releaseBuffer(gradQuery.buffer);
    releaseBuffer(splitGradGate.buffer);
  }
}

async function runGqaCase() {
  const options = {
    seqLen: 4,
    numHeads: 4,
    numKVHeads: 2,
    headDim: 3,
    scale: 1 / Math.sqrt(3),
    causal: true,
  };
  const queryValues = values(options.seqLen * options.numHeads * options.headDim, 5, 0.35);
  const keyValues = values(options.seqLen * options.numKVHeads * options.headDim, 37, 0.3);
  const valueValues = values(options.seqLen * options.numKVHeads * options.headDim, 61, 0.4);
  const gradOutputValues = values(
    options.seqLen * options.numHeads * options.headDim,
    83,
    0.25
  );
  const query = makeTensor(
    queryValues,
    [options.seqLen, options.numHeads, options.headDim],
    'full_attention_gqa_query'
  );
  const key = makeTensor(
    keyValues,
    [options.seqLen, options.numKVHeads, options.headDim],
    'full_attention_gqa_key'
  );
  const value = makeTensor(
    valueValues,
    [options.seqLen, options.numKVHeads, options.headDim],
    'full_attention_gqa_value'
  );
  const gradOutput = makeTensor(
    gradOutputValues,
    [options.seqLen, options.numHeads, options.headDim],
    'full_attention_gqa_grad_output'
  );
  let result = null;
  try {
    result = await runQwenGqaAttentionBackward(query, key, value, gradOutput, options);
    const softmax = computeAttentionSoftmaxData(queryValues, keyValues, options);
    const expected = computeAttentionBackwardData(
      queryValues,
      keyValues,
      valueValues,
      softmax,
      gradOutputValues,
      options
    );
    return {
      query: compare(await readF32(result.query), expected.dQ),
      key: compare(await readF32(result.key), expected.dK),
      value: compare(await readF32(result.value), expected.dV),
    };
  } finally {
    if (result) {
      releaseBuffer(result.query.buffer);
      releaseBuffer(result.key.buffer);
      releaseBuffer(result.value.buffer);
    }
    releaseBuffer(query.buffer);
    releaseBuffer(key.buffer);
    releaseBuffer(value.buffer);
    releaseBuffer(gradOutput.buffer);
  }
}

async function runPartialRopeCase() {
  const options = {
    numTokens: 3,
    numHeads: 2,
    headDim: 8,
    rotaryDim: 4,
    pairSpanDim: 4,
    interleaved: true,
    startPos: 0,
  };
  const inputValues = values(options.numTokens * options.numHeads * options.headDim, 5, 0.4);
  const gradOutputValues = values(inputValues.length, 37, 0.3);
  const cosValues = Float32Array.from(
    { length: options.numTokens * (options.rotaryDim / 2) },
    (_, index) => Math.cos(index * 0.17)
  );
  const sinValues = Float32Array.from(
    { length: cosValues.length },
    (_, index) => Math.sin(index * 0.17)
  );
  const input = makeTensor(
    inputValues,
    [options.numTokens, options.numHeads, options.headDim],
    'full_attention_rope_input'
  );
  const gradOutput = makeTensor(
    gradOutputValues,
    [options.numTokens, options.numHeads, options.headDim],
    'full_attention_rope_grad_output'
  );
  const cos = makeTensor(
    cosValues,
    [options.numTokens, options.rotaryDim / 2],
    'full_attention_rope_cos'
  );
  const sin = makeTensor(
    sinValues,
    [options.numTokens, options.rotaryDim / 2],
    'full_attention_rope_sin'
  );
  let backward = null;
  try {
    const forward = await runRoPE(input, cos, sin, options.numTokens, options);
    backward = await runRoPEBackward(gradOutput, cos, sin, {
      ...options,
      seqLen: options.numTokens,
    });
    return {
      forward: compare(
        await readF32(forward),
        partialRopeForward(inputValues, cosValues, sinValues, options)
      ),
      backward: compare(
        await readF32(backward),
        partialRopeBackward(gradOutputValues, cosValues, sinValues, options)
      ),
    };
  } finally {
    if (backward) releaseBuffer(backward.buffer);
    releaseBuffer(input.buffer);
    releaseBuffer(gradOutput.buffer);
    releaseBuffer(cos.buffer);
    releaseBuffer(sin.buffer);
  }
}

async function runIntegratedModuleCase(qAdapterPerturbation = 0) {
  const options = {
    seqLen: 2,
    hiddenSize: 8,
    numHeads: 2,
    numKVHeads: 1,
    headDim: 256,
    rotaryDim: 64,
    pairSpanDim: 64,
    interleaved: true,
    startPos: 0,
    rmsEps: 1e-6,
  };
  const querySize = options.numHeads * options.headDim;
  const kvSize = options.numKVHeads * options.headDim;
  const rank = 2;
  const alpha = 4;
  const weightBits = {
    qWeight: f32ToF16Array(values(querySize * 2 * options.hiddenSize, 3, 0.08)),
    kWeight: f32ToF16Array(values(kvSize * options.hiddenSize, 8201, 0.08)),
    vWeight: f32ToF16Array(values(kvSize * options.hiddenSize, 10253, 0.08)),
    oWeight: f32ToF16Array(values(options.hiddenSize * querySize, 12307, 0.08)),
    qNormWeight: f32ToF16Array(values(options.headDim, 16411, 0.04)),
    kNormWeight: f32ToF16Array(values(options.headDim, 16673, 0.04)),
  };
  const cosValues = Float32Array.from(
    { length: options.seqLen * (options.rotaryDim / 2) },
    (_, index) => Math.cos(index * 0.013)
  );
  const sinValues = Float32Array.from(
    { length: cosValues.length },
    (_, index) => Math.sin(index * 0.013)
  );
  const loraValues = {
    q: {
      A: values(options.hiddenSize * rank, 16741, 0.08),
      B: values(rank * querySize * 2, 16763, 0.06),
      rank,
      alpha,
    },
    k: {
      A: values(options.hiddenSize * rank, 16811, 0.08),
      B: values(rank * kvSize, 16829, 0.06),
      rank,
      alpha,
    },
    v: {
      A: values(options.hiddenSize * rank, 16843, 0.08),
      B: values(rank * kvSize, 16871, 0.06),
      rank,
      alpha,
    },
    o: {
      A: values(querySize * rank, 16889, 0.08),
      B: values(rank * options.hiddenSize, 16901, 0.06),
      rank,
      alpha,
    },
  };
  loraValues.q.B[1] += qAdapterPerturbation;
  const inputValues = {
    hidden: values(options.seqLen * options.hiddenSize, 16931, 0.2),
    qWeight: f16ToF32Array(weightBits.qWeight),
    kWeight: f16ToF32Array(weightBits.kWeight),
    vWeight: f16ToF32Array(weightBits.vWeight),
    oWeight: f16ToF32Array(weightBits.oWeight),
    qNormWeight: f16ToF32Array(weightBits.qNormWeight),
    kNormWeight: f16ToF32Array(weightBits.kNormWeight),
    cos: cosValues,
    sin: sinValues,
    lora: loraValues,
  };
  const referenceOptions = { ...options, numTokens: options.seqLen };
  const gradOutputValues = values(options.seqLen * options.hiddenSize, 17191, 0.25);
  const loraTensors = {
    q: {
      A: makeTensor(loraValues.q.A, [options.hiddenSize, rank], 'full_module_q_lora_a'),
      B: makeTensor(loraValues.q.B, [rank, querySize * 2], 'full_module_q_lora_b'),
      rank,
      alpha,
    },
    k: {
      A: makeTensor(loraValues.k.A, [options.hiddenSize, rank], 'full_module_k_lora_a'),
      B: makeTensor(loraValues.k.B, [rank, kvSize], 'full_module_k_lora_b'),
      rank,
      alpha,
    },
    v: {
      A: makeTensor(loraValues.v.A, [options.hiddenSize, rank], 'full_module_v_lora_a'),
      B: makeTensor(loraValues.v.B, [rank, kvSize], 'full_module_v_lora_b'),
      rank,
      alpha,
    },
    o: {
      A: makeTensor(loraValues.o.A, [querySize, rank], 'full_module_o_lora_a'),
      B: makeTensor(loraValues.o.B, [rank, options.hiddenSize], 'full_module_o_lora_b'),
      rank,
      alpha,
    },
  };
  const tensors = {
    hidden: makeTensor(inputValues.hidden, [options.seqLen, options.hiddenSize], 'full_module_hidden'),
    qWeight: makeTypedTensor(
      weightBits.qWeight,
      'f16',
      [querySize * 2, options.hiddenSize],
      'full_module_q_weight'
    ),
    kWeight: makeTypedTensor(
      weightBits.kWeight,
      'f16',
      [kvSize, options.hiddenSize],
      'full_module_k_weight'
    ),
    vWeight: makeTypedTensor(
      weightBits.vWeight,
      'f16',
      [kvSize, options.hiddenSize],
      'full_module_v_weight'
    ),
    oWeight: makeTypedTensor(
      weightBits.oWeight,
      'f16',
      [options.hiddenSize, querySize],
      'full_module_o_weight'
    ),
    qNormWeight: makeTypedTensor(
      weightBits.qNormWeight,
      'f16',
      [options.headDim],
      'full_module_q_norm_weight'
    ),
    kNormWeight: makeTypedTensor(
      weightBits.kNormWeight,
      'f16',
      [options.headDim],
      'full_module_k_norm_weight'
    ),
    cos: makeTensor(cosValues, [options.seqLen, options.rotaryDim / 2], 'full_module_cos'),
    sin: makeTensor(sinValues, [options.seqLen, options.rotaryDim / 2], 'full_module_sin'),
    lora: loraTensors,
  };
  const gradOutput = makeTensor(
    gradOutputValues,
    [options.seqLen, options.hiddenSize],
    'full_module_grad_output'
  );
  const expectedForward = qwenFullAttentionModuleForward(inputValues, referenceOptions);
  const expectedBackward = qwenFullAttentionModuleBackward(
    inputValues,
    gradOutputValues,
    expectedForward.cache,
    referenceOptions
  );
  let forward = null;
  let backward = null;
  try {
    forward = await runQwenFullAttentionTrainingModuleForward(tensors, options);
    backward = await runQwenFullAttentionTrainingModuleBackward(
      tensors,
      gradOutput,
      forward.cache,
      options
    );
    const actualOutput = await readF32(forward.output);
    const comparisons = {
      integratedModuleForward: compare(actualOutput, expectedForward.output),
      integratedModuleBackwardHidden: compare(
        await readF32(backward.hidden),
        expectedBackward.hidden
      ),
    };
    for (const projection of ['q', 'k', 'v', 'o']) {
      for (const matrix of ['A', 'B']) {
        const actualGradient = await readF32(backward.lora[projection][matrix]);
        const label = `integratedModuleBackwardLora${projection.toUpperCase()}${matrix}`;
        comparisons[label] = {
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
      releaseQwenFullAttentionTrainingModuleCache(forward.cache);
    }
    for (const [name, tensor] of Object.entries(tensors)) {
      if (name !== 'lora') releaseBuffer(tensor.buffer);
    }
    for (const adapter of Object.values(loraTensors)) {
      releaseBuffer(adapter.A.buffer);
      releaseBuffer(adapter.B.buffer);
    }
    releaseBuffer(gradOutput.buffer);
  }
}

export async function runQwenFullAttentionBackwardOracle() {
  const baseUrl = new URL('../../../src/config/', import.meta.url);
  setPlatformsBaseUrl(new URL('platforms/', baseUrl).toString());
  setRegistryUrl(new URL('kernels/registry.json', baseUrl).toString());
  await initDevice();

  const tolerance = 2e-5;
  const baseline = await executeCase(0);
  const perturbed = await executeCase(0.125);
  const gqa = await runGqaCase();
  const partialRope = await runPartialRopeCase();
  const integratedModule = await runIntegratedModuleCase();
  const perturbedIntegratedModule = await runIntegratedModuleCase(0.125);
  baseline.comparisons.gqaGradQuery = gqa.query;
  baseline.comparisons.gqaGradKey = gqa.key;
  baseline.comparisons.gqaGradValue = gqa.value;
  baseline.comparisons.partialInterleavedRopeForward = partialRope.forward;
  baseline.comparisons.partialInterleavedRopeBackward = partialRope.backward;
  Object.assign(baseline.comparisons, integratedModule.comparisons);
  const gatePerturbation = compare(perturbed.actualGated, baseline.actualGated);
  const adapterPerturbation = compare(
    perturbedIntegratedModule.actualOutput,
    integratedModule.actualOutput
  );
  const passed = Object.values(baseline.comparisons).every(
    (entry) => entry.allFinite
      && entry.maxAbsError <= tolerance
      && (entry.maxAbsValue == null || entry.maxAbsValue > 1e-10)
  ) && gatePerturbation.maxAbsError > 1e-4
    && adapterPerturbation.maxAbsError > 1e-7;
  const capabilities = getKernelCapabilities();
  return {
    artifactType: 'qwen_full_attention_component_backward_oracle',
    schemaVersion: 1,
    passed,
    tolerance: { maxAbsError: tolerance },
    comparisons: baseline.comparisons,
    negativeControl: {
      gatePerturbation: 'q_projection_gate_index_1_plus_0.125',
      gatedOutputDifference: gatePerturbation,
      adapterPerturbation: 'q_proj_lora_b_index_1_plus_0.125',
      moduleOutputDifference: adapterPerturbation,
      passed: gatePerturbation.maxAbsError > 1e-4
        && adapterPerturbation.maxAbsError > 1e-7,
    },
    adapterInfo: capabilities.adapterInfo || null,
    claimBoundary: 'Tiny integrated Qwen full-attention module with frozen F16 projections, Q/K/V/O LoRA gradients, offset Q/K RMSNorm, partial interleaved RoPE, sigmoid output gate, and causal GQA backward; residuals, MLP, complete decoder integration, and production performance remain absent.',
  };
}
