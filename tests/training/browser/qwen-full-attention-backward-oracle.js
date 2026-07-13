import { setRegistryUrl } from '../../../src/config/kernels/registry.js';
import { setPlatformsBaseUrl } from '../../../src/config/platforms/loader.js';
import {
  qwenAttentionSplitQGateBackward,
  qwenAttentionSplitQGateForward,
  sigmoidGateBackward,
  sigmoidGateForward,
} from '../../../src/experimental/training/qwen-full-attention-reference.js';
import { getKernelCapabilities, initDevice } from '../../../src/gpu/device.js';
import {
  runQwenAttentionSplitQGate,
  runSiLU,
} from '../../../src/gpu/kernels/index.js';
import {
  runQwenAttentionSplitQGateBackward,
  runSigmoidGatedBackward,
} from '../../../src/gpu/kernels/backward/index.js';
import { createTensor } from '../../../src/gpu/tensor.js';
import { acquireBuffer, readBuffer, releaseBuffer, uploadData } from '../../../src/memory/buffer-pool.js';

function values(length, offset, scale) {
  return Float32Array.from(
    { length },
    (_, index) => Math.sin((index + offset) * 0.47) * scale
  );
}

function makeTensor(data, shape, label) {
  const buffer = acquireBuffer(data.byteLength, undefined, label);
  uploadData(buffer, data);
  return createTensor(buffer, 'f32', shape, label);
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

export async function runQwenFullAttentionBackwardOracle() {
  const baseUrl = new URL('../../../src/config/', import.meta.url);
  setPlatformsBaseUrl(new URL('platforms/', baseUrl).toString());
  setRegistryUrl(new URL('kernels/registry.json', baseUrl).toString());
  await initDevice();

  const tolerance = 2e-5;
  const baseline = await executeCase(0);
  const perturbed = await executeCase(0.125);
  const perturbation = compare(perturbed.actualGated, baseline.actualGated);
  const passed = Object.values(baseline.comparisons).every(
    (entry) => entry.allFinite && entry.maxAbsError <= tolerance
  ) && perturbation.maxAbsError > 1e-4;
  const capabilities = getKernelCapabilities();
  return {
    artifactType: 'qwen_full_attention_component_backward_oracle',
    schemaVersion: 1,
    passed,
    tolerance: { maxAbsError: tolerance },
    comparisons: baseline.comparisons,
    negativeControl: {
      perturbation: 'q_projection_gate_index_1_plus_0.125',
      gatedOutputDifference: perturbation,
      passed: perturbation.maxAbsError > 1e-4,
    },
    adapterInfo: capabilities.adapterInfo || null,
    claimBoundary: 'Qwen per-head query/output-gate split and sigmoid output-gate GPU forward/backward only; Q/K norm, RoPE, GQA, projections, LoRA, and full layer integration remain absent.',
  };
}
