import { setRegistryUrl } from '../../../src/config/kernels/registry.js';
import { setPlatformsBaseUrl } from '../../../src/config/platforms/loader.js';
import { runQwenHybridSftMicrostep } from '../../../src/experimental/training/qwen-hybrid-sft-microstep.js';
import {
  createQwenSftBackendParityFixture,
} from '../../../src/experimental/training/qwen-sft-backend-parity-fixture.js';
import { AdamOptimizer } from '../../../src/experimental/training/optimizer.js';
import { getKernelCapabilities, initDevice } from '../../../src/gpu/device.js';
import { createTensor } from '../../../src/gpu/tensor.js';
import { f32ToF16Array } from '../../../src/inference/kv-cache/types.js';
import { acquireBuffer, readBuffer, releaseBuffer, uploadData } from '../../../src/memory/buffer-pool.js';

const MASKED_TARGET = 0xffffffff;

export function makeTensorFactory(ownedTensors) {
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

export async function readF32(tensor) {
  const count = tensor.shape.reduce((product, value) => product * value, 1);
  return new Float32Array(await readBuffer(tensor.buffer, count * 4));
}

export function compare(actual, expected) {
  if (actual.length !== expected.length) {
    throw new Error(`Parity array length mismatch: ${actual.length} != ${expected.length}.`);
  }
  let maxAbsError = 0;
  let squaredError = 0;
  let allFinite = true;
  for (let index = 0; index < expected.length; index += 1) {
    const error = Math.abs(actual[index] - expected[index]);
    maxAbsError = Math.max(maxAbsError, error);
    squaredError += error * error;
    allFinite = allFinite && Number.isFinite(actual[index]) && Number.isFinite(expected[index]);
  }
  return {
    elementCount: expected.length,
    allFinite,
    maxAbsError,
    rmse: Math.sqrt(squaredError / expected.length),
  };
}

export function updateSimilarity(actual, expected, initial) {
  let dot = 0;
  let actualSquared = 0;
  let expectedSquared = 0;
  let differenceSquared = 0;
  for (let index = 0; index < initial.length; index += 1) {
    const actualUpdate = actual[index] - initial[index];
    const expectedUpdate = expected[index] - initial[index];
    dot += actualUpdate * expectedUpdate;
    actualSquared += actualUpdate * actualUpdate;
    expectedSquared += expectedUpdate * expectedUpdate;
    const difference = actualUpdate - expectedUpdate;
    differenceSquared += difference * difference;
  }
  return {
    cosine: dot / Math.max(Math.sqrt(actualSquared * expectedSquared), 1e-30),
    relativeL2Error: Math.sqrt(differenceSquared) / Math.max(Math.sqrt(expectedSquared), 1e-30),
    actualUpdateL2: Math.sqrt(actualSquared),
    expectedUpdateL2: Math.sqrt(expectedSquared),
  };
}

function uploadFrozen(makeTensor, fixture, name, dtype = 'f16') {
  const spec = fixture.frozen[name];
  const values = new Float32Array(spec.data);
  return dtype === 'f16'
    ? makeTensor(f32ToF16Array(values), 'f16', spec.shape, `parity_${name}`)
    : makeTensor(values, 'f32', spec.shape, `parity_${name}`);
}

function uploadAdapter(makeTensor, spec, label) {
  return {
    A: makeTensor(new Float32Array(spec.A.data), 'f32', spec.A.shape, `${label}_a`),
    B: makeTensor(new Float32Array(spec.B.data), 'f32', spec.B.shape, `${label}_b`),
    rank: spec.rank,
    alpha: spec.alpha,
  };
}

export function buildGpuFixture(makeTensor, fixture) {
  const byPath = Object.fromEntries(
    Object.entries(fixture.adapters).map(([name, spec]) => [
      name,
      uploadAdapter(makeTensor, spec, name.replaceAll('.', '_')),
    ])
  );
  const attention = {
    q: byPath['layers.0.self_attn.q_proj'],
    k: byPath['layers.0.self_attn.k_proj'],
    v: byPath['layers.0.self_attn.v_proj'],
    o: byPath['layers.0.self_attn.o_proj'],
  };
  const mlp = {
    gate: byPath['layers.0.mlp.gate_proj'],
    up: byPath['layers.0.mlp.up_proj'],
    down: byPath['layers.0.mlp.down_proj'],
  };
  const targets = Uint32Array.from(
    fixture.targets,
    (value) => value < 0 ? MASKED_TARGET : value
  );
  return {
    tokenIds: makeTensor(new Uint32Array(fixture.tokenIds), 'u32', [fixture.model.numTokens], 'parity_tokens'),
    targets: makeTensor(targets, 'u32', [fixture.model.numTokens], 'parity_targets'),
    embedding: uploadFrozen(makeTensor, fixture, 'embedding'),
    finalNorm: uploadFrozen(makeTensor, fixture, 'finalNorm'),
    lmHead: uploadFrozen(makeTensor, fixture, 'lmHead'),
    adapters: byPath,
    layer: {
      type: 'full_attention',
      options: fixture.layer,
      inputs: {
        inputNormWeight: uploadFrozen(makeTensor, fixture, 'inputNorm'),
        postAttentionNormWeight: uploadFrozen(makeTensor, fixture, 'postAttentionNorm'),
        attention: {
          qWeight: uploadFrozen(makeTensor, fixture, 'qWeight'),
          kWeight: uploadFrozen(makeTensor, fixture, 'kWeight'),
          vWeight: uploadFrozen(makeTensor, fixture, 'vWeight'),
          oWeight: uploadFrozen(makeTensor, fixture, 'oWeight'),
          qNormWeight: uploadFrozen(makeTensor, fixture, 'qNorm'),
          kNormWeight: uploadFrozen(makeTensor, fixture, 'kNorm'),
          cos: uploadFrozen(makeTensor, fixture, 'cosine', 'f32'),
          sin: uploadFrozen(makeTensor, fixture, 'sine', 'f32'),
          lora: attention,
        },
        mlp: {
          gateWeight: uploadFrozen(makeTensor, fixture, 'gateWeight'),
          upWeight: uploadFrozen(makeTensor, fixture, 'upWeight'),
          downWeight: uploadFrozen(makeTensor, fixture, 'downWeight'),
          lora: mlp,
        },
      },
    },
  };
}

export function buildQwenParityAdapterEntries(gpu) {
  return Object.entries(gpu.adapters).flatMap(([prefix, adapter]) => [
    { name: `${prefix}.lora_A`, parameter: adapter.A },
    { name: `${prefix}.lora_B`, parameter: adapter.B },
  ]);
}

export function uploadQwenParityRow(makeTensor, fixture, row, label) {
  const targets = Uint32Array.from(
    row.targets,
    (value) => value < 0 ? MASKED_TARGET : value
  );
  return {
    rowId: row.rowId,
    tokenIds: makeTensor(
      new Uint32Array(row.tokenIds),
      'u32',
      [fixture.model.numTokens],
      `${label}_tokens`
    ),
    targets: makeTensor(
      targets,
      'u32',
      [fixture.model.numTokens],
      `${label}_targets`
    ),
  };
}

export async function runQwenGammaSftMicrostepParityOracle(input) {
  if (!input?.gammaReference || !input?.gammaIdentity) {
    throw new Error('Qwen Gamma parity oracle requires Gamma reference data and identity.');
  }
  const baseUrl = new URL('../../../src/config/', import.meta.url);
  setPlatformsBaseUrl(new URL('platforms/', baseUrl).toString());
  setRegistryUrl(new URL('kernels/registry.json', baseUrl).toString());
  await initDevice();

  const fixture = createQwenSftBackendParityFixture({ rank: 32, alpha: 64 });
  const gamma = input.gammaReference;
  if (gamma.rank !== 32 || gamma.alpha !== 64 || gamma.parameterCount !== 14) {
    throw new Error('Gamma reference does not match the frozen rank-32 adapter contract.');
  }
  if (JSON.stringify(gamma.architectureContract) !== JSON.stringify(fixture.architectureContract)) {
    throw new Error('Gamma reference does not match the pinned Qwen architecture contract.');
  }
  const ownedTensors = [];
  const makeTensor = makeTensorFactory(ownedTensors);
  const gpu = buildGpuFixture(makeTensor, fixture);
  const optimizerOptions = {
    ...fixture.optimizer,
    scheduler: { enabled: false },
  };
  const trainingConfig = { training: { optimizer: optimizerOptions } };
  const optimizer = new AdamOptimizer(trainingConfig);
  const initial = Object.fromEntries(
    Object.entries(fixture.adapters).flatMap(([prefix, adapter]) => [
      [`${prefix}.lora_A`, new Float32Array(adapter.A.data)],
      [`${prefix}.lora_B`, new Float32Array(adapter.B.data)],
    ])
  );
  try {
    const actual = await runQwenHybridSftMicrostep({
      tokenIds: gpu.tokenIds,
      embeddingWeight: gpu.embedding,
      layers: [gpu.layer],
      finalNormWeight: gpu.finalNorm,
      lmHeadWeight: gpu.lmHead,
      targets: gpu.targets,
    }, {
      ...fixture.model,
      optimizer,
      trainingConfig,
      captureGradients: true,
    });
    if (JSON.stringify(actual.parameterNames) !== JSON.stringify(gamma.parameterNames)) {
      throw new Error('Doppler and Gamma parameter order differs.');
    }
    const comparisons = {
      meanLoss: {
        elementCount: 1,
        allFinite: Number.isFinite(actual.meanLoss) && Number.isFinite(gamma.meanLoss),
        maxAbsError: Math.abs(actual.meanLoss - gamma.meanLoss),
        rmse: Math.abs(actual.meanLoss - gamma.meanLoss),
      },
    };
    const updateSimilarities = {};
    for (const name of actual.parameterNames) {
      const prefix = name.replace(/\.lora_[AB]$/, '');
      const kind = name.endsWith('lora_A') ? 'A' : 'B';
      const tensor = gpu.adapters[prefix][kind];
      const parameter = await readF32(tensor);
      const state = optimizer.getState(tensor);
      const reference = gamma.tensors[name];
      comparisons[`${name}.gradient`] = compare(
        actual.gradientSnapshots[name], reference.gradient
      );
      comparisons[`${name}.parameter`] = compare(parameter, reference.parameter);
      comparisons[`${name}.moment1`] = compare(await readF32(state.m), reference.moment1);
      comparisons[`${name}.moment2`] = compare(await readF32(state.v), reference.moment2);
      updateSimilarities[name] = updateSimilarity(
        parameter,
        reference.parameter,
        initial[name]
      );
    }
    const maximumAbsError = Math.max(
      ...Object.values(comparisons).map((comparison) => comparison.maxAbsError)
    );
    const minimumUpdateCosine = Math.min(
      ...Object.values(updateSimilarities).map((comparison) => comparison.cosine)
    );
    const maximumUpdateRelativeL2Error = Math.max(
      ...Object.values(updateSimilarities).map((comparison) => comparison.relativeL2Error)
    );
    const allGradientsNonzero = Object.values(actual.gradientSnapshots).every(
      (gradient) => gradient.some((value) => value !== 0)
    );
    const everyParameterChanged = Object.entries(gpu.adapters).every(([prefix, adapter]) => (
      ['A', 'B'].every((kind) => {
        const name = `${prefix}.lora_${kind}`;
        return updateSimilarities[name].actualUpdateL2 > 0;
      })
    ));
    const thresholds = {
      maximumAbsError: 5e-5,
      minimumAdapterUpdateCosine: 0.999,
      maximumAdapterUpdateRelativeL2Error: 0.02,
    };
    const maskDifference = Math.abs(gamma.meanLoss - gamma.unmaskedMeanLoss);
    const passed = actual.activeTokenCount === fixture.model.activeTokenCount
      && optimizer.stepCount === 1
      && allGradientsNonzero
      && everyParameterChanged
      && gamma.allGradientsNonzero === true
      && gamma.everyParameterChanged === true
      && Object.values(comparisons).every(
        (comparison) => comparison.allFinite
          && comparison.maxAbsError <= thresholds.maximumAbsError
      )
      && minimumUpdateCosine >= thresholds.minimumAdapterUpdateCosine
      && maximumUpdateRelativeL2Error <= thresholds.maximumAdapterUpdateRelativeL2Error
      && maskDifference > 1e-4;
    const capabilities = getKernelCapabilities();
    return {
      artifactType: 'qwen_gamma_sft_microstep_parity_oracle',
      schemaVersion: 1,
      passed,
      fixtureSha256: gamma.fixtureSha256,
      rank: gamma.rank,
      alpha: gamma.alpha,
      activeTokenCount: actual.activeTokenCount,
      parameterCount: actual.parameterNames.length,
      parameterNames: actual.parameterNames,
      optimizerStepCount: optimizer.stepCount,
      allGradientsNonzero,
      everyParameterChanged,
      thresholds,
      maximumAbsError,
      minimumUpdateCosine,
      maximumUpdateRelativeL2Error,
      comparisons,
      updateSimilarities,
      negativeControl: {
        control: 'gamma_same_logits_with_prompt_token_included_in_loss_mean',
        maskedVsUnmaskedMeanLossDifference: maskDifference,
        passed: maskDifference > 1e-4,
      },
      gammaIdentity: input.gammaIdentity,
      gammaReferenceImplementation: gamma.referenceImplementation,
      dopplerAdapterInfo: capabilities.adapterInfo || null,
      precisionContract: fixture.precisionContract,
      architectureContract: fixture.architectureContract,
      claimBoundary: 'Tiny one-full-layer rank-32, zero-dropout, token-aligned completion-masked microstep parity between the Transformers Qwen3_5DecoderLayer on Gamma PyTorch/ROCm and Doppler WebGPU/Vulkan, using the pinned residual-before-post-attention-norm and split-half partial-RoPE contracts. This is not production Qwen geometry, PEFT default initialization, V12 dropout, accumulation/resume, compiler capability, or semantic WGSL evidence.',
    };
  } finally {
    for (const state of optimizer.state.values()) {
      releaseBuffer(state.m.buffer);
      releaseBuffer(state.v.buffer);
    }
    for (const tensor of ownedTensors) releaseBuffer(tensor.buffer);
  }
}
