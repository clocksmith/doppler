import { setRegistryUrl } from '../../../src/config/kernels/registry.js';
import { setPlatformsBaseUrl } from '../../../src/config/platforms/loader.js';
import {
  captureQwenAdapterTrainingState,
  restoreQwenAdapterTrainingState,
  validateQwenAdapterTrainingState,
} from '../../../src/experimental/training/qwen-adapter-training-state.js';
import { QwenGradientAccumulator } from '../../../src/experimental/training/qwen-gradient-accumulator.js';
import { runQwenHybridSftMicrostep } from '../../../src/experimental/training/qwen-hybrid-sft-microstep.js';
import {
  createQwenSftBackendParityFixture,
} from '../../../src/experimental/training/qwen-sft-backend-parity-fixture.js';
import { AdamOptimizer } from '../../../src/experimental/training/optimizer.js';
import { getKernelCapabilities, initDevice } from '../../../src/gpu/device.js';
import { releaseBuffer } from '../../../src/memory/buffer-pool.js';
import {
  buildGpuFixture,
  buildQwenParityAdapterEntries,
  compare,
  makeTensorFactory,
  readF32,
  updateSimilarity,
  uploadQwenParityRow,
} from './qwen-gamma-sft-microstep-parity-oracle.js';

function createRun(fixture, trainingConfig, ownedRuns) {
  const ownedTensors = [];
  const makeTensor = makeTensorFactory(ownedTensors);
  const gpu = buildGpuFixture(makeTensor, fixture);
  const optimizer = new AdamOptimizer(trainingConfig);
  const accumulator = new QwenGradientAccumulator({
    accumSteps: fixture.prefixContract.accumulationSteps,
  });
  const run = {
    ownedTensors,
    makeTensor,
    gpu,
    entries: buildQwenParityAdapterEntries(gpu),
    optimizer,
    accumulator,
  };
  ownedRuns.push(run);
  return run;
}

async function runRows(run, fixture, rows, trainingConfig, label) {
  const losses = [];
  for (let index = 0; index < rows.length; index += 1) {
    const row = uploadQwenParityRow(
      run.makeTensor,
      fixture,
      rows[index],
      `${label}_${index}`
    );
    const result = await runQwenHybridSftMicrostep({
      tokenIds: row.tokenIds,
      embeddingWeight: run.gpu.embedding,
      layers: [run.gpu.layer],
      finalNormWeight: run.gpu.finalNorm,
      lmHeadWeight: run.gpu.lmHead,
      targets: row.targets,
    }, {
      ...fixture.model,
      applyOptimizer: false,
      gradientAccumulator: run.accumulator,
      trainingConfig,
    });
    losses.push(result.meanLoss);
    if (run.accumulator.ready) {
      await run.accumulator.step(run.optimizer, trainingConfig);
    }
  }
  if (run.accumulator.microstepCount !== 0) {
    throw new Error(`${label} ended with a partial accumulation window.`);
  }
  return losses;
}

async function captureLiveState(run) {
  const tensors = {};
  for (const entry of run.entries) {
    const state = run.optimizer.getState(entry.parameter);
    tensors[entry.name] = {
      parameter: await readF32(entry.parameter),
      moment1: await readF32(state.m),
      moment2: await readF32(state.v),
    };
  }
  return {
    optimizerStepCount: run.optimizer.stepCount,
    tensors,
  };
}

function compareStates(actual, expected, suffix, comparisons) {
  for (const [name, tensor] of Object.entries(actual.tensors)) {
    const reference = expected.tensors[name];
    if (!reference) {
      throw new Error(`Qwen prefix reference is missing ${name}.`);
    }
    comparisons[`${name}.parameter.${suffix}`] = compare(
      tensor.parameter,
      reference.parameter
    );
    comparisons[`${name}.moment1.${suffix}`] = compare(
      tensor.moment1,
      reference.moment1
    );
    comparisons[`${name}.moment2.${suffix}`] = compare(
      tensor.moment2,
      reference.moment2
    );
  }
}

function compareLosses(actual, expected) {
  return compare(new Float32Array(actual), new Float32Array(expected));
}

export async function runQwenGammaSftPrefixParityOracle(input) {
  if (!input?.gammaReference || !input?.gammaIdentity) {
    throw new Error('Qwen Gamma prefix parity oracle requires Gamma reference data and identity.');
  }
  const baseUrl = new URL('../../../src/config/', import.meta.url);
  setPlatformsBaseUrl(new URL('platforms/', baseUrl).toString());
  setRegistryUrl(new URL('kernels/registry.json', baseUrl).toString());
  await initDevice();

  const fixture = createQwenSftBackendParityFixture({ rank: 32, alpha: 64 });
  const gamma = input.gammaReference;
  if (gamma.rank !== 32 || gamma.alpha !== 64 || gamma.parameterCount !== 14) {
    throw new Error('Gamma prefix reference does not match the frozen rank-32 adapter contract.');
  }
  if (JSON.stringify(gamma.architectureContract) !== JSON.stringify(fixture.architectureContract)) {
    throw new Error('Gamma prefix reference does not match the pinned Qwen architecture contract.');
  }
  if (gamma.consumedPrefixSha256 !== fixture.prefixContract.consumedPrefixSha256
    || gamma.accumulationSteps !== fixture.prefixContract.accumulationSteps) {
    throw new Error('Gamma prefix reference does not match the frozen row-order contract.');
  }

  const optimizerOptions = {
    ...fixture.optimizer,
    scheduler: { enabled: false },
  };
  const trainingConfig = { training: { optimizer: optimizerOptions } };
  const ownedRuns = [];
  const continuous = createRun(fixture, trainingConfig, ownedRuns);
  const beforeCheckpoint = createRun(fixture, trainingConfig, ownedRuns);
  let resumed = null;
  try {
    const continuousLosses = await runRows(
      continuous,
      fixture,
      fixture.prefixRows,
      trainingConfig,
      'continuous'
    );
    const checkpointRows = fixture.prefixRows.slice(
      0,
      fixture.prefixContract.checkpointAfterMicrostep
    );
    const remainingRows = fixture.prefixRows.slice(
      fixture.prefixContract.checkpointAfterMicrostep
    );
    const checkpointLosses = await runRows(
      beforeCheckpoint,
      fixture,
      checkpointRows,
      trainingConfig,
      'before_checkpoint'
    );
    const checkpointLive = await captureLiveState(beforeCheckpoint);
    const checkpoint = await captureQwenAdapterTrainingState(
      beforeCheckpoint.entries,
      beforeCheckpoint.optimizer,
      {
        microstepCount: checkpointRows.length,
        consumedRowIds: checkpointRows.map((row) => row.rowId),
      }
    );
    let tamperedCheckpointRejected = false;
    try {
      validateQwenAdapterTrainingState({
        ...checkpoint,
        payloadSha256: '0'.repeat(64),
      });
    } catch {
      tamperedCheckpointRejected = true;
    }

    resumed = createRun(fixture, trainingConfig, ownedRuns);
    const restoredProgress = await restoreQwenAdapterTrainingState(
      resumed.entries,
      resumed.optimizer,
      checkpoint
    );
    const resumedLosses = await runRows(
      resumed,
      fixture,
      remainingRows,
      trainingConfig,
      'after_resume'
    );
    const continuousState = await captureLiveState(continuous);
    const resumedState = await captureLiveState(resumed);
    const comparisons = {
      lossesVsGamma: compareLosses(continuousLosses, gamma.losses),
      lossesResumeVsContinuous: compareLosses(
        [...checkpointLosses, ...resumedLosses],
        continuousLosses
      ),
    };
    compareStates(continuousState, gamma, 'continuousVsGamma', comparisons);
    compareStates(resumedState, continuousState, 'resumeVsContinuous', comparisons);
    compareStates(checkpointLive, gamma.checkpoint, 'checkpointVsGamma', comparisons);

    const initial = Object.fromEntries(
      Object.entries(fixture.adapters).flatMap(([prefix, adapter]) => [
        [`${prefix}.lora_A`, new Float32Array(adapter.A.data)],
        [`${prefix}.lora_B`, new Float32Array(adapter.B.data)],
      ])
    );
    const updateSimilarities = {};
    for (const [name, tensor] of Object.entries(continuousState.tensors)) {
      updateSimilarities[name] = updateSimilarity(
        tensor.parameter,
        new Float32Array(gamma.tensors[name].parameter),
        initial[name]
      );
    }
    const maximumAbsError = Math.max(
      ...Object.values(comparisons).map((value) => value.maxAbsError)
    );
    const resumeComparisons = Object.entries(comparisons)
      .filter(([name]) => name.endsWith('resumeVsContinuous'))
      .map(([, value]) => value);
    const maximumResumeAbsError = Math.max(
      comparisons.lossesResumeVsContinuous.maxAbsError,
      ...resumeComparisons.map((value) => value.maxAbsError)
    );
    const minimumUpdateCosine = Math.min(
      ...Object.values(updateSimilarities).map((value) => value.cosine)
    );
    const maximumUpdateRelativeL2Error = Math.max(
      ...Object.values(updateSimilarities).map((value) => value.relativeL2Error)
    );
    const everyParameterChanged = Object.values(updateSimilarities).every(
      (value) => value.actualUpdateL2 > 0
    );
    const thresholds = {
      maximumAbsError: 1e-4,
      maximumResumeAbsError: 1e-7,
      minimumAdapterUpdateCosine: 0.999,
      maximumAdapterUpdateRelativeL2Error: 0.02,
    };
    const passed = continuousState.optimizerStepCount === 2
      && resumedState.optimizerStepCount === 2
      && restoredProgress.microstepCount === checkpointRows.length
      && restoredProgress.optimizerStepCount === 1
      && restoredProgress.consumedPrefixSha256 === checkpoint.progress.consumedPrefixSha256
      && checkpoint.payloadSha256.length === 64
      && tamperedCheckpointRejected
      && gamma.optimizerStepCount === 2
      && gamma.allGradientsNonzero === true
      && gamma.everyParameterChanged === true
      && everyParameterChanged
      && Object.values(comparisons).every(
        (value) => value.allFinite && value.maxAbsError <= thresholds.maximumAbsError
      )
      && maximumResumeAbsError <= thresholds.maximumResumeAbsError
      && minimumUpdateCosine >= thresholds.minimumAdapterUpdateCosine
      && maximumUpdateRelativeL2Error <= thresholds.maximumAdapterUpdateRelativeL2Error;
    const capabilities = getKernelCapabilities();
    return {
      artifactType: 'qwen_gamma_sft_prefix_parity_oracle',
      schemaVersion: 1,
      passed,
      fixtureSha256: gamma.fixtureSha256,
      rank: gamma.rank,
      alpha: gamma.alpha,
      accumulationSteps: fixture.prefixContract.accumulationSteps,
      microstepCount: fixture.prefixRows.length,
      optimizerStepCount: continuousState.optimizerStepCount,
      consumedRowIds: fixture.prefixRows.map((row) => row.rowId),
      consumedPrefixSha256: fixture.prefixContract.consumedPrefixSha256,
      checkpoint: {
        afterMicrostep: checkpointRows.length,
        optimizerStepCount: checkpointLive.optimizerStepCount,
        payloadSha256: checkpoint.payloadSha256,
        consumedPrefixSha256: checkpoint.progress.consumedPrefixSha256,
        restoredProgress,
        tamperedCheckpointRejected,
      },
      everyParameterChanged,
      thresholds,
      maximumAbsError,
      maximumResumeAbsError,
      minimumUpdateCosine,
      maximumUpdateRelativeL2Error,
      comparisons,
      updateSimilarities,
      gammaIdentity: input.gammaIdentity,
      gammaReferenceImplementation: gamma.referenceImplementation,
      dopplerAdapterInfo: capabilities.adapterInfo || null,
      precisionContract: fixture.precisionContract,
      architectureContract: fixture.architectureContract,
      claimBoundary: 'Deterministic tiny four-row rank-32 matched-prefix parity: Doppler WebGPU/Vulkan and Gamma PyTorch/ROCm consume identical rows through two-step gradient accumulation and AdamW, while a hashed Doppler adapter/moment checkpoint reproduces uninterrupted Doppler state. This is not production Qwen geometry, partial-window resume, durable large-state I/O performance, V12 training, compiler capability, or semantic WGSL evidence.',
    };
  } finally {
    for (const run of ownedRuns) {
      run.accumulator.dispose();
      for (const state of run.optimizer.state.values()) {
        releaseBuffer(state.m.buffer);
        releaseBuffer(state.v.buffer);
      }
      for (const tensor of run.ownedTensors) releaseBuffer(tensor.buffer);
    }
  }
}
