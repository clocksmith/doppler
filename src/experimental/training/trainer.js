import { AutogradTape } from './autograd.js';
import { loadBackwardRegistry } from '../../config/backward-registry-loader.js';
import { runScale } from '../../gpu/kernels/index.js';
import { releaseBuffer } from '../../memory/buffer-pool.js';
import { createCrossEntropyObjective } from './objectives/cross_entropy.js';
import { createUploadedTensor } from './tensor-factory.js';

function createLossGradient(loss, lossScale) {
  const lossElements = loss.shape.reduce((acc, value) => acc * value, 1);
  const gradData = new Float32Array(lossElements);
  gradData.fill(lossScale);
  return createUploadedTensor(gradData, 'f32', loss.shape, 'loss_grad_output');
}

function normalizeLossResult(value) {
  if (value && typeof value === 'object' && value.loss) {
    return value;
  }
  return { loss: value, components: null };
}

function resolveTrainingObjective(options) {
  if (options.trainingObjective) {
    return options.trainingObjective;
  }
  return createCrossEntropyObjective({ crossEntropyLoss: options.crossEntropyLoss });
}

function resolveModelParamGroups(model) {
  if (model && typeof model.paramGroups === 'function') {
    const groups = model.paramGroups();
    if (!groups || typeof groups !== 'object') {
      throw new Error('model.paramGroups() must return an object of tensor arrays.');
    }
    return groups;
  }
  if (model && typeof model.loraParams === 'function') {
    return { lora: model.loraParams() };
  }
  return {};
}

function selectTrainableParamGroups(paramGroups, freezeMap) {
  const trainableGroups = {};
  const frozenGroups = [];
  for (const [groupName, params] of Object.entries(paramGroups)) {
    const normalizedParams = Array.isArray(params) ? params.filter(Boolean) : [];
    if (freezeMap?.[groupName] === true) {
      frozenGroups.push(groupName);
      continue;
    }
    trainableGroups[groupName] = normalizedParams;
  }
  return { trainableGroups, frozenGroups };
}

function flattenUniqueParams(paramGroups) {
  const unique = new Set();
  const output = [];
  for (const params of Object.values(paramGroups)) {
    for (const tensor of params) {
      if (!tensor || unique.has(tensor)) continue;
      unique.add(tensor);
      output.push(tensor);
    }
  }
  return output;
}

function isTensorLike(value) {
  return !!value
    && typeof value === 'object'
    && Array.isArray(value.shape)
    && value.buffer != null;
}

function normalizeBackwardSeedPayload(payload, loss, lossScale) {
  const collectSeedTensors = (value) => {
    const seedTensors = [];
    if (value instanceof Map) {
      for (const grad of value.values()) {
        if (isTensorLike(grad)) {
          seedTensors.push(grad);
        }
      }
      return seedTensors;
    }
    if (Array.isArray(value)) {
      for (const seed of value) {
        if (isTensorLike(seed?.grad)) {
          seedTensors.push(seed.grad);
        }
      }
      return seedTensors;
    }
    if (value && typeof value === 'object' && Array.isArray(value.seeds)) {
      for (const seed of value.seeds) {
        if (isTensorLike(seed?.grad)) {
          seedTensors.push(seed.grad);
        }
      }
      return seedTensors;
    }
    if (isTensorLike(value)) {
      seedTensors.push(value);
    }
    return seedTensors;
  };

  if (payload instanceof Map) {
    return { seedArg: payload, seedTensors: collectSeedTensors(payload) };
  }
  if (Array.isArray(payload)) {
    return { seedArg: payload, seedTensors: collectSeedTensors(payload) };
  }
  if (payload && typeof payload === 'object' && Array.isArray(payload.seeds)) {
    return { seedArg: payload, seedTensors: collectSeedTensors(payload) };
  }
  if (isTensorLike(payload)) {
    return { seedArg: payload, seedTensors: [payload] };
  }
  const fallback = createLossGradient(loss, lossScale);
  return { seedArg: fallback, seedTensors: [fallback] };
}

function filterGradientsForParams(grads, params) {
  const filtered = new Map();
  for (const param of params) {
    const grad = grads.get(param);
    if (grad) {
      filtered.set(param, grad);
    }
  }
  return filtered;
}

export async function trainStep(
  model,
  batch,
  config,
  options = {}
) {
  const {
    registry = loadBackwardRegistry(),
    crossEntropyLoss,
    clipGradients,
    optimizer,
    lossScale = 1,
    applyClip = true,
    applyOptimizer = true,
    trainingObjective = null,
  } = options;

  if (!crossEntropyLoss || !clipGradients || !optimizer) {
    throw new Error('trainStep requires crossEntropyLoss, clipGradients, and optimizer');
  }

  const objective = resolveTrainingObjective({
    trainingObjective,
    crossEntropyLoss,
  });
  const tape = new AutogradTape(registry);
  let preparedBatch = batch;
  if (typeof objective.prepareBatch === 'function') {
    const nextBatch = await objective.prepareBatch({
      model,
      batch,
      config,
      tape,
      options,
      lossScale,
    });
    if (nextBatch && typeof nextBatch === 'object') {
      preparedBatch = nextBatch;
    }
  }

  const t0 = globalThis.performance.now();
  const forwardState = await objective.forward({
    model,
    batch: preparedBatch,
    config,
    tape,
    options,
    lossScale,
  });
  const lossResult = normalizeLossResult(await objective.computeLoss({
    model,
    batch: preparedBatch,
    config,
    tape,
    options,
    lossScale,
    forwardState,
  }));
  const loss = lossResult.loss;
  const gradOutput = typeof objective.backwardTargets === 'function'
    ? await objective.backwardTargets({
      model,
      batch: preparedBatch,
      config,
      tape,
      options,
      lossScale,
      loss,
      lossResult,
      forwardState,
    })
    : createLossGradient(loss, lossScale);
  const forward_ms = globalThis.performance.now() - t0;

  const t1 = globalThis.performance.now();
  const backwardSeed = normalizeBackwardSeedPayload(gradOutput, loss, lossScale);
  const grads = await tape.backward(backwardSeed.seedArg);
  const backward_ms = globalThis.performance.now() - t1;
  for (const seedTensor of backwardSeed.seedTensors) {
    if (seedTensor?.buffer) {
      releaseBuffer(seedTensor.buffer);
    }
  }
  let processed = grads;
  if (lossScale !== 1) {
    const invScale = 1 / lossScale;
    const unscaled = new Map();
    for (const [param, grad] of grads.entries()) {
      const scaled = await runScale(grad, invScale, { inplace: true });
      unscaled.set(param, scaled);
    }
    processed = unscaled;
  }
  const paramGroups = resolveModelParamGroups(model);
  const freezeMap = config.training?.ul?.freeze
    ?? config.training?.distill?.freeze
    ?? {};
  const { trainableGroups, frozenGroups } = selectTrainableParamGroups(paramGroups, freezeMap);
  const trainableParams = flattenUniqueParams(trainableGroups);
  processed = filterGradientsForParams(processed, trainableParams);

  let clipMetrics = null;
  if (applyClip) {
    clipMetrics = await clipGradients(processed, config);
    processed = clipMetrics.clippedGrads;
  }

  let optimizerMetrics = null;
  let paramGroupMetrics = null;
  if (applyOptimizer) {
    optimizerMetrics = await optimizer.step(trainableParams, processed, config, {
      trainableGroups: Object.keys(trainableGroups),
      frozenGroups,
      allGroups: Object.keys(paramGroups),
    });
    paramGroupMetrics = {
      trainableGroups: Object.keys(trainableGroups),
      frozenGroups,
      allGroups: Object.keys(paramGroups),
      trainableParamCount: trainableParams.length,
    };
  }

  let objectiveMetrics = lossResult.components || null;
  if (typeof objective.metrics === 'function') {
    objectiveMetrics = await objective.metrics({
      model,
      batch: preparedBatch,
      config,
      tape,
      options,
      lossScale,
      loss,
      lossResult,
      forwardState,
    });
  }

  if (typeof objective.cleanup === 'function') {
    await objective.cleanup({
      model,
      batch,
      preparedBatch,
      config,
      tape,
      options,
      lossScale,
    });
  }

  return {
    loss,
    grads: processed,
    forward_ms,
    backward_ms,
    clipMetrics,
    optimizerMetrics,
    objectiveName: objective.name,
    objectiveMetrics,
    paramGroupMetrics,
  };
}
