import { crossEntropyLoss as defaultCrossEntropyLoss } from '../loss.js';
import { createTrainingObjective } from './base.js';
import { releaseBuffer } from '../../memory/buffer-pool.js';
import { createUploadedTensor } from '../tensor-factory.js';

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

function createF32Tensor(values, shape, label) {
  const data = values instanceof Float32Array ? values : new Float32Array(values);
  return createUploadedTensor(data, 'f32', shape, label);
}

function createU32TokenTensor(values, shape, label) {
  const data = values instanceof Uint32Array ? values : new Uint32Array(values);
  // Token targets are consumed as raw u32 bytes by loss kernels.
  return createUploadedTensor(data, 'f32', shape, label);
}

function releaseTensor(tensor) {
  if (tensor?.buffer) {
    releaseBuffer(tensor.buffer);
  }
}

function computeMeanStd(values) {
  if (!Array.isArray(values) || values.length === 0) {
    return { mean: 0, std: 0 };
  }
  let sum = 0;
  for (const value of values) {
    sum += Number(value) || 0;
  }
  const mean = sum / values.length;
  let variance = 0;
  for (const value of values) {
    const delta = (Number(value) || 0) - mean;
    variance += delta * delta;
  }
  return {
    mean,
    std: Math.sqrt(variance / values.length),
  };
}

function resolveStage1LatentEntry(stage1Context, stepIndex) {
  const entries = Array.isArray(stage1Context?.latentDataset?.entries)
    ? stage1Context.latentDataset.entries
    : [];
  if (entries.length === 0) {
    throw new Error('UL stage2 objective requires stage1 latent entries.');
  }
  const index = Math.max(0, Math.floor(Number(stepIndex) || 0)) % entries.length;
  const entry = entries[index];
  const shape = Array.isArray(entry?.latent_shape) ? entry.latent_shape : null;
  const values = Array.isArray(entry?.latent_noisy_values) ? entry.latent_noisy_values : null;
  if (!shape || !values) {
    throw new Error('UL stage2 objective requires stage1 latent entries with shape and noisy values.');
  }
  const elementCount = shape.reduce((acc, value) => acc * Math.max(1, Math.floor(Number(value) || 1)), 1);
  if (elementCount !== values.length) {
    throw new Error('UL stage2 objective stage1 latent entry has mismatched shape/value lengths.');
  }
  return { entry, shape, values };
}

export function createUlStage2BaseObjective(options = {}) {
  const lossFn = options.crossEntropyLoss || defaultCrossEntropyLoss;
  if (typeof lossFn !== 'function') {
    throw new Error('UL stage2 objective requires crossEntropyLoss(logits, targets, config, tape).');
  }

  return createTrainingObjective({
    name: 'ul_stage2_base',
    async prepareBatch({ batch, config, options: runOptions }) {
      const ulConfig = config.training?.ul;
      if (!ulConfig?.enabled) {
        throw new Error('UL stage2 objective requires training.ul.enabled=true.');
      }
      if (ulConfig.stage !== 'stage2_base') {
        throw new Error('UL stage2 objective requires training.ul.stage="stage2_base".');
      }
      if (!ulConfig.stage1Artifact) {
        throw new Error('UL stage2 objective requires training.ul.stage1Artifact.');
      }
      const stage1Context = runOptions?.stage1ArtifactContext || null;
      const latent = resolveStage1LatentEntry(stage1Context, runOptions?.stepIndex);
      const latentStats = computeMeanStd(latent.values);
      const latentInput = createF32Tensor(
        latent.values,
        latent.shape,
        'ul_stage2_input_latent'
      );
      const baseBatch = (batch && typeof batch === 'object') ? batch : {};
      const temporary = [latentInput];
      let targets = baseBatch.targets;
      if (!targets) {
        const tokenCount = Math.max(1, Math.floor(Number(latent.shape[0]) || 1));
        targets = createU32TokenTensor(
          new Uint32Array(tokenCount),
          [tokenCount],
          'ul_stage2_targets_fallback'
        );
        temporary.push(targets);
      }
      const existingTemp = Array.isArray(baseBatch._ulTemporaryTensors)
        ? baseBatch._ulTemporaryTensors
        : [];
      return {
        ...baseBatch,
        input: latentInput,
        targets,
        ulStage2: {
          stage1Entry: latent.entry,
          latentMean: latentStats.mean,
          latentStd: latentStats.std,
          latentShape: latent.shape,
        },
        _ulTemporaryTensors: [...existingTemp, ...temporary],
      };
    },
    async forward({ model, batch, tape }) {
      const logits = await model.forward(batch.input, tape);
      return { logits };
    },
    async computeLoss({ batch, config, tape, forwardState, options: runOptions }) {
      const loss = await lossFn(forwardState.logits, batch.targets, config, tape);
      const ulConfig = config.training.ul;
      const stage1Context = runOptions?.stage1ArtifactContext || null;
      const stage1Entries = Array.isArray(stage1Context?.latentDataset?.entries)
        ? stage1Context.latentDataset.entries
        : [];
      const stage1Summary = stage1Context?.latentDataset?.summary || null;
      const stage1Entry = batch?.ulStage2?.stage1Entry || null;
      const latentMean = Number.isFinite(batch?.ulStage2?.latentMean) ? batch.ulStage2.latentMean : 0;
      const latentStd = Number.isFinite(batch?.ulStage2?.latentStd) ? batch.ulStage2.latentStd : 0;
      const referenceLambda = Number.isFinite(stage1Summary?.lambdaMean)
        ? stage1Summary.lambdaMean
        : (Number.isFinite(stage1Entry?.lambda) ? stage1Entry.lambda : ulConfig.lambda0);
      const referenceLatentShape = Array.isArray(stage1Entry?.latent_shape)
        ? stage1Entry.latent_shape
        : null;
      const weighting = ulConfig.decoderSigmoidWeight;
      const sigmoidWeight = weighting.enabled
        ? sigmoid(weighting.slope * (referenceLambda - weighting.midpoint))
        : 1;
      const lossWeights = ulConfig.lossWeights || {};
      const ceCoef = Number.isFinite(lossWeights.ce) ? lossWeights.ce : 1;
      const decoderCoef = sigmoidWeight * (Number.isFinite(lossWeights.decoder) ? lossWeights.decoder : 1);
      const priorCoef = Number.isFinite(lossWeights.prior) ? lossWeights.prior : 1;
      const reconCoef = Number.isFinite(lossWeights.recon) ? lossWeights.recon : 1;
      const inputShape = referenceLatentShape || (Array.isArray(batch?.input?.shape) ? batch.input.shape : []);
      const latentSize = inputShape.reduce((acc, value) => acc * value, 1);
      const latentBitrate = Math.log2(1 + Math.max(latentSize, 1));
      const stage1NoisyMean = Number.isFinite(stage1Entry?.latent_noisy_mean) ? stage1Entry.latent_noisy_mean : 0;
      const stage1NoisyStd = Number.isFinite(stage1Entry?.latent_noisy_std) ? stage1Entry.latent_noisy_std : 0;
      const stage1CleanMean = Number.isFinite(stage1Entry?.latent_clean_mean) ? stage1Entry.latent_clean_mean : 0;
      const stage1CleanStd = Number.isFinite(stage1Entry?.latent_clean_std) ? stage1Entry.latent_clean_std : 0;
      const stage1NoiseStd = Number.isFinite(stage1Entry?.latent_noise_std) ? stage1Entry.latent_noise_std : 0;

      const priorBase = ((latentMean - stage1NoisyMean) ** 2) + ((latentStd - stage1NoisyStd) ** 2);
      const reconBase = ((latentMean - stage1CleanMean) ** 2) + ((latentStd - stage1CleanStd) ** 2);
      const decoderBase = reconBase + Math.abs(stage1NoiseStd);

      const lossPrior = priorCoef * priorBase;
      const lossRecon = reconCoef * reconBase;
      const lossDecoder = decoderCoef * decoderBase;
      const lossTotal = lossPrior + lossRecon + lossDecoder;

      return {
        loss,
        components: {
          loss_total: lossTotal,
          loss_prior: lossPrior,
          loss_decoder: lossDecoder,
          loss_recon: lossRecon,
          lambda: referenceLambda,
          latent_bitrate_proxy: latentBitrate,
          coeff_ce: ceCoef,
          coeff_prior: priorCoef,
          coeff_decoder: decoderCoef,
          coeff_recon: reconCoef,
          stage1_latent_count: stage1Entries.length,
        },
      };
    },
    metrics({ config, lossResult }) {
      const ulConfig = config.training.ul;
      const components = lossResult.components || {};
      return {
        loss_total: Number.isFinite(components.loss_total) ? components.loss_total : 0,
        loss_prior: Number.isFinite(components.loss_prior) ? components.loss_prior : 0,
        loss_decoder: Number.isFinite(components.loss_decoder) ? components.loss_decoder : 0,
        loss_recon: Number.isFinite(components.loss_recon) ? components.loss_recon : 0,
        lambda: Number.isFinite(components.lambda) ? components.lambda : ulConfig.lambda0,
        latent_bitrate_proxy: Number.isFinite(components.latent_bitrate_proxy)
          ? components.latent_bitrate_proxy
          : 0,
        coeff_ce: Number.isFinite(components.coeff_ce) ? components.coeff_ce : 1,
        coeff_prior: Number.isFinite(components.coeff_prior) ? components.coeff_prior : 1,
        coeff_decoder: Number.isFinite(components.coeff_decoder) ? components.coeff_decoder : 1,
        coeff_recon: Number.isFinite(components.coeff_recon) ? components.coeff_recon : 1,
        stage1_latent_count: Number.isFinite(components.stage1_latent_count)
          ? components.stage1_latent_count
          : 0,
      };
    },
    cleanup({ preparedBatch }) {
      const tensors = Array.isArray(preparedBatch?._ulTemporaryTensors)
        ? preparedBatch._ulTemporaryTensors
        : [];
      for (const tensor of tensors) {
        releaseTensor(tensor);
      }
    },
  });
}
