import { crossEntropyLoss as defaultCrossEntropyLoss } from '../loss.js';
import { createTrainingObjective } from './base.js';
import {
  applyUlStage1Batch,
  cleanupUlPreparedBatch,
  computeLatentBitrateProxy,
  resolveUlScheduledLambda,
} from '../ul_dataset.js';

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

function meanSquare(values) {
  if (!Array.isArray(values) || values.length === 0) return null;
  let sum = 0;
  let count = 0;
  for (const value of values) {
    const n = Number(value);
    if (!Number.isFinite(n)) continue;
    sum += n * n;
    count += 1;
  }
  return count > 0 ? (sum / count) : null;
}

function msePair(leftValues, rightValues) {
  if (!Array.isArray(leftValues) || !Array.isArray(rightValues)) return null;
  const count = Math.min(leftValues.length, rightValues.length);
  if (count <= 0) return null;
  let sum = 0;
  let used = 0;
  for (let i = 0; i < count; i += 1) {
    const left = Number(leftValues[i]);
    const right = Number(rightValues[i]);
    if (!Number.isFinite(left) || !Number.isFinite(right)) continue;
    const delta = left - right;
    sum += delta * delta;
    used += 1;
  }
  return used > 0 ? (sum / used) : null;
}

export function createUlStage1JointObjective(options = {}) {
  const lossFn = options.crossEntropyLoss || defaultCrossEntropyLoss;
  if (typeof lossFn !== 'function') {
    throw new Error('UL stage1 objective requires crossEntropyLoss(logits, targets, config, tape).');
  }

  return createTrainingObjective({
    name: 'ul_stage1_joint',
    async prepareBatch({ batch, config, options: runOptions }) {
      const ulConfig = config.training?.ul;
      if (!ulConfig?.enabled) {
        throw new Error('UL stage1 objective requires training.ul.enabled=true.');
      }
      return applyUlStage1Batch(batch, ulConfig, {
        seed: ulConfig.seed,
        stepIndex: runOptions?.stepIndex,
        includeValues: runOptions?.persistStage1Latents !== false,
      });
    },
    async forward({ model, batch, tape }) {
      const logits = await model.forward(batch.input, tape);
      return { logits };
    },
    async computeLoss({ batch, config, tape, forwardState, options: runOptions }) {
      const loss = await lossFn(forwardState.logits, batch.targets, config, tape);
      const ulConfig = config.training.ul;
      const decoderWeightConfig = ulConfig.decoderSigmoidWeight;
      const lambda = Number.isFinite(batch?.ul?.lambda)
        ? batch.ul.lambda
        : resolveUlScheduledLambda(ulConfig, runOptions?.stepIndex);
      const sigmoidWeight = decoderWeightConfig.enabled
        ? sigmoid(decoderWeightConfig.slope * (lambda - decoderWeightConfig.midpoint))
        : 1;
      const lossWeights = ulConfig.lossWeights || {};
      const ceCoef = Number.isFinite(lossWeights.ce) ? lossWeights.ce : 1;
      const priorCoefBase = ulConfig.priorAlignment.enabled ? ulConfig.priorAlignment.weight : 0;
      const priorCoef = priorCoefBase * (Number.isFinite(lossWeights.prior) ? lossWeights.prior : 1);
      const reconCoef = Number.isFinite(lossWeights.recon) ? lossWeights.recon : 1;
      const decoderCoef = sigmoidWeight * (Number.isFinite(lossWeights.decoder) ? lossWeights.decoder : 1);
      const cleanStd = batch?.ul?.clean?.std ?? 0;
      const cleanMean = batch?.ul?.clean?.mean ?? 0;
      const noisyStd = batch?.ul?.noisy?.std ?? 0;
      const noisyMean = batch?.ul?.noisy?.mean ?? 0;
      const noiseMean = batch?.ul?.noise?.mean ?? 0;
      const noiseStd = batch?.ul?.noise?.std ?? 0;
      const latentShape = Array.isArray(batch?.ul?.shape) ? batch.ul.shape : null;
      const latentValues = batch?.ul?.values || null;
      const cleanValues = Array.isArray(latentValues?.clean) ? latentValues.clean : null;
      const noiseValues = Array.isArray(latentValues?.noise) ? latentValues.noise : null;
      const noisyValues = Array.isArray(latentValues?.noisy) ? latentValues.noisy : null;
      const scheduleStepIndex = Number.isFinite(batch?.ul?.stepIndex)
        ? batch.ul.stepIndex
        : (Number.isFinite(runOptions?.stepIndex) ? runOptions.stepIndex : 0);
      const priorBase = meanSquare(noiseValues)
        ?? ((noiseMean * noiseMean) + (noiseStd * noiseStd));
      const reconBase = msePair(cleanValues, noisyValues)
        ?? (((cleanStd - noisyStd) ** 2) + ((cleanMean - noisyMean) ** 2));
      const decoderBase = (msePair(noisyValues, cleanValues) ?? reconBase) + Math.abs(noiseStd);
      const priorLoss = priorCoef * priorBase;
      const reconLoss = reconCoef * reconBase;
      const decoderLoss = decoderCoef * decoderBase;
      const latentBitrate = computeLatentBitrateProxy(batch?.ul, ulConfig);
      const lossTotal = priorLoss + reconLoss + decoderLoss;

      return {
        loss,
        components: {
          loss_total: lossTotal,
          loss_prior: priorLoss,
          loss_decoder: decoderLoss,
          loss_recon: reconLoss,
          lambda,
          latent_bitrate_proxy: latentBitrate,
          coeff_ce: ceCoef,
          coeff_prior: priorCoef,
          coeff_decoder: decoderCoef,
          coeff_recon: reconCoef,
          schedule_step_index: scheduleStepIndex,
          latent_clean_mean: cleanMean,
          latent_clean_std: cleanStd,
          latent_noise_mean: noiseMean,
          latent_noise_std: noiseStd,
          latent_noisy_mean: noisyMean,
          latent_noisy_std: noisyStd,
          latent_shape: latentShape,
          latent_clean_values: Array.isArray(latentValues?.clean) ? latentValues.clean : null,
          latent_noise_values: Array.isArray(latentValues?.noise) ? latentValues.noise : null,
          latent_noisy_values: Array.isArray(latentValues?.noisy) ? latentValues.noisy : null,
        },
      };
    },
    metrics({ batch, config, lossResult }) {
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
          : computeLatentBitrateProxy(batch?.ul, ulConfig),
        coeff_ce: Number.isFinite(components.coeff_ce) ? components.coeff_ce : 1,
        coeff_prior: Number.isFinite(components.coeff_prior) ? components.coeff_prior : 0,
        coeff_decoder: Number.isFinite(components.coeff_decoder) ? components.coeff_decoder : 1,
        coeff_recon: Number.isFinite(components.coeff_recon) ? components.coeff_recon : 1,
        schedule_step_index: Number.isFinite(components.schedule_step_index)
          ? components.schedule_step_index
          : 0,
        latent_clean_mean: Number.isFinite(components.latent_clean_mean)
          ? components.latent_clean_mean
          : 0,
        latent_clean_std: Number.isFinite(components.latent_clean_std)
          ? components.latent_clean_std
          : 0,
        latent_noise_mean: Number.isFinite(components.latent_noise_mean)
          ? components.latent_noise_mean
          : 0,
        latent_noise_std: Number.isFinite(components.latent_noise_std)
          ? components.latent_noise_std
          : 0,
        latent_noisy_mean: Number.isFinite(components.latent_noisy_mean)
          ? components.latent_noisy_mean
          : 0,
        latent_noisy_std: Number.isFinite(components.latent_noisy_std)
          ? components.latent_noisy_std
          : 0,
        latent_shape: Array.isArray(components.latent_shape) ? components.latent_shape : null,
        latent_clean_values: Array.isArray(components.latent_clean_values)
          ? components.latent_clean_values
          : null,
        latent_noise_values: Array.isArray(components.latent_noise_values)
          ? components.latent_noise_values
          : null,
        latent_noisy_values: Array.isArray(components.latent_noisy_values)
          ? components.latent_noisy_values
          : null,
      };
    },
    cleanup({ preparedBatch }) {
      cleanupUlPreparedBatch(preparedBatch);
    },
  });
}
