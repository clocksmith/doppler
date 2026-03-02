import { crossEntropyLoss as defaultCrossEntropyLoss } from '../loss.js';
import { createTrainingObjective } from './base.js';

function toFinite(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function resolveDistillConfig(config) {
  return config?.training?.distill || {};
}

export function createDistillTripletObjective(options = {}) {
  const lossFn = options.crossEntropyLoss || defaultCrossEntropyLoss;
  if (typeof lossFn !== 'function') {
    throw new Error('Distill triplet objective requires crossEntropyLoss(logits, targets, config, tape).');
  }

  return createTrainingObjective({
    name: 'triplet',
    async prepareBatch({ batch, config, options: runOptions }) {
      const distill = resolveDistillConfig(config);
      if (distill.stage !== 'stage_b') {
        throw new Error('Distill triplet objective requires training.distill.stage="stage_b".');
      }
      if (!distill.stageAArtifact) {
        throw new Error('Distill triplet objective requires training.distill.stageAArtifact.');
      }
      const stageAContext = runOptions?.stageAArtifactContext;
      if (!stageAContext || typeof stageAContext !== 'object') {
        throw new Error('Distill triplet objective requires stageAArtifactContext.');
      }
      return batch;
    },
    async forward({ model, batch, tape }) {
      const logits = await model.forward(batch.input, tape);
      return { logits };
    },
    async computeLoss({ batch, config, tape, forwardState, options: runOptions }) {
      const loss = await lossFn(forwardState.logits, batch.targets, config, tape);
      const distill = resolveDistillConfig(config);
      const margin = Math.max(0, toFinite(batch?.distill?.tripletMargin, toFinite(distill.tripletMargin, 0.2)));
      const stageAContext = runOptions?.stageAArtifactContext;
      const referenceKdMean = toFinite(stageAContext?.metricsSummary?.kdMean, 0.08);
      const tripletHint = Math.max(0, toFinite(batch?.distill?.tripletHint, NaN));
      const fallbackHint = referenceKdMean + (Math.abs(Math.floor(toFinite(runOptions?.stepIndex, 0))) * 0.02);
      const lossTriplet = Math.max(0, (Number.isFinite(tripletHint) ? tripletHint : fallbackHint) + margin);
      return {
        loss,
        components: {
          loss_triplet: lossTriplet,
          distill_stage: 'stage_b',
          distill_triplet_margin: margin,
          distill_stage_a_step_count: toFinite(stageAContext?.metricsSummary?.stepCount, 0),
          distill_stage_a_kd_mean: referenceKdMean,
        },
      };
    },
    metrics({ config, lossResult }) {
      const distill = resolveDistillConfig(config);
      const components = lossResult.components || {};
      return {
        loss_triplet: Number.isFinite(components.loss_triplet) ? components.loss_triplet : 0,
        distill_stage: 'stage_b',
        distill_triplet_margin: Number.isFinite(components.distill_triplet_margin)
          ? components.distill_triplet_margin
          : toFinite(distill.tripletMargin, 0.2),
        distill_stage_a_step_count: Number.isFinite(components.distill_stage_a_step_count)
          ? components.distill_stage_a_step_count
          : 0,
        distill_stage_a_kd_mean: Number.isFinite(components.distill_stage_a_kd_mean)
          ? components.distill_stage_a_kd_mean
          : null,
      };
    },
  });
}
