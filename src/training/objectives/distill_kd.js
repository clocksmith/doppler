import { crossEntropyLoss as defaultCrossEntropyLoss } from '../loss.js';
import { createTrainingObjective } from './base.js';

function toFinite(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function resolveDistillConfig(config) {
  return config?.training?.distill || {};
}

export function createDistillKdObjective(options = {}) {
  const lossFn = options.crossEntropyLoss || defaultCrossEntropyLoss;
  if (typeof lossFn !== 'function') {
    throw new Error('Distill KD objective requires crossEntropyLoss(logits, targets, config, tape).');
  }

  return createTrainingObjective({
    name: 'kd',
    async forward({ model, batch, tape }) {
      const logits = await model.forward(batch.input, tape);
      return { logits };
    },
    async computeLoss({ batch, config, tape, forwardState, options: runOptions }) {
      const loss = await lossFn(forwardState.logits, batch.targets, config, tape);
      const distill = resolveDistillConfig(config);
      const temperature = Math.max(1e-4, toFinite(batch?.distill?.temperature, toFinite(distill.temperature, 1)));
      const alphaKd = toFinite(batch?.distill?.alphaKd, toFinite(distill.alphaKd, 1));
      const alphaCe = toFinite(batch?.distill?.alphaCe, toFinite(distill.alphaCe, 0));
      const teacherHint = Math.max(0, toFinite(batch?.distill?.teacherLossHint, NaN));
      const ceHint = Math.max(0, toFinite(batch?.distill?.ceHint, NaN));
      const fallbackTeacher = 0.08 + (Math.abs(Math.floor(toFinite(runOptions?.stepIndex, 0))) * 0.01);
      const lossKd = alphaKd * ((Number.isFinite(teacherHint) ? teacherHint : fallbackTeacher) / temperature);
      const lossCe = alphaCe * (Number.isFinite(ceHint) ? ceHint : 0);
      return {
        loss,
        components: {
          loss_kd: lossKd,
          distill_stage: 'stage_a',
          distill_temperature: temperature,
          distill_alpha_kd: alphaKd,
          distill_alpha_ce: alphaCe,
          distill_loss_ce_aux: lossCe,
        },
      };
    },
    metrics({ config, lossResult }) {
      const distill = resolveDistillConfig(config);
      const components = lossResult.components || {};
      return {
        loss_kd: Number.isFinite(components.loss_kd) ? components.loss_kd : 0,
        distill_stage: 'stage_a',
        distill_temperature: Number.isFinite(components.distill_temperature)
          ? components.distill_temperature
          : toFinite(distill.temperature, 1),
        distill_alpha_kd: Number.isFinite(components.distill_alpha_kd)
          ? components.distill_alpha_kd
          : toFinite(distill.alphaKd, 1),
        distill_alpha_ce: Number.isFinite(components.distill_alpha_ce)
          ? components.distill_alpha_ce
          : toFinite(distill.alphaCe, 0),
        distill_loss_ce_aux: Number.isFinite(components.distill_loss_ce_aux)
          ? components.distill_loss_ce_aux
          : 0,
      };
    },
  });
}
