import { crossEntropyLoss as defaultCrossEntropyLoss } from '../loss.js';
import { acquireBuffer, uploadData } from '../../memory/buffer-pool.js';
import { createTensor } from '../../gpu/tensor.js';
import { createTrainingObjective } from './base.js';

function createLossGradient(loss, lossScale) {
  const lossElements = loss.shape.reduce((acc, value) => acc * value, 1);
  const gradData = new Float32Array(lossElements);
  gradData.fill(lossScale);
  const gradBuf = acquireBuffer(gradData.byteLength, undefined, 'loss_grad_output');
  uploadData(gradBuf, gradData);
  return createTensor(gradBuf, 'f32', [...loss.shape], 'loss_grad_output');
}

export function createCrossEntropyObjective(options = {}) {
  const lossFn = options.crossEntropyLoss || defaultCrossEntropyLoss;
  if (typeof lossFn !== 'function') {
    throw new Error('Cross-entropy objective requires crossEntropyLoss(logits, targets, config, tape).');
  }

  return createTrainingObjective({
    name: 'cross_entropy',
    async forward({ model, batch, tape }) {
      const logits = await model.forward(batch.input, tape);
      return { logits };
    },
    async computeLoss({ batch, config, tape, forwardState }) {
      const loss = await lossFn(forwardState.logits, batch.targets, config, tape);
      return { loss };
    },
    backwardTargets({ loss, lossScale }) {
      return createLossGradient(loss, lossScale);
    },
  });
}

export const CROSS_ENTROPY_OBJECTIVE = createCrossEntropyObjective();
