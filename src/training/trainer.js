import { AutogradTape } from './autograd.js';
import { loadBackwardRegistry } from '../config/backward-registry-loader.js';
import { runScale } from '../gpu/kernels/index.js';
import { acquireBuffer, uploadData, releaseBuffer } from '../memory/buffer-pool.js';
import { createTensor } from '../gpu/tensor.js';

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
  } = options;

  if (!crossEntropyLoss || !clipGradients || !optimizer) {
    throw new Error('trainStep requires crossEntropyLoss, clipGradients, and optimizer');
  }

  const tape = new AutogradTape(registry);
  const logits = await model.forward(batch.input, tape);
  const loss = await crossEntropyLoss(logits, batch.targets, config, tape);
  const lossElements = loss.shape.reduce((acc, value) => acc * value, 1);
  const gradData = new Float32Array(lossElements);
  gradData.fill(lossScale);
  const gradBuf = acquireBuffer(gradData.byteLength, undefined, 'loss_grad_output');
  uploadData(gradBuf, gradData);
  const gradOutput = createTensor(gradBuf, 'f32', [...loss.shape], 'loss_grad_output');

  const grads = await tape.backward(gradOutput);
  releaseBuffer(gradOutput.buffer);
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

  if (applyClip) {
    processed = await clipGradients(processed, config);
  }

  if (applyOptimizer) {
    await optimizer.step(model.loraParams(), processed, config);
  }

  return { loss, grads: processed };
}
