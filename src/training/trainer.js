import { AutogradTape } from './autograd.js';
import { loadBackwardRegistry } from '../config/backward-registry-loader.js';

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
  } = options;

  if (!crossEntropyLoss || !clipGradients || !optimizer) {
    throw new Error('trainStep requires crossEntropyLoss, clipGradients, and optimizer');
  }

  const tape = new AutogradTape(registry);
  const logits = await model.forward(batch.input, tape);
  const loss = await crossEntropyLoss(logits, batch.targets, config);

  const grads = await tape.backward(loss);
  const clipped = await clipGradients(grads, config);

  await optimizer.step(model.loraParams(), clipped, config);

  return { loss, grads: clipped };
}
