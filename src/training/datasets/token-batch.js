
import { acquireBuffer, uploadData, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor } from '../../gpu/tensor.js';

function flattenTokenBatch(samples, key) {
  const offsets = [];
  let total = 0;
  for (const sample of samples) {
    offsets.push(total);
    total += sample[key].length;
  }
  const flat = new Uint32Array(total);
  for (let i = 0; i < samples.length; i += 1) {
    const ids = samples[i][key];
    flat.set(ids, offsets[i]);
  }
  return { flat, offsets };
}

export function buildTokenBatch(samples) {
  if (!samples.length) {
    throw new Error('buildTokenBatch requires at least one sample');
  }
  const { flat: inputFlat, offsets } = flattenTokenBatch(samples, 'inputIds');
  const { flat: targetFlat } = flattenTokenBatch(samples, 'targetIds');
  return { inputFlat, targetFlat, offsets };
}

export function createTokenBatchTensors(batch) {
  let inputBuf = null;
  let targetBuf = null;
  try {
    inputBuf = acquireBuffer(batch.inputFlat.byteLength, undefined, 'train_input_tokens');
    uploadData(inputBuf, batch.inputFlat);

    targetBuf = acquireBuffer(batch.targetFlat.byteLength, undefined, 'train_target_tokens');
    uploadData(targetBuf, batch.targetFlat);

    const input = createTensor(inputBuf, 'f32', [batch.inputFlat.length], 'train_input_tokens');
    const targets = createTensor(targetBuf, 'f32', [batch.targetFlat.length], 'train_target_tokens');

    return { input, targets, offsets: batch.offsets };
  } catch (error) {
    if (inputBuf) {
      releaseBuffer(inputBuf);
    }
    if (targetBuf) {
      releaseBuffer(targetBuf);
    }
    throw error;
  }
}
