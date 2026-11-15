import { readBuffer } from '../../memory/buffer-pool.js';
import { createTensor } from '../tensor.js';
import { validateChannelTensor } from './channel-gain.js';
import { unifiedKernelWrapper, withKernelOutput, withKernelOutputs } from './kernel-execution.js';

/** @type {typeof import('./channel-rotation.js').deriveChannelRotation} */
export async function deriveChannelRotation(recorder, input, options) {
  const { tokens, channels, start, stop, contentStart, contentStop, epsilon } = options;
  validateChannelTensor(input, tokens, channels);
  if (![start, stop, contentStart, contentStop].every(Number.isInteger)
      || contentStart < 0 || contentStop > tokens || start < contentStart || stop > contentStop
      || start >= stop || contentStop - contentStart <= stop - start
      || !Number.isFinite(epsilon) || !Number.isFinite(Math.fround(epsilon)) || Math.fround(epsilon) <= 0) {
    throw new Error('Channel rotation requires exact nested spans, a nonempty complement and finite positive f32 epsilon.');
  }
  return withKernelOutputs(recorder, {
    basis: { bytes: channels * 8, label: 'channel_rotation_basis' },
    status: { bytes: 4, label: 'channel_rotation_status' },
  }, async ({ basis, status }) => {
    await unifiedKernelWrapper('channel_rotation_derive', recorder, 'f32', [input, basis, status],
      { channels, start, stop, content_start: contentStart, content_stop: contentStop, epsilon }, 1);
    const validate = async () => {
      const value = new Float32Array(await readBuffer(status, 4))[0];
      if (value !== 0 && value !== 1) throw new Error('Channel rotation capture contains invalid or overflowing activation data.');
    };
    if (recorder) recorder.enqueueValidationTask(validate);
    else await validate();
    return { basis: createTensor(basis, 'f32', [2, channels], 'channel_rotation_basis'),
      status: createTensor(status, 'f32', [1], 'channel_rotation_status') };
  });
}

/** @type {typeof import('./channel-rotation.js').applyChannelRotation} */
export async function applyChannelRotation(recorder, input, rotation, options) {
  const { tokens, channels, theta } = options;
  validateChannelTensor(input, tokens, channels);
  validateChannelTensor(rotation.basis, 2, channels);
  validateChannelTensor(rotation.status, 1, 1);
  if (!Number.isFinite(theta) || !Number.isFinite(Math.fround(theta))) {
    throw new Error('Channel rotation requires an explicit finite f32 angle in radians.');
  }
  return withKernelOutput(recorder, null, tokens * channels * 4, 'channel_rotation_apply', async (output) => {
    await unifiedKernelWrapper('channel_rotation_apply', recorder, 'f32',
      [input, rotation.basis, rotation.status, output], { channels, tokens, theta }, tokens);
    return createTensor(output, 'f32', [tokens, channels], 'channel_rotation_apply');
  });
}
