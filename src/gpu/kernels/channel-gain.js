import { selectRuleValue } from './rule-registry.js';
import { createTensor } from '../tensor.js';
import { WORKGROUP_SIZES } from './constants.js';
import { unifiedKernelWrapper, withKernelOutput } from './kernel-execution.js';

/** @type {typeof import('./channel-gain.js').validateChannelTensor} */
export function validateChannelTensor(input, tokens, channels) {
  if (!Number.isSafeInteger(tokens) || tokens < 1 || !Number.isSafeInteger(channels) || channels < 1
      || tokens * channels > 0xffffffff || input.dtype !== 'f32' || input.precision !== undefined
      || input.buffer.size < tokens * channels * 4) {
    throw new Error('Channel gain requires explicit positive geometry and an ordinary f32 tensor of sufficient size.');
  }
}

/** @type {typeof import('./channel-gain.js').channelEnergy} */
export async function channelEnergy(recorder, input, options) {
  const { tokens, channels, start, stop } = options;
  validateChannelTensor(input, tokens, channels);
  if (!Number.isInteger(start) || !Number.isInteger(stop) || start < 0 || stop > tokens || start >= stop) {
    throw new Error('Channel energy requires an exact nonempty half-open token span.');
  }
  return withKernelOutput(recorder, null, channels * 4, 'channel_energy', async (output) => {
    await unifiedKernelWrapper('channel_energy', recorder, 'f32', [input, output],
      { channels, start, stop }, Math.ceil(channels / WORKGROUP_SIZES.DEFAULT));
    return createTensor(output, 'f32', [channels], 'channel_energy');
  });
}

/** @type {typeof import('./channel-gain.js').channelGainSwiGLU} */
export async function channelGainSwiGLU(recorder, up, gate, gains, options) {
  const { tokens, channels, swigluLimit } = options;
  validateChannelTensor(up, tokens, channels);
  validateChannelTensor(gate, tokens, channels);
  validateChannelTensor(gains, 1, channels);
  if (swigluLimit !== null && (!Number.isFinite(swigluLimit) || swigluLimit <= 0)) {
    throw new Error('Channel gain requires explicit null or positive finite swigluLimit.');
  }
  return withKernelOutput(recorder, null, tokens * channels * 4, 'channel_gain_swiglu', async (output) => {
    await unifiedKernelWrapper('channel_gain_swiglu', recorder, 'f32', [up, gate, gains, output],
      { tokens, channels, clamp_max: swigluLimit === null ? 0 : swigluLimit },
      [Math.ceil(channels / WORKGROUP_SIZES.DEFAULT), tokens, 1]);
    return createTensor(output, 'f32', [tokens, channels], 'channel_gain_swiglu');
  });
}

/** @type {typeof import('./channel-gain.js').deriveChannelGains} */
export async function deriveChannelGains(recorder, instruction, reference, options) {
  const { channels, beta, clip, epsilon } = options;
  validateChannelTensor(instruction, 1, channels);
  validateChannelTensor(reference, 1, channels);
  if (!Number.isFinite(beta) || !Number.isFinite(clip) || clip <= 0
      || !Number.isFinite(epsilon) || epsilon <= 0 || Math.fround(epsilon) <= 0) {
    throw new Error('Channel gain requires finite beta, positive clip and representable positive epsilon.');
  }
  return withKernelOutput(recorder, null, channels * 4, 'channel_gain_derive', async (output) => {
    await unifiedKernelWrapper('channel_gain_derive', recorder, 'f32', [instruction, reference, output],
      { channels, beta, clip, epsilon }, 1);
    return createTensor(output, 'f32', [channels], 'channel_gain_derive');
  });
}

/** @type {typeof import('./channel-gain.js').controlChannelGains} */
export async function controlChannelGains(recorder, gains, options) {
  const { channels, mode, seeds } = options;
  validateChannelTensor(gains, 1, channels);
  if ((mode !== 'permuted' && mode !== 'random-z') || !seeds || seeds.size < channels * 4) {
    throw new Error('Channel gain control requires explicit mode and a complete seed/index buffer.');
  }
  const variant = selectRuleValue('channel_gain_control', 'variant', { mode });
  const workgroups = mode === 'permuted' ? Math.ceil(channels / WORKGROUP_SIZES.DEFAULT) : 1;
  return withKernelOutput(recorder, null, channels * 4, 'channel_gain_control', async (output) => {
    await unifiedKernelWrapper('channel_gain_control', recorder, variant, [gains, seeds, output],
      { channels }, workgroups);
    return createTensor(output, 'f32', [channels], 'channel_gain_control');
  });
}
