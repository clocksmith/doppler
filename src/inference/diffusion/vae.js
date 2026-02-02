import { getDevice } from '../../gpu/device.js';
import { acquireBuffer, releaseBuffer, readBuffer } from '../../memory/buffer-pool.js';
import { createTensor, dtypeBytes } from '../../gpu/tensor.js';
import { runConv2D } from '../../gpu/kernels/conv2d.js';
import { runGroupNorm } from '../../gpu/kernels/groupnorm.js';
import { runSiLU } from '../../gpu/kernels/silu.js';
import { runResidualAdd } from '../../gpu/kernels/residual.js';
import { runUpsample2D } from '../../gpu/kernels/upsample2d.js';
import { castF32ToF16 } from '../../gpu/kernels/cast.js';
import { f16ToF32 } from '../../loader/dtype-utils.js';
import { log } from '../../debug/index.js';

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function reshapeTensor(tensor, shape, label) {
  return createTensor(tensor.buffer, tensor.dtype, shape, label ?? tensor.label);
}

function getWeight(weights, shapes, name) {
  const value = weights.get(name);
  if (!value) {
    throw new Error(`Missing VAE weight: ${name}`);
  }
  const shape = shapes.get(name);
  if (!shape) {
    throw new Error(`Missing VAE weight shape: ${name}`);
  }
  return { value, shape };
}

function getConvShape(shape) {
  if (!Array.isArray(shape) || shape.length !== 4) {
    throw new Error(`Conv2D weight shape must be [out,in,h,w], got ${shape}`);
  }
  return {
    outChannels: shape[0],
    inChannels: shape[1],
    kernelH: shape[2],
    kernelW: shape[3],
  };
}

function buildIndexList(weights, prefix) {
  const indices = new Set();
  for (const name of weights.keys()) {
    if (!name.startsWith(prefix)) continue;
    const rest = name.slice(prefix.length);
    const match = rest.match(/^(\d+)\./);
    if (!match) continue;
    const idx = Number.parseInt(match[1], 10);
    if (Number.isFinite(idx)) indices.add(idx);
  }
  return Array.from(indices).sort((a, b) => a - b);
}

async function applyConv2D(state, weights, shapes, namePrefix, options = {}) {
  const weightName = `${namePrefix}.weight`;
  const biasName = `${namePrefix}.bias`;
  const weight = getWeight(weights, shapes, weightName);
  const bias = getWeight(weights, shapes, biasName);
  const { outChannels, inChannels, kernelH, kernelW } = getConvShape(weight.shape);

  if (inChannels !== state.channels) {
    log.warn('Diffusion', `VAE conv channel mismatch: ${namePrefix} in=${inChannels} state=${state.channels}`);
  }

  const output = await runConv2D(
    state.tensor,
    weight.value,
    bias.value,
    {
      inChannels,
      outChannels,
      height: state.height,
      width: state.width,
      kernelH,
      kernelW,
      stride: options.stride ?? 1,
      pad: options.pad ?? 1,
    }
  );

  releaseBuffer(state.tensor.buffer);

  return {
    tensor: output,
    channels: outChannels,
    height: Math.floor((state.height + (options.pad ?? 1) * 2 - kernelH) / (options.stride ?? 1)) + 1,
    width: Math.floor((state.width + (options.pad ?? 1) * 2 - kernelW) / (options.stride ?? 1)) + 1,
  };
}

async function runResnetBlock(state, weights, shapes, prefix, config) {
  const numGroups = config.numGroups;
  const eps = config.eps;
  const channels = state.channels;

  const norm1 = getWeight(weights, shapes, `${prefix}.norm1.weight`);
  const norm1Bias = getWeight(weights, shapes, `${prefix}.norm1.bias`);
  const normed1 = await runGroupNorm(state.tensor, norm1.value, norm1Bias.value, {
    channels,
    height: state.height,
    width: state.width,
    numGroups,
    eps,
  });

  const silu1 = await runSiLU(normed1, { size: channels * state.height * state.width, swigluLimit: null });
  releaseBuffer(normed1.buffer);
  const silu1View = reshapeTensor(silu1, [channels, state.height, state.width], 'vae_resnet_silu1');

  const conv1 = await applyConv2D({ tensor: silu1View, channels, height: state.height, width: state.width }, weights, shapes, `${prefix}.conv1`, { pad: 1 });

  const norm2 = getWeight(weights, shapes, `${prefix}.norm2.weight`);
  const norm2Bias = getWeight(weights, shapes, `${prefix}.norm2.bias`);
  const normed2 = await runGroupNorm(conv1.tensor, norm2.value, norm2Bias.value, {
    channels: conv1.channels,
    height: conv1.height,
    width: conv1.width,
    numGroups,
    eps,
  });

  releaseBuffer(conv1.tensor.buffer);

  const silu2 = await runSiLU(normed2, { size: conv1.channels * conv1.height * conv1.width, swigluLimit: null });
  releaseBuffer(normed2.buffer);
  const silu2View = reshapeTensor(silu2, [conv1.channels, conv1.height, conv1.width], 'vae_resnet_silu2');

  const conv2 = await applyConv2D({ tensor: silu2View, channels: conv1.channels, height: conv1.height, width: conv1.width }, weights, shapes, `${prefix}.conv2`, { pad: 1 });

  let residualTensor = state.tensor;

  if (weights.has(`${prefix}.conv_shortcut.weight`)) {
    const shortcut = await applyConv2D(state, weights, shapes, `${prefix}.conv_shortcut`, { pad: 0 });
    residualTensor = shortcut.tensor;
  }

  const size = conv2.channels * conv2.height * conv2.width;
  const residual = reshapeTensor(residualTensor, [size], 'vae_resnet_residual');
  const output = await runResidualAdd(
    reshapeTensor(conv2.tensor, [size], 'vae_resnet_main'),
    residual,
    size,
    { useVec4: true }
  );

  if (residualTensor === state.tensor) {
    releaseBuffer(state.tensor.buffer);
  } else {
    releaseBuffer(residualTensor.buffer);
  }

  releaseBuffer(conv2.tensor.buffer);

  return {
    tensor: reshapeTensor(output, [conv2.channels, conv2.height, conv2.width], 'vae_resnet_output'),
    channels: conv2.channels,
    height: conv2.height,
    width: conv2.width,
  };
}

async function decodeLatentsGPU(latents, options) {
  const device = getDevice();
  if (!device) {
    throw new Error('VAE GPU decode requires a WebGPU device.');
  }

  const config = options.modelConfig?.components?.vae?.config || {};
  const runtime = options.runtime || {};
  const weightsEntry = options.weights;

  if (!weightsEntry?.weights || !weightsEntry?.shapes) {
    throw new Error('VAE GPU decode requires loaded weights.');
  }

  const weights = weightsEntry.weights;
  const shapes = weightsEntry.shapes;

  const scalingFactor = config.scaling_factor;
  if (!Number.isFinite(scalingFactor) || scalingFactor === 0) {
    throw new Error('VAE decode requires a valid scaling_factor in config.');
  }
  const shiftFactor = Number.isFinite(config.shift_factor) ? config.shift_factor : 0.0;
  const numGroups = config.norm_num_groups;
  if (!Number.isFinite(numGroups) || numGroups <= 0) {
    throw new Error('VAE decode requires norm_num_groups in config.');
  }
  const eps = runtime.decode?.groupNormEps;
  if (!Number.isFinite(eps)) {
    throw new Error('VAE decode requires runtime.decode.groupNormEps.');
  }

  const scaledLatents = new Float32Array(latents.length);
  for (let i = 0; i < latents.length; i++) {
    scaledLatents[i] = latents[i] / scalingFactor + shiftFactor;
  }

  const latentBuffer = acquireBuffer(scaledLatents.byteLength, undefined, 'vae_latents');
  device.queue.writeBuffer(latentBuffer, 0, scaledLatents);

  let state = {
    tensor: createTensor(latentBuffer, 'f32', [options.latentChannels, options.latentHeight, options.latentWidth], 'vae_latents_f32'),
    channels: options.latentChannels,
    height: options.latentHeight,
    width: options.latentWidth,
  };

  const computeDtype = runtime.latent?.dtype ?? 'f16';
  if (computeDtype !== 'f16') {
    log.warn('Diffusion', `VAE GPU decode supports f16 only (requested ${computeDtype}). Using f16.`);
  }
  const casted = await castF32ToF16(state.tensor);
  releaseBuffer(state.tensor.buffer);
  state = {
    tensor: reshapeTensor(casted, [state.channels, state.height, state.width], 'vae_latents_f16'),
    channels: state.channels,
    height: state.height,
    width: state.width,
  };

  state = await applyConv2D(state, weights, shapes, 'vae.decoder.conv_in', { pad: 1 });

  const midResnetPrefix = 'vae.decoder.mid_block.resnets.';
  const midResnetIds = buildIndexList(weights, midResnetPrefix);
  for (const idx of midResnetIds) {
    state = await runResnetBlock(state, weights, shapes, `${midResnetPrefix}${idx}`, { numGroups, eps });
  }

  if (weights.has('vae.decoder.mid_block.attentions.0.to_q.weight')) {
    log.warn('Diffusion', 'VAE mid-block attention is not implemented yet; skipping.');
  }

  const upBlockPrefix = 'vae.decoder.up_blocks.';
  const upBlocks = buildIndexList(weights, upBlockPrefix);
  for (const blockIdx of upBlocks) {
    const resnetPrefix = `${upBlockPrefix}${blockIdx}.resnets.`;
    const resnetIds = buildIndexList(weights, resnetPrefix);
    for (const idx of resnetIds) {
      state = await runResnetBlock(state, weights, shapes, `${resnetPrefix}${idx}`, { numGroups, eps });
    }

    const upsampleWeightName = `${upBlockPrefix}${blockIdx}.upsamplers.0.conv.weight`;
    if (weights.has(upsampleWeightName)) {
      const upsample = await runUpsample2D(state.tensor, {
        channels: state.channels,
        height: state.height,
        width: state.width,
        scale: 2,
      });
      releaseBuffer(state.tensor.buffer);
      state = {
        tensor: reshapeTensor(upsample, [state.channels, state.height * 2, state.width * 2], 'vae_upsample'),
        channels: state.channels,
        height: state.height * 2,
        width: state.width * 2,
      };

      state = await applyConv2D(state, weights, shapes, `${upBlockPrefix}${blockIdx}.upsamplers.0.conv`, { pad: 1 });
    }
  }

  const normOut = getWeight(weights, shapes, 'vae.decoder.conv_norm_out.weight');
  const normOutBias = getWeight(weights, shapes, 'vae.decoder.conv_norm_out.bias');
  const normed = await runGroupNorm(state.tensor, normOut.value, normOutBias.value, {
    channels: state.channels,
    height: state.height,
    width: state.width,
    numGroups,
    eps,
  });
  releaseBuffer(state.tensor.buffer);

  const siluOut = await runSiLU(normed, { size: state.channels * state.height * state.width, swigluLimit: null });
  releaseBuffer(normed.buffer);
  state = {
    tensor: reshapeTensor(siluOut, [state.channels, state.height, state.width], 'vae_norm_out'),
    channels: state.channels,
    height: state.height,
    width: state.width,
  };

  state = await applyConv2D(state, weights, shapes, 'vae.decoder.conv_out', { pad: 1 });

  const outputSize = state.channels * state.height * state.width * dtypeBytes(state.tensor.dtype);
  const outputRaw = await readBuffer(state.tensor.buffer, outputSize);
  releaseBuffer(state.tensor.buffer);

  const output = state.tensor.dtype === 'f16'
    ? new Uint16Array(outputRaw)
    : new Float32Array(outputRaw);

  const outHeight = state.height;
  const outWidth = state.width;
  if (outHeight !== options.height || outWidth !== options.width) {
    log.warn('Diffusion', `VAE output size ${outWidth}x${outHeight} differs from request ${options.width}x${options.height}.`);
  }
  const pixels = new Uint8ClampedArray(outWidth * outHeight * 4);
  const height = outHeight;
  const width = outWidth;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const outIndex = (y * width + x) * 4;
      const base = (y * width + x);
      const rIdx = base;
      const gIdx = base + height * width;
      const bIdx = base + 2 * height * width;

      const r = state.tensor.dtype === 'f16' ? f16ToF32(output[rIdx]) : output[rIdx];
      const g = state.tensor.dtype === 'f16' ? f16ToF32(output[gIdx]) : output[gIdx];
      const b = state.tensor.dtype === 'f16' ? f16ToF32(output[bIdx]) : output[bIdx];

      pixels[outIndex] = clamp(Math.round((r * 0.5 + 0.5) * 255), 0, 255);
      pixels[outIndex + 1] = clamp(Math.round((g * 0.5 + 0.5) * 255), 0, 255);
      pixels[outIndex + 2] = clamp(Math.round((b * 0.5 + 0.5) * 255), 0, 255);
      pixels[outIndex + 3] = 255;
    }
  }

  return pixels;
}

function decodeLatentsCPU(latents, options) {
  if (!options) {
    throw new Error('decodeLatents requires options');
  }
  const width = options.width;
  const height = options.height;
  const scale = options.latentScale;
  if (!Number.isFinite(width) || !Number.isFinite(height)) {
    throw new Error('decodeLatents requires width/height');
  }
  if (!Number.isFinite(scale) || scale <= 0) {
    throw new Error('decodeLatents requires latentScale');
  }
  const latentWidth = Number.isFinite(options.latentWidth)
    ? options.latentWidth
    : Math.max(1, Math.floor(width / scale));
  const latentHeight = Number.isFinite(options.latentHeight)
    ? options.latentHeight
    : Math.max(1, Math.floor(height / scale));
  const channels = Number.isFinite(options.latentChannels)
    ? options.latentChannels
    : 0;
  if (!Number.isFinite(channels) || channels <= 0) {
    throw new Error('decodeLatents requires latentChannels');
  }
  const output = new Uint8ClampedArray(width * height * 4);

  for (let y = 0; y < height; y++) {
    const ly = Math.min(latentHeight - 1, Math.floor(y / scale));
    for (let x = 0; x < width; x++) {
      const lx = Math.min(latentWidth - 1, Math.floor(x / scale));
      const latentIndex = (ly * latentWidth + lx) * channels;
      const r = latents[latentIndex] ?? 0;
      const g = latents[latentIndex + 1] ?? r;
      const b = latents[latentIndex + 2] ?? r;
      const outIndex = (y * width + x) * 4;
      output[outIndex] = clamp(Math.round((r * 0.5 + 0.5) * 255), 0, 255);
      output[outIndex + 1] = clamp(Math.round((g * 0.5 + 0.5) * 255), 0, 255);
      output[outIndex + 2] = clamp(Math.round((b * 0.5 + 0.5) * 255), 0, 255);
      output[outIndex + 3] = 255;
    }
  }

  return output;
}

export async function decodeLatents(latents, options) {
  if (options?.weights && getDevice()) {
    return decodeLatentsGPU(latents, options);
  }
  return decodeLatentsCPU(latents, options);
}
