import { getDevice } from '../../../gpu/device.js';
import { acquireBuffer, releaseBuffer, readBuffer, isBufferActive } from '../../../memory/buffer-pool.js';
import { createTensor, dtypeBytes } from '../../../gpu/tensor.js';
import { getBuffer, getWeightDtype } from '../../../gpu/weight-buffer.js';
import { CommandRecorder } from '../../../gpu/command-recorder.js';
import { runConv2D, recordConv2D } from '../../../gpu/kernels/conv2d.js';
import { runGroupNorm, recordGroupNorm } from '../../../gpu/kernels/groupnorm.js';
import { runSiLU, recordSiLU } from '../../../gpu/kernels/silu.js';
import { runMatmul, recordMatmul } from '../../../gpu/kernels/matmul.js';
import { runAttention, recordAttention } from '../../../gpu/kernels/attention.js';
import { runTranspose, recordTranspose } from '../../../gpu/kernels/transpose.js';
import { runResidualAdd, runBiasAdd, recordResidualAdd, recordBiasAdd } from '../../../gpu/kernels/residual.js';
import { runUpsample2D, recordUpsample2D } from '../../../gpu/kernels/upsample2d.js';
import { castF32ToF16, recordCastF32ToF16 } from '../../../gpu/kernels/cast.js';
import { f16ToF32 } from '../../../loader/dtype-utils.js';
import { log } from '../../../debug/index.js';

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

function getWeightOptional(weights, shapes, name) {
  const value = weights.get(name);
  if (!value) return null;
  const shape = shapes.get(name);
  if (!shape) return null;
  return { value, shape };
}

function getWeightByCandidates(weights, shapes, candidates, label) {
  for (const name of candidates) {
    const value = getWeightOptional(weights, shapes, name);
    if (value) {
      return { ...value, name };
    }
  }
  throw new Error(
    `Missing VAE weight: ${label}. Tried: ${candidates.join(', ')}`
  );
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

function getLinearShape(shape, label) {
  if (Array.isArray(shape) && shape.length === 2) {
    return {
      outFeatures: shape[0],
      inFeatures: shape[1],
    };
  }
  if (Array.isArray(shape) && shape.length === 4) {
    if (shape[2] !== 1 || shape[3] !== 1) {
      throw new Error(`Linear weight "${label}" with 4D shape must be 1x1, got ${shape}`);
    }
    return {
      outFeatures: shape[0],
      inFeatures: shape[1],
    };
  }
  throw new Error(`Linear weight shape must be [out,in] or [out,in,1,1], got ${shape}`);
}

function readPositiveInteger(value) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 1) return null;
  return parsed;
}

function resolveAttentionHeadShape(channels, config) {
  const rawHeadDim = Array.isArray(config?.attention_head_dim)
    ? config.attention_head_dim[0]
    : config?.attention_head_dim;
  const configuredHeadDim = readPositiveInteger(rawHeadDim);
  if (configuredHeadDim && channels % configuredHeadDim === 0) {
    return {
      numHeads: channels / configuredHeadDim,
      headDim: configuredHeadDim,
    };
  }

  const configuredNumHeads = readPositiveInteger(config?.num_attention_heads);
  if (configuredNumHeads && channels % configuredNumHeads === 0) {
    return {
      numHeads: configuredNumHeads,
      headDim: channels / configuredNumHeads,
    };
  }

  const fallbackHeadDims = [64, 40, 32, 24, 20, 16, 12, 10, 8, 6, 5, 4, 3, 2, 1];
  const headDim = fallbackHeadDims.find((candidate) => candidate <= channels && channels % candidate === 0) || 1;
  return {
    numHeads: Math.max(1, channels / headDim),
    headDim,
  };
}

function createBiasTensor(weight, label, fallbackDtype = 'f16') {
  if (!weight) return null;
  const dtype = getWeightDtype(weight.value) || fallbackDtype;
  const shape = Array.isArray(weight.shape) && weight.shape.length > 0
    ? weight.shape
    : [0];
  const size = shape.reduce((acc, value) => acc * value, 1);
  if (!Number.isInteger(size) || size < 1) {
    throw new Error(`Bias "${label}" has invalid shape: ${shape}`);
  }
  return createTensor(getBuffer(weight.value), dtype, [size], label);
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

function createKernelOps(recorder) {
  if (!recorder) {
    return {
      conv2d: runConv2D,
      groupNorm: runGroupNorm,
      silu: runSiLU,
      matmul: runMatmul,
      attention: runAttention,
      transpose: runTranspose,
      residualAdd: runResidualAdd,
      biasAdd: runBiasAdd,
      upsample2d: runUpsample2D,
      castF32ToF16,
    };
  }
  return {
    conv2d: (...args) => recordConv2D(recorder, ...args),
    groupNorm: (...args) => recordGroupNorm(recorder, ...args),
    silu: (...args) => recordSiLU(recorder, ...args),
    matmul: (...args) => recordMatmul(recorder, ...args),
    attention: (...args) => recordAttention(recorder, ...args),
    transpose: (...args) => recordTranspose(recorder, ...args),
    residualAdd: (...args) => recordResidualAdd(recorder, ...args),
    biasAdd: (...args) => recordBiasAdd(recorder, ...args),
    upsample2d: (...args) => recordUpsample2D(recorder, ...args),
    castF32ToF16: (...args) => recordCastF32ToF16(recorder, ...args),
  };
}

function createBufferReleaser(recorder) {
  if (!recorder) {
    return (buffer) => {
      if (!buffer || !isBufferActive(buffer)) return;
      releaseBuffer(buffer);
    };
  }
  return (buffer) => {
    if (!buffer) return;
    recorder.trackTemporaryBuffer(buffer);
  };
}

function sumProfileTimings(timings) {
  if (!timings) return null;
  return Object.values(timings).reduce((sum, value) => sum + value, 0);
}

async function applyConv2D(state, weights, shapes, namePrefix, options = {}, ops, release) {
  const weightName = `${namePrefix}.weight`;
  const biasName = `${namePrefix}.bias`;
  const weight = getWeight(weights, shapes, weightName);
  const bias = getWeight(weights, shapes, biasName);
  const { outChannels, inChannels, kernelH, kernelW } = getConvShape(weight.shape);

  if (inChannels !== state.channels) {
    log.warn('Diffusion', `VAE conv channel mismatch: ${namePrefix} in=${inChannels} state=${state.channels}`);
  }

  const output = await ops.conv2d(
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

  release(state.tensor.buffer);

  return {
    tensor: output,
    channels: outChannels,
    height: Math.floor((state.height + (options.pad ?? 1) * 2 - kernelH) / (options.stride ?? 1)) + 1,
    width: Math.floor((state.width + (options.pad ?? 1) * 2 - kernelW) / (options.stride ?? 1)) + 1,
  };
}

async function runResnetBlock(state, weights, shapes, prefix, config, ops, release) {
  const numGroups = config.numGroups;
  const eps = config.eps;
  const channels = state.channels;

  const norm1 = getWeight(weights, shapes, `${prefix}.norm1.weight`);
  const norm1Bias = getWeight(weights, shapes, `${prefix}.norm1.bias`);
  const normed1 = await ops.groupNorm(state.tensor, norm1.value, norm1Bias.value, {
    channels,
    height: state.height,
    width: state.width,
    numGroups,
    eps,
  });

  const silu1 = await ops.silu(normed1, { size: channels * state.height * state.width, swigluLimit: null });
  release(normed1.buffer);
  const silu1View = reshapeTensor(silu1, [channels, state.height, state.width], 'vae_resnet_silu1');

  const conv1 = await applyConv2D(
    { tensor: silu1View, channels, height: state.height, width: state.width },
    weights,
    shapes,
    `${prefix}.conv1`,
    { pad: 1 },
    ops,
    release
  );

  const norm2 = getWeight(weights, shapes, `${prefix}.norm2.weight`);
  const norm2Bias = getWeight(weights, shapes, `${prefix}.norm2.bias`);
  const normed2 = await ops.groupNorm(conv1.tensor, norm2.value, norm2Bias.value, {
    channels: conv1.channels,
    height: conv1.height,
    width: conv1.width,
    numGroups,
    eps,
  });

  release(conv1.tensor.buffer);

  const silu2 = await ops.silu(normed2, { size: conv1.channels * conv1.height * conv1.width, swigluLimit: null });
  release(normed2.buffer);
  const silu2View = reshapeTensor(silu2, [conv1.channels, conv1.height, conv1.width], 'vae_resnet_silu2');

  const conv2 = await applyConv2D(
    { tensor: silu2View, channels: conv1.channels, height: conv1.height, width: conv1.width },
    weights,
    shapes,
    `${prefix}.conv2`,
    { pad: 1 },
    ops,
    release
  );

  let residualTensor = state.tensor;

  if (weights.has(`${prefix}.conv_shortcut.weight`)) {
    const shortcut = await applyConv2D(state, weights, shapes, `${prefix}.conv_shortcut`, { pad: 0 }, ops, release);
    residualTensor = shortcut.tensor;
  }

  const size = conv2.channels * conv2.height * conv2.width;
  const residual = reshapeTensor(residualTensor, [size], 'vae_resnet_residual');
  const output = await ops.residualAdd(
    reshapeTensor(conv2.tensor, [size], 'vae_resnet_main'),
    residual,
    size,
    { useVec4: true }
  );

  if (residualTensor === state.tensor) {
    release(state.tensor.buffer);
  } else {
    release(residualTensor.buffer);
  }

  release(conv2.tensor.buffer);

  return {
    tensor: reshapeTensor(output, [conv2.channels, conv2.height, conv2.width], 'vae_resnet_output'),
    channels: conv2.channels,
    height: conv2.height,
    width: conv2.width,
  };
}

async function runMidBlockAttention(state, weights, shapes, prefix, config, ops, release) {
  const channels = state.channels;
  const height = state.height;
  const width = state.width;
  const spatial = height * width;
  if (!Number.isFinite(spatial) || spatial <= 0) {
    throw new Error('VAE mid-block attention requires a positive spatial size.');
  }

  const normWeight = getWeightByCandidates(
    weights,
    shapes,
    [`${prefix}.group_norm.weight`, `${prefix}.norm.weight`],
    `${prefix}.group_norm.weight`
  );
  const normBias = getWeightByCandidates(
    weights,
    shapes,
    [`${prefix}.group_norm.bias`, `${prefix}.norm.bias`],
    `${prefix}.group_norm.bias`
  );

  const normed = await ops.groupNorm(state.tensor, normWeight.value, normBias.value, {
    channels,
    height,
    width,
    numGroups: config.numGroups,
    eps: config.eps,
  });
  const normedChannelsSpatial = reshapeTensor(normed, [channels, spatial], 'vae_attn_norm_cs');
  const normedTokens = await ops.transpose(normedChannelsSpatial, channels, spatial);
  release(normed.buffer);

  const residualChannelsSpatial = reshapeTensor(state.tensor, [channels, spatial], 'vae_attn_residual_cs');
  const residualTokens = await ops.transpose(residualChannelsSpatial, channels, spatial);
  release(state.tensor.buffer);

  const qWeight = getWeightByCandidates(weights, shapes, [`${prefix}.to_q.weight`], `${prefix}.to_q.weight`);
  const kWeight = getWeightByCandidates(weights, shapes, [`${prefix}.to_k.weight`], `${prefix}.to_k.weight`);
  const vWeight = getWeightByCandidates(weights, shapes, [`${prefix}.to_v.weight`], `${prefix}.to_v.weight`);
  const qBias = getWeightOptional(weights, shapes, `${prefix}.to_q.bias`);
  const kBias = getWeightOptional(weights, shapes, `${prefix}.to_k.bias`);
  const vBias = getWeightOptional(weights, shapes, `${prefix}.to_v.bias`);
  const qShape = getLinearShape(qWeight.shape, qWeight.name);
  const kShape = getLinearShape(kWeight.shape, kWeight.name);
  const vShape = getLinearShape(vWeight.shape, vWeight.name);

  if (qShape.inFeatures !== channels || kShape.inFeatures !== channels || vShape.inFeatures !== channels) {
    throw new Error(
      `VAE mid-block attention projection mismatch: expected inFeatures=${channels}, ` +
      `got q=${qShape.inFeatures}, k=${kShape.inFeatures}, v=${vShape.inFeatures}.`
    );
  }
  if (qShape.outFeatures !== kShape.outFeatures || qShape.outFeatures !== vShape.outFeatures) {
    throw new Error(
      `VAE mid-block attention projection mismatch: q/k/v outFeatures differ ` +
      `(${qShape.outFeatures}, ${kShape.outFeatures}, ${vShape.outFeatures}).`
    );
  }

  const hiddenSize = qShape.outFeatures;
  const projectionDtype = normedTokens.dtype;
  let q = await ops.matmul(normedTokens, qWeight.value, spatial, hiddenSize, channels, {
    outputDtype: projectionDtype,
    transposeB: 'auto',
  });
  let k = await ops.matmul(normedTokens, kWeight.value, spatial, hiddenSize, channels, {
    outputDtype: projectionDtype,
    transposeB: 'auto',
  });
  let v = await ops.matmul(normedTokens, vWeight.value, spatial, hiddenSize, channels, {
    outputDtype: projectionDtype,
    transposeB: 'auto',
  });

  const qBiasTensor = createBiasTensor(qBias, `${prefix}.to_q.bias`, projectionDtype);
  const kBiasTensor = createBiasTensor(kBias, `${prefix}.to_k.bias`, projectionDtype);
  const vBiasTensor = createBiasTensor(vBias, `${prefix}.to_v.bias`, projectionDtype);
  if (qBiasTensor) q = await ops.biasAdd(q, qBiasTensor, spatial, hiddenSize);
  if (kBiasTensor) k = await ops.biasAdd(k, kBiasTensor, spatial, hiddenSize);
  if (vBiasTensor) v = await ops.biasAdd(v, vBiasTensor, spatial, hiddenSize);

  const { numHeads, headDim } = resolveAttentionHeadShape(hiddenSize, config.modelConfig);
  const attn = await ops.attention(
    q,
    k,
    v,
    null,
    numHeads,
    headDim,
    {
      seqLen: spatial,
      kvLen: spatial,
      numKVHeads: numHeads,
      causal: false,
    }
  );
  release(q.buffer);
  release(k.buffer);
  release(v.buffer);

  const outWeight = getWeightByCandidates(
    weights,
    shapes,
    [`${prefix}.to_out.0.weight`, `${prefix}.to_out.weight`],
    `${prefix}.to_out.0.weight`
  );
  const outBias = getWeightOptional(weights, shapes, `${prefix}.to_out.0.bias`)
    || getWeightOptional(weights, shapes, `${prefix}.to_out.bias`);
  const outShape = getLinearShape(outWeight.shape, outWeight.name);
  if (outShape.inFeatures !== hiddenSize) {
    throw new Error(
      `VAE mid-block attention output projection mismatch: expected inFeatures=${hiddenSize}, got ${outShape.inFeatures}.`
    );
  }
  if (outShape.outFeatures !== channels) {
    throw new Error(
      `VAE mid-block attention output projection mismatch: expected outFeatures=${channels}, got ${outShape.outFeatures}.`
    );
  }

  let projected = await ops.matmul(attn, outWeight.value, spatial, outShape.outFeatures, outShape.inFeatures, {
    outputDtype: projectionDtype,
    transposeB: 'auto',
  });
  release(attn.buffer);
  const outBiasTensor = createBiasTensor(outBias, `${prefix}.to_out.0.bias`, projectionDtype);
  if (outBiasTensor) {
    projected = await ops.biasAdd(projected, outBiasTensor, spatial, outShape.outFeatures);
  }

  const combined = await ops.residualAdd(projected, residualTokens, spatial * outShape.outFeatures, { useVec4: true });
  release(projected.buffer);
  release(residualTokens.buffer);
  release(normedTokens.buffer);

  const combinedChannelsSpatial = await ops.transpose(combined, spatial, outShape.outFeatures);
  release(combined.buffer);

  return {
    tensor: reshapeTensor(combinedChannelsSpatial, [outShape.outFeatures, height, width], 'vae_attn_out'),
    channels: outShape.outFeatures,
    height,
    width,
  };
}

async function decodeLatentsGPU(latents, options) {
  const device = getDevice();
  if (!device) {
    throw new Error('VAE GPU decode requires a WebGPU device.');
  }

  const profileTarget = options.profile ?? null;
  const wantsProfile = profileTarget === true || typeof profileTarget === 'object';
  const localRecorder = wantsProfile
    ? new CommandRecorder(device, 'vae_decode', { profile: true })
    : null;
  const recorder = localRecorder;
  const ops = createKernelOps(recorder);
  const release = createBufferReleaser(recorder);

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

  const computeDtype = runtime.latent?.dtype;
  if (!computeDtype) {
    throw new Error('VAE decode requires runtime.latent.dtype.');
  }
  if (computeDtype !== 'f16') {
    throw new Error(
      `VAE GPU decode requires runtime.latent.dtype="f16"; got "${computeDtype}".`
    );
  }
  const casted = await ops.castF32ToF16(state.tensor);
  release(state.tensor.buffer);
  state = {
    tensor: reshapeTensor(casted, [state.channels, state.height, state.width], 'vae_latents_f16'),
    channels: state.channels,
    height: state.height,
    width: state.width,
  };

  state = await applyConv2D(state, weights, shapes, 'vae.decoder.conv_in', { pad: 1 }, ops, release);

  const midResnetPrefix = 'vae.decoder.mid_block.resnets.';
  const midResnetIds = buildIndexList(weights, midResnetPrefix);
  for (const idx of midResnetIds) {
    state = await runResnetBlock(state, weights, shapes, `${midResnetPrefix}${idx}`, { numGroups, eps }, ops, release);
  }

  const midAttentionPrefix = 'vae.decoder.mid_block.attentions.';
  const midAttentionIds = buildIndexList(weights, midAttentionPrefix);
  for (const idx of midAttentionIds) {
    state = await runMidBlockAttention(
      state,
      weights,
      shapes,
      `${midAttentionPrefix}${idx}`,
      {
        numGroups,
        eps,
        modelConfig: config,
      },
      ops,
      release
    );
  }

  const upBlockPrefix = 'vae.decoder.up_blocks.';
  const upBlocks = buildIndexList(weights, upBlockPrefix);
  for (const blockIdx of upBlocks) {
    const resnetPrefix = `${upBlockPrefix}${blockIdx}.resnets.`;
    const resnetIds = buildIndexList(weights, resnetPrefix);
    for (const idx of resnetIds) {
      state = await runResnetBlock(state, weights, shapes, `${resnetPrefix}${idx}`, { numGroups, eps }, ops, release);
    }

    const upsampleWeightName = `${upBlockPrefix}${blockIdx}.upsamplers.0.conv.weight`;
    if (weights.has(upsampleWeightName)) {
      const upsample = await ops.upsample2d(state.tensor, {
        channels: state.channels,
        height: state.height,
        width: state.width,
        scale: 2,
      });
      release(state.tensor.buffer);
      state = {
        tensor: reshapeTensor(upsample, [state.channels, state.height * 2, state.width * 2], 'vae_upsample'),
        channels: state.channels,
        height: state.height * 2,
        width: state.width * 2,
      };

      state = await applyConv2D(state, weights, shapes, `${upBlockPrefix}${blockIdx}.upsamplers.0.conv`, { pad: 1 }, ops, release);
    }
  }

  const normOut = getWeight(weights, shapes, 'vae.decoder.conv_norm_out.weight');
  const normOutBias = getWeight(weights, shapes, 'vae.decoder.conv_norm_out.bias');
  const normed = await ops.groupNorm(state.tensor, normOut.value, normOutBias.value, {
    channels: state.channels,
    height: state.height,
    width: state.width,
    numGroups,
    eps,
  });
  release(state.tensor.buffer);

  const siluOut = await ops.silu(normed, { size: state.channels * state.height * state.width, swigluLimit: null });
  release(normed.buffer);
  state = {
    tensor: reshapeTensor(siluOut, [state.channels, state.height, state.width], 'vae_norm_out'),
    channels: state.channels,
    height: state.height,
    width: state.width,
  };

  state = await applyConv2D(state, weights, shapes, 'vae.decoder.conv_out', { pad: 1 }, ops, release);

  const outputSize = state.channels * state.height * state.width * dtypeBytes(state.tensor.dtype);
  if (localRecorder) {
    localRecorder.submit();
  }
  const outputRaw = await readBuffer(state.tensor.buffer, outputSize);
  releaseBuffer(state.tensor.buffer);

  if (localRecorder) {
    const timings = await localRecorder.resolveProfileTimings();
    if (profileTarget && typeof profileTarget === 'object') {
      profileTarget.totalMs = sumProfileTimings(timings) ?? null;
      profileTarget.timings = timings ?? null;
    }
  }

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

export async function decodeLatents(latents, options) {
  if (!options?.weights || !getDevice()) {
    throw new Error(
      'Diffusion decode requires GPU VAE weights and a WebGPU device. ' +
      'CPU decode fallback is unsupported.'
    );
  }
  return decodeLatentsGPU(latents, options);
}
