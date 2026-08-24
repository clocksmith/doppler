import { getDevice } from '../../../../gpu/device.js';
import { acquireBuffer, releaseBuffer, readBuffer, isBufferActive } from '../../../../memory/buffer-pool.js';
import { createTensor, dtypeBytes } from '../../../../gpu/tensor.js';
import { getBuffer, getWeightDtype } from '../../../../gpu/weight-buffer.js';
import { CommandRecorder } from '../../../../gpu/command-recorder.js';
import { runConv2D, recordConv2D } from '../../../../gpu/kernels/conv2d.js';
import { runGroupNorm, recordGroupNorm } from '../../../../gpu/kernels/groupnorm.js';
import { runRMSNorm, recordRMSNorm } from '../../../../gpu/kernels/rmsnorm.js';
import { runSiLU, runSiLURowSplit, recordSiLU, recordSiLURowSplit } from '../../../../gpu/kernels/silu.js';
import { runMatmul, recordMatmul } from '../../../../gpu/kernels/matmul.js';
import { runAttention, recordAttention } from '../../../../gpu/kernels/attention.js';
import { runTranspose, recordTranspose } from '../../../../gpu/kernels/transpose.js';
import { runResidualAdd, runBiasAdd, recordResidualAdd, recordBiasAdd } from '../../../../gpu/kernels/residual.js';
import { runUpsample2D, recordUpsample2D } from '../../../../gpu/kernels/upsample2d.js';
import { runDepthwiseConv2D, recordDepthwiseConv2D } from '../../../../gpu/kernels/depthwise_conv2d.js';
import { runGroupedPointwiseConv2D, recordGroupedPointwiseConv2D } from '../../../../gpu/kernels/grouped_pointwise_conv2d.js';
import { runLinearAttention, recordLinearAttention } from '../../../../gpu/kernels/linear_attention.js';
import { runPixelShuffle, recordPixelShuffle } from '../../../../gpu/kernels/pixel_shuffle.js';
import { runRepeatChannels, recordRepeatChannels } from '../../../../gpu/kernels/repeat_channels.js';
import { runReLU, recordReLU } from '../../../../gpu/kernels/relu.js';
import { castF32ToF16, recordCastF32ToF16 } from '../../../../gpu/kernels/cast.js';
import { f16ToF32 } from '../../../../loader/dtype-utils.js';
import { log } from '../../../../debug/index.js';

export function reshapeTensor(tensor, shape, label) {
  return createTensor(tensor.buffer, tensor.dtype, shape, label ?? tensor.label);
}

export function getWeight(weights, shapes, name) {
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

export function getWeightOptional(weights, shapes, name) {
  const value = weights.get(name);
  if (!value) return null;
  const shape = shapes.get(name);
  if (!shape) return null;
  return { value, shape };
}

export function getConvShape(shape) {
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

export function getLinearShape(shape, label) {
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

export function createBiasTensor(weight, label, fallbackDtype = 'f16') {
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

export function normalizePerBlockValue(value, count, label) {
  if (Array.isArray(value)) {
    if (value.length !== count) {
      throw new Error(`${label} must have ${count} entries, got ${value.length}.`);
    }
    return value;
  }
  return Array.from({ length: count }, () => value);
}

export function tensorElementCount(tensor) {
  if (!Array.isArray(tensor?.shape) || tensor.shape.length === 0) {
    throw new Error('Tensor shape is required.');
  }
  return tensor.shape.reduce((acc, value) => acc * value, 1);
}

export async function applyConv2D(state, weights, shapes, namePrefix, options = {}, ops, release) {
  const weightName = `${namePrefix}.weight`;
  const biasName = `${namePrefix}.bias`;
  const weight = getWeight(weights, shapes, weightName);
  const bias = getWeightOptional(weights, shapes, biasName);
  const { outChannels, inChannels, kernelH, kernelW } = getConvShape(weight.shape);

  if (inChannels !== state.channels) {
    log.warn('Diffusion', `VAE conv channel mismatch: ${namePrefix} in=${inChannels} state=${state.channels}`);
  }

  const output = await ops.conv2d(
    state.tensor,
    weight.value,
    bias?.value ?? null,
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

export async function submitCopyWork(device, recorder, encoder) {
  if (recorder) {
    return;
  }
  device.queue.submit([encoder.finish()]);
}

export async function concatChannelTensors(tensors, height, width, recorder) {
  if (!Array.isArray(tensors) || tensors.length === 0) {
    throw new Error('concatChannelTensors requires at least one tensor.');
  }
  const device = getDevice();
  if (!device) {
    throw new Error('Channel tensor concatenation requires a WebGPU device.');
  }
  const dtype = tensors[0].dtype;
  const bytesPerElement = dtypeBytes(dtype);
  let totalChannels = 0;
  for (const tensor of tensors) {
    if (tensor.dtype !== dtype) {
      throw new Error('concatChannelTensors requires matching dtypes.');
    }
    if (tensor.shape[1] !== height || tensor.shape[2] !== width) {
      throw new Error('concatChannelTensors requires matching spatial dimensions.');
    }
    totalChannels += tensor.shape[0];
  }

  const output = acquireBuffer(totalChannels * height * width * bytesPerElement, undefined, 'vae_concat_channels');
  const encoder = recorder ? recorder.getEncoder() : device.createCommandEncoder({ label: 'vae_concat_channels' });
  let channelOffset = 0;
  for (const tensor of tensors) {
    const byteLength = tensor.shape[0] * height * width * bytesPerElement;
    encoder.copyBufferToBuffer(
      tensor.buffer,
      0,
      output,
      channelOffset * height * width * bytesPerElement,
      byteLength
    );
    channelOffset += tensor.shape[0];
  }
  await submitCopyWork(device, recorder, encoder);
  return createTensor(output, dtype, [totalChannels, height, width], 'vae_concat_channels');
}

export async function sliceChannelTensor(tensor, startChannel, channelCount, height, width, recorder) {
  const device = getDevice();
  if (!device) {
    throw new Error('Channel tensor slicing requires a WebGPU device.');
  }
  const bytesPerElement = dtypeBytes(tensor.dtype);
  const channelSize = height * width * bytesPerElement;
  const output = acquireBuffer(channelCount * channelSize, undefined, 'vae_slice_channels');
  const encoder = recorder ? recorder.getEncoder() : device.createCommandEncoder({ label: 'vae_slice_channels' });
  encoder.copyBufferToBuffer(
    tensor.buffer,
    startChannel * channelSize,
    output,
    0,
    channelCount * channelSize
  );
  await submitCopyWork(device, recorder, encoder);
  return createTensor(output, tensor.dtype, [channelCount, height, width], 'vae_slice_channels');
}

export async function runChannelwiseRmsNorm(state, normWeight, normBias, eps, ops, release) {
  const spatial = state.height * state.width;
  const channelsSpatial = reshapeTensor(state.tensor, [state.channels, spatial], 'vae_rmsnorm_channels_spatial');
  const tokens = await ops.transpose(channelsSpatial, state.channels, spatial);
  const normed = await ops.rmsNorm(tokens, normWeight.value, eps, {
    batchSize: spatial,
    hiddenSize: state.channels,
  });
  release(tokens.buffer);
  let shifted = normed;
  if (normBias) {
    const biasTensor = createBiasTensor(normBias, `${normBias.name ?? 'vae_rmsnorm_bias'}`, normed.dtype);
    shifted = await ops.biasAdd(normed, biasTensor, spatial, state.channels);
    release(normed.buffer);
  }
  const channelsFirst = await ops.transpose(shifted, spatial, state.channels);
  release(shifted.buffer);
  return {
    tensor: reshapeTensor(channelsFirst, [state.channels, state.height, state.width], 'vae_rmsnorm_output'),
    channels: state.channels,
    height: state.height,
    width: state.width,
  };
}

export async function channelsToTokens(state, ops) {
  const spatial = state.height * state.width;
  const channelsSpatial = reshapeTensor(state.tensor, [state.channels, spatial], 'vae_channels_spatial');
  const tokens = await ops.transpose(channelsSpatial, state.channels, spatial);
  return {
    tensor: tokens,
    numTokens: spatial,
  };
}

export async function tokensToChannels(tokens, channels, height, width, ops) {
  const channelsSpatial = await ops.transpose(tokens, height * width, channels);
  return {
    tensor: reshapeTensor(channelsSpatial, [channels, height, width], 'vae_tokens_channels'),
    channels,
    height,
    width,
  };
}

export async function runAutoencoderDCInputProjection(state, weights, shapes, config, ops, release) {
  const blockOutChannels = config.decoder_block_out_channels;
  if (!Array.isArray(blockOutChannels) || blockOutChannels.length === 0) {
    throw new Error('AutoencoderDC decode requires decoder_block_out_channels.');
  }
  const outChannels = blockOutChannels[blockOutChannels.length - 1];
  const repeats = outChannels / state.channels;
  if (!Number.isInteger(repeats) || repeats < 1) {
    throw new Error(
      `AutoencoderDC input shortcut requires an integer repeat factor; got ${outChannels}/${state.channels}.`
    );
  }

  const shortcut = await ops.repeatChannels(state.tensor, {
    inChannels: state.channels,
    height: state.height,
    width: state.width,
    repeats,
  });
  const projected = await applyConv2D(state, weights, shapes, 'vae.decoder.conv_in', { pad: 1 }, ops, release);
  const size = projected.channels * projected.height * projected.width;
  const combined = await ops.residualAdd(
    reshapeTensor(projected.tensor, [size], 'vae_dc_conv_in'),
    reshapeTensor(shortcut, [size], 'vae_dc_conv_in_shortcut'),
    size,
    { useVec4: true }
  );
  release(projected.tensor.buffer);
  release(shortcut.buffer);
  return {
    tensor: reshapeTensor(combined, [projected.channels, projected.height, projected.width], 'vae_dc_conv_in_out'),
    channels: projected.channels,
    height: projected.height,
    width: projected.width,
  };
}

export async function runAutoencoderDCUpBlock(state, weights, shapes, prefix, config, ops, release, recorder) {
  const convWeight = getWeight(weights, shapes, `${prefix}.conv.weight`);
  const convBias = getWeightOptional(weights, shapes, `${prefix}.conv.bias`);
  const { outChannels, inChannels, kernelH, kernelW } = getConvShape(convWeight.shape);
  if (inChannels !== state.channels) {
    throw new Error(
      `AutoencoderDC up block "${prefix}" expected ${inChannels} input channels, got ${state.channels}.`
    );
  }
  const factor = 2;
  const outHeight = state.height * factor;
  const outWidth = state.width * factor;
  const shortcutRepeats = outChannels * factor * factor / state.channels;
  if (!Number.isInteger(shortcutRepeats) || shortcutRepeats < 1) {
    throw new Error(
      `AutoencoderDC up block "${prefix}" requires integer shortcut repeats; got ${outChannels}/${state.channels}.`
    );
  }

  let projected;
  if (config.upsample_block_type === 'interpolate') {
    const upsampled = await ops.upsample2d(state.tensor, {
      channels: state.channels,
      height: state.height,
      width: state.width,
      scale: factor,
    });
    projected = await ops.conv2d(
      reshapeTensor(upsampled, [state.channels, outHeight, outWidth], 'vae_dc_upsample'),
      convWeight.value,
      convBias?.value ?? null,
      {
        inChannels: state.channels,
        outChannels,
        height: outHeight,
        width: outWidth,
        kernelH,
        kernelW,
        stride: 1,
        pad: 1,
      }
    );
    release(upsampled.buffer);
  } else if (config.upsample_block_type === 'pixel_shuffle') {
    const conv = await ops.conv2d(state.tensor, convWeight.value, convBias?.value ?? null, {
      inChannels: state.channels,
      outChannels: outChannels * factor * factor,
      height: state.height,
      width: state.width,
      kernelH,
      kernelW,
      stride: 1,
      pad: 1,
    });
    projected = await ops.pixelShuffle(conv, {
      outChannels,
      outHeight,
      outWidth,
      gridWidth: state.width,
      gridHeight: state.height,
      patchSize: factor,
      patchChannels: outChannels * factor * factor,
    });
    release(conv.buffer);
  } else {
    throw new Error(
      `Unsupported AutoencoderDC upsample_block_type "${config.upsample_block_type}".`
    );
  }

  const repeated = await ops.repeatChannels(state.tensor, {
    inChannels: state.channels,
    height: state.height,
    width: state.width,
    repeats: shortcutRepeats,
  });
  const shortcut = await ops.pixelShuffle(repeated, {
    outChannels,
    outHeight,
    outWidth,
    gridWidth: state.width,
    gridHeight: state.height,
    patchSize: factor,
    patchChannels: state.channels * shortcutRepeats,
  });
  release(repeated.buffer);
  release(state.tensor.buffer);

  const size = outChannels * outHeight * outWidth;
  const combined = await ops.residualAdd(
    reshapeTensor(projected, [size], 'vae_dc_up_main'),
    reshapeTensor(shortcut, [size], 'vae_dc_up_shortcut'),
    size,
    { useVec4: true }
  );
  release(projected.buffer);
  release(shortcut.buffer);
  return {
    tensor: reshapeTensor(combined, [outChannels, outHeight, outWidth], 'vae_dc_up_out'),
    channels: outChannels,
    height: outHeight,
    width: outWidth,
  };
}

export async function runAutoencoderDCResBlock(state, weights, shapes, prefix, eps, ops, release) {
  const conv1Weight = getWeight(weights, shapes, `${prefix}.conv1.weight`);
  const conv1Bias = getWeightOptional(weights, shapes, `${prefix}.conv1.bias`);
  const conv1Shape = getConvShape(conv1Weight.shape);
  const conv1Tensor = await ops.conv2d(state.tensor, conv1Weight.value, conv1Bias?.value ?? null, {
    inChannels: conv1Shape.inChannels,
    outChannels: conv1Shape.outChannels,
    height: state.height,
    width: state.width,
    kernelH: conv1Shape.kernelH,
    kernelW: conv1Shape.kernelW,
    stride: 1,
    pad: 1,
  });
  const conv1 = {
    tensor: reshapeTensor(conv1Tensor, [conv1Shape.outChannels, state.height, state.width], 'vae_dc_resblock_conv1'),
    channels: conv1Shape.outChannels,
    height: state.height,
    width: state.width,
  };
  const activated = await ops.silu(conv1.tensor, {
    size: conv1.channels * conv1.height * conv1.width,
    swigluLimit: null,
  });
  release(conv1.tensor.buffer);
  const conv2Weight = getWeight(weights, shapes, `${prefix}.conv2.weight`);
  const conv2Shape = getConvShape(conv2Weight.shape);
  const conv2 = await ops.conv2d(
    reshapeTensor(activated, [conv1.channels, conv1.height, conv1.width], 'vae_dc_resblock_act'),
    conv2Weight.value,
    null,
    {
      inChannels: conv1.channels,
      outChannels: conv2Shape.outChannels,
      height: conv1.height,
      width: conv1.width,
      kernelH: 3,
      kernelW: 3,
      stride: 1,
      pad: 1,
    }
  );
  release(activated.buffer);

  const normed = await runChannelwiseRmsNorm(
    {
      tensor: reshapeTensor(conv2, [conv2Shape.outChannels, conv1.height, conv1.width], 'vae_dc_resblock_conv2'),
      channels: conv2Shape.outChannels,
      height: conv1.height,
      width: conv1.width,
    },
    getWeight(weights, shapes, `${prefix}.norm.weight`),
    getWeightOptional(weights, shapes, `${prefix}.norm.bias`),
    eps,
    ops,
    release
  );
  release(conv2.buffer);

  const size = normed.channels * normed.height * normed.width;
  const combined = await ops.residualAdd(
    reshapeTensor(normed.tensor, [size], 'vae_dc_resblock_main'),
    reshapeTensor(state.tensor, [size], 'vae_dc_resblock_residual'),
    size,
    { useVec4: true }
  );
  release(normed.tensor.buffer);
  release(state.tensor.buffer);
  return {
    tensor: reshapeTensor(combined, [normed.channels, normed.height, normed.width], 'vae_dc_resblock_out'),
    channels: normed.channels,
    height: normed.height,
    width: normed.width,
  };
}

export async function runAutoencoderDCAttention(state, weights, shapes, prefix, attentionHeadDim, qkvMultiscales, eps, ops, release, recorder) {
  const qWeight = getWeight(weights, shapes, `${prefix}.attn.to_q.weight`);
  const kWeight = getWeight(weights, shapes, `${prefix}.attn.to_k.weight`);
  const vWeight = getWeight(weights, shapes, `${prefix}.attn.to_v.weight`);
  const qShape = getLinearShape(qWeight.shape, `${prefix}.attn.to_q.weight`);
  const innerDim = qShape.outFeatures;
  if (qShape.inFeatures !== state.channels || innerDim !== getLinearShape(kWeight.shape, `${prefix}.attn.to_k.weight`).outFeatures || innerDim !== getLinearShape(vWeight.shape, `${prefix}.attn.to_v.weight`).outFeatures) {
    throw new Error(`AutoencoderDC attention "${prefix}" has incompatible q/k/v projection shapes.`);
  }
  if (!Number.isFinite(attentionHeadDim) || attentionHeadDim <= 0 || innerDim % attentionHeadDim !== 0) {
    throw new Error(`AutoencoderDC attention "${prefix}" requires innerDim divisible by attentionHeadDim.`);
  }
  const numHeads = innerDim / attentionHeadDim;
  const baseOptions = {
    inChannels: state.channels,
    outChannels: innerDim,
    height: state.height,
    width: state.width,
    groups: 1,
  };
  const qBase = await ops.groupedPointwiseConv2d(state.tensor, qWeight.value, null, baseOptions);
  const kBase = await ops.groupedPointwiseConv2d(state.tensor, kWeight.value, null, baseOptions);
  const vBase = await ops.groupedPointwiseConv2d(state.tensor, vWeight.value, null, baseOptions);
  const qVariants = [qBase];
  const kVariants = [kBase];
  const vVariants = [vBase];

  if (Array.isArray(qkvMultiscales)) {
    const qkvBase = await concatChannelTensors([qBase, kBase, vBase], state.height, state.width, recorder);
    for (let scaleIdx = 0; scaleIdx < qkvMultiscales.length; scaleIdx++) {
      const depthWeight = getWeight(weights, shapes, `${prefix}.attn.to_qkv_multiscale.${scaleIdx}.proj_in.weight`);
      const pointWeight = getWeight(weights, shapes, `${prefix}.attn.to_qkv_multiscale.${scaleIdx}.proj_out.weight`);
      const depthShape = getConvShape(depthWeight.shape);
      const pointShape = getConvShape(pointWeight.shape);
      const groups = pointShape.outChannels / pointShape.inChannels;
      const depth = await ops.depthwiseConv2d(qkvBase, depthWeight.value, null, {
        channels: qkvBase.shape[0],
        height: state.height,
        width: state.width,
        kernelH: depthShape.kernelH,
        kernelW: depthShape.kernelW,
        stride: 1,
        pad: Math.floor(depthShape.kernelH / 2),
      });
      const projected = await ops.groupedPointwiseConv2d(depth, pointWeight.value, null, {
        inChannels: qkvBase.shape[0],
        outChannels: pointShape.outChannels,
        height: state.height,
        width: state.width,
        groups,
      });
      release(depth.buffer);

      const qScale = await sliceChannelTensor(projected, 0, innerDim, state.height, state.width, recorder);
      const kScale = await sliceChannelTensor(projected, innerDim, innerDim, state.height, state.width, recorder);
      const vScale = await sliceChannelTensor(projected, innerDim * 2, innerDim, state.height, state.width, recorder);
      release(projected.buffer);
      qVariants.push(qScale);
      kVariants.push(kScale);
      vVariants.push(vScale);
    }
    release(qkvBase.buffer);
  }

  const qAll = await concatChannelTensors(qVariants, state.height, state.width, recorder);
  const kAll = await concatChannelTensors(kVariants, state.height, state.width, recorder);
  const vAll = await concatChannelTensors(vVariants, state.height, state.width, recorder);
  for (const tensor of qVariants) release(tensor.buffer);
  for (const tensor of kVariants) release(tensor.buffer);
  for (const tensor of vVariants) release(tensor.buffer);

  const qTokens = await channelsToTokens({ tensor: qAll, channels: qAll.shape[0], height: state.height, width: state.width }, ops);
  const kTokens = await channelsToTokens({ tensor: kAll, channels: kAll.shape[0], height: state.height, width: state.width }, ops);
  const vTokens = await channelsToTokens({ tensor: vAll, channels: vAll.shape[0], height: state.height, width: state.width }, ops);
  release(qAll.buffer);
  release(kAll.buffer);
  release(vAll.buffer);

  const qRelu = await ops.relu(qTokens.tensor, { count: tensorElementCount(qTokens.tensor) });
  const kRelu = await ops.relu(kTokens.tensor, { count: tensorElementCount(kTokens.tensor) });
  release(qTokens.tensor.buffer);
  release(kTokens.tensor.buffer);

  const allHeads = numHeads * qVariants.length;
  const attention = await ops.linearAttention(qRelu, kRelu, vTokens.tensor, {
    numHeads: allHeads,
    headDim: attentionHeadDim,
    numTokens: qTokens.numTokens,
    hiddenSize: allHeads * attentionHeadDim,
    eps,
  });
  release(qRelu.buffer);
  release(kRelu.buffer);
  release(vTokens.tensor.buffer);

  const attended = await tokensToChannels(attention, allHeads * attentionHeadDim, state.height, state.width, ops);
  release(attention.buffer);
  const outWeight = getWeight(weights, shapes, `${prefix}.attn.to_out.weight`);
  const outShape = getLinearShape(outWeight.shape, `${prefix}.attn.to_out.weight`);
  const projected = await ops.groupedPointwiseConv2d(attended.tensor, outWeight.value, null, {
    inChannels: attended.channels,
    outChannels: outShape.outFeatures,
    height: state.height,
    width: state.width,
    groups: 1,
  });
  release(attended.tensor.buffer);
  const normed = await runChannelwiseRmsNorm(
    {
      tensor: reshapeTensor(projected, [outShape.outFeatures, state.height, state.width], 'vae_dc_attn_projected'),
      channels: outShape.outFeatures,
      height: state.height,
      width: state.width,
    },
    getWeight(weights, shapes, `${prefix}.attn.norm_out.weight`),
    getWeightOptional(weights, shapes, `${prefix}.attn.norm_out.bias`),
    1e-5,
    ops,
    release
  );
  release(projected.buffer);

  const size = normed.channels * normed.height * normed.width;
  const combined = await ops.residualAdd(
    reshapeTensor(normed.tensor, [size], 'vae_dc_attn_main'),
    reshapeTensor(state.tensor, [size], 'vae_dc_attn_residual'),
    size,
    { useVec4: true }
  );
  release(normed.tensor.buffer);
  release(state.tensor.buffer);
  return {
    tensor: reshapeTensor(combined, [normed.channels, normed.height, normed.width], 'vae_dc_attn_out'),
    channels: normed.channels,
    height: normed.height,
    width: normed.width,
  };
}

export async function runAutoencoderDCGlumbConv(state, weights, shapes, prefix, eps, ops, release) {
  const invertedWeight = getWeight(weights, shapes, `${prefix}.conv_out.conv_inverted.weight`);
  const invertedBias = getWeightOptional(weights, shapes, `${prefix}.conv_out.conv_inverted.bias`);
  const invertedShape = getLinearShape(invertedWeight.shape, `${prefix}.conv_out.conv_inverted.weight`);
  const hiddenChannels = Math.floor(invertedShape.outFeatures / 2);
  const inverted = await ops.groupedPointwiseConv2d(state.tensor, invertedWeight.value, invertedBias?.value ?? null, {
    inChannels: state.channels,
    outChannels: invertedShape.outFeatures,
    height: state.height,
    width: state.width,
    groups: 1,
  });
  const activated = await ops.silu(inverted, {
    size: invertedShape.outFeatures * state.height * state.width,
    swigluLimit: null,
  });
  release(inverted.buffer);
  const depthWeight = getWeight(weights, shapes, `${prefix}.conv_out.conv_depth.weight`);
  const depthBias = getWeightOptional(weights, shapes, `${prefix}.conv_out.conv_depth.bias`);
  const depthShape = getConvShape(depthWeight.shape);
  const depth = await ops.depthwiseConv2d(
    reshapeTensor(activated, [invertedShape.outFeatures, state.height, state.width], 'vae_dc_glumb_act'),
    depthWeight.value,
    depthBias?.value ?? null,
    {
      channels: invertedShape.outFeatures,
      height: state.height,
      width: state.width,
      kernelH: depthShape.kernelH,
      kernelW: depthShape.kernelW,
      stride: 1,
      pad: 1,
    }
  );
  release(activated.buffer);
  const depthTokens = await channelsToTokens({ tensor: depth, channels: invertedShape.outFeatures, height: state.height, width: state.width }, ops);
  release(depth.buffer);
  const gated = await ops.siluRowSplit(depthTokens.tensor, {
    numTokens: depthTokens.numTokens,
    dim: hiddenChannels,
    activation: 'silu',
    swigluLimit: null,
  });
  release(depthTokens.tensor.buffer);
  const gatedChannels = await tokensToChannels(gated, hiddenChannels, state.height, state.width, ops);
  release(gated.buffer);

  const pointWeight = getWeight(weights, shapes, `${prefix}.conv_out.conv_point.weight`);
  const pointShape = getLinearShape(pointWeight.shape, `${prefix}.conv_out.conv_point.weight`);
  const projected = await ops.groupedPointwiseConv2d(gatedChannels.tensor, pointWeight.value, null, {
    inChannels: hiddenChannels,
    outChannels: pointShape.outFeatures,
    height: state.height,
    width: state.width,
    groups: 1,
  });
  release(gatedChannels.tensor.buffer);
  const normed = await runChannelwiseRmsNorm(
    {
      tensor: reshapeTensor(projected, [pointShape.outFeatures, state.height, state.width], 'vae_dc_glumb_projected'),
      channels: pointShape.outFeatures,
      height: state.height,
      width: state.width,
    },
    getWeight(weights, shapes, `${prefix}.conv_out.norm.weight`),
    getWeightOptional(weights, shapes, `${prefix}.conv_out.norm.bias`),
    eps,
    ops,
    release
  );
  release(projected.buffer);

  const size = normed.channels * normed.height * normed.width;
  const combined = await ops.residualAdd(
    reshapeTensor(normed.tensor, [size], 'vae_dc_glumb_main'),
    reshapeTensor(state.tensor, [size], 'vae_dc_glumb_residual'),
    size,
    { useVec4: true }
  );
  release(normed.tensor.buffer);
  release(state.tensor.buffer);
  return {
    tensor: reshapeTensor(combined, [normed.channels, normed.height, normed.width], 'vae_dc_glumb_out'),
    channels: normed.channels,
    height: normed.height,
    width: normed.width,
  };
}

export async function runAutoencoderDCEfficientVitBlock(state, weights, shapes, prefix, attentionHeadDim, qkvMultiscales, eps, ops, release, recorder) {
  const attended = await runAutoencoderDCAttention(
    state,
    weights,
    shapes,
    prefix,
    attentionHeadDim,
    qkvMultiscales,
    1e-15,
    ops,
    release,
    recorder
  );
  return runAutoencoderDCGlumbConv(attended, weights, shapes, prefix, eps, ops, release);
}

export async function decodeLatentsAutoencoderDC(state, config, weights, shapes, ops, release, recorder) {
  const blockTypes = normalizePerBlockValue(config.decoder_block_types, config.decoder_block_out_channels.length, 'decoder_block_types');
  const layersPerBlock = normalizePerBlockValue(config.decoder_layers_per_block, config.decoder_block_out_channels.length, 'decoder_layers_per_block');
  const qkvMultiscales = normalizePerBlockValue(config.decoder_qkv_multiscales, config.decoder_block_out_channels.length, 'decoder_qkv_multiscales');
  const normTypes = normalizePerBlockValue(config.decoder_norm_types, config.decoder_block_out_channels.length, 'decoder_norm_types');
  const actFns = normalizePerBlockValue(config.decoder_act_fns, config.decoder_block_out_channels.length, 'decoder_act_fns');
  const rmsNormEps = 1e-5;

  state = await runAutoencoderDCInputProjection(state, weights, shapes, config, ops, release);

  for (let blockIdx = blockTypes.length - 1; blockIdx >= 0; blockIdx--) {
    const prefix = `vae.decoder.up_blocks.${blockIdx}`;
    const hasUpsample = weights.has(`${prefix}.0.conv.weight`);
    if (hasUpsample) {
      state = await runAutoencoderDCUpBlock(state, weights, shapes, `${prefix}.0`, config, ops, release, recorder);
    }

    if (normTypes[blockIdx] !== 'rms_norm') {
      throw new Error(
        `Unsupported AutoencoderDC norm type "${normTypes[blockIdx]}" in block ${blockIdx}.`
      );
    }
    if (actFns[blockIdx] !== 'silu') {
      throw new Error(
        `Unsupported AutoencoderDC activation "${actFns[blockIdx]}" in block ${blockIdx}.`
      );
    }

    const startIndex = hasUpsample ? 1 : 0;
    const blockType = blockTypes[blockIdx];
    const numLayers = layersPerBlock[blockIdx];
    for (let layerOffset = 0; layerOffset < numLayers; layerOffset++) {
      const layerPrefix = `${prefix}.${startIndex + layerOffset}`;
      if (blockType === 'ResBlock') {
        state = await runAutoencoderDCResBlock(state, weights, shapes, layerPrefix, rmsNormEps, ops, release);
        continue;
      }
      if (blockType === 'EfficientViTBlock') {
        state = await runAutoencoderDCEfficientVitBlock(
          state,
          weights,
          shapes,
          layerPrefix,
          config.attention_head_dim,
          qkvMultiscales[blockIdx],
          rmsNormEps,
          ops,
          release,
          recorder
        );
        continue;
      }
      throw new Error(`Unsupported AutoencoderDC block type "${blockType}" in block ${blockIdx}.`);
    }
  }

  const normed = await runChannelwiseRmsNorm(
    state,
    getWeight(weights, shapes, 'vae.decoder.norm_out.weight'),
    getWeightOptional(weights, shapes, 'vae.decoder.norm_out.bias'),
    rmsNormEps,
    ops,
    release
  );
  release(state.tensor.buffer);
  const activated = await ops.relu(normed.tensor, {
    count: normed.channels * normed.height * normed.width,
  });
  release(normed.tensor.buffer);
  return applyConv2D(
    {
      tensor: reshapeTensor(activated, [normed.channels, normed.height, normed.width], 'vae_dc_norm_out'),
      channels: normed.channels,
      height: normed.height,
      width: normed.width,
    },
    weights,
    shapes,
    'vae.decoder.conv_out',
    { pad: 1 },
    ops,
    release
  );
}
