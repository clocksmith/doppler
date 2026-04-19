import { getDevice } from '../../../gpu/device.js';
import { createTensor } from '../../../gpu/tensor.js';
import { getBuffer } from '../../../gpu/weight-buffer.js';
import { acquireBuffer } from '../../../memory/buffer-pool.js';
import {
  runConv2D,
  runDepthwiseConv2D,
  runGroupedPointwiseConv2D,
  runLayerNorm,
  runRMSNorm,
  runMatmul,
  runAttention,
  runSiLU,
  runSiLURowSplit,
  runResidualAdd,
  runBiasAdd,
  runModulate,
  runSanaLinearAttention,
  recordConv2D,
  recordDepthwiseConv2D,
  recordGroupedPointwiseConv2D,
  recordLayerNorm,
  recordRMSNorm,
  recordMatmul,
  recordAttention,
  recordSiLU,
  recordSiLURowSplit,
  recordResidualAdd,
  recordBiasAdd,
  recordModulate,
  recordSanaLinearAttention,
} from '../../../gpu/kernels/index.js';
import { log } from '../../../debug/index.js';
import {
  resolveDiffusionActivationDtype,
  createDiffusionBufferReleaser,
  createDiffusionBufferDestroyer,
  normalizeDiffusionMatmulLocationDtype,
  inferDiffusionMatmulDtypeFromBuffer,
  expectDiffusionWeight,
} from './helpers.js';

function reshapeTensor(tensor, shape, label) {
  return createTensor(tensor.buffer, tensor.dtype, shape, label ?? tensor.label);
}

function getWeight(weightsEntry, name) {
  return weightsEntry?.weights?.get(`transformer.${name}`) ?? null;
}

function getWeightShape(weightsEntry, name) {
  return weightsEntry?.shapes?.get(`transformer.${name}`) ?? null;
}

function getWeightDtype(weightsEntry, name) {
  return weightsEntry?.dtypes?.get(`transformer.${name}`) ?? null;
}

function createKernelOps(recorder) {
  if (!recorder) {
    return {
      conv2d: runConv2D,
      depthwiseConv2d: runDepthwiseConv2D,
      groupedPointwiseConv2d: runGroupedPointwiseConv2D,
      layerNorm: runLayerNorm,
      rmsNorm: runRMSNorm,
      attention: runAttention,
      silu: runSiLU,
      siluRowSplit: runSiLURowSplit,
      residualAdd: runResidualAdd,
      biasAdd: runBiasAdd,
      modulate: runModulate,
      sanaLinearAttention: runSanaLinearAttention,
    };
  }
  return {
    conv2d: (...args) => recordConv2D(recorder, ...args),
    depthwiseConv2d: (...args) => recordDepthwiseConv2D(recorder, ...args),
    groupedPointwiseConv2d: (...args) => recordGroupedPointwiseConv2D(recorder, ...args),
    layerNorm: (...args) => recordLayerNorm(recorder, ...args),
    rmsNorm: (...args) => recordRMSNorm(recorder, ...args),
    attention: (...args) => recordAttention(recorder, ...args),
    silu: (...args) => recordSiLU(recorder, ...args),
    siluRowSplit: (...args) => recordSiLURowSplit(recorder, ...args),
    residualAdd: (...args) => recordResidualAdd(recorder, ...args),
    biasAdd: (...args) => recordBiasAdd(recorder, ...args),
    modulate: (...args) => recordModulate(recorder, ...args),
    sanaLinearAttention: (...args) => recordSanaLinearAttention(recorder, ...args),
  };
}

function createVectorBuffer(device, values, label) {
  const buffer = acquireBuffer(values.byteLength, undefined, label);
  device.queue.writeBuffer(buffer, 0, values);
  return buffer;
}

function createBiasTensor(weight, size, label) {
  if (!weight) return null;
  return createTensor(getBuffer(weight), 'f32', [size], label);
}

function countMask(mask) {
  if (!mask) return null;
  let count = 0;
  for (const value of mask) {
    if (value) count += 1;
  }
  return count;
}

function trimContext(context, attentionMask) {
  const validTokens = countMask(attentionMask);
  if (!Number.isFinite(validTokens) || validTokens <= 0 || validTokens >= context.shape[0]) {
    return context;
  }
  return createTensor(context.buffer, context.dtype, [validTokens, context.shape[1]], 'sana_trimmed_context');
}

function buildSinusoidalEmbedding(value, dim = 256) {
  const half = Math.floor(dim / 2);
  const out = new Float32Array(dim);
  const maxPeriod = 10000;
  for (let i = 0; i < half; i++) {
    const freq = Math.exp(-Math.log(maxPeriod) * i / half);
    const angle = value * freq;
    out[2 * i] = Math.cos(angle);
    out[2 * i + 1] = Math.sin(angle);
  }
  return out;
}

function resolveMatmulDtype(weightsEntry, name, N, K) {
  const weight = getWeight(weightsEntry, name);
  const preferred = normalizeDiffusionMatmulLocationDtype(getWeightDtype(weightsEntry, name));
  return inferDiffusionMatmulDtypeFromBuffer(weight, N, K, preferred);
}

async function runMatmulResolved(input, weightsEntry, name, M, N, K, recorder, options = {}) {
  const weight = expectDiffusionWeight(getWeight(weightsEntry, name), name);
  const bDtype = resolveMatmulDtype(weightsEntry, name, N, K);
  if (recorder) {
    return recordMatmul(recorder, input, weight, M, N, K, { ...options, bDtype, transposeB: 'auto' });
  }
  return runMatmul(input, weight, M, N, K, { ...options, bDtype, transposeB: 'auto' });
}

async function runTwoLayerEmbedding(inputTensor, weightsEntry, prefix, outDim, recorder, runtime, ops, release) {
  const activationDtype = resolveDiffusionActivationDtype(runtime);

  let output = await runMatmulResolved(
    inputTensor,
    weightsEntry,
    `${prefix}.linear_1.weight`,
    1,
    getWeightShape(weightsEntry, `${prefix}.linear_1.weight`)[0],
    getWeightShape(weightsEntry, `${prefix}.linear_1.weight`)[1],
    recorder,
    { outputDtype: activationDtype }
  );
  const bias1 = createBiasTensor(getWeight(weightsEntry, `${prefix}.linear_1.bias`), output.shape[1], `${prefix}_bias1`);
  if (bias1) {
    output = await ops.biasAdd(output, bias1, 1, output.shape[1]);
  }

  const act = await ops.silu(output, { size: output.shape[1], swigluLimit: null });
  release(output.buffer);

  let projected = await runMatmulResolved(
    act,
    weightsEntry,
    `${prefix}.linear_2.weight`,
    1,
    outDim,
    getWeightShape(weightsEntry, `${prefix}.linear_2.weight`)[1],
    recorder,
    { outputDtype: activationDtype }
  );
  const bias2 = createBiasTensor(getWeight(weightsEntry, `${prefix}.linear_2.bias`), outDim, `${prefix}_bias2`);
  if (bias2) {
    projected = await ops.biasAdd(projected, bias2, 1, outDim);
  }
  release(act.buffer);
  return projected;
}

export async function buildSanaTimestepConditioning(timestep, guidanceScale, weightsEntry, config, runtime, options = {}) {
  const device = getDevice();
  if (!device) {
    throw new Error('Sana timestep conditioning requires a WebGPU device.');
  }
  const recorder = options.recorder ?? null;
  const ops = createKernelOps(recorder);
  const release = createDiffusionBufferReleaser(recorder);
  const activationDtype = resolveDiffusionActivationDtype(runtime);
  const hiddenSize = config.num_attention_heads * config.attention_head_dim;

  const timeTensor = createTensor(
    createVectorBuffer(device, buildSinusoidalEmbedding(timestep, 256), 'sana_timestep'),
    activationDtype,
    [1, 256],
    'sana_timestep'
  );
  const timeEmbedding = await runTwoLayerEmbedding(
    timeTensor,
    weightsEntry,
    'time_embed.timestep_embedder',
    hiddenSize,
    recorder,
    runtime,
    ops,
    release
  );
  release(timeTensor.buffer);

  let conditioning = timeEmbedding;
  if (config.guidance_embeds === true) {
    const guidanceTensor = createTensor(
      createVectorBuffer(device, buildSinusoidalEmbedding(guidanceScale, 256), 'sana_guidance'),
      activationDtype,
      [1, 256],
      'sana_guidance'
    );
    const guidanceEmbedding = await runTwoLayerEmbedding(
      guidanceTensor,
      weightsEntry,
      'time_embed.guidance_embedder',
      hiddenSize,
      recorder,
      runtime,
      ops,
      release
    );
    release(guidanceTensor.buffer);
    conditioning = await ops.residualAdd(timeEmbedding, guidanceEmbedding, hiddenSize, { useVec4: true });
    release(timeEmbedding.buffer);
    release(guidanceEmbedding.buffer);
  }

  const conditioningAct = await ops.silu(conditioning, { size: hiddenSize, swigluLimit: null });
  let modulation = await runMatmulResolved(
    conditioningAct,
    weightsEntry,
    'time_embed.linear.weight',
    1,
    hiddenSize * 6,
    hiddenSize,
    recorder,
    { outputDtype: activationDtype }
  );
  const modulationBias = createBiasTensor(getWeight(weightsEntry, 'time_embed.linear.bias'), hiddenSize * 6, 'sana_time_linear_bias');
  if (modulationBias) {
    modulation = await ops.biasAdd(modulation, modulationBias, 1, hiddenSize * 6);
  }
  release(conditioningAct.buffer);

  return {
    modulation,
    embeddedTimestep: conditioning,
  };
}

export async function projectSanaContext(context, attentionMask, weightsEntry, config, runtime, options = {}) {
  const recorder = options.recorder ?? null;
  const ops = createKernelOps(recorder);
  const release = createDiffusionBufferReleaser(recorder);
  const trimmed = trimContext(context, attentionMask);
  const tokenCount = trimmed.shape[0];
  const inputDim = trimmed.shape[1];
  const hiddenSize = config.num_attention_heads * config.attention_head_dim;
  const activationDtype = resolveDiffusionActivationDtype(runtime);

  let hidden = await runMatmulResolved(
    trimmed,
    weightsEntry,
    'caption_projection.linear_1.weight',
    tokenCount,
    hiddenSize,
    inputDim,
    recorder,
    { outputDtype: activationDtype }
  );
  const bias1 = createBiasTensor(getWeight(weightsEntry, 'caption_projection.linear_1.bias'), hiddenSize, 'sana_caption_bias1');
  if (bias1) {
    hidden = await ops.biasAdd(hidden, bias1, tokenCount, hiddenSize);
  }

  // PixArtAlphaTextProjection uses GELU(tanh). Reuse the existing GeLU kernel here.
  const { runGeLU, recordGeLU } = await import('../../../gpu/kernels/gelu.js');
  const gelu = recorder
    ? (input, options = {}) => recordGeLU(recorder, input, options)
    : runGeLU;
  const activated = await gelu(hidden, { size: tokenCount * hiddenSize });
  release(hidden.buffer);

  let projected = await runMatmulResolved(
    activated,
    weightsEntry,
    'caption_projection.linear_2.weight',
    tokenCount,
    hiddenSize,
    hiddenSize,
    recorder,
    { outputDtype: activationDtype }
  );
  const bias2 = createBiasTensor(getWeight(weightsEntry, 'caption_projection.linear_2.bias'), hiddenSize, 'sana_caption_bias2');
  if (bias2) {
    projected = await ops.biasAdd(projected, bias2, tokenCount, hiddenSize);
  }
  release(activated.buffer);

  const normWeight = expectDiffusionWeight(getWeight(weightsEntry, 'caption_norm.weight'), 'caption_norm.weight');
  const normed = await ops.rmsNorm(projected, getBuffer(normWeight), 1e-5, {
    batchSize: tokenCount,
    hiddenSize,
  });
  release(projected.buffer);
  return normed;
}

async function duplicateVectorTensor(tensor, times, recorder) {
  const device = getDevice();
  const output = acquireBuffer(tensor.buffer.size * times, undefined, 'sana_duplicate_vector');
  const encoder = recorder ? recorder.getEncoder() : device.createCommandEncoder();
  for (let i = 0; i < times; i++) {
    encoder.copyBufferToBuffer(tensor.buffer, 0, output, i * tensor.buffer.size, tensor.buffer.size);
  }
  if (!recorder) {
    device.queue.submit([encoder.finish()]);
  }
  return createTensor(output, tensor.dtype, [1, tensor.shape[1] * times], 'sana_duplicate_vector');
}

async function buildLayerModulation(baseModulation, tableWeight, tableShape, recorder, ops, release) {
  const segments = Array.isArray(tableShape) ? tableShape.reduce((acc, value) => acc * value, 1) : 0;
  let combined = await ops.biasAdd(
    baseModulation,
    createTensor(getBuffer(tableWeight), baseModulation.dtype, [segments], 'sana_layer_table'),
    1,
    segments
  );
  return combined;
}

async function runSelfAttention(hiddenStates, layerIdx, weightsEntry, numTokens, hiddenSize, numHeads, headDim, eps, recorder, runtime, ops, release) {
  let query = await runMatmulResolved(
    hiddenStates,
    weightsEntry,
    `transformer_blocks.${layerIdx}.attn1.to_q.weight`,
    numTokens,
    hiddenSize,
    hiddenSize,
    recorder,
    { outputDtype: hiddenStates.dtype }
  );
  let key = await runMatmulResolved(
    hiddenStates,
    weightsEntry,
    `transformer_blocks.${layerIdx}.attn1.to_k.weight`,
    numTokens,
    hiddenSize,
    hiddenSize,
    recorder,
    { outputDtype: hiddenStates.dtype }
  );
  let value = await runMatmulResolved(
    hiddenStates,
    weightsEntry,
    `transformer_blocks.${layerIdx}.attn1.to_v.weight`,
    numTokens,
    hiddenSize,
    hiddenSize,
    recorder,
    { outputDtype: hiddenStates.dtype }
  );

  const normQ = getWeight(weightsEntry, `transformer_blocks.${layerIdx}.attn1.norm_q.weight`);
  const normK = getWeight(weightsEntry, `transformer_blocks.${layerIdx}.attn1.norm_k.weight`);
  if (normQ) {
    const next = await ops.rmsNorm(query, getBuffer(normQ), eps, { batchSize: numTokens, hiddenSize });
    release(query.buffer);
    query = next;
  }
  if (normK) {
    const next = await ops.rmsNorm(key, getBuffer(normK), eps, { batchSize: numTokens, hiddenSize });
    release(key.buffer);
    key = next;
  }

  let output = await ops.sanaLinearAttention(query, key, value, {
    numHeads,
    headDim,
    numTokens,
    hiddenSize,
    eps: 1e-15,
  });
  release(query.buffer);
  release(key.buffer);
  release(value.buffer);

  let projected = await runMatmulResolved(
    output,
    weightsEntry,
    `transformer_blocks.${layerIdx}.attn1.to_out.0.weight`,
    numTokens,
    hiddenSize,
    hiddenSize,
    recorder,
    { outputDtype: hiddenStates.dtype }
  );
  release(output.buffer);
  const outBias = createBiasTensor(getWeight(weightsEntry, `transformer_blocks.${layerIdx}.attn1.to_out.0.bias`), hiddenSize, 'sana_self_attn_bias');
  if (outBias) {
    projected = await ops.biasAdd(projected, outBias, numTokens, hiddenSize);
  }
  return projected;
}

async function runCrossAttention(hiddenStates, context, layerIdx, weightsEntry, numTokens, hiddenSize, config, recorder, ops, release) {
  const qHeads = getWeightShape(weightsEntry, `transformer_blocks.${layerIdx}.attn2.to_q.weight`)[0];
  const kHeads = getWeightShape(weightsEntry, `transformer_blocks.${layerIdx}.attn2.to_k.weight`)[0];
  const vHeads = getWeightShape(weightsEntry, `transformer_blocks.${layerIdx}.attn2.to_v.weight`)[0];
  const contextTokens = context.shape[0];
  let query = await runMatmulResolved(
    hiddenStates,
    weightsEntry,
    `transformer_blocks.${layerIdx}.attn2.to_q.weight`,
    numTokens,
    qHeads,
    hiddenSize,
    recorder,
    { outputDtype: hiddenStates.dtype }
  );
  let key = await runMatmulResolved(
    context,
    weightsEntry,
    `transformer_blocks.${layerIdx}.attn2.to_k.weight`,
    contextTokens,
    kHeads,
    context.shape[1],
    recorder,
    { outputDtype: hiddenStates.dtype }
  );
  let value = await runMatmulResolved(
    context,
    weightsEntry,
    `transformer_blocks.${layerIdx}.attn2.to_v.weight`,
    contextTokens,
    vHeads,
    context.shape[1],
    recorder,
    { outputDtype: hiddenStates.dtype }
  );

  const crossHeads = config.num_cross_attention_heads;
  const headDim = config.cross_attention_head_dim;
  const normQ = getWeight(weightsEntry, `transformer_blocks.${layerIdx}.attn2.norm_q.weight`);
  const normK = getWeight(weightsEntry, `transformer_blocks.${layerIdx}.attn2.norm_k.weight`);
  if (normQ) {
    const next = await ops.rmsNorm(query, getBuffer(normQ), Number(config.norm_eps ?? 1e-6), {
      batchSize: numTokens,
      hiddenSize: qHeads,
    });
    release(query.buffer);
    query = next;
  }
  if (normK) {
    const next = await ops.rmsNorm(key, getBuffer(normK), Number(config.norm_eps ?? 1e-6), {
      batchSize: contextTokens,
      hiddenSize: kHeads,
    });
    release(key.buffer);
    key = next;
  }

  const attention = await ops.attention(query, key, value, null, crossHeads, headDim, {
    seqLen: numTokens,
    kvLen: contextTokens,
    numKVHeads: crossHeads,
    causal: false,
  });
  release(query.buffer);
  release(key.buffer);
  release(value.buffer);

  let projected = await runMatmulResolved(
    attention,
    weightsEntry,
    `transformer_blocks.${layerIdx}.attn2.to_out.0.weight`,
    numTokens,
    hiddenSize,
    hiddenSize,
    recorder,
    { outputDtype: hiddenStates.dtype }
  );
  release(attention.buffer);
  const outBias = createBiasTensor(getWeight(weightsEntry, `transformer_blocks.${layerIdx}.attn2.to_out.0.bias`), hiddenSize, 'sana_cross_attn_bias');
  if (outBias) {
    projected = await ops.biasAdd(projected, outBias, numTokens, hiddenSize);
  }
  return projected;
}

async function runGlumbConv(hiddenStates, layerIdx, weightsEntry, gridHeight, gridWidth, hiddenSize, recorder, runtime, ops, release) {
  const expandRatio = Number(getWeightShape(weightsEntry, `transformer_blocks.${layerIdx}.ff.conv_inverted.weight`)[0]) / hiddenSize / 2;
  const hiddenChannels = Math.floor(hiddenSize * expandRatio);
  let inverted = await runMatmulResolved(
    hiddenStates,
    weightsEntry,
    `transformer_blocks.${layerIdx}.ff.conv_inverted.weight`,
    hiddenStates.shape[0],
    hiddenChannels * 2,
    hiddenSize,
    recorder,
    { outputDtype: hiddenStates.dtype }
  );
  const invertedBias = createBiasTensor(getWeight(weightsEntry, `transformer_blocks.${layerIdx}.ff.conv_inverted.bias`), hiddenChannels * 2, 'sana_ff_inverted_bias');
  if (invertedBias) {
    inverted = await ops.biasAdd(inverted, invertedBias, hiddenStates.shape[0], hiddenChannels * 2);
  }
  const invertedAct = await ops.silu(inverted, { size: hiddenStates.shape[0] * hiddenChannels * 2, swigluLimit: null });
  release(inverted.buffer);

  const convInput = reshapeTensor(invertedAct, [hiddenChannels * 2, gridHeight, gridWidth], 'sana_ff_conv_input');
  const depthWeight = expectDiffusionWeight(getWeight(weightsEntry, `transformer_blocks.${layerIdx}.ff.conv_depth.weight`), `transformer_blocks.${layerIdx}.ff.conv_depth.weight`);
  const depthBias = getWeight(weightsEntry, `transformer_blocks.${layerIdx}.ff.conv_depth.bias`);
  const depth = await ops.depthwiseConv2d(convInput, depthWeight, depthBias, {
    channels: hiddenChannels * 2,
    height: gridHeight,
    width: gridWidth,
    kernelH: 3,
    kernelW: 3,
    stride: 1,
    pad: 1,
  });
  release(invertedAct.buffer);

  const depthTokens = reshapeTensor(depth, [hiddenStates.shape[0], hiddenChannels * 2], 'sana_ff_depth_tokens');
  const gated = await ops.siluRowSplit(depthTokens, {
    numTokens: hiddenStates.shape[0],
    dim: hiddenChannels,
    activation: 'silu',
    swigluLimit: null,
  });
  release(depth.buffer);

  let projected = await runMatmulResolved(
    gated,
    weightsEntry,
    `transformer_blocks.${layerIdx}.ff.conv_point.weight`,
    hiddenStates.shape[0],
    hiddenSize,
    hiddenChannels,
    recorder,
    { outputDtype: hiddenStates.dtype }
  );
  release(gated.buffer);
  return projected;
}

export async function runSanaTransformer(latents, context, timeState, weightsEntry, modelConfig, runtime, options = {}) {
  const device = getDevice();
  if (!device) {
    throw new Error('Sana transformer requires a WebGPU device.');
  }

  const recorder = options.recorder ?? null;
  const ops = createKernelOps(recorder);
  const release = createDiffusionBufferReleaser(recorder);
  const destroy = createDiffusionBufferDestroyer(recorder);
  const config = modelConfig?.components?.transformer?.config || {};
  const hiddenSize = config.num_attention_heads * config.attention_head_dim;
  const numHeads = config.num_attention_heads;
  const headDim = config.attention_head_dim;
  const patchSize = config.patch_size ?? 1;
  const normEps = Number(config.norm_eps ?? 1e-6);
  const latentHeight = latents.shape[1];
  const latentWidth = latents.shape[2];
  const gridHeight = Math.floor(latentHeight / patchSize);
  const gridWidth = Math.floor(latentWidth / patchSize);
  const numTokens = gridHeight * gridWidth;

  const patchWeight = expectDiffusionWeight(getWeight(weightsEntry, 'patch_embed.proj.weight'), 'patch_embed.proj.weight');
  const patchBias = getWeight(weightsEntry, 'patch_embed.proj.bias');
  const conv = await ops.conv2d(latents, patchWeight, patchBias, {
    inChannels: latents.shape[0],
    outChannels: hiddenSize,
    height: latentHeight,
    width: latentWidth,
    kernelH: patchSize,
    kernelW: patchSize,
    stride: patchSize,
    pad: 0,
  });
  let hidden = await import('../../../gpu/kernels/transpose.js').then(({ runTranspose, recordTranspose }) => {
    const transpose = recorder ? (input, rows, cols) => recordTranspose(recorder, input, rows, cols) : runTranspose;
    return transpose(conv, hiddenSize, numTokens);
  });
  release(conv.buffer);

  const ones = new Float32Array(hiddenSize).fill(1.0);
  const zeros = new Float32Array(hiddenSize);
  const onesBuf = createVectorBuffer(device, ones, 'sana_norm_ones');
  const zerosBuf = createVectorBuffer(device, zeros, 'sana_norm_zeros');

  for (let layerIdx = 0; layerIdx < config.num_layers; layerIdx++) {
    const layerTable = expectDiffusionWeight(getWeight(weightsEntry, `transformer_blocks.${layerIdx}.scale_shift_table`), `transformer_blocks.${layerIdx}.scale_shift_table`);
    const modulation = await buildLayerModulation(
      timeState.modulation,
      layerTable,
      getWeightShape(weightsEntry, `transformer_blocks.${layerIdx}.scale_shift_table`),
      recorder,
      ops,
      release
    );

    let norm1 = await ops.layerNorm(hidden, onesBuf, zerosBuf, normEps, { batchSize: numTokens, hiddenSize });
    norm1 = await ops.modulate(norm1, modulation, {
      numTokens,
      hiddenSize,
      scaleOffset: hiddenSize,
      shiftOffset: 0,
      gateOffset: hiddenSize * 2,
      hasGate: false,
      addOne: true,
    });
    const selfAttn = await runSelfAttention(norm1, layerIdx, weightsEntry, numTokens, hiddenSize, numHeads, headDim, normEps, recorder, runtime, ops, release);
    release(norm1.buffer);
    const gatedSelf = await ops.modulate(selfAttn, modulation, {
      numTokens,
      hiddenSize,
      scaleOffset: hiddenSize * 2,
      shiftOffset: hiddenSize * 6,
      gateOffset: hiddenSize * 2,
      hasGate: false,
      addOne: false,
    });
    release(selfAttn.buffer);
    let nextHidden = await ops.residualAdd(hidden, gatedSelf, numTokens * hiddenSize, { useVec4: true });
    release(hidden.buffer);
    release(gatedSelf.buffer);
    hidden = createTensor(nextHidden.buffer, nextHidden.dtype, [numTokens, hiddenSize], 'sana_hidden_after_self');

    let norm2 = await ops.layerNorm(hidden, onesBuf, zerosBuf, normEps, { batchSize: numTokens, hiddenSize });
    const crossAttn = await runCrossAttention(norm2, context, layerIdx, weightsEntry, numTokens, hiddenSize, config, recorder, ops, release);
    release(norm2.buffer);
    nextHidden = await ops.residualAdd(hidden, crossAttn, numTokens * hiddenSize, { useVec4: true });
    release(hidden.buffer);
    release(crossAttn.buffer);
    hidden = createTensor(nextHidden.buffer, nextHidden.dtype, [numTokens, hiddenSize], 'sana_hidden_after_cross');

    let normFf = await ops.layerNorm(hidden, onesBuf, zerosBuf, normEps, { batchSize: numTokens, hiddenSize });
    normFf = await ops.modulate(normFf, modulation, {
      numTokens,
      hiddenSize,
      scaleOffset: hiddenSize * 4,
      shiftOffset: hiddenSize * 3,
      gateOffset: hiddenSize * 5,
      hasGate: false,
      addOne: true,
    });
    const ff = await runGlumbConv(normFf, layerIdx, weightsEntry, gridHeight, gridWidth, hiddenSize, recorder, runtime, ops, release);
    release(normFf.buffer);
    const gatedFf = await ops.modulate(ff, modulation, {
      numTokens,
      hiddenSize,
      scaleOffset: hiddenSize * 5,
      shiftOffset: hiddenSize * 6,
      gateOffset: hiddenSize * 5,
      hasGate: false,
      addOne: false,
    });
    release(ff.buffer);
    release(modulation.buffer);
    nextHidden = await ops.residualAdd(hidden, gatedFf, numTokens * hiddenSize, { useVec4: true });
    release(hidden.buffer);
    release(gatedFf.buffer);
    hidden = createTensor(nextHidden.buffer, nextHidden.dtype, [numTokens, hiddenSize], 'sana_hidden_after_ff');
  }
  release(timeState.modulation.buffer);

  const finalTable = expectDiffusionWeight(getWeight(weightsEntry, 'scale_shift_table'), 'scale_shift_table');
  const duplicated = await duplicateVectorTensor(timeState.embeddedTimestep, 2, recorder);
  release(timeState.embeddedTimestep.buffer);
  let finalMod = await ops.biasAdd(
    duplicated,
    createTensor(getBuffer(finalTable), duplicated.dtype, [hiddenSize * 2], 'sana_final_table'),
    1,
    hiddenSize * 2
  );
  release(duplicated.buffer);
  let normed = await ops.layerNorm(hidden, onesBuf, zerosBuf, 1e-6, { batchSize: numTokens, hiddenSize });
  normed = await ops.modulate(normed, finalMod, {
    numTokens,
    hiddenSize,
    scaleOffset: hiddenSize,
    shiftOffset: 0,
    gateOffset: hiddenSize,
    hasGate: false,
    addOne: true,
  });
  release(hidden.buffer);
  release(finalMod.buffer);

  let projected = await runMatmulResolved(
    normed,
    weightsEntry,
    'proj_out.weight',
    numTokens,
    config.out_channels ?? latents.shape[0],
    hiddenSize,
    recorder,
    { outputDtype: normed.dtype }
  );
  release(normed.buffer);
  const projBias = createBiasTensor(getWeight(weightsEntry, 'proj_out.bias'), projected.shape[1], 'sana_proj_out_bias');
  if (projBias) {
    projected = await ops.biasAdd(projected, projBias, numTokens, projected.shape[1]);
  }

  const { runTranspose, recordTranspose } = await import('../../../gpu/kernels/transpose.js');
  const transpose = recorder ? (input, rows, cols) => recordTranspose(recorder, input, rows, cols) : runTranspose;
  const channelsFirst = await transpose(projected, numTokens, projected.shape[1]);
  release(projected.buffer);
  destroy(onesBuf);
  destroy(zerosBuf);
  return reshapeTensor(channelsFirst, [config.out_channels ?? latents.shape[0], gridHeight, gridWidth], 'sana_output');
}

