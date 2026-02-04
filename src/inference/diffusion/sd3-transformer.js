import { getDevice, getKernelCapabilities } from '../../gpu/device.js';
import { createTensor, dtypeBytes } from '../../gpu/tensor.js';
import { getBuffer } from '../../gpu/weight-buffer.js';
import { acquireBuffer, releaseBuffer, isBufferActive } from '../../memory/buffer-pool.js';
import {
  runConv2D,
  runTranspose,
  runGather,
  runLayerNorm,
  runRMSNorm,
  runMatmul,
  runAttention,
  runGeLU,
  runSiLURowSplit,
  runResidualAdd,
  runBiasAdd,
  runModulate,
  runPixelShuffle,
  recordConv2D,
  recordTranspose,
  recordGather,
  recordLayerNorm,
  recordRMSNorm,
  recordMatmul,
  recordAttention,
  recordGeLU,
  recordSiLURowSplit,
  recordResidualAdd,
  recordBiasAdd,
  recordModulate,
  recordPixelShuffle,
} from '../../gpu/kernels/index.js';
import { log } from '../../debug/index.js';
import { createSD3WeightResolver } from './sd3-weights.js';

function reshapeTensor(tensor, shape, label) {
  return createTensor(tensor.buffer, tensor.dtype, shape, label);
}

function resolveActivationDtype(runtime) {
  const caps = getKernelCapabilities();
  const wantsF16 = runtime?.latent?.dtype === 'f16';
  return wantsF16 && caps.hasF16 ? 'f16' : 'f32';
}

function createKernelOps(recorder) {
  if (!recorder) {
    return {
      conv2d: runConv2D,
      transpose: runTranspose,
      gather: runGather,
      layerNorm: runLayerNorm,
      rmsNorm: runRMSNorm,
      attention: runAttention,
      gelu: runGeLU,
      siluRowSplit: runSiLURowSplit,
      residualAdd: runResidualAdd,
      biasAdd: runBiasAdd,
      modulate: runModulate,
      pixelShuffle: runPixelShuffle,
    };
  }
  return {
    conv2d: (...args) => recordConv2D(recorder, ...args),
    transpose: (...args) => recordTranspose(recorder, ...args),
    gather: (...args) => recordGather(recorder, ...args),
    layerNorm: (...args) => recordLayerNorm(recorder, ...args),
    rmsNorm: (...args) => recordRMSNorm(recorder, ...args),
    attention: (...args) => recordAttention(recorder, ...args),
    gelu: (...args) => recordGeLU(recorder, ...args),
    siluRowSplit: (...args) => recordSiLURowSplit(recorder, ...args),
    residualAdd: (...args) => recordResidualAdd(recorder, ...args),
    biasAdd: (...args) => recordBiasAdd(recorder, ...args),
    modulate: (...args) => recordModulate(recorder, ...args),
    pixelShuffle: (...args) => recordPixelShuffle(recorder, ...args),
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

function createBufferDestroyer(recorder) {
  if (!recorder) return (buffer) => buffer?.destroy();
  return (buffer) => {
    if (!buffer) return;
    recorder.trackTemporaryBuffer(buffer);
  };
}

function createVectorBuffer(device, data, label) {
  const buffer = acquireBuffer(data.byteLength, undefined, label);
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

function createIndexBuffer(device, indices, label) {
  const buffer = device.createBuffer({
    label,
    size: indices.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, indices);
  return buffer;
}

function normalizeLocationDtype(dtype) {
  if (!dtype) return null;
  const normalized = String(dtype).toLowerCase();
  if (normalized === 'f16' || normalized === 'float16') return 'f16';
  if (normalized === 'f32' || normalized === 'float32') return 'f32';
  if (normalized === 'bf16' || normalized === 'bfloat16') return 'f32';
  return null;
}

function resolveEmbeddingDtype(weight, weightsEntry, key, runtime) {
  if (weight && weight.dtype) return weight.dtype;
  const locationDtype = weightsEntry?.dtypes?.get(key);
  const mapped = normalizeLocationDtype(locationDtype);
  if (!mapped) return null;
  if (mapped !== 'f16') return mapped;
  const allowUpcast = runtime?.loading?.allowF32UpcastNonMatmul !== false;
  return allowUpcast ? 'f32' : 'f16';
}

function normalizeMatmulLocationDtype(dtype) {
  if (!dtype) return null;
  const normalized = String(dtype).toLowerCase();
  if (normalized === 'f16' || normalized === 'float16') return 'f16';
  if (normalized === 'bf16' || normalized === 'bfloat16') return 'bf16';
  if (normalized === 'f32' || normalized === 'float32') return 'f32';
  if (normalized === 'q4_k' || normalized === 'q4_k_m') return 'q4k';
  return normalized;
}

function resolveMatmulDtype(weight, resolver, name) {
  if (weight && weight.dtype) return weight.dtype;
  if (!resolver || !name) return null;
  const locationDtype = resolver.dtype(name);
  return normalizeMatmulLocationDtype(locationDtype);
}

function resolveBiasDtype(weight, resolver, name) {
  if (weight && weight.dtype) return weight.dtype;
  if (!resolver || !name) return 'f32';
  const locationDtype = resolver.dtype(name);
  const mapped = normalizeLocationDtype(locationDtype);
  return mapped || 'f32';
}

function inferMatmulDtypeFromBuffer(weight, N, K, preferred) {
  const buffer = getBuffer(weight);
  if (!buffer || !Number.isFinite(N) || !Number.isFinite(K)) return preferred;
  if (preferred === 'q4k') return preferred;
  const expectedF16 = N * K * 2;
  const expectedF32 = N * K * 4;
  if (preferred === 'f32' && buffer.size < expectedF32 && buffer.size >= expectedF16) {
    return 'f16';
  }
  if (!preferred) {
    if (buffer.size >= expectedF32) return 'f32';
    if (buffer.size >= expectedF16) return 'f16';
  }
  return preferred;
}

async function runMatmulResolved(input, weight, resolver, name, M, N, K, options = {}) {
  const { recorder = null, ...rest } = options;
  const resolved = resolveMatmulDtype(weight, resolver, name);
  const bDtype = inferMatmulDtypeFromBuffer(weight, N, K, resolved);
  const nextOptions = bDtype ? { ...rest, bDtype } : rest;
  if (recorder) {
    return recordMatmul(recorder, input, weight, M, N, K, nextOptions);
  }
  return runMatmul(input, weight, M, N, K, nextOptions);
}

function expectWeight(weight, label) {
  if (!weight) {
    throw new Error(`Missing diffusion weight: ${label}`);
  }
  return weight;
}

function createBiasTensorWithDtype(weight, size, label, resolver, name) {
  if (!weight) return null;
  const dtype = resolveBiasDtype(weight, resolver, name);
  return createTensor(getBuffer(weight), dtype, [size], label);
}

async function splitQKV(qkv, numTokens, hiddenSize, label, recorder) {
  const device = getDevice();
  const bytesPerElement = dtypeBytes(qkv.dtype);
  const sliceBytes = numTokens * hiddenSize * bytesPerElement;
  const qBuf = acquireBuffer(sliceBytes, undefined, `${label}_q`);
  const kBuf = acquireBuffer(sliceBytes, undefined, `${label}_k`);
  const vBuf = acquireBuffer(sliceBytes, undefined, `${label}_v`);

  const encoder = recorder ? recorder.getEncoder() : device.createCommandEncoder();
  encoder.copyBufferToBuffer(qkv.buffer, 0, qBuf, 0, sliceBytes);
  encoder.copyBufferToBuffer(qkv.buffer, sliceBytes, kBuf, 0, sliceBytes);
  encoder.copyBufferToBuffer(qkv.buffer, sliceBytes * 2, vBuf, 0, sliceBytes);
  if (!recorder) {
    device.queue.submit([encoder.finish()]);
  }

  return {
    q: createTensor(qBuf, qkv.dtype, [numTokens, hiddenSize], `${label}_q`),
    k: createTensor(kBuf, qkv.dtype, [numTokens, hiddenSize], `${label}_k`),
    v: createTensor(vBuf, qkv.dtype, [numTokens, hiddenSize], `${label}_v`),
  };
}

async function runFusedQKV(input, weight, biasTensor, numTokens, hiddenSize, outputDtype, label, matmul, weightName, ops, release, recorder) {
  const qkv = await matmul(input, weight, weightName, numTokens, hiddenSize * 3, hiddenSize, {
    outputDtype,
    transposeB: 'auto',
  });

  let qkvTensor = qkv;
  if (biasTensor) {
    qkvTensor = await ops.biasAdd(qkv, biasTensor, numTokens, hiddenSize * 3);
  }

  const split = await splitQKV(qkvTensor, numTokens, hiddenSize, label, recorder);
  release(qkvTensor.buffer);
  return split;
}

async function runQKV(input, weights, bias, numTokens, hiddenSize, label, matmul, weightNames, ops, release, recorder) {
  const outputDtype = input.dtype;
  if (weights.qkv) {
    return runFusedQKV(
      input,
      weights.qkv,
      bias?.qkv ?? null,
      numTokens,
      hiddenSize,
      outputDtype,
      label,
      matmul,
      weightNames?.qkv ?? null,
      ops,
      release,
      recorder
    );
  }

  const qWeight = expectWeight(weights.q, `${label}.q`);
  const kWeight = expectWeight(weights.k, `${label}.k`);
  const vWeight = expectWeight(weights.v, `${label}.v`);

  let q = await matmul(input, qWeight, weightNames?.q ?? null, numTokens, hiddenSize, hiddenSize, {
    outputDtype,
    transposeB: 'auto',
  });
  let k = await matmul(input, kWeight, weightNames?.k ?? null, numTokens, hiddenSize, hiddenSize, {
    outputDtype,
    transposeB: 'auto',
  });
  let v = await matmul(input, vWeight, weightNames?.v ?? null, numTokens, hiddenSize, hiddenSize, {
    outputDtype,
    transposeB: 'auto',
  });

  if (bias?.q) q = await ops.biasAdd(q, bias.q, numTokens, hiddenSize);
  if (bias?.k) k = await ops.biasAdd(k, bias.k, numTokens, hiddenSize);
  if (bias?.v) v = await ops.biasAdd(v, bias.v, numTokens, hiddenSize);

  return { q, k, v };
}

async function applyQKNorm(tensor, weight, numTokens, numHeads, headDim, eps, ops) {
  const flattened = createTensor(tensor.buffer, tensor.dtype, [numTokens * numHeads, headDim], 'qk_norm_in');
  const normed = await ops.rmsNorm(flattened, getBuffer(weight), eps, {
    batchSize: numTokens * numHeads,
    hiddenSize: headDim,
  });
  return reshapeTensor(normed, [numTokens, numHeads, headDim], 'qk_norm_out');
}

async function concatKV(a, b, numTokensA, numTokensB, hiddenSize, recorder) {
  const device = getDevice();
  const bytesPerElement = a.dtype === 'f16' ? 2 : 4;
  const outputSize = (numTokensA + numTokensB) * hiddenSize * bytesPerElement;
  const output = acquireBuffer(outputSize, undefined, 'kv_concat');
  const encoder = recorder ? recorder.getEncoder() : device.createCommandEncoder();
  encoder.copyBufferToBuffer(a.buffer, 0, output, 0, numTokensA * hiddenSize * bytesPerElement);
  encoder.copyBufferToBuffer(b.buffer, 0, output, numTokensA * hiddenSize * bytesPerElement, numTokensB * hiddenSize * bytesPerElement);
  if (!recorder) {
    device.queue.submit([encoder.finish()]);
  }
  return createTensor(output, a.dtype, [numTokensA + numTokensB, hiddenSize], 'kv_concat');
}

async function runAttentionBlock(input, weights, bias, numTokens, hiddenSize, numHeads, headDim, normWeights, eps, matmul, weightNames, ops, release) {
  let q = await matmul(input, weights.q, weightNames?.q ?? null, numTokens, hiddenSize, hiddenSize, {
    outputDtype: input.dtype,
    transposeB: 'auto',
  });
  let k = await matmul(input, weights.k, weightNames?.k ?? null, numTokens, hiddenSize, hiddenSize, {
    outputDtype: input.dtype,
    transposeB: 'auto',
  });
  let v = await matmul(input, weights.v, weightNames?.v ?? null, numTokens, hiddenSize, hiddenSize, {
    outputDtype: input.dtype,
    transposeB: 'auto',
  });

  if (bias?.q) q = await ops.biasAdd(q, bias.q, numTokens, hiddenSize);
  if (bias?.k) k = await ops.biasAdd(k, bias.k, numTokens, hiddenSize);
  if (bias?.v) v = await ops.biasAdd(v, bias.v, numTokens, hiddenSize);

  if (normWeights?.q) {
    const normed = await applyQKNorm(q, normWeights.q, numTokens, numHeads, headDim, eps, ops);
    release(q.buffer);
    q = normed;
  }
  if (normWeights?.k) {
    const normed = await applyQKNorm(k, normWeights.k, numTokens, numHeads, headDim, eps, ops);
    release(k.buffer);
    k = normed;
  }

  const attn = await ops.attention(q, k, v, null, numHeads, headDim, {
    seqLen: numTokens,
    kvLen: numTokens,
    numKVHeads: numHeads,
    causal: false,
  });

  release(q.buffer);
  release(k.buffer);
  release(v.buffer);

  return attn;
}

function resolveModulationSegments(weight, hiddenSize, fallbackSegments, resolver, name) {
  const shape = weight?.shape || (resolver && name ? resolver.shape(name) : null);
  const rows = Array.isArray(shape) ? shape[0] : null;
  if (Number.isFinite(rows) && Number.isFinite(hiddenSize) && hiddenSize > 0) {
    const segments = rows / hiddenSize;
    if (Number.isInteger(segments) && segments > 0) {
      return segments;
    }
    log.warn(
      'Diffusion',
      `Modulation segments mismatch for ${name || 'unknown'}: rows=${rows}, hidden=${hiddenSize}, fallback=${fallbackSegments}`
    );
  }
  return fallbackSegments;
}

function resolveModulationOffsets(segments, hiddenSize) {
  if (segments >= 9) {
    return {
      attn: { scale: 0, shift: hiddenSize, gate: hiddenSize * 2 },
      attn2: { scale: hiddenSize * 3, shift: hiddenSize * 4, gate: hiddenSize * 5 },
      ff: { scale: hiddenSize * 6, shift: hiddenSize * 7, gate: hiddenSize * 8 },
    };
  }
  if (segments >= 6) {
    if (segments !== 6) {
      log.warn('Diffusion', `Unexpected modulation segment count=${segments}; using 6-segment layout.`);
    }
    const attn = { scale: 0, shift: hiddenSize, gate: hiddenSize * 2 };
    return {
      attn,
      attn2: { ...attn },
      ff: { scale: hiddenSize * 3, shift: hiddenSize * 4, gate: hiddenSize * 5 },
    };
  }
  throw new Error(`Unsupported modulation segments=${segments} (expected >= 6).`);
}

async function buildModulation(timeText, weight, bias, hiddenSize, segments, runtime, matmul, weightName, ops) {
  const device = getDevice();
  const activationDtype = resolveActivationDtype(runtime);
  const outDim = hiddenSize * segments;
  const bytesPerElement = activationDtype === 'f16' ? 2 : 4;
  const bufferSize = (outDim + hiddenSize) * bytesPerElement;
  const outputBuffer = acquireBuffer(bufferSize, undefined, 'sd3_modulate');

  const mod = await matmul(timeText, weight, weightName, 1, outDim, hiddenSize, {
    outputDtype: activationDtype,
    transposeB: 'auto',
    outputBuffer,
  });

  if (bias) {
    await ops.biasAdd(mod, bias, 1, outDim);
  }

  const zeroOffset = outDim * bytesPerElement;
  device.queue.writeBuffer(outputBuffer, zeroOffset, new Uint8Array(hiddenSize * bytesPerElement));

  return {
    tensor: createTensor(outputBuffer, activationDtype, [1, outDim], 'sd3_mod'),
    zeroOffset: outDim,
  };
}

async function applyAdaLayerNorm(input, weight, bias, eps, mod, offsets, runtime, ops, release, options = {}) {
  const { numTokens, hiddenSize } = options;
  const normed = await ops.layerNorm(input, weight, bias, eps, { batchSize: numTokens, hiddenSize });
  const modulated = await ops.modulate(normed, mod.tensor, {
    numTokens,
    hiddenSize,
    scaleOffset: offsets.scale,
    shiftOffset: offsets.shift,
    gateOffset: offsets.gate,
    hasGate: false,
    addOne: true,
  });
  release(normed.buffer);
  return modulated;
}

async function applyGate(output, mod, offsets, ops, release, options = {}) {
  const { numTokens, hiddenSize, zeroOffset } = options;
  const gated = await ops.modulate(output, mod.tensor, {
    numTokens,
    hiddenSize,
    scaleOffset: offsets.gate,
    shiftOffset: zeroOffset,
    gateOffset: offsets.gate,
    hasGate: false,
    addOne: false,
  });
  release(output.buffer);
  return gated;
}

async function runFFN(input, weights, bias, numTokens, hiddenSize, runtime, matmul, weightNames, ops, release) {
  const activationDtype = resolveActivationDtype(runtime);
  const upDim = weights.up.shape[0];
  const downInput = weights.down.shape[1];
  let up = await matmul(input, weights.up, weightNames?.up ?? null, numTokens, upDim, hiddenSize, {
    outputDtype: activationDtype,
    transposeB: 'auto',
  });
  if (bias?.up) up = await ops.biasAdd(up, bias.up, numTokens, upDim);

  let act = null;
  let intermediate = upDim;
  if (Number.isFinite(downInput) && upDim === downInput * 2) {
    act = await ops.siluRowSplit(up, {
      numTokens,
      dim: downInput,
      activation: 'gelu',
      swigluLimit: null,
    });
    intermediate = downInput;
  } else {
    act = await ops.gelu(up, { size: numTokens * upDim });
  }
  release(up.buffer);

  let down = await matmul(act, weights.down, weightNames?.down ?? null, numTokens, hiddenSize, intermediate, {
    outputDtype: activationDtype,
    transposeB: 'auto',
  });
  if (bias?.down) down = await ops.biasAdd(down, bias.down, numTokens, hiddenSize);
  release(act.buffer);
  return down;
}

export async function runSD3Transformer(latents, context, timeText, weightsEntry, modelConfig, runtime, options = {}) {
  const device = getDevice();
  if (!device) {
    throw new Error('SD3 transformer requires a WebGPU device.');
  }

  const resolver = createSD3WeightResolver(weightsEntry, modelConfig);
  const recorder = options.recorder ?? null;
  const ops = createKernelOps(recorder);
  const release = createBufferReleaser(recorder);
  const destroy = createBufferDestroyer(recorder);
  const matmul = (input, weight, name, M, N, K, options = {}) =>
    runMatmulResolved(input, weight, resolver, name, M, N, K, { ...options, recorder });
  const config = modelConfig?.components?.transformer?.config || {};
  const hiddenSize = config.num_attention_heads * config.attention_head_dim;
  const numHeads = config.num_attention_heads;
  const headDim = config.attention_head_dim;
  const patchSize = config.patch_size;
  const layerNormEps = runtime?.backend?.scaffold?.layerNormEps;
  if (!Number.isFinite(layerNormEps)) {
    throw new Error('Diffusion backend.layerNormEps is required.');
  }

  const latentChannels = latents.shape[0];
  const latentHeight = latents.shape[1];
  const latentWidth = latents.shape[2];
  const gridHeight = Math.floor(latentHeight / patchSize);
  const gridWidth = Math.floor(latentWidth / patchSize);
  const tokenCount = gridHeight * gridWidth;

  const projWeight = expectWeight(resolver.get('pos_embed.proj.weight'), 'pos_embed.proj.weight');
  const projBias = resolver.get('pos_embed.proj.bias');

  const conv = await ops.conv2d(latents, projWeight, projBias, {
    inChannels: latentChannels,
    outChannels: hiddenSize,
    height: latentHeight,
    width: latentWidth,
    kernelH: patchSize,
    kernelW: patchSize,
    stride: patchSize,
    pad: 0,
  });

  const tokens = await ops.transpose(conv, hiddenSize, tokenCount);
  release(conv.buffer);

  const posEmbed = expectWeight(resolver.get('pos_embed.pos_embed'), 'pos_embed.pos_embed');
  const posShape = resolver.shape('pos_embed.pos_embed') || [1, tokenCount, hiddenSize];
  const maxTokens = posShape[1];
  const maxGrid = Math.floor(Math.sqrt(maxTokens));

  if (maxGrid * maxGrid !== maxTokens) {
    log.warn('Diffusion', 'pos_embed size is not square; using sequential indices.');
  }

  const posIndices = new Uint32Array(tokenCount);
  for (let y = 0; y < gridHeight; y++) {
    const srcY = maxGrid * (y / Math.max(1, gridHeight));
    const srcYIdx = Math.min(maxGrid - 1, Math.floor(srcY));
    for (let x = 0; x < gridWidth; x++) {
      const srcX = maxGrid * (x / Math.max(1, gridWidth));
      const srcXIdx = Math.min(maxGrid - 1, Math.floor(srcX));
      posIndices[y * gridWidth + x] = srcYIdx * maxGrid + srcXIdx;
    }
  }

  const posBuffer = createIndexBuffer(device, posIndices, 'sd3_pos_idx');
  const posEmbedKey = resolver.key('pos_embed.pos_embed');
  const posEmbedDtype = resolveEmbeddingDtype(posEmbed, weightsEntry, posEmbedKey, runtime);
  const pos = await ops.gather(
    posBuffer,
    getBuffer(posEmbed),
    tokenCount,
    hiddenSize,
    maxTokens,
    {
      embeddingDtype: posEmbedDtype,
      outputDtype: tokens.dtype,
      transpose: false,
    }
  );
  destroy(posBuffer);

  const xCombined = await ops.residualAdd(tokens, pos, tokenCount * hiddenSize, { useVec4: true });
  release(tokens.buffer);
  release(pos.buffer);

  let x = createTensor(xCombined.buffer, xCombined.dtype, [tokenCount, hiddenSize], 'sd3_tokens');
  let ctx = context;
  let ctxOwned = false;

  const ones = new Float32Array(hiddenSize).fill(1.0);
  const zeros = new Float32Array(hiddenSize);
  const onesBuf = createVectorBuffer(device, ones, 'sd3_ln_weight');
  const zerosBuf = createVectorBuffer(device, zeros, 'sd3_ln_bias');

  const dualLayers = new Set(config.dual_attention_layers || []);
  const attn2Layers = Array.isArray(config.attn2_layers)
    ? new Set(config.attn2_layers)
    : null;
  const numLayers = config.num_layers;

  for (let layerIdx = 0; layerIdx < numLayers; layerIdx++) {
    const modWeightName = `transformer_blocks.${layerIdx}.norm1.linear.weight`;
    const modBiasName = `transformer_blocks.${layerIdx}.norm1.linear.bias`;
    const modWeight = expectWeight(
      resolver.get(modWeightName),
      modWeightName
    );
    const modBias = resolver.get(modBiasName);
    const modSegments = resolveModulationSegments(modWeight, hiddenSize, 9, resolver, modWeightName);
    if (modSegments < 6) {
      throw new Error(`Unsupported modulation segments=${modSegments} for ${modWeightName}`);
    }
    const modBiasTensor = createBiasTensorWithDtype(
      modBias,
      hiddenSize * modSegments,
      'sd3_mod_bias',
      resolver,
      modBiasName
    );
    const mod = await buildModulation(timeText, modWeight, modBiasTensor, hiddenSize, modSegments, runtime, matmul, modWeightName, ops);

    const offsets = resolveModulationOffsets(modSegments, hiddenSize);
    const attnOffsets = offsets.attn;
    const attn2Offsets = offsets.attn2;
    const ffOffsets = offsets.ff;

    let ctxMod = null;
    let ctxOffsets = null;
    let ctxAttnOffsets = null;
    let ctxFfOffsets = null;
    if (dualLayers.has(layerIdx)) {
      const ctxWeightName = `transformer_blocks.${layerIdx}.norm1_context.linear.weight`;
      const ctxBiasName = `transformer_blocks.${layerIdx}.norm1_context.linear.bias`;
      const ctxWeight = expectWeight(
        resolver.get(ctxWeightName),
        ctxWeightName
      );
      const ctxBias = resolver.get(ctxBiasName);
      const ctxSegments = resolveModulationSegments(ctxWeight, hiddenSize, 6, resolver, ctxWeightName);
      if (ctxSegments < 6) {
        throw new Error(`Unsupported modulation segments=${ctxSegments} for ${ctxWeightName}`);
      }
      const ctxBiasTensor = createBiasTensorWithDtype(
        ctxBias,
        hiddenSize * ctxSegments,
        'sd3_ctx_mod_bias',
        resolver,
        ctxBiasName
      );
      ctxMod = await buildModulation(timeText, ctxWeight, ctxBiasTensor, hiddenSize, ctxSegments, runtime, matmul, ctxWeightName, ops);
      ctxOffsets = resolveModulationOffsets(ctxSegments, hiddenSize);
      ctxAttnOffsets = ctxOffsets.attn;
      ctxFfOffsets = ctxOffsets.ff;
    }

    const xAttnIn = await applyAdaLayerNorm(
      x,
      onesBuf,
      zerosBuf,
      layerNormEps,
      mod,
      attnOffsets,
      runtime,
      ops,
      release,
      { numTokens: tokenCount, hiddenSize }
    );

    if (dualLayers.has(layerIdx)) {
      const ctxAttnIn = await applyAdaLayerNorm(
        ctx,
        onesBuf,
        zerosBuf,
        layerNormEps,
        ctxMod,
        ctxAttnOffsets,
        runtime,
        ops,
        release,
        { numTokens: ctx.shape[0], hiddenSize }
      );

      const attnWeightNames = {
        q: `transformer_blocks.${layerIdx}.attn.to_q.weight`,
        k: `transformer_blocks.${layerIdx}.attn.to_k.weight`,
        v: `transformer_blocks.${layerIdx}.attn.to_v.weight`,
        qkv: `transformer_blocks.${layerIdx}.attn.qkv.weight`,
      };
      const attnWeights = {
        q: resolver.get(attnWeightNames.q),
        k: resolver.get(attnWeightNames.k),
        v: resolver.get(attnWeightNames.v),
        qkv: resolver.get(attnWeightNames.qkv),
      };
      const attnBiasNames = {
        q: `transformer_blocks.${layerIdx}.attn.to_q.bias`,
        k: `transformer_blocks.${layerIdx}.attn.to_k.bias`,
        v: `transformer_blocks.${layerIdx}.attn.to_v.bias`,
        qkv: `transformer_blocks.${layerIdx}.attn.qkv.bias`,
      };
      const attnBias = {
        q: createBiasTensorWithDtype(
          resolver.get(attnBiasNames.q),
          hiddenSize,
          'sd3_attn_q_bias',
          resolver,
          attnBiasNames.q
        ),
        k: createBiasTensorWithDtype(
          resolver.get(attnBiasNames.k),
          hiddenSize,
          'sd3_attn_k_bias',
          resolver,
          attnBiasNames.k
        ),
        v: createBiasTensorWithDtype(
          resolver.get(attnBiasNames.v),
          hiddenSize,
          'sd3_attn_v_bias',
          resolver,
          attnBiasNames.v
        ),
        qkv: createBiasTensorWithDtype(
          resolver.get(attnBiasNames.qkv),
          hiddenSize * 3,
          'sd3_attn_qkv_bias',
          resolver,
          attnBiasNames.qkv
        ),
      };
      const addWeightNames = {
        q: `transformer_blocks.${layerIdx}.attn.add_q_proj.weight`,
        k: `transformer_blocks.${layerIdx}.attn.add_k_proj.weight`,
        v: `transformer_blocks.${layerIdx}.attn.add_v_proj.weight`,
        qkv: `transformer_blocks.${layerIdx}.attn.add_qkv.weight`,
      };
      const addWeights = {
        q: resolver.get(addWeightNames.q),
        k: resolver.get(addWeightNames.k),
        v: resolver.get(addWeightNames.v),
        qkv: resolver.get(addWeightNames.qkv),
      };
      const addBiasNames = {
        q: `transformer_blocks.${layerIdx}.attn.add_q_proj.bias`,
        k: `transformer_blocks.${layerIdx}.attn.add_k_proj.bias`,
        v: `transformer_blocks.${layerIdx}.attn.add_v_proj.bias`,
        qkv: `transformer_blocks.${layerIdx}.attn.add_qkv.bias`,
      };
      const addBias = {
        q: createBiasTensorWithDtype(
          resolver.get(addBiasNames.q),
          hiddenSize,
          'sd3_attn_add_q_bias',
          resolver,
          addBiasNames.q
        ),
        k: createBiasTensorWithDtype(
          resolver.get(addBiasNames.k),
          hiddenSize,
          'sd3_attn_add_k_bias',
          resolver,
          addBiasNames.k
        ),
        v: createBiasTensorWithDtype(
          resolver.get(addBiasNames.v),
          hiddenSize,
          'sd3_attn_add_v_bias',
          resolver,
          addBiasNames.v
        ),
        qkv: createBiasTensorWithDtype(
          resolver.get(addBiasNames.qkv),
          hiddenSize * 3,
          'sd3_attn_add_qkv_bias',
          resolver,
          addBiasNames.qkv
        ),
      };

      const normWeights = {
        q: resolver.get(`transformer_blocks.${layerIdx}.attn.norm_q.weight`),
        k: resolver.get(`transformer_blocks.${layerIdx}.attn.norm_k.weight`),
        qAdd: resolver.get(`transformer_blocks.${layerIdx}.attn.norm_added_q.weight`),
        kAdd: resolver.get(`transformer_blocks.${layerIdx}.attn.norm_added_k.weight`),
      };

      let { q: qx, k: kx, v: vx } = await runQKV(
        xAttnIn,
        attnWeights,
        attnBias,
        tokenCount,
        hiddenSize,
        `sd3_attn_${layerIdx}`,
        matmul,
        attnWeightNames,
        ops,
        release,
        recorder
      );

      let { q: qc, k: kc, v: vc } = await runQKV(
        ctxAttnIn,
        addWeights,
        addBias,
        ctx.shape[0],
        hiddenSize,
        `sd3_attn_add_${layerIdx}`,
        matmul,
        addWeightNames,
        ops,
        release,
        recorder
      );

      if (normWeights.q) {
        const normed = await applyQKNorm(qx, normWeights.q, tokenCount, numHeads, headDim, layerNormEps, ops);
        release(qx.buffer);
        qx = normed;
      }
      if (normWeights.k) {
        const normed = await applyQKNorm(kx, normWeights.k, tokenCount, numHeads, headDim, layerNormEps, ops);
        release(kx.buffer);
        kx = normed;
      }
      if (normWeights.qAdd) {
        const normed = await applyQKNorm(qc, normWeights.qAdd, ctx.shape[0], numHeads, headDim, layerNormEps, ops);
        release(qc.buffer);
        qc = normed;
      }
      if (normWeights.kAdd) {
        const normed = await applyQKNorm(kc, normWeights.kAdd, ctx.shape[0], numHeads, headDim, layerNormEps, ops);
        release(kc.buffer);
        kc = normed;
      }

      const kAll = await concatKV(kx, kc, tokenCount, ctx.shape[0], hiddenSize, recorder);
      const vAll = await concatKV(vx, vc, tokenCount, ctx.shape[0], hiddenSize, recorder);

      const attnX = await ops.attention(qx, kAll, vAll, null, numHeads, headDim, {
        seqLen: tokenCount,
        kvLen: tokenCount + ctx.shape[0],
        numKVHeads: numHeads,
        causal: false,
      });

      const attnC = await ops.attention(qc, kAll, vAll, null, numHeads, headDim, {
        seqLen: ctx.shape[0],
        kvLen: tokenCount + ctx.shape[0],
        numKVHeads: numHeads,
        causal: false,
      });

      const outWeightName = `transformer_blocks.${layerIdx}.attn.to_out.0.weight`;
      const outWeight = expectWeight(
        resolver.get(outWeightName),
        outWeightName
      );
      const outBiasName = `transformer_blocks.${layerIdx}.attn.to_out.0.bias`;
      const outBias = resolver.get(outBiasName);
      const outAddWeightName = `transformer_blocks.${layerIdx}.attn.to_add_out.weight`;
      const outAddWeight = expectWeight(
        resolver.get(outAddWeightName),
        outAddWeightName
      );
      const outAddBiasName = `transformer_blocks.${layerIdx}.attn.to_add_out.bias`;
      const outAddBias = resolver.get(outAddBiasName);

      let attnOutX = await matmul(attnX, outWeight, outWeightName, tokenCount, hiddenSize, hiddenSize, {
        outputDtype: attnX.dtype,
        transposeB: 'auto',
      });
      if (outBias) {
        attnOutX = await ops.biasAdd(
          attnOutX,
          createBiasTensorWithDtype(outBias, hiddenSize, 'sd3_attn_out_bias', resolver, outBiasName),
          tokenCount,
          hiddenSize
        );
      }

      let attnOutC = await matmul(attnC, outAddWeight, outAddWeightName, ctx.shape[0], hiddenSize, hiddenSize, {
        outputDtype: attnC.dtype,
        transposeB: 'auto',
      });
      if (outAddBias) {
        attnOutC = await ops.biasAdd(
          attnOutC,
          createBiasTensorWithDtype(outAddBias, hiddenSize, 'sd3_attn_out_add_bias', resolver, outAddBiasName),
          ctx.shape[0],
          hiddenSize
        );
      }

      const gatedX = await applyGate(attnOutX, mod, attnOffsets, ops, release, { numTokens: tokenCount, hiddenSize, zeroOffset: mod.zeroOffset });
      const gatedC = await applyGate(attnOutC, ctxMod, ctxAttnOffsets, ops, release, { numTokens: ctx.shape[0], hiddenSize, zeroOffset: ctxMod.zeroOffset });

      const xRes = await ops.residualAdd(x, gatedX, tokenCount * hiddenSize, { useVec4: true });
      const cRes = await ops.residualAdd(ctx, gatedC, ctx.shape[0] * hiddenSize, { useVec4: true });

      release(xAttnIn.buffer);
      release(ctxAttnIn.buffer);
      release(qx.buffer);
      release(kx.buffer);
      release(vx.buffer);
      release(qc.buffer);
      release(kc.buffer);
      release(vc.buffer);
      release(kAll.buffer);
      release(vAll.buffer);
      release(attnX.buffer);
      release(attnC.buffer);
      release(gatedX.buffer);
      release(gatedC.buffer);
      release(x.buffer);
      if (ctxOwned) {
        release(ctx.buffer);
      }

      x = createTensor(xRes.buffer, xRes.dtype, [tokenCount, hiddenSize], 'sd3_x');
      ctx = createTensor(cRes.buffer, cRes.dtype, [ctx.shape[0], hiddenSize], 'sd3_ctx');
      ctxOwned = true;

      const ctxFfIn = await applyAdaLayerNorm(
        ctx,
        onesBuf,
        zerosBuf,
        layerNormEps,
        ctxMod,
        ctxFfOffsets,
        runtime,
        ops,
        release,
        { numTokens: ctx.shape[0], hiddenSize }
      );

      const ffCtxWeightNames = {
        up: `transformer_blocks.${layerIdx}.ff_context.net.0.proj.weight`,
        down: `transformer_blocks.${layerIdx}.ff_context.net.2.weight`,
      };
      const ffCtxWeights = {
        up: expectWeight(
          resolver.get(ffCtxWeightNames.up),
          ffCtxWeightNames.up
        ),
        down: expectWeight(
          resolver.get(ffCtxWeightNames.down),
          ffCtxWeightNames.down
        ),
      };
      const ffCtxBiasNames = {
        up: `transformer_blocks.${layerIdx}.ff_context.net.0.proj.bias`,
        down: `transformer_blocks.${layerIdx}.ff_context.net.2.bias`,
      };
      const ffCtxBias = {
        up: createBiasTensorWithDtype(
          resolver.get(ffCtxBiasNames.up),
          ffCtxWeights.up.shape[0],
          'sd3_ff_ctx_up_bias',
          resolver,
          ffCtxBiasNames.up
        ),
        down: createBiasTensorWithDtype(
          resolver.get(ffCtxBiasNames.down),
          hiddenSize,
          'sd3_ff_ctx_down_bias',
          resolver,
          ffCtxBiasNames.down
        ),
      };
      const ffCtxOut = await runFFN(
        ctxFfIn,
        ffCtxWeights,
        ffCtxBias,
        ctx.shape[0],
        hiddenSize,
        runtime,
        matmul,
        ffCtxWeightNames,
        ops,
        release
      );
      const ffCtxGated = await applyGate(ffCtxOut, ctxMod, ctxFfOffsets, ops, release, { numTokens: ctx.shape[0], hiddenSize, zeroOffset: ctxMod.zeroOffset });
      const ctxRes2 = await ops.residualAdd(ctx, ffCtxGated, ctx.shape[0] * hiddenSize, { useVec4: true });

      release(ctxFfIn.buffer);
      release(ffCtxGated.buffer);
      if (ctxOwned) {
        release(ctx.buffer);
      }
      ctx = createTensor(ctxRes2.buffer, ctxRes2.dtype, [ctx.shape[0], hiddenSize], 'sd3_ctx');
      ctxOwned = true;

    } else {
      release(xAttnIn.buffer);
    }

    const hasAttn2 = attn2Layers ? attn2Layers.has(layerIdx) : config.dual_attention_layers ? dualLayers.has(layerIdx) : true;
    if (hasAttn2) {
      const xAttn2In = await applyAdaLayerNorm(
        x,
        onesBuf,
        zerosBuf,
        layerNormEps,
        mod,
        attn2Offsets,
        runtime,
        ops,
        release,
        { numTokens: tokenCount, hiddenSize }
      );

      const attn2WeightNames = {
        q: `transformer_blocks.${layerIdx}.attn2.to_q.weight`,
        k: `transformer_blocks.${layerIdx}.attn2.to_k.weight`,
        v: `transformer_blocks.${layerIdx}.attn2.to_v.weight`,
        qkv: `transformer_blocks.${layerIdx}.attn2.qkv.weight`,
      };
      const attn2Weights = {
        q: resolver.get(attn2WeightNames.q),
        k: resolver.get(attn2WeightNames.k),
        v: resolver.get(attn2WeightNames.v),
        qkv: resolver.get(attn2WeightNames.qkv),
      };
      const attn2BiasNames = {
        q: `transformer_blocks.${layerIdx}.attn2.to_q.bias`,
        k: `transformer_blocks.${layerIdx}.attn2.to_k.bias`,
        v: `transformer_blocks.${layerIdx}.attn2.to_v.bias`,
        qkv: `transformer_blocks.${layerIdx}.attn2.qkv.bias`,
      };
      const attn2Bias = {
        q: createBiasTensorWithDtype(
          resolver.get(attn2BiasNames.q),
          hiddenSize,
          'sd3_attn2_q_bias',
          resolver,
          attn2BiasNames.q
        ),
        k: createBiasTensorWithDtype(
          resolver.get(attn2BiasNames.k),
          hiddenSize,
          'sd3_attn2_k_bias',
          resolver,
          attn2BiasNames.k
        ),
        v: createBiasTensorWithDtype(
          resolver.get(attn2BiasNames.v),
          hiddenSize,
          'sd3_attn2_v_bias',
          resolver,
          attn2BiasNames.v
        ),
        qkv: createBiasTensorWithDtype(
          resolver.get(attn2BiasNames.qkv),
          hiddenSize * 3,
          'sd3_attn2_qkv_bias',
          resolver,
          attn2BiasNames.qkv
        ),
      };

      let { q: q2, k: k2, v: v2 } = await runQKV(
        xAttn2In,
        attn2Weights,
        attn2Bias,
        tokenCount,
        hiddenSize,
        `sd3_attn2_${layerIdx}`,
        matmul,
        attn2WeightNames,
        ops,
        release,
        recorder
      );

      const normQ2 = resolver.get(`transformer_blocks.${layerIdx}.attn2.norm_q.weight`);
      const normK2 = resolver.get(`transformer_blocks.${layerIdx}.attn2.norm_k.weight`);
      if (normQ2) {
        const normed = await applyQKNorm(q2, normQ2, tokenCount, numHeads, headDim, layerNormEps, ops);
        release(q2.buffer);
        q2 = normed;
      }
      if (normK2) {
        const normed = await applyQKNorm(k2, normK2, tokenCount, numHeads, headDim, layerNormEps, ops);
        release(k2.buffer);
        k2 = normed;
      }

      const attn2 = await ops.attention(q2, k2, v2, null, numHeads, headDim, {
        seqLen: tokenCount,
        kvLen: tokenCount,
        numKVHeads: numHeads,
        causal: false,
      });

      const attn2OutWeightName = `transformer_blocks.${layerIdx}.attn2.to_out.0.weight`;
      const attn2OutWeight = expectWeight(
        resolver.get(attn2OutWeightName),
        attn2OutWeightName
      );
      const attn2OutBiasName = `transformer_blocks.${layerIdx}.attn2.to_out.0.bias`;
      const attn2OutBias = resolver.get(attn2OutBiasName);
      let attn2Out = await matmul(attn2, attn2OutWeight, attn2OutWeightName, tokenCount, hiddenSize, hiddenSize, {
        outputDtype: attn2.dtype,
        transposeB: 'auto',
      });
      if (attn2OutBias) {
        attn2Out = await ops.biasAdd(
          attn2Out,
          createBiasTensorWithDtype(attn2OutBias, hiddenSize, 'sd3_attn2_out_bias', resolver, attn2OutBiasName),
          tokenCount,
          hiddenSize
        );
      }

      const gated2 = await applyGate(attn2Out, mod, attn2Offsets, ops, release, { numTokens: tokenCount, hiddenSize, zeroOffset: mod.zeroOffset });
      const xRes2 = await ops.residualAdd(x, gated2, tokenCount * hiddenSize, { useVec4: true });

      release(xAttn2In.buffer);
      release(q2.buffer);
      release(k2.buffer);
      release(v2.buffer);
      release(attn2.buffer);
      release(attn2Out.buffer);
      release(gated2.buffer);
      release(x.buffer);

      x = createTensor(xRes2.buffer, xRes2.dtype, [tokenCount, hiddenSize], 'sd3_x');
    }

    const xFfIn = await applyAdaLayerNorm(
      x,
      onesBuf,
      zerosBuf,
      layerNormEps,
      mod,
      ffOffsets,
      runtime,
      ops,
      release,
      { numTokens: tokenCount, hiddenSize }
    );

    const ffWeightNames = {
      up: `transformer_blocks.${layerIdx}.ff.net.0.proj.weight`,
      down: `transformer_blocks.${layerIdx}.ff.net.2.weight`,
    };
    const ffWeights = {
      up: expectWeight(
        resolver.get(ffWeightNames.up),
        ffWeightNames.up
      ),
      down: expectWeight(
        resolver.get(ffWeightNames.down),
        ffWeightNames.down
      ),
    };
    const ffBiasNames = {
      up: `transformer_blocks.${layerIdx}.ff.net.0.proj.bias`,
      down: `transformer_blocks.${layerIdx}.ff.net.2.bias`,
    };
    const ffBias = {
      up: createBiasTensorWithDtype(
        resolver.get(ffBiasNames.up),
        ffWeights.up.shape[0],
        'sd3_ff_up_bias',
        resolver,
        ffBiasNames.up
      ),
      down: createBiasTensorWithDtype(
        resolver.get(ffBiasNames.down),
        hiddenSize,
        'sd3_ff_down_bias',
        resolver,
        ffBiasNames.down
      ),
    };

    const ffOut = await runFFN(
      xFfIn,
      ffWeights,
      ffBias,
      tokenCount,
      hiddenSize,
      runtime,
      matmul,
      ffWeightNames,
      ops,
      release
    );
    const ffGated = await applyGate(ffOut, mod, ffOffsets, ops, release, { numTokens: tokenCount, hiddenSize, zeroOffset: mod.zeroOffset });
    const xRes3 = await ops.residualAdd(x, ffGated, tokenCount * hiddenSize, { useVec4: true });

    release(xFfIn.buffer);
    release(ffGated.buffer);
    release(x.buffer);

    x = createTensor(xRes3.buffer, xRes3.dtype, [tokenCount, hiddenSize], 'sd3_x');

    release(mod.tensor.buffer);
    if (ctxMod?.tensor?.buffer) {
      release(ctxMod.tensor.buffer);
    }
  }

  const normOutWeightName = 'norm_out.linear.weight';
  const normOutWeight = expectWeight(resolver.get(normOutWeightName), normOutWeightName);
  const normOutBias = resolver.get('norm_out.linear.bias');
  const normOutSegments = resolveModulationSegments(normOutWeight, hiddenSize, 2, resolver, normOutWeightName);
  const normOutBiasTensor = createBiasTensorWithDtype(
    normOutBias,
    hiddenSize * normOutSegments,
    'sd3_norm_out_bias',
    resolver,
    'norm_out.linear.bias'
  );
  const normOut = await buildModulation(timeText, normOutWeight, normOutBiasTensor, hiddenSize, normOutSegments, runtime, matmul, normOutWeightName, ops);

  const xNorm = await ops.layerNorm(x, onesBuf, zerosBuf, layerNormEps, { batchSize: tokenCount, hiddenSize });
  const xMod = await ops.modulate(xNorm, normOut.tensor, {
    numTokens: tokenCount,
    hiddenSize,
    scaleOffset: 0,
    shiftOffset: hiddenSize,
    gateOffset: 0,
    hasGate: false,
    addOne: true,
  });

  release(xNorm.buffer);
  release(x.buffer);
  release(normOut.tensor.buffer);
  if (ctxOwned) {
    release(ctx.buffer);
  }
  release(onesBuf);
  release(zerosBuf);

  const projOutWeightName = 'proj_out.weight';
  const projOutWeight = expectWeight(resolver.get(projOutWeightName), projOutWeightName);
  const projOutBiasName = 'proj_out.bias';
  const projOutBias = resolver.get(projOutBiasName);
  let patch = await matmul(xMod, projOutWeight, projOutWeightName, tokenCount, projOutWeight.shape[0], hiddenSize, {
    outputDtype: xMod.dtype,
    transposeB: 'auto',
  });
  if (projOutBias) {
    patch = await ops.biasAdd(
      patch,
      createBiasTensorWithDtype(projOutBias, projOutWeight.shape[0], 'sd3_proj_out_bias', resolver, projOutBiasName),
      tokenCount,
      projOutWeight.shape[0]
    );
  }

  release(xMod.buffer);

  const patchChannels = projOutWeight.shape[0];
  const output = await ops.pixelShuffle(patch, {
    outChannels: latentChannels,
    outHeight: latentHeight,
    outWidth: latentWidth,
    gridWidth,
    gridHeight,
    patchSize,
    patchChannels,
  });

  release(patch.buffer);

  return output;
}
