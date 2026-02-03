import { getDevice, getKernelCapabilities } from '../../gpu/device.js';
import { createTensor, dtypeBytes } from '../../gpu/tensor.js';
import { getBuffer } from '../../gpu/weight-buffer.js';
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
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

async function runMatmulResolved(input, weight, resolver, name, M, N, K, options = {}) {
  const bDtype = resolveMatmulDtype(weight, resolver, name);
  const nextOptions = bDtype ? { ...options, bDtype } : options;
  return runMatmul(input, weight, M, N, K, nextOptions);
}

function expectWeight(weight, label) {
  if (!weight) {
    throw new Error(`Missing diffusion weight: ${label}`);
  }
  return weight;
}

function createBiasTensor(weight, size, label) {
  if (!weight) return null;
  return createTensor(getBuffer(weight), 'f32', [size], label);
}

async function splitQKV(qkv, numTokens, hiddenSize, label) {
  const device = getDevice();
  const bytesPerElement = dtypeBytes(qkv.dtype);
  const sliceBytes = numTokens * hiddenSize * bytesPerElement;
  const qBuf = acquireBuffer(sliceBytes, undefined, `${label}_q`);
  const kBuf = acquireBuffer(sliceBytes, undefined, `${label}_k`);
  const vBuf = acquireBuffer(sliceBytes, undefined, `${label}_v`);

  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(qkv.buffer, 0, qBuf, 0, sliceBytes);
  encoder.copyBufferToBuffer(qkv.buffer, sliceBytes, kBuf, 0, sliceBytes);
  encoder.copyBufferToBuffer(qkv.buffer, sliceBytes * 2, vBuf, 0, sliceBytes);
  device.queue.submit([encoder.finish()]);

  return {
    q: createTensor(qBuf, qkv.dtype, [numTokens, hiddenSize], `${label}_q`),
    k: createTensor(kBuf, qkv.dtype, [numTokens, hiddenSize], `${label}_k`),
    v: createTensor(vBuf, qkv.dtype, [numTokens, hiddenSize], `${label}_v`),
  };
}

async function runFusedQKV(input, weight, bias, numTokens, hiddenSize, outputDtype, label, matmul, weightName) {
  const qkv = await matmul(input, weight, weightName, numTokens, hiddenSize * 3, hiddenSize, {
    outputDtype,
    transposeB: 'auto',
  });

  let qkvTensor = qkv;
  if (bias) {
    const biasTensor = createBiasTensor(bias, hiddenSize * 3, `${label}_qkv_bias`);
    qkvTensor = await runBiasAdd(qkv, biasTensor, numTokens, hiddenSize * 3);
  }

  const split = await splitQKV(qkvTensor, numTokens, hiddenSize, label);
  releaseBuffer(qkvTensor.buffer);
  return split;
}

async function runQKV(input, weights, bias, numTokens, hiddenSize, label, matmul, weightNames) {
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
      weightNames?.qkv ?? null
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

  if (bias?.q) q = await runBiasAdd(q, bias.q, numTokens, hiddenSize);
  if (bias?.k) k = await runBiasAdd(k, bias.k, numTokens, hiddenSize);
  if (bias?.v) v = await runBiasAdd(v, bias.v, numTokens, hiddenSize);

  return { q, k, v };
}

async function applyQKNorm(tensor, weight, numTokens, numHeads, headDim, eps) {
  const flattened = createTensor(tensor.buffer, tensor.dtype, [numTokens * numHeads, headDim], 'qk_norm_in');
  const normed = await runRMSNorm(flattened, getBuffer(weight), eps, {
    batchSize: numTokens * numHeads,
    hiddenSize: headDim,
  });
  return reshapeTensor(normed, [numTokens, numHeads, headDim], 'qk_norm_out');
}

async function concatKV(a, b, numTokensA, numTokensB, hiddenSize) {
  const device = getDevice();
  const bytesPerElement = a.dtype === 'f16' ? 2 : 4;
  const outputSize = (numTokensA + numTokensB) * hiddenSize * bytesPerElement;
  const output = acquireBuffer(outputSize, undefined, 'kv_concat');
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(a.buffer, 0, output, 0, numTokensA * hiddenSize * bytesPerElement);
  encoder.copyBufferToBuffer(b.buffer, 0, output, numTokensA * hiddenSize * bytesPerElement, numTokensB * hiddenSize * bytesPerElement);
  device.queue.submit([encoder.finish()]);
  return createTensor(output, a.dtype, [numTokensA + numTokensB, hiddenSize], 'kv_concat');
}

async function runAttentionBlock(input, weights, bias, numTokens, hiddenSize, numHeads, headDim, normWeights, eps, matmul, weightNames) {
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

  if (bias?.q) q = await runBiasAdd(q, bias.q, numTokens, hiddenSize);
  if (bias?.k) k = await runBiasAdd(k, bias.k, numTokens, hiddenSize);
  if (bias?.v) v = await runBiasAdd(v, bias.v, numTokens, hiddenSize);

  if (normWeights?.q) {
    const normed = await applyQKNorm(q, normWeights.q, numTokens, numHeads, headDim, eps);
    releaseBuffer(q.buffer);
    q = normed;
  }
  if (normWeights?.k) {
    const normed = await applyQKNorm(k, normWeights.k, numTokens, numHeads, headDim, eps);
    releaseBuffer(k.buffer);
    k = normed;
  }

  const attn = await runAttention(q, k, v, null, numHeads, headDim, {
    seqLen: numTokens,
    kvLen: numTokens,
    numKVHeads: numHeads,
    causal: false,
  });

  releaseBuffer(q.buffer);
  releaseBuffer(k.buffer);
  releaseBuffer(v.buffer);

  return attn;
}

async function buildModulation(timeText, weight, bias, hiddenSize, segments, runtime, matmul, weightName) {
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
    await runBiasAdd(mod, bias, 1, outDim);
  }

  const zeroOffset = outDim * bytesPerElement;
  device.queue.writeBuffer(outputBuffer, zeroOffset, new Uint8Array(hiddenSize * bytesPerElement));

  return {
    tensor: createTensor(outputBuffer, activationDtype, [1, outDim], 'sd3_mod'),
    zeroOffset: outDim,
  };
}

async function applyAdaLayerNorm(input, weight, bias, eps, mod, offsets, runtime, options = {}) {
  const { numTokens, hiddenSize } = options;
  const normed = await runLayerNorm(input, weight, bias, eps, { batchSize: numTokens, hiddenSize });
  const modulated = await runModulate(normed, mod.tensor, {
    numTokens,
    hiddenSize,
    scaleOffset: offsets.scale,
    shiftOffset: offsets.shift,
    gateOffset: offsets.gate,
    hasGate: false,
    addOne: true,
  });
  releaseBuffer(normed.buffer);
  return modulated;
}

async function applyGate(output, mod, offsets, options = {}) {
  const { numTokens, hiddenSize, zeroOffset } = options;
  const gated = await runModulate(output, mod.tensor, {
    numTokens,
    hiddenSize,
    scaleOffset: offsets.gate,
    shiftOffset: zeroOffset,
    gateOffset: offsets.gate,
    hasGate: false,
    addOne: false,
  });
  releaseBuffer(output.buffer);
  return gated;
}

async function runFFN(input, weights, bias, numTokens, hiddenSize, runtime, matmul, weightNames) {
  const activationDtype = resolveActivationDtype(runtime);
  const upDim = weights.up.shape[0];
  const downInput = weights.down.shape[1];
  let up = await matmul(input, weights.up, weightNames?.up ?? null, numTokens, upDim, hiddenSize, {
    outputDtype: activationDtype,
    transposeB: 'auto',
  });
  if (bias?.up) up = await runBiasAdd(up, bias.up, numTokens, upDim);

  let act = null;
  let intermediate = upDim;
  if (Number.isFinite(downInput) && upDim === downInput * 2) {
    act = await runSiLURowSplit(up, {
      numTokens,
      dim: downInput,
      activation: 'gelu',
      swigluLimit: null,
    });
    intermediate = downInput;
  } else {
    act = await runGeLU(up, { size: numTokens * upDim });
  }
  releaseBuffer(up.buffer);

  let down = await matmul(act, weights.down, weightNames?.down ?? null, numTokens, hiddenSize, intermediate, {
    outputDtype: activationDtype,
    transposeB: 'auto',
  });
  if (bias?.down) down = await runBiasAdd(down, bias.down, numTokens, hiddenSize);
  releaseBuffer(act.buffer);
  return down;
}

export async function runSD3Transformer(latents, context, timeText, weightsEntry, modelConfig, runtime) {
  const device = getDevice();
  if (!device) {
    throw new Error('SD3 transformer requires a WebGPU device.');
  }

  const resolver = createSD3WeightResolver(weightsEntry, modelConfig);
  const matmul = (input, weight, name, M, N, K, options = {}) =>
    runMatmulResolved(input, weight, resolver, name, M, N, K, options);
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

  const conv = await runConv2D(latents, projWeight, projBias, {
    inChannels: latentChannels,
    outChannels: hiddenSize,
    height: latentHeight,
    width: latentWidth,
    kernelH: patchSize,
    kernelW: patchSize,
    stride: patchSize,
    pad: 0,
  });

  const tokens = await runTranspose(conv, hiddenSize, tokenCount);
  releaseBuffer(conv.buffer);

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
  const pos = await runGather(
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
  posBuffer.destroy();

  const xCombined = await runResidualAdd(tokens, pos, tokenCount * hiddenSize, { useVec4: true });
  releaseBuffer(tokens.buffer);
  releaseBuffer(pos.buffer);

  let x = createTensor(xCombined.buffer, xCombined.dtype, [tokenCount, hiddenSize], 'sd3_tokens');
  let ctx = context;

  const ones = new Float32Array(hiddenSize).fill(1.0);
  const zeros = new Float32Array(hiddenSize);
  const onesBuf = createVectorBuffer(device, ones, 'sd3_ln_weight');
  const zerosBuf = createVectorBuffer(device, zeros, 'sd3_ln_bias');

  const dualLayers = new Set(config.dual_attention_layers || []);
  const numLayers = config.num_layers;

  for (let layerIdx = 0; layerIdx < numLayers; layerIdx++) {
    const modWeightName = `transformer_blocks.${layerIdx}.norm1.linear.weight`;
    const modWeight = expectWeight(
      resolver.get(modWeightName),
      modWeightName
    );
    const modBias = resolver.get(`transformer_blocks.${layerIdx}.norm1.linear.bias`);
    const mod = await buildModulation(timeText, modWeight, modBias, hiddenSize, 9, runtime, matmul, modWeightName);

    const offsets = {
      attn: { scale: 0, shift: hiddenSize, gate: hiddenSize * 2 },
      attn2: { scale: hiddenSize * 3, shift: hiddenSize * 4, gate: hiddenSize * 5 },
      ff: { scale: hiddenSize * 6, shift: hiddenSize * 7, gate: hiddenSize * 8 },
    };

    let ctxMod = null;
    let ctxOffsets = null;
    if (dualLayers.has(layerIdx)) {
      const ctxWeightName = `transformer_blocks.${layerIdx}.norm1_context.linear.weight`;
      const ctxWeight = expectWeight(
        resolver.get(ctxWeightName),
        ctxWeightName
      );
      const ctxBias = resolver.get(`transformer_blocks.${layerIdx}.norm1_context.linear.bias`);
      ctxMod = await buildModulation(timeText, ctxWeight, ctxBias, hiddenSize, 6, runtime, matmul, ctxWeightName);
      ctxOffsets = {
        attn: { scale: 0, shift: hiddenSize, gate: hiddenSize * 2 },
        ff: { scale: hiddenSize * 3, shift: hiddenSize * 4, gate: hiddenSize * 5 },
      };
    }

    const xAttnIn = await applyAdaLayerNorm(
      x,
      onesBuf,
      zerosBuf,
      layerNormEps,
      mod,
      offsets.attn,
      runtime,
      { numTokens: tokenCount, hiddenSize }
    );

    if (dualLayers.has(layerIdx)) {
      const ctxAttnIn = await applyAdaLayerNorm(
        ctx,
        onesBuf,
        zerosBuf,
        layerNormEps,
        ctxMod,
        ctxOffsets.attn,
        runtime,
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
      const attnBias = {
        q: createBiasTensor(resolver.get(`transformer_blocks.${layerIdx}.attn.to_q.bias`), hiddenSize, 'sd3_attn_q_bias'),
        k: createBiasTensor(resolver.get(`transformer_blocks.${layerIdx}.attn.to_k.bias`), hiddenSize, 'sd3_attn_k_bias'),
        v: createBiasTensor(resolver.get(`transformer_blocks.${layerIdx}.attn.to_v.bias`), hiddenSize, 'sd3_attn_v_bias'),
        qkv: resolver.get(`transformer_blocks.${layerIdx}.attn.qkv.bias`),
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
      const addBias = {
        q: createBiasTensor(resolver.get(`transformer_blocks.${layerIdx}.attn.add_q_proj.bias`), hiddenSize, 'sd3_attn_add_q_bias'),
        k: createBiasTensor(resolver.get(`transformer_blocks.${layerIdx}.attn.add_k_proj.bias`), hiddenSize, 'sd3_attn_add_k_bias'),
        v: createBiasTensor(resolver.get(`transformer_blocks.${layerIdx}.attn.add_v_proj.bias`), hiddenSize, 'sd3_attn_add_v_bias'),
        qkv: resolver.get(`transformer_blocks.${layerIdx}.attn.add_qkv.bias`),
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
        attnWeightNames
      );

      let { q: qc, k: kc, v: vc } = await runQKV(
        ctxAttnIn,
        addWeights,
        addBias,
        ctx.shape[0],
        hiddenSize,
        `sd3_attn_add_${layerIdx}`,
        matmul,
        addWeightNames
      );

      if (normWeights.q) {
        const normed = await applyQKNorm(qx, normWeights.q, tokenCount, numHeads, headDim, layerNormEps);
        releaseBuffer(qx.buffer);
        qx = normed;
      }
      if (normWeights.k) {
        const normed = await applyQKNorm(kx, normWeights.k, tokenCount, numHeads, headDim, layerNormEps);
        releaseBuffer(kx.buffer);
        kx = normed;
      }
      if (normWeights.qAdd) {
        const normed = await applyQKNorm(qc, normWeights.qAdd, ctx.shape[0], numHeads, headDim, layerNormEps);
        releaseBuffer(qc.buffer);
        qc = normed;
      }
      if (normWeights.kAdd) {
        const normed = await applyQKNorm(kc, normWeights.kAdd, ctx.shape[0], numHeads, headDim, layerNormEps);
        releaseBuffer(kc.buffer);
        kc = normed;
      }

      const kAll = await concatKV(kx, kc, tokenCount, ctx.shape[0], hiddenSize);
      const vAll = await concatKV(vx, vc, tokenCount, ctx.shape[0], hiddenSize);

      const attnX = await runAttention(qx, kAll, vAll, null, numHeads, headDim, {
        seqLen: tokenCount,
        kvLen: tokenCount + ctx.shape[0],
        numKVHeads: numHeads,
        causal: false,
      });

      const attnC = await runAttention(qc, kAll, vAll, null, numHeads, headDim, {
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
      const outBias = resolver.get(`transformer_blocks.${layerIdx}.attn.to_out.0.bias`);
      const outAddWeightName = `transformer_blocks.${layerIdx}.attn.to_add_out.weight`;
      const outAddWeight = expectWeight(
        resolver.get(outAddWeightName),
        outAddWeightName
      );
      const outAddBias = resolver.get(`transformer_blocks.${layerIdx}.attn.to_add_out.bias`);

      let attnOutX = await matmul(attnX, outWeight, outWeightName, tokenCount, hiddenSize, hiddenSize, {
        outputDtype: attnX.dtype,
        transposeB: 'auto',
      });
      if (outBias) attnOutX = await runBiasAdd(attnOutX, createBiasTensor(outBias, hiddenSize, 'sd3_attn_out_bias'), tokenCount, hiddenSize);

      let attnOutC = await matmul(attnC, outAddWeight, outAddWeightName, ctx.shape[0], hiddenSize, hiddenSize, {
        outputDtype: attnC.dtype,
        transposeB: 'auto',
      });
      if (outAddBias) attnOutC = await runBiasAdd(attnOutC, createBiasTensor(outAddBias, hiddenSize, 'sd3_attn_out_add_bias'), ctx.shape[0], hiddenSize);

      const gatedX = await applyGate(attnOutX, mod, offsets.attn, { numTokens: tokenCount, hiddenSize, zeroOffset: mod.zeroOffset });
      const gatedC = await applyGate(attnOutC, ctxMod, ctxOffsets.attn, { numTokens: ctx.shape[0], hiddenSize, zeroOffset: ctxMod.zeroOffset });

      const xRes = await runResidualAdd(x, gatedX, tokenCount * hiddenSize, { useVec4: true });
      const cRes = await runResidualAdd(ctx, gatedC, ctx.shape[0] * hiddenSize, { useVec4: true });

      releaseBuffer(xAttnIn.buffer);
      releaseBuffer(ctxAttnIn.buffer);
      releaseBuffer(qx.buffer);
      releaseBuffer(kx.buffer);
      releaseBuffer(vx.buffer);
      releaseBuffer(qc.buffer);
      releaseBuffer(kc.buffer);
      releaseBuffer(vc.buffer);
      releaseBuffer(kAll.buffer);
      releaseBuffer(vAll.buffer);
      releaseBuffer(attnX.buffer);
      releaseBuffer(attnC.buffer);
      releaseBuffer(gatedX.buffer);
      releaseBuffer(gatedC.buffer);
      releaseBuffer(x.buffer);
      releaseBuffer(ctx.buffer);

      x = createTensor(xRes.buffer, xRes.dtype, [tokenCount, hiddenSize], 'sd3_x');
      ctx = createTensor(cRes.buffer, cRes.dtype, [ctx.shape[0], hiddenSize], 'sd3_ctx');

      const ctxFfIn = await applyAdaLayerNorm(
        ctx,
        onesBuf,
        zerosBuf,
        layerNormEps,
        ctxMod,
        ctxOffsets.ff,
        runtime,
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
      const ffCtxBias = {
        up: createBiasTensor(
          resolver.get(`transformer_blocks.${layerIdx}.ff_context.net.0.proj.bias`),
          ffCtxWeights.up.shape[0],
          'sd3_ff_ctx_up_bias'
        ),
        down: createBiasTensor(
          resolver.get(`transformer_blocks.${layerIdx}.ff_context.net.2.bias`),
          hiddenSize,
          'sd3_ff_ctx_down_bias'
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
        ffCtxWeightNames
      );
      const ffCtxGated = await applyGate(ffCtxOut, ctxMod, ctxOffsets.ff, { numTokens: ctx.shape[0], hiddenSize, zeroOffset: ctxMod.zeroOffset });
      const ctxRes2 = await runResidualAdd(ctx, ffCtxGated, ctx.shape[0] * hiddenSize, { useVec4: true });

      releaseBuffer(ctxFfIn.buffer);
      releaseBuffer(ffCtxGated.buffer);
      releaseBuffer(ctx.buffer);
      ctx = createTensor(ctxRes2.buffer, ctxRes2.dtype, [ctx.shape[0], hiddenSize], 'sd3_ctx');

    } else {
      releaseBuffer(xAttnIn.buffer);
    }

    const xAttn2In = await applyAdaLayerNorm(
      x,
      onesBuf,
      zerosBuf,
      layerNormEps,
      mod,
      offsets.attn2,
      runtime,
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
    const attn2Bias = {
      q: createBiasTensor(resolver.get(`transformer_blocks.${layerIdx}.attn2.to_q.bias`), hiddenSize, 'sd3_attn2_q_bias'),
      k: createBiasTensor(resolver.get(`transformer_blocks.${layerIdx}.attn2.to_k.bias`), hiddenSize, 'sd3_attn2_k_bias'),
      v: createBiasTensor(resolver.get(`transformer_blocks.${layerIdx}.attn2.to_v.bias`), hiddenSize, 'sd3_attn2_v_bias'),
      qkv: resolver.get(`transformer_blocks.${layerIdx}.attn2.qkv.bias`),
    };

    let { q: q2, k: k2, v: v2 } = await runQKV(
      xAttn2In,
      attn2Weights,
      attn2Bias,
      tokenCount,
      hiddenSize,
      `sd3_attn2_${layerIdx}`,
      matmul,
      attn2WeightNames
    );

    const normQ2 = resolver.get(`transformer_blocks.${layerIdx}.attn2.norm_q.weight`);
    const normK2 = resolver.get(`transformer_blocks.${layerIdx}.attn2.norm_k.weight`);
    if (normQ2) {
      const normed = await applyQKNorm(q2, normQ2, tokenCount, numHeads, headDim, layerNormEps);
      releaseBuffer(q2.buffer);
      q2 = normed;
    }
    if (normK2) {
      const normed = await applyQKNorm(k2, normK2, tokenCount, numHeads, headDim, layerNormEps);
      releaseBuffer(k2.buffer);
      k2 = normed;
    }

    const attn2 = await runAttention(q2, k2, v2, null, numHeads, headDim, {
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
    const attn2OutBias = resolver.get(`transformer_blocks.${layerIdx}.attn2.to_out.0.bias`);
    let attn2Out = await matmul(attn2, attn2OutWeight, attn2OutWeightName, tokenCount, hiddenSize, hiddenSize, {
      outputDtype: attn2.dtype,
      transposeB: 'auto',
    });
    if (attn2OutBias) attn2Out = await runBiasAdd(attn2Out, createBiasTensor(attn2OutBias, hiddenSize, 'sd3_attn2_out_bias'), tokenCount, hiddenSize);

    const gated2 = await applyGate(attn2Out, mod, offsets.attn2, { numTokens: tokenCount, hiddenSize, zeroOffset: mod.zeroOffset });
    const xRes2 = await runResidualAdd(x, gated2, tokenCount * hiddenSize, { useVec4: true });

    releaseBuffer(xAttn2In.buffer);
    releaseBuffer(q2.buffer);
    releaseBuffer(k2.buffer);
    releaseBuffer(v2.buffer);
    releaseBuffer(attn2.buffer);
    releaseBuffer(attn2Out.buffer);
    releaseBuffer(gated2.buffer);
    releaseBuffer(x.buffer);

    x = createTensor(xRes2.buffer, xRes2.dtype, [tokenCount, hiddenSize], 'sd3_x');

    const xFfIn = await applyAdaLayerNorm(
      x,
      onesBuf,
      zerosBuf,
      layerNormEps,
      mod,
      offsets.ff,
      runtime,
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
    const ffBias = {
      up: createBiasTensor(
        resolver.get(`transformer_blocks.${layerIdx}.ff.net.0.proj.bias`),
        ffWeights.up.shape[0],
        'sd3_ff_up_bias'
      ),
      down: createBiasTensor(
        resolver.get(`transformer_blocks.${layerIdx}.ff.net.2.bias`),
        hiddenSize,
        'sd3_ff_down_bias'
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
      ffWeightNames
    );
    const ffGated = await applyGate(ffOut, mod, offsets.ff, { numTokens: tokenCount, hiddenSize, zeroOffset: mod.zeroOffset });
    const xRes3 = await runResidualAdd(x, ffGated, tokenCount * hiddenSize, { useVec4: true });

    releaseBuffer(xFfIn.buffer);
    releaseBuffer(ffGated.buffer);
    releaseBuffer(x.buffer);

    x = createTensor(xRes3.buffer, xRes3.dtype, [tokenCount, hiddenSize], 'sd3_x');

    releaseBuffer(mod.tensor.buffer);
    if (ctxMod?.tensor?.buffer) {
      releaseBuffer(ctxMod.tensor.buffer);
    }
  }

  const normOutWeightName = 'norm_out.linear.weight';
  const normOutWeight = expectWeight(resolver.get(normOutWeightName), normOutWeightName);
  const normOutBias = resolver.get('norm_out.linear.bias');
  const normOut = await buildModulation(timeText, normOutWeight, normOutBias, hiddenSize, 2, runtime, matmul, normOutWeightName);

  const xNorm = await runLayerNorm(x, onesBuf, zerosBuf, layerNormEps, { batchSize: tokenCount, hiddenSize });
  const xMod = await runModulate(xNorm, normOut.tensor, {
    numTokens: tokenCount,
    hiddenSize,
    scaleOffset: 0,
    shiftOffset: hiddenSize,
    gateOffset: 0,
    hasGate: false,
    addOne: true,
  });

  releaseBuffer(xNorm.buffer);
  releaseBuffer(x.buffer);
  releaseBuffer(normOut.tensor.buffer);
  releaseBuffer(onesBuf);
  releaseBuffer(zerosBuf);

  const projOutWeightName = 'proj_out.weight';
  const projOutWeight = expectWeight(resolver.get(projOutWeightName), projOutWeightName);
  const projOutBias = resolver.get('proj_out.bias');
  let patch = await matmul(xMod, projOutWeight, projOutWeightName, tokenCount, projOutWeight.shape[0], hiddenSize, {
    outputDtype: xMod.dtype,
    transposeB: 'auto',
  });
  if (projOutBias) patch = await runBiasAdd(patch, createBiasTensor(projOutBias, projOutWeight.shape[0], 'sd3_proj_out_bias'), tokenCount, projOutWeight.shape[0]);

  releaseBuffer(xMod.buffer);

  const patchChannels = projOutWeight.shape[0];
  const output = await runPixelShuffle(patch, {
    outChannels: latentChannels,
    outHeight: latentHeight,
    outWidth: latentWidth,
    gridWidth,
    gridHeight,
    patchSize,
    patchChannels,
  });

  releaseBuffer(patch.buffer);

  return output;
}
