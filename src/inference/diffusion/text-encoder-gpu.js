import { getDevice, getKernelCapabilities } from '../../gpu/device.js';
import { createTensor } from '../../gpu/tensor.js';
import { getBuffer } from '../../gpu/weight-buffer.js';
import { acquireBuffer, releaseBuffer, readBuffer } from '../../memory/buffer-pool.js';
import {
  runGather,
  runLayerNorm,
  runRMSNorm,
  runMatmul,
  runAttention,
  runGeLU,
  runSiLU,
  runSiLURowSplit,
  runResidualAdd,
  runBiasAdd,
} from '../../gpu/kernels/index.js';
import { log } from '../../debug/index.js';
import { createSD3WeightResolver } from './sd3-weights.js';

function resolveActivationDtype(runtime) {
  const caps = getKernelCapabilities();
  const wantsF16 = runtime?.latent?.dtype === 'f16';
  return wantsF16 && caps.hasF16 ? 'f16' : 'f32';
}

function padTokens(tokens, maxLength, padTokenId) {
  const length = Math.max(1, Math.min(tokens.length, maxLength));
  const out = new Uint32Array(maxLength);
  for (let i = 0; i < maxLength; i++) {
    out[i] = i < length ? tokens[i] : padTokenId;
  }
  return out;
}

function findEosIndex(tokens, eosTokenId) {
  if (eosTokenId == null) return tokens.length - 1;
  for (let i = 0; i < tokens.length; i++) {
    if (tokens[i] === eosTokenId) return i;
  }
  return tokens.length - 1;
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

function createVectorTensor(device, data, dtype, label) {
  const buffer = acquireBuffer(data.byteLength, undefined, label);
  device.queue.writeBuffer(buffer, 0, data);
  return createTensor(buffer, dtype, [1, data.length], label);
}

function createBiasTensor(buffer, size, label) {
  return createTensor(buffer, 'f32', [size], label);
}

function getWeight(weights, prefix, name) {
  const key = `${prefix}.${name}`;
  return weights.get(key) || null;
}

function expectWeight(weight, label) {
  if (!weight) {
    throw new Error(`Missing diffusion weight: ${label}`);
  }
  return weight;
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

function resolveMatmulDtype(weight, weightsEntry, key) {
  if (weight && weight.dtype) return weight.dtype;
  const locationDtype = weightsEntry?.dtypes?.get(key);
  return normalizeMatmulLocationDtype(locationDtype);
}

async function runMatmulResolved(input, weight, weightsEntry, key, M, N, K, options = {}) {
  const bDtype = resolveMatmulDtype(weight, weightsEntry, key);
  const nextOptions = bDtype ? { ...options, bDtype } : options;
  return runMatmul(input, weight, M, N, K, nextOptions);
}

async function runClipTextEncoder(tokens, weightsEntry, config, runtime, options = {}) {
  const device = getDevice();
  if (!device) throw new Error('CLIP encoder requires a WebGPU device.');
  if (!weightsEntry?.weights || !weightsEntry?.shapes) {
    throw new Error('CLIP encoder requires loaded weights.');
  }

  const prefix = options.prefix;
  const weights = weightsEntry.weights;
  const hiddenSize = config.hidden_size;
  const numHeads = config.num_attention_heads;
  const headDim = Math.floor(hiddenSize / numHeads);
  const maxLength = config.max_position_embeddings;
  const padTokenId = config.pad_token_id ?? 0;
  const eosTokenId = config.eos_token_id ?? null;
  const activationDtype = resolveActivationDtype(runtime);
  const matmul = (input, weight, key, M, N, K, options = {}) =>
    runMatmulResolved(input, weight, weightsEntry, key, M, N, K, options);

  const padded = padTokens(tokens, maxLength, padTokenId);
  const tokenBuffer = createIndexBuffer(device, padded, `${prefix}_tokens`);

  const tokenEmbedWeight = expectWeight(
    getWeight(weights, prefix, 'text_model.embeddings.token_embedding.weight'),
    `${prefix}.text_model.embeddings.token_embedding.weight`
  );
  const posEmbedWeight = expectWeight(
    getWeight(weights, prefix, 'text_model.embeddings.position_embedding.weight'),
    `${prefix}.text_model.embeddings.position_embedding.weight`
  );

  const tokenEmbedKey = `${prefix}.text_model.embeddings.token_embedding.weight`;
  const posEmbedKey = `${prefix}.text_model.embeddings.position_embedding.weight`;
  const tokenEmbedDtype = resolveEmbeddingDtype(tokenEmbedWeight, weightsEntry, tokenEmbedKey, runtime);
  const posEmbedDtype = resolveEmbeddingDtype(posEmbedWeight, weightsEntry, posEmbedKey, runtime);

  let hidden = await runGather(
    tokenBuffer,
    getBuffer(tokenEmbedWeight),
    maxLength,
    hiddenSize,
    config.vocab_size,
    {
      embeddingDtype: tokenEmbedDtype,
      outputDtype: activationDtype,
      transpose: false,
    }
  );

  const posIndices = new Uint32Array(maxLength);
  for (let i = 0; i < maxLength; i++) posIndices[i] = i;
  const posBuffer = createIndexBuffer(device, posIndices, `${prefix}_pos_idx`);
  const pos = await runGather(
    posBuffer,
    getBuffer(posEmbedWeight),
    maxLength,
    hiddenSize,
    config.max_position_embeddings,
    {
      embeddingDtype: posEmbedDtype,
      outputDtype: activationDtype,
      transpose: false,
    }
  );

  posBuffer.destroy();
  tokenBuffer.destroy();

  const combined = await runResidualAdd(hidden, pos, maxLength * hiddenSize, { useVec4: true });
  releaseBuffer(hidden.buffer);
  releaseBuffer(pos.buffer);
  hidden = createTensor(combined.buffer, combined.dtype, [maxLength, hiddenSize], 'clip_embed');

  const layerCount = config.num_hidden_layers;
  for (let layerIdx = 0; layerIdx < layerCount; layerIdx++) {
    const ln1Weight = expectWeight(
      getWeight(weights, prefix, `text_model.encoder.layers.${layerIdx}.layer_norm1.weight`),
      `${prefix}.text_model.encoder.layers.${layerIdx}.layer_norm1.weight`
    );
    const ln1Bias = expectWeight(
      getWeight(weights, prefix, `text_model.encoder.layers.${layerIdx}.layer_norm1.bias`),
      `${prefix}.text_model.encoder.layers.${layerIdx}.layer_norm1.bias`
    );
    const ln2Weight = expectWeight(
      getWeight(weights, prefix, `text_model.encoder.layers.${layerIdx}.layer_norm2.weight`),
      `${prefix}.text_model.encoder.layers.${layerIdx}.layer_norm2.weight`
    );
    const ln2Bias = expectWeight(
      getWeight(weights, prefix, `text_model.encoder.layers.${layerIdx}.layer_norm2.bias`),
      `${prefix}.text_model.encoder.layers.${layerIdx}.layer_norm2.bias`
    );

    const norm1 = await runLayerNorm(hidden, getBuffer(ln1Weight), getBuffer(ln1Bias), config.layer_norm_eps, {
      batchSize: maxLength,
      hiddenSize,
    });

    const qKey = `${prefix}.text_model.encoder.layers.${layerIdx}.self_attn.q_proj.weight`;
    const kKey = `${prefix}.text_model.encoder.layers.${layerIdx}.self_attn.k_proj.weight`;
    const vKey = `${prefix}.text_model.encoder.layers.${layerIdx}.self_attn.v_proj.weight`;
    const qWeight = expectWeight(
      getWeight(weights, prefix, `text_model.encoder.layers.${layerIdx}.self_attn.q_proj.weight`),
      qKey
    );
    const kWeight = expectWeight(
      getWeight(weights, prefix, `text_model.encoder.layers.${layerIdx}.self_attn.k_proj.weight`),
      kKey
    );
    const vWeight = expectWeight(
      getWeight(weights, prefix, `text_model.encoder.layers.${layerIdx}.self_attn.v_proj.weight`),
      vKey
    );
    const qBias = expectWeight(
      getWeight(weights, prefix, `text_model.encoder.layers.${layerIdx}.self_attn.q_proj.bias`),
      `${prefix}.text_model.encoder.layers.${layerIdx}.self_attn.q_proj.bias`
    );
    const kBias = expectWeight(
      getWeight(weights, prefix, `text_model.encoder.layers.${layerIdx}.self_attn.k_proj.bias`),
      `${prefix}.text_model.encoder.layers.${layerIdx}.self_attn.k_proj.bias`
    );
    const vBias = expectWeight(
      getWeight(weights, prefix, `text_model.encoder.layers.${layerIdx}.self_attn.v_proj.bias`),
      `${prefix}.text_model.encoder.layers.${layerIdx}.self_attn.v_proj.bias`
    );
    const outKey = `${prefix}.text_model.encoder.layers.${layerIdx}.self_attn.out_proj.weight`;
    const outWeight = expectWeight(
      getWeight(weights, prefix, `text_model.encoder.layers.${layerIdx}.self_attn.out_proj.weight`),
      outKey
    );
    const outBias = expectWeight(
      getWeight(weights, prefix, `text_model.encoder.layers.${layerIdx}.self_attn.out_proj.bias`),
      `${prefix}.text_model.encoder.layers.${layerIdx}.self_attn.out_proj.bias`
    );

    let q = await matmul(norm1, qWeight, qKey, maxLength, hiddenSize, hiddenSize, { outputDtype: activationDtype, transposeB: 'auto' });
    let k = await matmul(norm1, kWeight, kKey, maxLength, hiddenSize, hiddenSize, { outputDtype: activationDtype, transposeB: 'auto' });
    let v = await matmul(norm1, vWeight, vKey, maxLength, hiddenSize, hiddenSize, { outputDtype: activationDtype, transposeB: 'auto' });
    if (qBias) q = await runBiasAdd(q, createBiasTensor(getBuffer(qBias), hiddenSize, `${prefix}_q_bias`), maxLength, hiddenSize);
    if (kBias) k = await runBiasAdd(k, createBiasTensor(getBuffer(kBias), hiddenSize, `${prefix}_k_bias`), maxLength, hiddenSize);
    if (vBias) v = await runBiasAdd(v, createBiasTensor(getBuffer(vBias), hiddenSize, `${prefix}_v_bias`), maxLength, hiddenSize);

    const attn = await runAttention(q, k, v, null, numHeads, headDim, {
      seqLen: maxLength,
      kvLen: maxLength,
      numKVHeads: numHeads,
      causal: false,
    });

    let attnOut = await matmul(attn, outWeight, outKey, maxLength, hiddenSize, hiddenSize, { outputDtype: activationDtype, transposeB: 'auto' });
    if (outBias) attnOut = await runBiasAdd(attnOut, createBiasTensor(getBuffer(outBias), hiddenSize, `${prefix}_out_bias`), maxLength, hiddenSize);

    const attnResidual = await runResidualAdd(hidden, attnOut, maxLength * hiddenSize, { useVec4: true });

    releaseBuffer(norm1.buffer);
    releaseBuffer(q.buffer);
    releaseBuffer(k.buffer);
    releaseBuffer(v.buffer);
    releaseBuffer(attn.buffer);
    releaseBuffer(attnOut.buffer);
    releaseBuffer(hidden.buffer);

    hidden = createTensor(attnResidual.buffer, attnResidual.dtype, [maxLength, hiddenSize], 'clip_attn_out');

    const norm2 = await runLayerNorm(hidden, getBuffer(ln2Weight), getBuffer(ln2Bias), config.layer_norm_eps, {
      batchSize: maxLength,
      hiddenSize,
    });

    const fc1Key = `${prefix}.text_model.encoder.layers.${layerIdx}.mlp.fc1.weight`;
    const fc1Weight = expectWeight(
      getWeight(weights, prefix, `text_model.encoder.layers.${layerIdx}.mlp.fc1.weight`),
      fc1Key
    );
    const fc1Bias = expectWeight(
      getWeight(weights, prefix, `text_model.encoder.layers.${layerIdx}.mlp.fc1.bias`),
      `${prefix}.text_model.encoder.layers.${layerIdx}.mlp.fc1.bias`
    );
    const fc2Key = `${prefix}.text_model.encoder.layers.${layerIdx}.mlp.fc2.weight`;
    const fc2Weight = expectWeight(
      getWeight(weights, prefix, `text_model.encoder.layers.${layerIdx}.mlp.fc2.weight`),
      fc2Key
    );
    const fc2Bias = expectWeight(
      getWeight(weights, prefix, `text_model.encoder.layers.${layerIdx}.mlp.fc2.bias`),
      `${prefix}.text_model.encoder.layers.${layerIdx}.mlp.fc2.bias`
    );

    const intermediate = fc1Weight.shape[0];
    let mlp = await matmul(norm2, fc1Weight, fc1Key, maxLength, intermediate, hiddenSize, { outputDtype: activationDtype, transposeB: 'auto' });
    if (fc1Bias) mlp = await runBiasAdd(mlp, createBiasTensor(getBuffer(fc1Bias), intermediate, `${prefix}_fc1_bias`), maxLength, intermediate);

    const gelu = await runGeLU(mlp, { size: maxLength * intermediate });
    releaseBuffer(mlp.buffer);

    let mlpOut = await matmul(gelu, fc2Weight, fc2Key, maxLength, hiddenSize, intermediate, { outputDtype: activationDtype, transposeB: 'auto' });
    if (fc2Bias) mlpOut = await runBiasAdd(mlpOut, createBiasTensor(getBuffer(fc2Bias), hiddenSize, `${prefix}_fc2_bias`), maxLength, hiddenSize);

    const mlpResidual = await runResidualAdd(hidden, mlpOut, maxLength * hiddenSize, { useVec4: true });

    releaseBuffer(norm2.buffer);
    releaseBuffer(gelu.buffer);
    releaseBuffer(mlpOut.buffer);
    releaseBuffer(hidden.buffer);

    hidden = createTensor(mlpResidual.buffer, mlpResidual.dtype, [maxLength, hiddenSize], 'clip_mlp_out');
  }

  const finalLnWeight = expectWeight(
    getWeight(weights, prefix, 'text_model.final_layer_norm.weight'),
    `${prefix}.text_model.final_layer_norm.weight`
  );
  const finalLnBias = expectWeight(
    getWeight(weights, prefix, 'text_model.final_layer_norm.bias'),
    `${prefix}.text_model.final_layer_norm.bias`
  );
  const final = await runLayerNorm(hidden, getBuffer(finalLnWeight), getBuffer(finalLnBias), config.layer_norm_eps, {
    batchSize: maxLength,
    hiddenSize,
  });
  releaseBuffer(hidden.buffer);

  const eosIndex = findEosIndex(padded, eosTokenId);
  const eosIdxBuffer = createIndexBuffer(device, new Uint32Array([eosIndex]), `${prefix}_eos_idx`);
  const pooledToken = await runGather(
    eosIdxBuffer,
    final.buffer,
    1,
    hiddenSize,
    maxLength,
    {
      embeddingDtype: final.dtype,
      outputDtype: activationDtype,
      transpose: false,
    }
  );
  eosIdxBuffer.destroy();

  const textProjKey = `${prefix}.text_projection.weight`;
  const textProj = expectWeight(
    getWeight(weights, prefix, 'text_projection.weight'),
    textProjKey
  );
  let pooled = await matmul(pooledToken, textProj, textProjKey, 1, hiddenSize, hiddenSize, { outputDtype: activationDtype, transposeB: 'auto' });
  releaseBuffer(pooledToken.buffer);

  const pooledData = await readBuffer(pooled.buffer, hiddenSize * (pooled.dtype === 'f16' ? 2 : 4));
  releaseBuffer(pooled.buffer);

  const pooledView = pooled.dtype === 'f16'
    ? new Float32Array(new Uint16Array(pooledData).length)
    : new Float32Array(pooledData);

  if (pooled.dtype === 'f16') {
    const u16 = new Uint16Array(pooledData);
    for (let i = 0; i < u16.length; i++) {
      const h = u16[i];
      const sign = (h & 0x8000) ? -1 : 1;
      const exp = (h >> 10) & 0x1f;
      const mant = h & 0x3ff;
      if (exp === 0) {
        pooledView[i] = sign * mant * Math.pow(2, -24);
      } else if (exp === 31) {
        pooledView[i] = mant ? NaN : sign * Infinity;
      } else {
        pooledView[i] = sign * (1 + mant / 1024) * Math.pow(2, exp - 15);
      }
    }
  }

  return {
    hidden: final,
    pooled: pooledView,
    maxLength,
    hiddenSize,
  };
}

async function runT5Encoder(tokens, weightsEntry, config, runtime, options = {}) {
  const device = getDevice();
  if (!device) throw new Error('T5 encoder requires a WebGPU device.');
  if (!weightsEntry?.weights || !weightsEntry?.shapes) {
    throw new Error('T5 encoder requires loaded weights.');
  }

  const prefix = options.prefix;
  const weights = weightsEntry.weights;
  const hiddenSize = config.d_model;
  const numHeads = config.num_heads;
  const headDim = config.d_kv;
  const maxLength = options.maxLength;
  const padTokenId = config.pad_token_id ?? 0;
  const activationDtype = resolveActivationDtype(runtime);
  const matmul = (input, weight, key, M, N, K, options = {}) =>
    runMatmulResolved(input, weight, weightsEntry, key, M, N, K, options);

  const padded = padTokens(tokens, maxLength, padTokenId);
  const tokenBuffer = createIndexBuffer(device, padded, `${prefix}_tokens`);

  const embedWeight = getWeight(weights, prefix, 'shared.weight');
  if (!embedWeight) {
    throw new Error('T5 shared.weight missing.');
  }

  const embedKey = `${prefix}.shared.weight`;
  const embedDtype = resolveEmbeddingDtype(embedWeight, weightsEntry, embedKey, runtime);

  let hidden = await runGather(
    tokenBuffer,
    getBuffer(embedWeight),
    maxLength,
    hiddenSize,
    config.vocab_size,
    {
      embeddingDtype: embedDtype,
      outputDtype: activationDtype,
      transpose: false,
    }
  );
  tokenBuffer.destroy();

  const layerCount = config.num_layers;
  for (let layerIdx = 0; layerIdx < layerCount; layerIdx++) {
    const lnWeight = expectWeight(
      getWeight(weights, prefix, `encoder.block.${layerIdx}.layer.0.layer_norm.weight`),
      `${prefix}.encoder.block.${layerIdx}.layer.0.layer_norm.weight`
    );
    const normed = await runRMSNorm(hidden, getBuffer(lnWeight), config.layer_norm_epsilon, {
      batchSize: maxLength,
      hiddenSize,
    });

    const qWeight = expectWeight(
      getWeight(weights, prefix, `encoder.block.${layerIdx}.layer.0.SelfAttention.q.weight`),
      `${prefix}.encoder.block.${layerIdx}.layer.0.SelfAttention.q.weight`
    );
    const kWeight = expectWeight(
      getWeight(weights, prefix, `encoder.block.${layerIdx}.layer.0.SelfAttention.k.weight`),
      `${prefix}.encoder.block.${layerIdx}.layer.0.SelfAttention.k.weight`
    );
    const vWeight = expectWeight(
      getWeight(weights, prefix, `encoder.block.${layerIdx}.layer.0.SelfAttention.v.weight`),
      `${prefix}.encoder.block.${layerIdx}.layer.0.SelfAttention.v.weight`
    );
    const oWeight = expectWeight(
      getWeight(weights, prefix, `encoder.block.${layerIdx}.layer.0.SelfAttention.o.weight`),
      `${prefix}.encoder.block.${layerIdx}.layer.0.SelfAttention.o.weight`
    );

    let q = await runMatmul(normed, qWeight, maxLength, hiddenSize, hiddenSize, { outputDtype: activationDtype, transposeB: 'auto' });
    let k = await runMatmul(normed, kWeight, maxLength, hiddenSize, hiddenSize, { outputDtype: activationDtype, transposeB: 'auto' });
    let v = await runMatmul(normed, vWeight, maxLength, hiddenSize, hiddenSize, { outputDtype: activationDtype, transposeB: 'auto' });

    const attn = await runAttention(q, k, v, null, numHeads, headDim, {
      seqLen: maxLength,
      kvLen: maxLength,
      numKVHeads: numHeads,
      causal: false,
    });

    const attnOut = await runMatmul(attn, oWeight, maxLength, hiddenSize, hiddenSize, { outputDtype: activationDtype, transposeB: 'auto' });
    const attnResidual = await runResidualAdd(hidden, attnOut, maxLength * hiddenSize, { useVec4: true });

    releaseBuffer(normed.buffer);
    releaseBuffer(q.buffer);
    releaseBuffer(k.buffer);
    releaseBuffer(v.buffer);
    releaseBuffer(attn.buffer);
    releaseBuffer(attnOut.buffer);
    releaseBuffer(hidden.buffer);

    hidden = createTensor(attnResidual.buffer, attnResidual.dtype, [maxLength, hiddenSize], 't5_attn_out');

    const ln2Weight = expectWeight(
      getWeight(weights, prefix, `encoder.block.${layerIdx}.layer.1.layer_norm.weight`),
      `${prefix}.encoder.block.${layerIdx}.layer.1.layer_norm.weight`
    );
    const norm2 = await runRMSNorm(hidden, getBuffer(ln2Weight), config.layer_norm_epsilon, {
      batchSize: maxLength,
      hiddenSize,
    });

    const wi0 = expectWeight(
      getWeight(weights, prefix, `encoder.block.${layerIdx}.layer.1.DenseReluDense.wi_0.weight`),
      `${prefix}.encoder.block.${layerIdx}.layer.1.DenseReluDense.wi_0.weight`
    );
    const wi1 = expectWeight(
      getWeight(weights, prefix, `encoder.block.${layerIdx}.layer.1.DenseReluDense.wi_1.weight`),
      `${prefix}.encoder.block.${layerIdx}.layer.1.DenseReluDense.wi_1.weight`
    );
    const wo = expectWeight(
      getWeight(weights, prefix, `encoder.block.${layerIdx}.layer.1.DenseReluDense.wo.weight`),
      `${prefix}.encoder.block.${layerIdx}.layer.1.DenseReluDense.wo.weight`
    );

    const dff = wi0.shape[0];
    const bytesPerElement = activationDtype === 'f16' ? 2 : 4;
    const combinedSize = maxLength * dff * 2 * bytesPerElement;
    const combinedBuffer = acquireBuffer(combinedSize, undefined, 't5_ff_combined');

    const wi0Out = await runMatmul(norm2, wi0, maxLength, dff, hiddenSize, {
      outputDtype: activationDtype,
      transposeB: 'auto',
      outputBuffer: combinedBuffer,
      cOffset: 0,
    });
    const wi1Out = await runMatmul(norm2, wi1, maxLength, dff, hiddenSize, {
      outputDtype: activationDtype,
      transposeB: 'auto',
      outputBuffer: combinedBuffer,
      cOffset: maxLength * dff * bytesPerElement,
    });

    const combinedTensor = createTensor(combinedBuffer, activationDtype, [maxLength, dff * 2], 't5_ff_combined');
    const gated = await runSiLURowSplit(combinedTensor, {
      numTokens: maxLength,
      dim: dff,
      activation: 'gelu',
      swigluLimit: null,
    });

    releaseBuffer(combinedTensor.buffer);

    const ffOut = await runMatmul(gated, wo, maxLength, hiddenSize, dff, { outputDtype: activationDtype, transposeB: 'auto' });
    const ffResidual = await runResidualAdd(hidden, ffOut, maxLength * hiddenSize, { useVec4: true });

    releaseBuffer(norm2.buffer);
    releaseBuffer(gated.buffer);
    releaseBuffer(ffOut.buffer);
    releaseBuffer(hidden.buffer);

    hidden = createTensor(ffResidual.buffer, ffResidual.dtype, [maxLength, hiddenSize], 't5_ff_out');
  }

  const finalLn = expectWeight(
    getWeight(weights, prefix, 'encoder.final_layer_norm.weight'),
    `${prefix}.encoder.final_layer_norm.weight`
  );
  const final = await runRMSNorm(hidden, getBuffer(finalLn), config.layer_norm_epsilon, {
    batchSize: maxLength,
    hiddenSize,
  });
  releaseBuffer(hidden.buffer);

  return {
    hidden: final,
    maxLength,
    hiddenSize,
  };
}

export async function runTextEncodersForPrompt(tokensByEncoder, weightsByComponent, modelConfig, runtime) {
  const clipConfig = modelConfig?.components?.text_encoder?.config || {};
  const clip2Config = modelConfig?.components?.text_encoder_2?.config || {};
  const t5Config = modelConfig?.components?.text_encoder_3?.config || {};
  const t5MaxLength = runtime?.textEncoder?.maxLength ?? 256;

  const clip = await runClipTextEncoder(tokensByEncoder.text_encoder, weightsByComponent.text_encoder, clipConfig, runtime, {
    prefix: 'text_encoder',
  });
  const clip2 = await runClipTextEncoder(tokensByEncoder.text_encoder_2, weightsByComponent.text_encoder_2, clip2Config, runtime, {
    prefix: 'text_encoder_2',
  });
  const t5 = await runT5Encoder(tokensByEncoder.text_encoder_3, weightsByComponent.text_encoder_3, t5Config, runtime, {
    prefix: 'text_encoder_3',
    maxLength: t5MaxLength,
  });

  const pooled = new Float32Array(clip.pooled.length + clip2.pooled.length);
  pooled.set(clip.pooled, 0);
  pooled.set(clip2.pooled, clip.pooled.length);

  releaseBuffer(clip.hidden.buffer);
  releaseBuffer(clip2.hidden.buffer);

  return {
    pooled,
    context: t5.hidden,
  };
}

export async function buildTimeTextEmbedding(pooled, weightsEntry, modelConfig, runtime) {
  const device = getDevice();
  if (!device) throw new Error('TimeText embedding requires a WebGPU device.');
  const activationDtype = resolveActivationDtype(runtime);

  const resolver = createSD3WeightResolver(weightsEntry, modelConfig);
  const textLinear1 = resolver.get('time_text_embed.text_embedder.linear_1.weight');
  const textLinear1Bias = resolver.get('time_text_embed.text_embedder.linear_1.bias');
  const textLinear2 = resolver.get('time_text_embed.text_embedder.linear_2.weight');
  const textLinear2Bias = resolver.get('time_text_embed.text_embedder.linear_2.bias');
  if (!textLinear1 || !textLinear2) {
    throw new Error('Missing diffusion time_text_embed text weights.');
  }

  const pooledTensor = createVectorTensor(device, pooled, activationDtype, 'sd3_pooled');
  let text = await runMatmul(pooledTensor, textLinear1, 1, textLinear1.shape[0], textLinear1.shape[1], {
    outputDtype: activationDtype,
    transposeB: 'auto',
  });
  if (textLinear1Bias) {
    text = await runBiasAdd(text, createBiasTensor(getBuffer(textLinear1Bias), textLinear1.shape[0], 'sd3_text_bias1'), 1, textLinear1.shape[0]);
  }
  const textAct = await runSiLU(text, { size: textLinear1.shape[0], swigluLimit: null });
  releaseBuffer(text.buffer);

  let textOut = await runMatmul(textAct, textLinear2, 1, textLinear2.shape[0], textLinear2.shape[1], {
    outputDtype: activationDtype,
    transposeB: 'auto',
  });
  if (textLinear2Bias) {
    textOut = await runBiasAdd(textOut, createBiasTensor(getBuffer(textLinear2Bias), textLinear2.shape[0], 'sd3_text_bias2'), 1, textLinear2.shape[0]);
  }

  releaseBuffer(textAct.buffer);
  releaseBuffer(pooledTensor.buffer);

  return textOut;
}

export async function buildTimestepEmbedding(timestep, weightsEntry, modelConfig, runtime, options = {}) {
  const device = getDevice();
  if (!device) throw new Error('Timestep embedding requires a WebGPU device.');

  const dim = options.dim ?? 256;
  const half = Math.floor(dim / 2);
  const emb = new Float32Array(dim);
  const maxPeriod = 10000;
  for (let i = 0; i < half; i++) {
    const freq = Math.exp(-Math.log(maxPeriod) * i / half);
    const angle = timestep * freq;
    emb[2 * i] = Math.cos(angle);
    emb[2 * i + 1] = Math.sin(angle);
  }

  const activationDtype = resolveActivationDtype(runtime);
  const embTensor = createVectorTensor(device, emb, activationDtype, 'sd3_timestep');

  const resolver = createSD3WeightResolver(weightsEntry, modelConfig);
  const linear1 = resolver.get('time_text_embed.timestep_embedder.linear_1.weight');
  const linear1Bias = resolver.get('time_text_embed.timestep_embedder.linear_1.bias');
  const linear2 = resolver.get('time_text_embed.timestep_embedder.linear_2.weight');
  const linear2Bias = resolver.get('time_text_embed.timestep_embedder.linear_2.bias');
  if (!linear1 || !linear2) {
    throw new Error('Missing diffusion time_text_embed timestep weights.');
  }

  let out = await runMatmul(embTensor, linear1, 1, linear1.shape[0], linear1.shape[1], {
    outputDtype: activationDtype,
    transposeB: 'auto',
  });
  if (linear1Bias) {
    out = await runBiasAdd(out, createBiasTensor(getBuffer(linear1Bias), linear1.shape[0], 'sd3_time_bias1'), 1, linear1.shape[0]);
  }
  const act = await runSiLU(out, { size: linear1.shape[0], swigluLimit: null });
  releaseBuffer(out.buffer);

  let out2 = await runMatmul(act, linear2, 1, linear2.shape[0], linear2.shape[1], {
    outputDtype: activationDtype,
    transposeB: 'auto',
  });
  if (linear2Bias) {
    out2 = await runBiasAdd(out2, createBiasTensor(getBuffer(linear2Bias), linear2.shape[0], 'sd3_time_bias2'), 1, linear2.shape[0]);
  }

  releaseBuffer(act.buffer);
  releaseBuffer(embTensor.buffer);

  return out2;
}

export async function combineTimeTextEmbeddings(time, text, hiddenSize) {
  const combined = await runResidualAdd(time, text, hiddenSize, { useVec4: true });
  releaseBuffer(time.buffer);
  releaseBuffer(text.buffer);
  return createTensor(combined.buffer, combined.dtype, [1, hiddenSize], 'sd3_time_text');
}

export async function projectContext(context, weightsEntry, modelConfig, runtime) {
  const resolver = createSD3WeightResolver(weightsEntry, modelConfig);
  const projWeight = resolver.get('context_embedder.weight');
  const projBias = resolver.get('context_embedder.bias');
  if (!projWeight) {
    throw new Error('Missing diffusion context_embedder weight.');
  }
  const numTokens = context.shape[0];
  const inDim = context.shape[1];
  const outDim = projWeight.shape[0];
  const activationDtype = resolveActivationDtype(runtime);
  let projected = await runMatmul(context, projWeight, numTokens, outDim, inDim, {
    outputDtype: activationDtype,
    transposeB: 'auto',
  });
  if (projBias) {
    projected = await runBiasAdd(projected, createBiasTensor(getBuffer(projBias), outDim, 'sd3_ctx_bias'), numTokens, outDim);
  }
  releaseBuffer(context.buffer);
  return projected;
}

export function logQuickGeluWarning(config) {
  if (config.hidden_act === 'quick_gelu') {
    log.warn('Diffusion', 'CLIP quick_gelu is approximated with gelu.');
  }
}
