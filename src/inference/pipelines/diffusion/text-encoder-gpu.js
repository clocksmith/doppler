import { getDevice } from '../../../gpu/device.js';
import { createTensor } from '../../../gpu/tensor.js';
import { getBuffer } from '../../../gpu/weight-buffer.js';
import { acquireBuffer, readBuffer } from '../../../memory/buffer-pool.js';
import { CommandRecorder } from '../../../gpu/command-recorder.js';
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
  recordGather,
  recordLayerNorm,
  recordRMSNorm,
  recordMatmul,
  recordAttention,
  recordGeLU,
  recordSiLU,
  recordSiLURowSplit,
  recordResidualAdd,
  recordBiasAdd,
} from '../../../gpu/kernels/index.js';
import { log } from '../../../debug/index.js';
import { createSD3WeightResolver } from './sd3-weights.js';
import { f32ToF16Array } from '../../kv-cache/types.js';
import {
  resolveDiffusionActivationDtype,
  createDiffusionBufferReleaser,
  createDiffusionBufferDestroyer,
  createDiffusionIndexBuffer,
  expectDiffusionWeight,
  normalizeDiffusionLocationDtype,
  normalizeDiffusionMatmulLocationDtype,
  inferDiffusionMatmulDtypeFromBuffer,
  sumDiffusionProfileTimings,
} from './helpers.js';

function padTokens(tokens, maxLength, padTokenId) {
  if (!Number.isFinite(maxLength) || maxLength <= 0) {
    throw new Error(`padTokens requires a positive maxLength (got ${maxLength}).`);
  }
  const length = Math.min(tokens.length, maxLength);
  const out = new Uint32Array(maxLength);
  for (let i = 0; i < maxLength; i++) {
    out[i] = i < length ? (tokens[i] ?? padTokenId) : padTokenId;
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

function createVectorTensor(device, data, dtype, label) {
  const length = data.length;
  let payload = data;
  if (dtype === 'f16') {
    if (!(data instanceof Uint16Array)) {
      const f32 = data instanceof Float32Array ? data : new Float32Array(data);
      payload = f32ToF16Array(f32);
    }
  } else if (!(data instanceof Float32Array)) {
    payload = new Float32Array(data);
  }
  const byteLength = payload.byteLength;
  const alignedLength = Math.ceil(byteLength / 4) * 4;
  const buffer = acquireBuffer(alignedLength, undefined, label);
  if (alignedLength === byteLength) {
    device.queue.writeBuffer(buffer, 0, payload);
  } else {
    const padded = new Uint8Array(alignedLength);
    padded.set(new Uint8Array(payload.buffer, payload.byteOffset, payload.byteLength));
    device.queue.writeBuffer(buffer, 0, padded);
  }
  return createTensor(buffer, dtype, [1, length], label);
}

function resolveBiasDtype(weight, weightsEntry, key) {
  if (weight && weight.dtype) return weight.dtype;
  const locationDtype = weightsEntry?.dtypes?.get(key);
  const mapped = normalizeDiffusionLocationDtype(locationDtype);
  return mapped || 'f32';
}

function createBiasTensorWithDtype(weight, weightsEntry, key, size, label) {
  if (!weight) return null;
  const dtype = resolveBiasDtype(weight, weightsEntry, key);
  return createTensor(getBuffer(weight), dtype, [size], label);
}

function createKernelOps(recorder) {
  if (!recorder) {
    return {
      gather: runGather,
      layerNorm: runLayerNorm,
      rmsNorm: runRMSNorm,
      matmul: runMatmul,
      attention: runAttention,
      gelu: runGeLU,
      silu: runSiLU,
      siluRowSplit: runSiLURowSplit,
      residualAdd: runResidualAdd,
      biasAdd: runBiasAdd,
    };
  }
  return {
    gather: (...args) => recordGather(recorder, ...args),
    layerNorm: (...args) => recordLayerNorm(recorder, ...args),
    rmsNorm: (...args) => recordRMSNorm(recorder, ...args),
    matmul: (...args) => recordMatmul(recorder, ...args),
    attention: (...args) => recordAttention(recorder, ...args),
    gelu: (...args) => recordGeLU(recorder, ...args),
    silu: (...args) => recordSiLU(recorder, ...args),
    siluRowSplit: (...args) => recordSiLURowSplit(recorder, ...args),
    residualAdd: (...args) => recordResidualAdd(recorder, ...args),
    biasAdd: (...args) => recordBiasAdd(recorder, ...args),
  };
}

function getWeight(weights, prefix, name) {
  const key = `${prefix}.${name}`;
  return weights.get(key) || null;
}

function resolveEmbeddingDtype(weight, weightsEntry, key, runtime) {
  if (weight && weight.dtype) return weight.dtype;
  const locationDtype = weightsEntry?.dtypes?.get(key);
  const mapped = normalizeDiffusionLocationDtype(locationDtype);
  if (!mapped) return null;
  if (mapped !== 'f16') return mapped;
  const allowUpcast = runtime?.loading?.allowF32UpcastNonMatmul !== false;
  return allowUpcast ? 'f32' : 'f16';
}

function resolveMatmulDtype(weight, weightsEntry, key) {
  if (weight && weight.dtype) return weight.dtype;
  const locationDtype = weightsEntry?.dtypes?.get(key);
  return normalizeDiffusionMatmulLocationDtype(locationDtype);
}

async function runMatmulResolved(input, weight, weightsEntry, key, M, N, K, options = {}) {
  const { recorder = null, ...rest } = options;
  const resolved = resolveMatmulDtype(weight, weightsEntry, key);
  const bDtype = inferDiffusionMatmulDtypeFromBuffer(weight, N, K, resolved);
  const nextOptions = bDtype ? { ...rest, bDtype } : rest;
  if (recorder) {
    return recordMatmul(recorder, input, weight, M, N, K, nextOptions);
  }
  return runMatmul(input, weight, M, N, K, nextOptions);
}

async function runClipTextEncoder(tokens, weightsEntry, config, runtime, options = {}) {
  const device = getDevice();
  if (!device) throw new Error('CLIP encoder requires a WebGPU device.');
  if (!weightsEntry?.weights || !weightsEntry?.shapes) {
    throw new Error('CLIP encoder requires loaded weights.');
  }

  const prefix = options.prefix;
  const localRecorder = options.recorder
    ? null
    : (options.profile ? new CommandRecorder(device, `${prefix || 'clip'}_encoder`, { profile: true }) : null);
  const recorder = options.recorder ?? localRecorder;
  const ops = createKernelOps(recorder);
  const release = createDiffusionBufferReleaser(recorder);
  const destroy = createDiffusionBufferDestroyer(recorder);
  const weights = weightsEntry.weights;
  const hiddenSize = config.hidden_size;
  const numHeads = config.num_attention_heads;
  const headDim = Math.floor(hiddenSize / numHeads);
  const maxLength = config.max_position_embeddings;
  const padTokenId = config.pad_token_id ?? 0;
  const eosTokenId = config.eos_token_id ?? null;
  const activationDtype = resolveDiffusionActivationDtype(runtime);
  const matmul = (input, weight, key, M, N, K, options = {}) =>
    runMatmulResolved(input, weight, weightsEntry, key, M, N, K, { ...options, recorder });

  const padded = padTokens(tokens, maxLength, padTokenId);
  const tokenBuffer = createDiffusionIndexBuffer(device, padded, `${prefix}_tokens`);

  const tokenEmbedWeight = expectDiffusionWeight(
    getWeight(weights, prefix, 'text_model.embeddings.token_embedding.weight'),
    `${prefix}.text_model.embeddings.token_embedding.weight`
  );
  const posEmbedWeight = expectDiffusionWeight(
    getWeight(weights, prefix, 'text_model.embeddings.position_embedding.weight'),
    `${prefix}.text_model.embeddings.position_embedding.weight`
  );

  const tokenEmbedKey = `${prefix}.text_model.embeddings.token_embedding.weight`;
  const posEmbedKey = `${prefix}.text_model.embeddings.position_embedding.weight`;
  const tokenEmbedDtype = resolveEmbeddingDtype(tokenEmbedWeight, weightsEntry, tokenEmbedKey, runtime);
  const posEmbedDtype = resolveEmbeddingDtype(posEmbedWeight, weightsEntry, posEmbedKey, runtime);

  let hidden = await ops.gather(
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
  const posBuffer = createDiffusionIndexBuffer(device, posIndices, `${prefix}_pos_idx`);
  const pos = await ops.gather(
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

  destroy(posBuffer);
  destroy(tokenBuffer);

  const combined = await ops.residualAdd(hidden, pos, maxLength * hiddenSize, { useVec4: true });
  release(hidden.buffer);
  release(pos.buffer);
  hidden = createTensor(combined.buffer, combined.dtype, [maxLength, hiddenSize], 'clip_embed');

  const layerCount = config.num_hidden_layers;
  for (let layerIdx = 0; layerIdx < layerCount; layerIdx++) {
    const ln1Weight = expectDiffusionWeight(
      getWeight(weights, prefix, `text_model.encoder.layers.${layerIdx}.layer_norm1.weight`),
      `${prefix}.text_model.encoder.layers.${layerIdx}.layer_norm1.weight`
    );
    const ln1Bias = expectDiffusionWeight(
      getWeight(weights, prefix, `text_model.encoder.layers.${layerIdx}.layer_norm1.bias`),
      `${prefix}.text_model.encoder.layers.${layerIdx}.layer_norm1.bias`
    );
    const ln2Weight = expectDiffusionWeight(
      getWeight(weights, prefix, `text_model.encoder.layers.${layerIdx}.layer_norm2.weight`),
      `${prefix}.text_model.encoder.layers.${layerIdx}.layer_norm2.weight`
    );
    const ln2Bias = expectDiffusionWeight(
      getWeight(weights, prefix, `text_model.encoder.layers.${layerIdx}.layer_norm2.bias`),
      `${prefix}.text_model.encoder.layers.${layerIdx}.layer_norm2.bias`
    );

    const norm1 = await ops.layerNorm(hidden, getBuffer(ln1Weight), getBuffer(ln1Bias), config.layer_norm_eps, {
      batchSize: maxLength,
      hiddenSize,
    });

    const qKey = `${prefix}.text_model.encoder.layers.${layerIdx}.self_attn.q_proj.weight`;
    const kKey = `${prefix}.text_model.encoder.layers.${layerIdx}.self_attn.k_proj.weight`;
    const vKey = `${prefix}.text_model.encoder.layers.${layerIdx}.self_attn.v_proj.weight`;
    const qWeight = expectDiffusionWeight(
      getWeight(weights, prefix, `text_model.encoder.layers.${layerIdx}.self_attn.q_proj.weight`),
      qKey
    );
    const kWeight = expectDiffusionWeight(
      getWeight(weights, prefix, `text_model.encoder.layers.${layerIdx}.self_attn.k_proj.weight`),
      kKey
    );
    const vWeight = expectDiffusionWeight(
      getWeight(weights, prefix, `text_model.encoder.layers.${layerIdx}.self_attn.v_proj.weight`),
      vKey
    );
    const qBiasKey = `${prefix}.text_model.encoder.layers.${layerIdx}.self_attn.q_proj.bias`;
    const kBiasKey = `${prefix}.text_model.encoder.layers.${layerIdx}.self_attn.k_proj.bias`;
    const vBiasKey = `${prefix}.text_model.encoder.layers.${layerIdx}.self_attn.v_proj.bias`;
    const qBias = expectDiffusionWeight(
      getWeight(weights, prefix, `text_model.encoder.layers.${layerIdx}.self_attn.q_proj.bias`),
      qBiasKey
    );
    const kBias = expectDiffusionWeight(
      getWeight(weights, prefix, `text_model.encoder.layers.${layerIdx}.self_attn.k_proj.bias`),
      kBiasKey
    );
    const vBias = expectDiffusionWeight(
      getWeight(weights, prefix, `text_model.encoder.layers.${layerIdx}.self_attn.v_proj.bias`),
      vBiasKey
    );
    const outKey = `${prefix}.text_model.encoder.layers.${layerIdx}.self_attn.out_proj.weight`;
    const outWeight = expectDiffusionWeight(
      getWeight(weights, prefix, `text_model.encoder.layers.${layerIdx}.self_attn.out_proj.weight`),
      outKey
    );
    const outBiasKey = `${prefix}.text_model.encoder.layers.${layerIdx}.self_attn.out_proj.bias`;
    const outBias = expectDiffusionWeight(
      getWeight(weights, prefix, `text_model.encoder.layers.${layerIdx}.self_attn.out_proj.bias`),
      outBiasKey
    );

    let q = await matmul(norm1, qWeight, qKey, maxLength, hiddenSize, hiddenSize, { outputDtype: activationDtype, transposeB: 'auto' });
    let k = await matmul(norm1, kWeight, kKey, maxLength, hiddenSize, hiddenSize, { outputDtype: activationDtype, transposeB: 'auto' });
    let v = await matmul(norm1, vWeight, vKey, maxLength, hiddenSize, hiddenSize, { outputDtype: activationDtype, transposeB: 'auto' });
    if (qBias) q = await ops.biasAdd(q, createBiasTensorWithDtype(qBias, weightsEntry, qBiasKey, hiddenSize, `${prefix}_q_bias`), maxLength, hiddenSize);
    if (kBias) k = await ops.biasAdd(k, createBiasTensorWithDtype(kBias, weightsEntry, kBiasKey, hiddenSize, `${prefix}_k_bias`), maxLength, hiddenSize);
    if (vBias) v = await ops.biasAdd(v, createBiasTensorWithDtype(vBias, weightsEntry, vBiasKey, hiddenSize, `${prefix}_v_bias`), maxLength, hiddenSize);

    const attn = await ops.attention(q, k, v, null, numHeads, headDim, {
      seqLen: maxLength,
      kvLen: maxLength,
      numKVHeads: numHeads,
      causal: false,
    });

    let attnOut = await matmul(attn, outWeight, outKey, maxLength, hiddenSize, hiddenSize, { outputDtype: activationDtype, transposeB: 'auto' });
    if (outBias) attnOut = await ops.biasAdd(attnOut, createBiasTensorWithDtype(outBias, weightsEntry, outBiasKey, hiddenSize, `${prefix}_out_bias`), maxLength, hiddenSize);

    const attnResidual = await ops.residualAdd(hidden, attnOut, maxLength * hiddenSize, { useVec4: true });

    release(norm1.buffer);
    release(q.buffer);
    release(k.buffer);
    release(v.buffer);
    release(attn.buffer);
    release(attnOut.buffer);
    release(hidden.buffer);

    hidden = createTensor(attnResidual.buffer, attnResidual.dtype, [maxLength, hiddenSize], 'clip_attn_out');

    const norm2 = await ops.layerNorm(hidden, getBuffer(ln2Weight), getBuffer(ln2Bias), config.layer_norm_eps, {
      batchSize: maxLength,
      hiddenSize,
    });

    const fc1Key = `${prefix}.text_model.encoder.layers.${layerIdx}.mlp.fc1.weight`;
    const fc1Weight = expectDiffusionWeight(
      getWeight(weights, prefix, `text_model.encoder.layers.${layerIdx}.mlp.fc1.weight`),
      fc1Key
    );
    const fc1BiasKey = `${prefix}.text_model.encoder.layers.${layerIdx}.mlp.fc1.bias`;
    const fc1Bias = expectDiffusionWeight(
      getWeight(weights, prefix, `text_model.encoder.layers.${layerIdx}.mlp.fc1.bias`),
      fc1BiasKey
    );
    const fc2Key = `${prefix}.text_model.encoder.layers.${layerIdx}.mlp.fc2.weight`;
    const fc2Weight = expectDiffusionWeight(
      getWeight(weights, prefix, `text_model.encoder.layers.${layerIdx}.mlp.fc2.weight`),
      fc2Key
    );
    const fc2BiasKey = `${prefix}.text_model.encoder.layers.${layerIdx}.mlp.fc2.bias`;
    const fc2Bias = expectDiffusionWeight(
      getWeight(weights, prefix, `text_model.encoder.layers.${layerIdx}.mlp.fc2.bias`),
      fc2BiasKey
    );

    const intermediate = fc1Weight.shape[0];
    let mlp = await matmul(norm2, fc1Weight, fc1Key, maxLength, intermediate, hiddenSize, { outputDtype: activationDtype, transposeB: 'auto' });
    if (fc1Bias) mlp = await ops.biasAdd(mlp, createBiasTensorWithDtype(fc1Bias, weightsEntry, fc1BiasKey, intermediate, `${prefix}_fc1_bias`), maxLength, intermediate);

    const gelu = await ops.gelu(mlp, { size: maxLength * intermediate });
    release(mlp.buffer);

    let mlpOut = await matmul(gelu, fc2Weight, fc2Key, maxLength, hiddenSize, intermediate, { outputDtype: activationDtype, transposeB: 'auto' });
    if (fc2Bias) mlpOut = await ops.biasAdd(mlpOut, createBiasTensorWithDtype(fc2Bias, weightsEntry, fc2BiasKey, hiddenSize, `${prefix}_fc2_bias`), maxLength, hiddenSize);

    const mlpResidual = await ops.residualAdd(hidden, mlpOut, maxLength * hiddenSize, { useVec4: true });

    release(norm2.buffer);
    release(gelu.buffer);
    release(mlpOut.buffer);
    release(hidden.buffer);

    hidden = createTensor(mlpResidual.buffer, mlpResidual.dtype, [maxLength, hiddenSize], 'clip_mlp_out');
  }

  const finalLnWeight = expectDiffusionWeight(
    getWeight(weights, prefix, 'text_model.final_layer_norm.weight'),
    `${prefix}.text_model.final_layer_norm.weight`
  );
  const finalLnBias = expectDiffusionWeight(
    getWeight(weights, prefix, 'text_model.final_layer_norm.bias'),
    `${prefix}.text_model.final_layer_norm.bias`
  );
  const final = await ops.layerNorm(hidden, getBuffer(finalLnWeight), getBuffer(finalLnBias), config.layer_norm_eps, {
    batchSize: maxLength,
    hiddenSize,
  });
  release(hidden.buffer);

  const eosIndex = findEosIndex(padded, eosTokenId);
  const eosIdxBuffer = createDiffusionIndexBuffer(device, new Uint32Array([eosIndex]), `${prefix}_eos_idx`);
  const pooledToken = await ops.gather(
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
  destroy(eosIdxBuffer);

  const textProjKey = `${prefix}.text_projection.weight`;
  const textProj = expectDiffusionWeight(
    getWeight(weights, prefix, 'text_projection.weight'),
    textProjKey
  );
  let pooled = await matmul(pooledToken, textProj, textProjKey, 1, hiddenSize, hiddenSize, { outputDtype: activationDtype, transposeB: 'auto' });
  release(pooledToken.buffer);

  const pooledBuffer = pooled.buffer;
  if (recorder) {
    recorder.submit();
  }
  const pooledData = await readBuffer(pooledBuffer, hiddenSize * (pooled.dtype === 'f16' ? 2 : 4));
  if (recorder) {
    releaseBuffer(pooledBuffer);
  } else {
    release(pooledBuffer);
  }

  let profile = null;
  if (localRecorder) {
    const timings = await localRecorder.resolveProfileTimings();
    profile = timings ? { totalMs: sumDiffusionProfileTimings(timings) ?? 0, timings } : { totalMs: null };
  }

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
    profile,
  };
}

async function runT5Encoder(tokens, weightsEntry, config, runtime, options = {}) {
  const device = getDevice();
  if (!device) throw new Error('T5 encoder requires a WebGPU device.');
  if (!weightsEntry?.weights || !weightsEntry?.shapes) {
    throw new Error('T5 encoder requires loaded weights.');
  }

  const prefix = options.prefix;
  const localRecorder = options.recorder
    ? null
    : (options.profile ? new CommandRecorder(device, `${prefix || 't5'}_encoder`, { profile: true }) : null);
  const recorder = options.recorder ?? localRecorder;
  const ops = createKernelOps(recorder);
  const release = createDiffusionBufferReleaser(recorder);
  const destroy = createDiffusionBufferDestroyer(recorder);
  const weights = weightsEntry.weights;
  const hiddenSize = config.d_model;
  const numHeads = config.num_heads;
  const headDim = config.d_kv;
  const maxLength = options.maxLength;
  const padTokenId = config.pad_token_id ?? 0;
  const activationDtype = resolveDiffusionActivationDtype(runtime);
  const matmul = (input, weight, key, M, N, K, options = {}) =>
    runMatmulResolved(input, weight, weightsEntry, key, M, N, K, { ...options, recorder });

  const padded = padTokens(tokens, maxLength, padTokenId);
  const tokenBuffer = createDiffusionIndexBuffer(device, padded, `${prefix}_tokens`);

  const embedWeight = getWeight(weights, prefix, 'shared.weight');
  if (!embedWeight) {
    throw new Error('T5 shared.weight missing.');
  }

  const embedKey = `${prefix}.shared.weight`;
  const embedDtype = resolveEmbeddingDtype(embedWeight, weightsEntry, embedKey, runtime);

  let hidden = await ops.gather(
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
  destroy(tokenBuffer);

  const layerCount = config.num_layers;
  for (let layerIdx = 0; layerIdx < layerCount; layerIdx++) {
    const lnWeight = expectDiffusionWeight(
      getWeight(weights, prefix, `encoder.block.${layerIdx}.layer.0.layer_norm.weight`),
      `${prefix}.encoder.block.${layerIdx}.layer.0.layer_norm.weight`
    );
    const normed = await ops.rmsNorm(hidden, getBuffer(lnWeight), config.layer_norm_epsilon, {
      batchSize: maxLength,
      hiddenSize,
    });

    const qName = `encoder.block.${layerIdx}.layer.0.SelfAttention.q.weight`;
    const kName = `encoder.block.${layerIdx}.layer.0.SelfAttention.k.weight`;
    const vName = `encoder.block.${layerIdx}.layer.0.SelfAttention.v.weight`;
    const oName = `encoder.block.${layerIdx}.layer.0.SelfAttention.o.weight`;
    const qKey = `${prefix}.${qName}`;
    const kKey = `${prefix}.${kName}`;
    const vKey = `${prefix}.${vName}`;
    const oKey = `${prefix}.${oName}`;

    const qWeight = expectDiffusionWeight(getWeight(weights, prefix, qName), qKey);
    const kWeight = expectDiffusionWeight(getWeight(weights, prefix, kName), kKey);
    const vWeight = expectDiffusionWeight(getWeight(weights, prefix, vName), vKey);
    const oWeight = expectDiffusionWeight(getWeight(weights, prefix, oName), oKey);

    let q = await matmul(normed, qWeight, qKey, maxLength, hiddenSize, hiddenSize, {
      outputDtype: activationDtype,
      transposeB: 'auto',
    });
    let k = await matmul(normed, kWeight, kKey, maxLength, hiddenSize, hiddenSize, {
      outputDtype: activationDtype,
      transposeB: 'auto',
    });
    let v = await matmul(normed, vWeight, vKey, maxLength, hiddenSize, hiddenSize, {
      outputDtype: activationDtype,
      transposeB: 'auto',
    });

    const attn = await ops.attention(q, k, v, null, numHeads, headDim, {
      seqLen: maxLength,
      kvLen: maxLength,
      numKVHeads: numHeads,
      causal: false,
    });

    const attnOut = await matmul(attn, oWeight, oKey, maxLength, hiddenSize, hiddenSize, {
      outputDtype: activationDtype,
      transposeB: 'auto',
    });
    const attnResidual = await ops.residualAdd(hidden, attnOut, maxLength * hiddenSize, { useVec4: true });

    release(normed.buffer);
    release(q.buffer);
    release(k.buffer);
    release(v.buffer);
    release(attn.buffer);
    release(attnOut.buffer);
    release(hidden.buffer);

    hidden = createTensor(attnResidual.buffer, attnResidual.dtype, [maxLength, hiddenSize], 't5_attn_out');

    const ln2Weight = expectDiffusionWeight(
      getWeight(weights, prefix, `encoder.block.${layerIdx}.layer.1.layer_norm.weight`),
      `${prefix}.encoder.block.${layerIdx}.layer.1.layer_norm.weight`
    );
    const norm2 = await ops.rmsNorm(hidden, getBuffer(ln2Weight), config.layer_norm_epsilon, {
      batchSize: maxLength,
      hiddenSize,
    });

    const wi0Name = `encoder.block.${layerIdx}.layer.1.DenseReluDense.wi_0.weight`;
    const wi1Name = `encoder.block.${layerIdx}.layer.1.DenseReluDense.wi_1.weight`;
    const woName = `encoder.block.${layerIdx}.layer.1.DenseReluDense.wo.weight`;
    const wi0Key = `${prefix}.${wi0Name}`;
    const wi1Key = `${prefix}.${wi1Name}`;
    const woKey = `${prefix}.${woName}`;
    const wi0 = expectDiffusionWeight(getWeight(weights, prefix, wi0Name), wi0Key);
    const wi1 = expectDiffusionWeight(getWeight(weights, prefix, wi1Name), wi1Key);
    const wo = expectDiffusionWeight(getWeight(weights, prefix, woName), woKey);

    const dff = wi0.shape[0];
    const bytesPerElement = activationDtype === 'f16' ? 2 : 4;
    const combinedSize = maxLength * dff * 2 * bytesPerElement;
    const combinedBuffer = acquireBuffer(combinedSize, undefined, 't5_ff_combined');

    const wi0Out = await matmul(norm2, wi0, wi0Key, maxLength, dff, hiddenSize, {
      outputDtype: activationDtype,
      transposeB: 'auto',
      outputBuffer: combinedBuffer,
      cOffset: 0,
    });
    const wi1Out = await matmul(norm2, wi1, wi1Key, maxLength, dff, hiddenSize, {
      outputDtype: activationDtype,
      transposeB: 'auto',
      outputBuffer: combinedBuffer,
      cOffset: maxLength * dff * bytesPerElement,
    });

    const combinedTensor = createTensor(combinedBuffer, activationDtype, [maxLength, dff * 2], 't5_ff_combined');
    const gated = await ops.siluRowSplit(combinedTensor, {
      numTokens: maxLength,
      dim: dff,
      activation: 'gelu',
      swigluLimit: null,
    });

    release(combinedTensor.buffer);

    const ffOut = await matmul(gated, wo, woKey, maxLength, hiddenSize, dff, {
      outputDtype: activationDtype,
      transposeB: 'auto',
    });
    const ffResidual = await ops.residualAdd(hidden, ffOut, maxLength * hiddenSize, { useVec4: true });

    release(norm2.buffer);
    release(gated.buffer);
    release(ffOut.buffer);
    release(hidden.buffer);

    hidden = createTensor(ffResidual.buffer, ffResidual.dtype, [maxLength, hiddenSize], 't5_ff_out');
  }

  const finalLn = expectDiffusionWeight(
    getWeight(weights, prefix, 'encoder.final_layer_norm.weight'),
    `${prefix}.encoder.final_layer_norm.weight`
  );
  const final = await ops.rmsNorm(hidden, getBuffer(finalLn), config.layer_norm_epsilon, {
    batchSize: maxLength,
    hiddenSize,
  });
  release(hidden.buffer);

  let profile = null;
  if (localRecorder) {
    localRecorder.submit();
    const timings = await localRecorder.resolveProfileTimings();
    profile = timings ? { totalMs: sumDiffusionProfileTimings(timings) ?? 0, timings } : { totalMs: null };
  }

  return {
    hidden: final,
    maxLength,
    hiddenSize,
    profile,
  };
}

export async function runTextEncodersForPrompt(tokensByEncoder, weightsByComponent, modelConfig, runtime, options = {}) {
  const clipConfig = modelConfig?.components?.text_encoder?.config || {};
  const clip2Config = modelConfig?.components?.text_encoder_2?.config || {};
  const t5Config = modelConfig?.components?.text_encoder_3?.config || {};
  const t5MaxLength = runtime?.textEncoder?.t5MaxLength ?? runtime?.textEncoder?.maxLength;
  if (!Number.isFinite(t5MaxLength) || t5MaxLength <= 0) {
    throw new Error('T5 encoder requires runtime.textEncoder.t5MaxLength (or runtime.textEncoder.maxLength).');
  }
  const profileEnabled = options.profile === true;

  const clip = await runClipTextEncoder(tokensByEncoder.text_encoder, weightsByComponent.text_encoder, clipConfig, runtime, {
    prefix: 'text_encoder',
    profile: profileEnabled,
  });
  const clip2 = await runClipTextEncoder(tokensByEncoder.text_encoder_2, weightsByComponent.text_encoder_2, clip2Config, runtime, {
    prefix: 'text_encoder_2',
    profile: profileEnabled,
  });
  const t5 = await runT5Encoder(tokensByEncoder.text_encoder_3, weightsByComponent.text_encoder_3, t5Config, runtime, {
    prefix: 'text_encoder_3',
    maxLength: t5MaxLength,
    profile: profileEnabled,
  });

  const pooled = new Float32Array(clip.pooled.length + clip2.pooled.length);
  pooled.set(clip.pooled, 0);
  pooled.set(clip2.pooled, clip.pooled.length);

  releaseBuffer(clip.hidden.buffer);
  releaseBuffer(clip2.hidden.buffer);

  const clipMs = clip.profile?.totalMs ?? null;
  const clip2Ms = clip2.profile?.totalMs ?? null;
  const t5Ms = t5.profile?.totalMs ?? null;
  const totalMs = [clipMs, clip2Ms, t5Ms].reduce((sum, value) => sum + (Number.isFinite(value) ? value : 0), 0);
  const profile = profileEnabled
    ? {
        totalMs: Number.isFinite(totalMs) ? totalMs : null,
        clipMs,
        clip2Ms,
        t5Ms,
      }
    : null;

  return {
    pooled,
    context: t5.hidden,
    profile,
  };
}

export async function buildTimeTextEmbedding(pooled, weightsEntry, modelConfig, runtime, options = {}) {
  const device = getDevice();
  if (!device) throw new Error('TimeText embedding requires a WebGPU device.');
  const activationDtype = resolveDiffusionActivationDtype(runtime);
  const recorder = options.recorder ?? null;
  const ops = createKernelOps(recorder);
  const release = createDiffusionBufferReleaser(recorder);

  const resolver = createSD3WeightResolver(weightsEntry, modelConfig);
  const matmul = (input, weight, name, M, N, K, options = {}) =>
    runMatmulResolved(input, weight, weightsEntry, resolver.key(name), M, N, K, { ...options, recorder });
  const textLinear1Name = 'time_text_embed.text_embedder.linear_1.weight';
  const textLinear2Name = 'time_text_embed.text_embedder.linear_2.weight';
  const textLinear1 = resolver.get(textLinear1Name);
  const textLinear1BiasName = 'time_text_embed.text_embedder.linear_1.bias';
  const textLinear1Bias = resolver.get(textLinear1BiasName);
  const textLinear1BiasKey = resolver.key(textLinear1BiasName);
  const textLinear2 = resolver.get(textLinear2Name);
  const textLinear2BiasName = 'time_text_embed.text_embedder.linear_2.bias';
  const textLinear2Bias = resolver.get(textLinear2BiasName);
  const textLinear2BiasKey = resolver.key(textLinear2BiasName);
  if (!textLinear1 || !textLinear2) {
    throw new Error('Missing diffusion time_text_embed text weights.');
  }

  const pooledTensor = createVectorTensor(device, pooled, activationDtype, 'sd3_pooled');
  let text = await matmul(pooledTensor, textLinear1, textLinear1Name, 1, textLinear1.shape[0], textLinear1.shape[1], {
    outputDtype: activationDtype,
    transposeB: 'auto',
  });
  if (textLinear1Bias) {
    text = await ops.biasAdd(
      text,
      createBiasTensorWithDtype(textLinear1Bias, weightsEntry, textLinear1BiasKey, textLinear1.shape[0], 'sd3_text_bias1'),
      1,
      textLinear1.shape[0]
    );
  }
  const textAct = await ops.silu(text, { size: textLinear1.shape[0], swigluLimit: null });
  release(text.buffer);

  let textOut = await matmul(textAct, textLinear2, textLinear2Name, 1, textLinear2.shape[0], textLinear2.shape[1], {
    outputDtype: activationDtype,
    transposeB: 'auto',
  });
  if (textLinear2Bias) {
    textOut = await ops.biasAdd(
      textOut,
      createBiasTensorWithDtype(textLinear2Bias, weightsEntry, textLinear2BiasKey, textLinear2.shape[0], 'sd3_text_bias2'),
      1,
      textLinear2.shape[0]
    );
  }

  release(textAct.buffer);
  release(pooledTensor.buffer);

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

  const activationDtype = resolveDiffusionActivationDtype(runtime);
  const recorder = options.recorder ?? null;
  const ops = createKernelOps(recorder);
  const release = createDiffusionBufferReleaser(recorder);
  const embTensor = createVectorTensor(device, emb, activationDtype, 'sd3_timestep');

  const resolver = createSD3WeightResolver(weightsEntry, modelConfig);
  const matmul = (input, weight, name, M, N, K, options = {}) =>
    runMatmulResolved(input, weight, weightsEntry, resolver.key(name), M, N, K, { ...options, recorder });
  const linear1Name = 'time_text_embed.timestep_embedder.linear_1.weight';
  const linear2Name = 'time_text_embed.timestep_embedder.linear_2.weight';
  const linear1 = resolver.get(linear1Name);
  const linear1BiasName = 'time_text_embed.timestep_embedder.linear_1.bias';
  const linear1Bias = resolver.get(linear1BiasName);
  const linear1BiasKey = resolver.key(linear1BiasName);
  const linear2 = resolver.get(linear2Name);
  const linear2BiasName = 'time_text_embed.timestep_embedder.linear_2.bias';
  const linear2Bias = resolver.get(linear2BiasName);
  const linear2BiasKey = resolver.key(linear2BiasName);
  if (!linear1 || !linear2) {
    throw new Error('Missing diffusion time_text_embed timestep weights.');
  }

  let out = await matmul(embTensor, linear1, linear1Name, 1, linear1.shape[0], linear1.shape[1], {
    outputDtype: activationDtype,
    transposeB: 'auto',
  });
  if (linear1Bias) {
    out = await ops.biasAdd(
      out,
      createBiasTensorWithDtype(linear1Bias, weightsEntry, linear1BiasKey, linear1.shape[0], 'sd3_time_bias1'),
      1,
      linear1.shape[0]
    );
  }
  const act = await ops.silu(out, { size: linear1.shape[0], swigluLimit: null });
  release(out.buffer);

  let out2 = await matmul(act, linear2, linear2Name, 1, linear2.shape[0], linear2.shape[1], {
    outputDtype: activationDtype,
    transposeB: 'auto',
  });
  if (linear2Bias) {
    out2 = await ops.biasAdd(
      out2,
      createBiasTensorWithDtype(linear2Bias, weightsEntry, linear2BiasKey, linear2.shape[0], 'sd3_time_bias2'),
      1,
      linear2.shape[0]
    );
  }

  release(act.buffer);
  release(embTensor.buffer);

  return out2;
}

export async function combineTimeTextEmbeddings(time, text, hiddenSize, options = {}) {
  const recorder = options.recorder ?? null;
  const ops = createKernelOps(recorder);
  const release = createDiffusionBufferReleaser(recorder);
  const combined = await ops.residualAdd(time, text, hiddenSize, { useVec4: true });
  release(time.buffer);
  release(text.buffer);
  return createTensor(combined.buffer, combined.dtype, [1, hiddenSize], 'sd3_time_text');
}

export async function projectContext(context, weightsEntry, modelConfig, runtime, options = {}) {
  const resolver = createSD3WeightResolver(weightsEntry, modelConfig);
  const recorder = options.recorder ?? null;
  const ops = createKernelOps(recorder);
  const release = createDiffusionBufferReleaser(recorder);
  const matmul = (input, weight, name, M, N, K, options = {}) =>
    runMatmulResolved(input, weight, weightsEntry, resolver.key(name), M, N, K, { ...options, recorder });
  const projWeightName = 'context_embedder.weight';
  const projWeight = resolver.get(projWeightName);
  const projBiasName = 'context_embedder.bias';
  const projBias = resolver.get(projBiasName);
  const projBiasKey = resolver.key(projBiasName);
  if (!projWeight) {
    throw new Error('Missing diffusion context_embedder weight.');
  }
  const numTokens = context.shape[0];
  const inDim = context.shape[1];
  const outDim = projWeight.shape[0];
  const activationDtype = resolveDiffusionActivationDtype(runtime);
  let projected = await matmul(context, projWeight, projWeightName, numTokens, outDim, inDim, {
    outputDtype: activationDtype,
    transposeB: 'auto',
  });
  if (projBias) {
    projected = await ops.biasAdd(
      projected,
      createBiasTensorWithDtype(projBias, weightsEntry, projBiasKey, outDim, 'sd3_ctx_bias'),
      numTokens,
      outDim
    );
  }
  release(context.buffer);
  return projected;
}

export function logQuickGeluWarning(config) {
  if (config.hidden_act === 'quick_gelu') {
    log.warn('Diffusion', 'CLIP quick_gelu is approximated with gelu.');
  }
}
