import { getDevice } from '../../gpu/device.js';
import { createTensor } from '../../gpu/tensor.js';
import { createWeightBuffer } from '../../gpu/weight-buffer.js';
import { releaseBuffer } from '../../memory/buffer-pool.js';
import {
  runMatmul,
  runSoftmax,
  runGeLU,
  runBiasAdd,
  runResidualAdd,
  runLayerNorm,
} from '../../gpu/kernels/index.js';
import { log, trace } from '../../debug/index.js';

function createRng(seed) {
  let state = seed >>> 0;
  if (!state) state = 0x6d2b79f5;
  return () => {
    state |= 0;
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function createBufferFromArray(device, data, label) {
  const buffer = device.createBuffer({
    label,
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

function createWeight(device, data, shape, label) {
  const buffer = createBufferFromArray(device, data, label);
  return createWeightBuffer(buffer, 'f32', 'row', shape, label);
}

function createBias(device, data, label) {
  return createBufferFromArray(device, data, label);
}

function createTensorFromArray(device, data, shape, label) {
  const buffer = createBufferFromArray(device, data, label);
  return createTensor(buffer, 'f32', shape, label);
}

function generateRandomArray(count, seed) {
  const out = new Float32Array(count);
  const rand = createRng(seed);
  for (let i = 0; i < count; i++) {
    const u = rand();
    const v = rand();
    const z = Math.sqrt(-2.0 * Math.log(Math.max(u, 1e-6))) * Math.cos(2.0 * Math.PI * v);
    out[i] = z * 0.02;
  }
  return out;
}

export function initializeDiffusionGpuScaffold(runtime) {
  const device = getDevice();
  if (!device) {
    throw new Error('GPU scaffold requires an initialized device.');
  }

  const config = runtime?.backend?.scaffold || {};
  const tokens = Math.floor(config.tokens);
  const hiddenSize = Math.floor(config.hiddenSize);
  const numHeads = Math.floor(config.numHeads);
  const eps = config.layerNormEps;

  if (!Number.isFinite(tokens) || tokens <= 0 ||
      !Number.isFinite(hiddenSize) || hiddenSize <= 0 ||
      !Number.isFinite(numHeads) || numHeads <= 0) {
    throw new Error('Diffusion GPU scaffold requires tokens, hiddenSize, and numHeads > 0.');
  }
  if (!Number.isFinite(eps) || eps <= 0) {
    throw new Error('Diffusion GPU scaffold requires a valid layerNormEps.');
  }

  const seed = config.seed ?? 1337;
  const inputSize = tokens * hiddenSize;
  const input = createTensorFromArray(device, generateRandomArray(inputSize, seed), [tokens, hiddenSize], 'diffusion_scaffold_input');

  const weightSize = hiddenSize * hiddenSize;
  const qWeight = createWeight(device, generateRandomArray(weightSize, seed + 1), [hiddenSize, hiddenSize], 'diffusion_scaffold_wq');
  const kWeight = createWeight(device, generateRandomArray(weightSize, seed + 2), [hiddenSize, hiddenSize], 'diffusion_scaffold_wk');
  const vWeight = createWeight(device, generateRandomArray(weightSize, seed + 3), [hiddenSize, hiddenSize], 'diffusion_scaffold_wv');
  const outWeight = createWeight(device, generateRandomArray(weightSize, seed + 4), [hiddenSize, hiddenSize], 'diffusion_scaffold_wo');

  const ffnUp = createWeight(device, generateRandomArray(weightSize, seed + 5), [hiddenSize, hiddenSize], 'diffusion_scaffold_ffn_up');
  const ffnDown = createWeight(device, generateRandomArray(weightSize, seed + 6), [hiddenSize, hiddenSize], 'diffusion_scaffold_ffn_down');

  const lnWeight = createBias(device, generateRandomArray(hiddenSize, seed + 7), 'diffusion_scaffold_ln_weight');
  const lnBias = createBias(device, generateRandomArray(hiddenSize, seed + 8), 'diffusion_scaffold_ln_bias');

  return {
    device,
    input,
    tokens,
    hiddenSize,
    numHeads,
    eps,
    weights: {
      q: qWeight,
      k: kWeight,
      v: vWeight,
      out: outWeight,
      ffnUp,
      ffnDown,
      lnWeight,
      lnBias,
    },
  };
}

export async function runLinear(input, weight, bias, options = {}) {
  const { tokens, outDim, inDim, outputDtype = 'f32' } = options;
  const output = await runMatmul(
    input,
    weight,
    tokens,
    outDim,
    inDim,
    { outputDtype }
  );
  if (bias) {
    await runBiasAdd(output, bias, tokens, outDim);
  }
  return output;
}

export async function runAttentionScaffold(input, weights, options = {}) {
  const tokens = options.tokens;
  const hiddenSize = options.hiddenSize;
  if (!Number.isFinite(tokens) || !Number.isFinite(hiddenSize)) {
    throw new Error('Attention scaffold requires tokens and hiddenSize.');
  }

  const q = await runMatmul(input, weights.q, tokens, hiddenSize, hiddenSize, { outputDtype: 'f32' });
  const k = await runMatmul(input, weights.k, tokens, hiddenSize, hiddenSize, { outputDtype: 'f32' });
  const v = await runMatmul(input, weights.v, tokens, hiddenSize, hiddenSize, { outputDtype: 'f32' });

  const scores = await runMatmul(
    q,
    k.buffer,
    tokens,
    tokens,
    hiddenSize,
    { bDtype: k.dtype, transposeB: true, outputDtype: 'f32' }
  );

  const softmax = await runSoftmax(scores, -1, {
    batchSize: tokens,
    size: tokens,
    temperature: Math.sqrt(hiddenSize),
  });

  const context = await runMatmul(
    softmax,
    v.buffer,
    tokens,
    hiddenSize,
    tokens,
    { bDtype: v.dtype, transposeB: false, outputDtype: 'f32' }
  );

  const out = await runMatmul(context, weights.out, tokens, hiddenSize, hiddenSize, { outputDtype: 'f32' });

  releaseBuffer(q.buffer);
  releaseBuffer(k.buffer);
  releaseBuffer(v.buffer);
  releaseBuffer(scores.buffer);
  releaseBuffer(softmax.buffer);
  releaseBuffer(context.buffer);

  return out;
}

export async function runFeedForwardScaffold(input, weights, options = {}) {
  const tokens = options.tokens;
  const hiddenSize = options.hiddenSize;

  const up = await runMatmul(input, weights.ffnUp, tokens, hiddenSize, hiddenSize, { outputDtype: 'f32' });
  const act = await runGeLU(up, { size: tokens * hiddenSize });
  const down = await runMatmul(act, weights.ffnDown, tokens, hiddenSize, hiddenSize, { outputDtype: 'f32' });

  releaseBuffer(up.buffer);
  releaseBuffer(act.buffer);

  return down;
}

export async function runDiffusionGpuScaffold(scaffold, options = {}) {
  if (!scaffold) return null;
  const { tokens, hiddenSize, eps, weights } = scaffold;
  const step = options.stepIndex ?? 0;

  const normed = await runLayerNorm(scaffold.input, weights.lnWeight, weights.lnBias, eps, {
    batchSize: tokens,
    hiddenSize,
  });

  const attn = await runAttentionScaffold(normed, weights, { tokens, hiddenSize });
  const ffn = await runFeedForwardScaffold(attn, weights, { tokens, hiddenSize });
  const combined = await runResidualAdd(attn, ffn, tokens * hiddenSize, { useVec4: true });

  releaseBuffer(normed.buffer);
  releaseBuffer(attn.buffer);
  releaseBuffer(ffn.buffer);

  releaseBuffer(scaffold.input.buffer);
  scaffold.input = combined;

  trace.perf('Diffusion scaffold step', {
    step,
    tokens,
    hiddenSize,
  });

  return combined;
}

export function logDiffusionGpuScaffold(scaffold) {
  if (!scaffold) return;
  const { tokens, hiddenSize, numHeads } = scaffold;
  log.warn('Diffusion', `GPU scaffold enabled (diagnostic only): tokens=${tokens}, hidden=${hiddenSize}, heads=${numHeads}`);
}
