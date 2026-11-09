import { getDevice } from '../device.js';
import { acquireBuffer, releaseBuffer, uploadData } from '../../memory/buffer-pool.js';
import { createPipeline } from './pipeline-cache.js';
import { createUniformBufferWithView } from './uniform-utils.js';
import { createBindGroupWithValidation } from './bind-group-validation.js';
import { dispatchKernel } from './dispatch.js';
import { requirePositiveInteger as kernelPositiveInteger } from './kernel-geometry.js';

const UNIFORM_SIZE = 32;

const assertPositiveInteger = (value, label) => kernelPositiveInteger(value, label, 'DiffusionGemma sampling');

function assertFinitePositiveNumber(value, label) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    throw new Error(`[DiffusionGemmaSampling] ${label} must be a positive finite number.`);
  }
}

function assertFiniteNonNegativeNumber(value, label) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    throw new Error(`[DiffusionGemmaSampling] ${label} must be a non-negative finite number.`);
  }
}

function createStatsUniformBuffer(device, options) {
  return createUniformBufferWithView(
    'diffusion_gemma_sampling_uniforms',
    UNIFORM_SIZE,
    (view) => {
      view.setUint32(0, options.vocabSize, true);
      view.setUint32(4, options.canvasLength, true);
      view.setUint32(8, options.padTokenId == null ? 0xffffffff : options.padTokenId, true);
      view.setUint32(12, 0, true);
      view.setFloat32(16, options.temperature, true);
      view.setFloat32(20, options.logitSoftcap, true);
      view.setFloat32(24, 0, true);
      view.setFloat32(28, 0, true);
    },
    null,
    device
  );
}

export async function runDiffusionGemmaCanvasStats(logitsBuffer, options = {}) {
  const device = getDevice();
  if (!device) {
    throw new Error('[DiffusionGemmaSampling] GPU device is not initialized.');
  }
  if (!logitsBuffer) {
    throw new Error('[DiffusionGemmaSampling] logitsBuffer is required.');
  }
  assertPositiveInteger(options.canvasLength, 'canvasLength');
  assertPositiveInteger(options.vocabSize, 'vocabSize');
  assertFinitePositiveNumber(options.temperature, 'temperature');
  assertFiniteNonNegativeNumber(options.logitSoftcap, 'logitSoftcap');
  if (options.padTokenId != null && (!Number.isInteger(options.padTokenId) || options.padTokenId < 0)) {
    throw new Error('[DiffusionGemmaSampling] padTokenId must be null or a non-negative integer.');
  }
  const uniformDraws = options.uniformDraws;
  if (!(uniformDraws instanceof Float32Array) || uniformDraws.length !== options.canvasLength) {
    throw new Error('[DiffusionGemmaSampling] uniformDraws must be a Float32Array with one draw per canvas position.');
  }
  for (let i = 0; i < uniformDraws.length; i += 1) {
    if (!(uniformDraws[i] >= 0 && uniformDraws[i] < 1)) {
      throw new Error(`[DiffusionGemmaSampling] uniformDraws[${i}]=${uniformDraws[i]} must lie in [0, 1).`);
    }
  }

  const tokenBytes = options.canvasLength * Uint32Array.BYTES_PER_ELEMENT;
  const entropyBytes = options.canvasLength * Float32Array.BYTES_PER_ELEMENT;
  const argmaxBuffer = options.argmaxBuffer ?? acquireBuffer(tokenBytes, undefined, 'diffusion_gemma_argmax_canvas');
  const entropyBuffer = options.entropyBuffer ?? acquireBuffer(entropyBytes, undefined, 'diffusion_gemma_entropy_canvas');
  const sampledBuffer = options.sampledBuffer ?? acquireBuffer(tokenBytes, undefined, 'diffusion_gemma_sampled_canvas');
  const ownsArgmax = options.argmaxBuffer == null;
  const ownsEntropy = options.entropyBuffer == null;
  const ownsSampled = options.sampledBuffer == null;
  let uniformBuffer = null;
  let drawsBuffer = null;
  let completed = false;

  try {
    if (argmaxBuffer.size < tokenBytes) {
      throw new Error('[DiffusionGemmaSampling] argmaxBuffer is too small.');
    }
    if (entropyBuffer.size < entropyBytes) {
      throw new Error('[DiffusionGemmaSampling] entropyBuffer is too small.');
    }
    if (sampledBuffer.size < tokenBytes) {
      throw new Error('[DiffusionGemmaSampling] sampledBuffer is too small.');
    }
    const pipeline = await createPipeline('diffusion_gemma_sampling', 'entropy_stats');
    uniformBuffer = createStatsUniformBuffer(device, options);
    drawsBuffer = acquireBuffer(entropyBytes, undefined, 'diffusion_gemma_uniform_draws');
    uploadData(drawsBuffer, uniformDraws);
    const bindGroup = await createBindGroupWithValidation(device, {
      label: 'diffusion_gemma_sampling_bind_group',
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: logitsBuffer } },
        { binding: 2, resource: { buffer: argmaxBuffer } },
        { binding: 3, resource: { buffer: entropyBuffer } },
        { binding: 4, resource: { buffer: drawsBuffer } },
        { binding: 5, resource: { buffer: sampledBuffer } },
      ],
    }, 'diffusion_gemma_sampling/entropy_stats');

    dispatchKernel(null, pipeline, bindGroup, options.canvasLength, 'diffusion_gemma_entropy_stats');
    completed = true;
    return { argmaxBuffer, entropyBuffer, sampledBuffer };
  } finally {
    uniformBuffer?.destroy?.();
    // The dispatch is submitted; the pool reuses this buffer only behind it in queue order.
    if (drawsBuffer) releaseBuffer(drawsBuffer);
    if (!completed) {
      if (ownsArgmax) releaseBuffer(argmaxBuffer);
      if (ownsEntropy) releaseBuffer(entropyBuffer);
      if (ownsSampled) releaseBuffer(sampledBuffer);
    }
  }
}
