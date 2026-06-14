import { getDevice } from '../device.js';
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import {
  createPipeline,
  createUniformBufferWithView,
  createBindGroupWithValidation,
} from './utils.js';
import { dispatchKernel } from './dispatch.js';

const UNIFORM_SIZE = 32;

function assertPositiveInteger(value, label) {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`[DiffusionGemmaSampling] ${label} must be a positive integer.`);
  }
}

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

  const tokenBytes = options.canvasLength * Uint32Array.BYTES_PER_ELEMENT;
  const entropyBytes = options.canvasLength * Float32Array.BYTES_PER_ELEMENT;
  const argmaxBuffer = options.argmaxBuffer ?? acquireBuffer(tokenBytes, undefined, 'diffusion_gemma_argmax_canvas');
  const entropyBuffer = options.entropyBuffer ?? acquireBuffer(entropyBytes, undefined, 'diffusion_gemma_entropy_canvas');
  const ownsArgmax = options.argmaxBuffer == null;
  const ownsEntropy = options.entropyBuffer == null;
  let uniformBuffer = null;
  let completed = false;

  try {
    if (argmaxBuffer.size < tokenBytes) {
      throw new Error('[DiffusionGemmaSampling] argmaxBuffer is too small.');
    }
    if (entropyBuffer.size < entropyBytes) {
      throw new Error('[DiffusionGemmaSampling] entropyBuffer is too small.');
    }
    const pipeline = await createPipeline('diffusion_gemma_sampling', 'entropy_stats');
    uniformBuffer = createStatsUniformBuffer(device, options);
    const bindGroup = await createBindGroupWithValidation(device, {
      label: 'diffusion_gemma_sampling_bind_group',
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: logitsBuffer } },
        { binding: 2, resource: { buffer: argmaxBuffer } },
        { binding: 3, resource: { buffer: entropyBuffer } },
      ],
    }, 'diffusion_gemma_sampling/entropy_stats');

    dispatchKernel(null, pipeline, bindGroup, options.canvasLength, 'diffusion_gemma_entropy_stats');
    completed = true;
    return { argmaxBuffer, entropyBuffer };
  } finally {
    uniformBuffer?.destroy?.();
    if (!completed) {
      if (ownsArgmax) releaseBuffer(argmaxBuffer);
      if (ownsEntropy) releaseBuffer(entropyBuffer);
    }
  }
}
