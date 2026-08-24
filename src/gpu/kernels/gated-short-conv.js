import { getDevice, getDeviceEpoch } from '../device.js';
import { WORKGROUP_SIZES } from './constants.js';
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor } from '../tensor.js';
import { createUniformBufferFromData } from './uniform-utils.js';
import {
  getOrCreateBindGroupLayout,
  getOrCreatePipelineLayout,
} from './pipeline-cache.js';
import { recordDispatch } from './dispatch.js';
import { getShaderModule } from './shader-cache.js';

const CONV_WORKGROUP_SIZE = WORKGROUP_SIZES.DEFAULT;

// ======================================================================
// UNIFORM BUFFER
// ======================================================================

const UNIFORM_LAYOUT = {
  numTokens: { offset: 0, size: 4 },
  hiddenSize: { offset: 4, size: 4 },
  kernelSize: { offset: 8, size: 4 },
  _pad: { offset: 12, size: 4 },
};

const UNIFORM_SIZE = 16;

function buildParamsData(numTokens, hiddenSize, kernelSize) {
  const data = new ArrayBuffer(UNIFORM_SIZE);
  const view = new DataView(data);
  view.setUint32(UNIFORM_LAYOUT.numTokens.offset, numTokens, true);
  view.setUint32(UNIFORM_LAYOUT.hiddenSize.offset, hiddenSize, true);
  view.setUint32(UNIFORM_LAYOUT.kernelSize.offset, kernelSize, true);
  view.setUint32(UNIFORM_LAYOUT._pad.offset, 0, true);
  return data;
}

// ======================================================================
// PIPELINE CACHE
// ======================================================================

let cachedEpoch = -1;
let pipeline = null;
let bindGroupLayout = null;

async function createPipeline(device) {
  bindGroupLayout = getOrCreateBindGroupLayout(
    'gated_short_conv_layout',
    [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
    device
  );

  const module = await getShaderModule(device, 'gated_short_conv.wgsl', 'gated_short_conv');

  pipeline = device.createComputePipeline({
    label: 'gated_short_conv_pipeline',
    layout: getOrCreatePipelineLayout('gated_short_conv_pipeline_layout', [bindGroupLayout], device),
    compute: {
      module,
      entryPoint: 'main',
      constants: {
        WORKGROUP_SIZE: CONV_WORKGROUP_SIZE,
      },
    },
  });
}

async function ensurePipeline(device) {
  const epoch = getDeviceEpoch();
  if (epoch !== cachedEpoch || !pipeline) {
    await createPipeline(device);
    cachedEpoch = epoch;
  }
}

// ======================================================================
// VALIDATION
// ======================================================================

function requireGpuBuffer(buffer, label) {
  if (!(buffer instanceof GPUBuffer)) {
    throw new Error(`gated_short_conv kernel requires GPUBuffer for ${label}.`);
  }
}

// ======================================================================
// DISPATCH
// ======================================================================

export async function runGatedShortConvGPU(inputTensor, layerState, options = {}) {
  const device = getDevice();
  if (!device) {
    throw new Error('No GPU device available for gated_short_conv.');
  }

  const recorder = options.recorder ?? null;
  const useRecorder = recorder
    && typeof recorder.getEncoder === 'function'
    && typeof recorder.trackTemporaryBuffer === 'function';

  requireGpuBuffer(inputTensor?.buffer, 'inputTensor');
  requireGpuBuffer(layerState?.convWeightGPU, 'convWeightGPU');
  requireGpuBuffer(layerState?.convStateGPU, 'convStateGPU');

  const numTokens = Number(options.numTokens ?? 0);
  if (!Number.isFinite(numTokens) || numTokens <= 0) {
    throw new Error('runGatedShortConvGPU requires numTokens > 0.');
  }

  const hiddenSize = Number(layerState.hiddenSize ?? 0);
  if (!Number.isFinite(hiddenSize) || hiddenSize <= 0) {
    throw new Error('runGatedShortConvGPU requires hiddenSize > 0.');
  }

  const kernelSize = Number(layerState.kernelSize ?? 0);
  if (!Number.isFinite(kernelSize) || kernelSize < 2) {
    throw new Error('runGatedShortConvGPU requires kernelSize >= 2.');
  }

  await ensurePipeline(device);

  const outputSize = numTokens * hiddenSize * Float32Array.BYTES_PER_ELEMENT;
  const outputBuffer = acquireBuffer(outputSize, undefined, `L${options.layerIdx ?? 0}.gated_short_conv_out`);

  if (useRecorder) {
    const paramsBuffer = createUniformBufferFromData(
      'gated_short_conv_params',
      buildParamsData(numTokens, hiddenSize, kernelSize),
      recorder
    );

    try {
      const bg = device.createBindGroup({
        label: 'gated_short_conv_bind_group',
        layout: bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: inputTensor.buffer } },
          { binding: 2, resource: { buffer: layerState.convWeightGPU } },
          { binding: 3, resource: { buffer: layerState.convStateGPU } },
          { binding: 4, resource: { buffer: outputBuffer } },
        ],
      });

      recordDispatch(
        recorder,
        pipeline,
        bg,
        [Math.ceil(hiddenSize / CONV_WORKGROUP_SIZE), 1, 1],
        'gated_short_conv'
      );

      return createTensor(
        outputBuffer,
        'f32',
        [numTokens, hiddenSize],
        `L${options.layerIdx ?? 0}.gated_short_conv`
      );
    } catch (error) {
      releaseBuffer(outputBuffer);
      throw error;
    }
  }

  // Non-recorder path
  const paramsBuffer = createUniformBufferFromData(
    'gated_short_conv_params',
    buildParamsData(numTokens, hiddenSize, kernelSize),
    null,
    device,
    { useCache: false }
  );
  let submitted = false;

  try {
    const bg = device.createBindGroup({
      label: 'gated_short_conv_bind_group',
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: paramsBuffer } },
        { binding: 1, resource: { buffer: inputTensor.buffer } },
        { binding: 2, resource: { buffer: layerState.convWeightGPU } },
        { binding: 3, resource: { buffer: layerState.convStateGPU } },
        { binding: 4, resource: { buffer: outputBuffer } },
      ],
    });

    const encoder = device.createCommandEncoder({ label: 'gated_short_conv' });
    const pass = encoder.beginComputePass({ label: 'gated_short_conv_pass' });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(hiddenSize / CONV_WORKGROUP_SIZE), 1, 1);
    pass.end();
    device.queue.submit([encoder.finish()]);
    submitted = true;

    return createTensor(
      outputBuffer,
      'f32',
      [numTokens, hiddenSize],
      `L${options.layerIdx ?? 0}.gated_short_conv`
    );
  } catch (error) {
    releaseBuffer(outputBuffer);
    throw error;
  } finally {
    if (submitted) {
      device.queue.onSubmittedWorkDone()
        .then(() => {
          paramsBuffer.destroy();
        })
        .catch(() => {
          paramsBuffer.destroy();
        });
    } else {
      paramsBuffer.destroy();
    }
  }
}
