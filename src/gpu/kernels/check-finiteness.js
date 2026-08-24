import { getDevice, getDeviceEpoch } from '../device.js';
import { dispatchKernel } from './dispatch.js';
import { createUniformBufferWithView } from './uniform-utils.js';
import {
  getOrCreateBindGroupLayout,
  getOrCreatePipelineLayout,
} from './pipeline-cache.js';
import { getShaderModule } from './shader-cache.js';

export const DEFAULT_FINITENESS_ABS_THRESHOLD = 65500;

export function resolveFinitenessAbsThreshold(value) {
  if (Number.isFinite(value) && value > 0) {
    return value;
  }
  return DEFAULT_FINITENESS_ABS_THRESHOLD;
}

export function shouldTriggerFinitenessValue(value, absThreshold = DEFAULT_FINITENESS_ABS_THRESHOLD) {
  if (!Number.isFinite(value)) {
    return true;
  }
  return Math.abs(value) > resolveFinitenessAbsThreshold(absThreshold);
}

let checkFinitenessPipeline = null;
let checkFinitenessPipelineEpoch = -1;

function getCheckFinitenessBindGroupLayout(device) {
    return getOrCreateBindGroupLayout(
        'check_finiteness_bind_group_layout',
        [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ],
        device
    );
}

async function getCheckFinitenessPipeline() {
    const epoch = getDeviceEpoch();
    if (checkFinitenessPipeline && checkFinitenessPipelineEpoch === epoch) return checkFinitenessPipeline;
    const device = getDevice();
    const shaderModule = await getShaderModule(device, 'check_finiteness.wgsl', 'check_finiteness');
    const bindGroupLayout = getCheckFinitenessBindGroupLayout(device);

    checkFinitenessPipeline = device.createComputePipeline({
        layout: getOrCreatePipelineLayout('check_finiteness_pipeline_layout', [bindGroupLayout], device),
        compute: {
            module: shaderModule,
            entryPoint: 'main',
            constants: { WORKGROUP_SIZE: 256 },
        },
    });
    checkFinitenessPipelineEpoch = epoch;

    return checkFinitenessPipeline;
}

export async function recordCheckFiniteness(
    target,
    inputBuffer,
    size,
    statusBuffer,
    layerIdx = 0,
    step = 0,
    absThreshold = DEFAULT_FINITENESS_ABS_THRESHOLD
) {
    const isRecorder = target && typeof target.beginComputePass === 'function';
    const device = isRecorder ? target.device : getDevice();
    const pipeline = await getCheckFinitenessPipeline();
    const resolvedAbsThreshold = resolveFinitenessAbsThreshold(absThreshold);

    const uniformBuffer = createUniformBufferWithView(
        'check_finiteness_uniforms',
        16,
        (view) => {
            view.setUint32(0, size, true);
            view.setUint32(4, layerIdx, true);
            view.setUint32(8, step, true);
            view.setFloat32(12, resolvedAbsThreshold, true);
        },
        isRecorder ? target : null,
        device
    );

    // Create bind group
    const bindGroup = device.createBindGroup({
        layout: getCheckFinitenessBindGroupLayout(device),
        entries: [
            { binding: 0, resource: { buffer: uniformBuffer } },
            { binding: 1, resource: { buffer: inputBuffer } },
            { binding: 2, resource: { buffer: statusBuffer } },
        ],
    });

    const workgroups = Math.ceil(size / 256);
    dispatchKernel(target, pipeline, bindGroup, workgroups, 'check_finiteness');

    // Recorder-created uniform buffers come from the uniform cache and must not
    // be destroyed as temporaries. Non-recorder path uses direct allocations.
    if (!isRecorder) {
        device.queue.onSubmittedWorkDone()
          .then(() => {
            uniformBuffer.destroy();
          })
          .catch(() => {
            uniformBuffer.destroy();
          });
    }
}
