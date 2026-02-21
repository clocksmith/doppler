import { getDevice, getDeviceEpoch } from '../device.js';
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { dispatchKernel } from './dispatch.js';
import { createUniformBufferFromData, getOrCreateBindGroupLayout, getOrCreatePipelineLayout } from './utils.js';

let checkFinitenessPipeline = null;
let checkFinitenessPipelineEpoch = -1;

const SHADER = /* wgsl */ `
enable f16;

override WORKGROUP_SIZE: u32 = 256u;

struct Uniforms {
    size: u32,
    layer: u32,
    step: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f16>;
@group(0) @binding(2) var<storage, read_write> status: array<atomic<u32>>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;
    
    if (idx >= u.size) {
        return;
    }

    let val = f32(input[idx]);
    let bits = bitcast<u32>(val);
    
    // Check for NaN or Infinity (exponent bits all 1)
    if ((bits & 0x7F800000u) == 0x7F800000u) {
        let old = atomicCompareExchangeWeak(&status[0], 0u, 1u);
        if (old.exchanged) {
            atomicStore(&status[1], u.layer);
            atomicStore(&status[2], u.step);
        }
    }
}
`;

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

function getCheckFinitenessPipeline() {
    const epoch = getDeviceEpoch();
    if (checkFinitenessPipeline && checkFinitenessPipelineEpoch === epoch) return checkFinitenessPipeline;
    const device = getDevice();
    const shaderModule = device.createShaderModule({ code: SHADER });
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

export function recordCheckFiniteness(
    target,
    inputBuffer,
    size,
    statusBuffer,
    layerIdx = 0,
    step = 0
) {
    const isRecorder = target && typeof target.beginComputePass === 'function';
    const device = isRecorder ? target.device : getDevice();
    const pipeline = getCheckFinitenessPipeline();

    // Create uniform buffer (size: 3 uints, padded by createUniformBufferFromData)
    const uniformData = new Uint32Array([size, layerIdx, step]);
    const uniformBuffer = createUniformBufferFromData('check_finiteness_uniforms', uniformData, isRecorder ? target : null, device);

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

    // Need to cleanup uniform buffer
    if (isRecorder) {
        target.trackTemporaryBuffer(uniformBuffer);
    } else {
        setTimeout(() => uniformBuffer.destroy(), 1);
    }
}
