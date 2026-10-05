import assert from 'node:assert/strict';
import { setDevice } from '../../src/gpu/device.js';
import { CommandRecorder } from '../../src/gpu/command-recorder.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { recordRMSNorm } from '../../src/gpu/kernels/rmsnorm.js';
import { getBufferPool, releaseBuffer, destroyBufferPool } from '../../src/memory/buffer-pool.js';
import { registerShaderSources, clearShaderCaches } from '../../src/gpu/kernels/shader-cache.js';
import { clearPipelineCaches } from '../../src/gpu/kernels/pipeline-cache.js';

globalThis.GPUBufferUsage = { UNIFORM: 64, STORAGE: 128, COPY_SRC: 4, COPY_DST: 8, MAP_READ: 1 };
globalThis.GPUShaderStage = { COMPUTE: 2 };

for (const outcome of ['submit', 'abort', 'bind-failure']) {
  const buffers = [];
  const device = {
    features: new Set(), limits: { maxBufferSize: 1 << 20, maxStorageBufferBindingSize: 1 << 20,
      maxComputeWorkgroupsPerDimension: 65535, maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256, maxComputeWorkgroupSizeY: 1, maxComputeWorkgroupSizeZ: 1 },
    queue: { writeBuffer() {}, onSubmittedWorkDone: async () => {}, submit(commands) {
      for (const command of commands) for (const group of command.groups) for (const entry of group.entries) {
        assert.equal(entry.resource.buffer.destroyed, false, 'Every recorded binding must survive until submission');
      }
    } },
    createBuffer(descriptor) {
      const buffer = { ...descriptor, destroyed: false, destroy() { this.destroyed = true; } };
      buffers.push(buffer); return buffer;
    },
    createShaderModule: descriptor => descriptor,
    createBindGroupLayout: () => ({}), createPipelineLayout: () => ({}),
    async createComputePipelineAsync() { return { getBindGroupLayout: () => ({}) }; },
    createBindGroup(descriptor) {
      if (outcome === 'bind-failure') throw Error('injected binding failure');
      return descriptor;
    },
    createCommandEncoder() {
      const groups = [];
      return { beginComputePass: () => ({ setPipeline() {},
        setBindGroup: (_index, group) => groups.push(group), dispatchWorkgroups() {}, end() {} }),
      finish: () => ({ groups }) };
    },
  };
  destroyBufferPool(); clearPipelineCaches(); clearShaderCaches();
  setDevice(device, { platformConfig: null });
  registerShaderSources({ 'rmsnorm.wgsl': '@compute @workgroup_size(256) fn main() {}' });
  const recorder = new CommandRecorder(device, 'recorded_binding_lifetime', { profile: false });
  const borrowed = { size: 16, destroyed: false, destroy() { this.destroyed = true; } };
  const input = createTensor(borrowed, 'f32', [1, 4], 'input');
  let output;
  try {
    if (outcome === 'bind-failure') {
      await assert.rejects(recordRMSNorm(recorder, input, borrowed, 1e-6,
        { hiddenSize: 4, batchSize: 1 }), /injected binding failure/);
    } else {
      output = await recordRMSNorm(recorder, input, borrowed, 1e-6, { hiddenSize: 4, batchSize: 1 });
      // Memory pressure may evict every idle pooled buffer before this chunk submits.
      getBufferPool().clearPool();
      await new Promise(resolve => setImmediate(resolve));
      const placeholder = buffers.find(buffer => buffer.label?.startsWith('rmsnorm_prenorm_placeholder'));
      assert.equal(placeholder.destroyed, false, 'A recorded placeholder is submit-owned, never idle pooled storage');
      if (outcome === 'submit') await recorder.submitAndWait();
    }
  } finally {
    await recorder.abort();
    if (output) releaseBuffer(output.buffer);
    assert.equal(getBufferPool().getStats().activeBuffers, 0, `${outcome}: all owned buffers settle`);
    assert.equal(borrowed.destroyed, false, 'Input and weight remain borrowed');
    destroyBufferPool(); clearPipelineCaches(); clearShaderCaches();
    setDevice(null, { platformConfig: null });
  }
}
console.log('rmsnorm-recorded-binding-lifetime.test: submission, eviction, abort and failure passed');
