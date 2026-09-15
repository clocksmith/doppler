import assert from 'node:assert/strict';
import { setDevice } from '../../src/gpu/device.js';
import { getBufferPool, releaseBuffer, destroyBufferPool } from '../../src/memory/buffer-pool.js';
import { runRMSNormStats, recordRMSNormStats } from '../../src/gpu/kernels/rmsnorm-stats.js';
import { clearShaderCaches, registerShaderSources } from '../../src/gpu/kernels/shader-cache.js';
import { clearPipelineCaches } from '../../src/gpu/kernels/pipeline-cache.js';

globalThis.GPUBufferUsage = { UNIFORM: 64, STORAGE: 128, COPY_SRC: 4, COPY_DST: 8, MAP_READ: 1 };
for (const recorded of [false, true]) {
  for (const failure of ['inv_rms', 'uniform', 'compile', 'bind', 'dispatch', null]) {
    for (const borrowed of [false, true]) {
      const buffers = [], recordedUniforms = [];
      const fail = stage => { if (failure === stage) throw new Error(`injected ${stage}`); };
      const device = {
        features: new Set(), limits: { maxBufferSize: 1 << 20, maxStorageBufferBindingSize: 1 << 20,
          maxComputeWorkgroupsPerDimension: 65535 },
        queue: { writeBuffer() {}, submit() { fail('dispatch'); }, onSubmittedWorkDone: async () => {} },
        createBuffer(descriptor) {
          if (descriptor.label?.startsWith('rmsnorm_stats_inv_rms')) fail('inv_rms');
          const buffer = { ...descriptor, destroyed: false, destroy() { this.destroyed = true; } };
          buffers.push(buffer); return buffer;
        },
        createShaderModule: descriptor => descriptor,
        async createComputePipelineAsync() { fail('compile'); return { getBindGroupLayout: () => ({}) }; },
        createBindGroup() { fail('bind'); return {}; },
        createCommandEncoder: () => ({ finish: () => ({}), beginComputePass: () => ({
          setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {},
        }) }),
      };
      const writeBuffer = device.queue.writeBuffer;
      device.queue.writeBuffer = (buffer, ...args) => {
        if (buffer.label === 'rmsnorm_stats_uniforms') fail('uniform');
        writeBuffer(buffer, ...args);
      };
      destroyBufferPool(); setDevice(device, { platformConfig: null });
      clearShaderCaches(); clearPipelineCaches();
      registerShaderSources({ 'rmsnorm_stats.wgsl': '@compute @workgroup_size(256) fn main() {}' });
      const recorder = { device, createUniformBuffer(data, label) {
        fail('uniform');
        const buffer = device.createBuffer({ label, size: data.byteLength, usage: GPUBufferUsage.UNIFORM });
        recordedUniforms.push(buffer); return buffer;
      }, recordDispatch() { fail('dispatch'); } };
      const external = { size: 256, destroyed: false, destroy() { this.destroyed = true; } };
      const input = { dtype: 'f32', buffer: external };
      const options = { batchSize: 2, hiddenSize: 32, ...(borrowed ? { outputBuffer: external } : {}) };
      const run = () => recorded
        ? recordRMSNormStats(recorder, input, input, 1e-5, options)
        : runRMSNormStats(input, input, 1e-5, options);
      try {
        if (failure) await assert.rejects(run(), new RegExp(`injected ${failure}`));
        else {
          const result = await run();
          if (!borrowed) releaseBuffer(result.prenormSum.buffer);
          releaseBuffer(result.invRmsBuffer);
        }
        assert.equal(getBufferPool().getStats().activeBuffers, 0, `${recorded}:${failure}:${borrowed}: pooled output leak`);
        assert.equal(external.destroyed, false, 'borrowed input/output never destroyed');
        for (const buffer of buffers.filter(buffer => buffer.label === 'rmsnorm_stats_uniforms')) {
          assert.equal(buffer.destroyed, !recorded, 'direct uniform released; recorder uniform retains its owner');
        }
      } finally {
        for (const buffer of recordedUniforms) buffer.destroy();
        destroyBufferPool(); clearPipelineCaches(); clearShaderCaches(); setDevice(null, { platformConfig: null });
      }
    }
  }
}
console.log('rmsnorm-stats-ownership.test: 24 allocation/compilation/dispatch/ownership cases passed');
