import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { loadHyperConnectionMixerWeights } from '../../src/loader/hyper-connection-loader.js';
import { createWeightBuffer, tagBufferDtype } from '../../src/gpu/weight-buffer.js';
import { acquireBuffer, releaseBuffer, readBuffer, getBufferPool, destroyBufferPool } from '../../src/memory/buffer-pool.js';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';
import { loadTensorToGPU } from '../../src/loader/tensors/tensor-loader.js';

const gpu = await probeNodeGPU();
if (!gpu.ready) {
  assert.ok(!process.argv.includes('--require-gpu'), gpu.reason);
  console.log(`hyper-connection vector GPU test skipped: ${gpu.reason}`);
} else {
  const gpuBuffers = new Set();
  try {
    const prefix = 'model.language_model.hyper_connection_mixer';
    const tensorLocations = new Map();
    const weights = new Map();
    const add = (suffix, shape, values) => {
      const name = `${prefix}.${suffix}`;
      const buffer = acquireBuffer(values.byteLength);
      gpuBuffers.add(buffer);
      getDevice().queue.writeBuffer(buffer, 0, values);
      tagBufferDtype(buffer, 'f16');
      tensorLocations.set(name, { dtype: 'F16', role: shape.length === 1 ? 'norm' : 'other', layout: 'row', shape, size: values.byteLength });
      weights.set(name, shape.length === 1 ? buffer : createWeightBuffer(buffer, 'f16', 'row', shape, name));
      return buffer;
    };
    const original = add('hc_norm.weight', [8], new Uint16Array([0, 0x3400, 0xb800, 0x3c00, 0x4000, 0xbc00, 0x3800, 0x3e00]));
    add('input_mix_weight_down.weight', [3, 8], new Uint16Array(24));
    add('input_mix_weight_up.weight', [8, 3], new Uint16Array(24));
    const result = await loadHyperConnectionMixerWeights({ tensorLocations, gpuBuffers,
      hyperConnections: { streamCount: 2, streamWidth: 4, projectionRank: 3 },
      loadTensor: async name => {
        const location = tensorLocations.get(name);
        if (location.shape.length === 1) return weights.get(name);
        const materialized = await loadTensorToGPU(new Uint8Array(location.size), location, name, {
          gpuCapabilities: { hasF16: true }, allowF32UpcastNonMatmul: false,
          representationRequirement: { complete: true, mode: 'dense-only', dtypes: ['f32'], consumers: [{ role: 'hc_mixer_down' }] },
        });
        for (const buffer of materialized.allocatedBuffers) gpuBuffers.add(buffer);
        return materialized.data;
      } });
    assert.equal(result.norm.dtype, 'f32');
    assert.notEqual(result.norm.buffer, original);
    assert.ok(gpuBuffers.has(result.norm.buffer), 'the loader must own the converted vector');
    assert.deepEqual([...new Float32Array(await readBuffer(result.norm.buffer))].slice(0, 8), [0, .25, -.5, 1, 2, -1, .5, 1.5]);
    assert.equal(result.down.dtype, 'f32', 'the projection follows its declared F32 consumer');
    console.log('physical GPU: F16 mechanism vector -> exact F32 values with tracked ownership');
  } finally {
    await getDevice().queue.onSubmittedWorkDone();
    for (const buffer of gpuBuffers) releaseBuffer(buffer);
    await getBufferPool().settleDeferredDestruction();
    assert.equal(getBufferPool().getStats().activeBuffers, 0);
    destroyBufferPool(); destroyDevice(); await releaseNodeWebGPU();
  }
}
