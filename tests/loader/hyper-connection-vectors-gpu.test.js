import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { loadHyperConnectionLayerWeights, loadHyperConnectionMixerWeights } from '../../src/loader/hyper-connection-loader.js';
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
    // Source GGUF HC injection/indexer projections are BF16 with generic
    // source role "other". The bound graph, not the source heuristic, owns
    // whether materialization produces a declared projection WeightBuffer.
    const bf16Values = new Uint16Array([0x3f80, 0x4000, 0x4040, 0x4080]);
    const bf16Projection = await loadTensorToGPU(new Uint8Array(bf16Values.buffer), {
      dtype: 'BF16', role: 'other', layout: 'row', shape: [2, 2], size: bf16Values.byteLength,
    }, 'blk.0.hc_attn_inject.weight', {
      gpuCapabilities: { hasF16: true }, keepF32Weights: true,
      representationRequirement: { complete: true, mode: 'dense-only', dtypes: ['f32'], consumers: [{ role: 'attn_hc_inject' }] },
    });
    for (const buffer of bf16Projection.allocatedBuffers) gpuBuffers.add(buffer);
    assert.equal(bf16Projection.data.dtype, 'f32');
    assert.equal(bf16Projection.data.layout, 'row');
    assert.deepEqual(bf16Projection.data.shape, [2, 2]);
    assert.deepEqual([...new Float32Array(await readBuffer(bf16Projection.data.buffer, 16))], [1, 2, 3, 4]);
    // Native split Q/K rows must fuse once, in Q-then-K order, without readback.
    const locations = new Map();
    const nativeWeights = new Map();
    const addNative = (name, shape, values, projection = true) => {
      const buffer = acquireBuffer(values.byteLength);
      gpuBuffers.add(buffer);
      getDevice().queue.writeBuffer(buffer, 0, values);
      tagBufferDtype(buffer, 'f32');
      locations.set(name, { dtype: 'F32', shape, size: values.byteLength, layout: 'row' });
      nativeWeights.set(name, projection ? createWeightBuffer(buffer, 'f32', 'row', shape, name) : buffer);
    };
    for (const branch of ['attn', 'ffn']) {
      for (const [key, shape] of [['norm', [8]], ['down', [3, 8]], ['up', [8, 3]], ['inject', [2, 8]]]) {
        addNative(`blk.1.hc_${branch}_${key}.weight`, shape, new Float32Array(shape.reduce((a, b) => a * b, 1)), key !== 'norm');
      }
    }
    addNative('blk.1.indexer.q_proj.weight', [4, 4], Float32Array.from({ length: 16 }, (_, i) => i));
    addNative('blk.1.indexer.k_proj.weight', [2, 4], Float32Array.from({ length: 8 }, (_, i) => 100 + i));
    addNative('blk.1.indexer.q_norm.weight', [2], new Float32Array(2), false);
    addNative('blk.1.indexer.k_norm.weight', [2], new Float32Array(2), false);
    const joined = await loadHyperConnectionLayerWeights({ tensorLocations: locations, gpuBuffers,
      hyperConnections: { streamCount: 2, streamWidth: 4, projectionRank: 3 },
      indexedAttention: { layers: [1], queryHeads: 2, keyHeads: 1, headDim: 2 },
      loadTensor: async name => nativeWeights.get(name) }, 1);
    assert.deepEqual(joined.qsaIndexer.projection.shape, [6, 4]);
    assert.deepEqual([...new Float32Array(await readBuffer(joined.qsaIndexer.projection.buffer, 24 * 4))],
      [...Array.from({ length: 16 }, (_, i) => i), ...Array.from({ length: 8 }, (_, i) => 100 + i)]);
    assert.ok(gpuBuffers.has(joined.qsaIndexer.projection.buffer));
    console.log('physical GPU: F16 mechanism vector -> exact F32 values with tracked ownership');
  } finally {
    await getDevice().queue.onSubmittedWorkDone();
    for (const buffer of gpuBuffers) releaseBuffer(buffer);
    await getBufferPool().settleDeferredDestruction();
    assert.equal(getBufferPool().getStats().activeBuffers, 0);
    destroyBufferPool(); destroyDevice(); await releaseNodeWebGPU();
  }
}
