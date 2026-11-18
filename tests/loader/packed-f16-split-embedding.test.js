import assert from 'node:assert/strict';

import { setRuntimeConfig, resetRuntimeConfig } from '../../src/config/runtime.js';
import { setDevice } from '../../src/gpu/device.js';
import { loadEmbeddings } from '../../src/loader/embedding-loader.js';
import { _loadEmbeddings } from '../../src/loader/model-executor.js';
import { resolveEmbeddingKernel, resolveManifestGpuResidentEmbeddingLimitError } from '../../src/loader/embedding-limit-preflight.js';
import { loadFloat } from '../../src/loader/tensors/tensor-loader.js';
import { destroyBufferPool, getBufferPool } from '../../src/memory/buffer-pool.js';

globalThis.GPUBufferUsage ??= {
  MAP_READ: 0x0001,
  MAP_WRITE: 0x0002,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  INDEX: 0x0010,
  VERTEX: 0x0020,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
  INDIRECT: 0x0100,
  QUERY_RESOLVE: 0x0200,
};

class FakeBuffer {
  constructor({ size, usage, label = '' }) {
    this.size = size;
    this.usage = usage;
    this.label = label;
    this.destroyed = false;
  }

  destroy() {
    this.destroyed = true;
  }
}

function createTinyStorageLimitDevice({
  maxStorageBufferBindingSize = 128,
  maxBufferSize = 128,
  maxStorageBuffersPerShaderStage = 8,
} = {}) {
  return {
    lost: new Promise(() => {}),
    queue: {
      submit() {},
      writeBuffer() {},
      onSubmittedWorkDone() {
        return Promise.resolve();
      },
    },
    features: new Set(),
    limits: {
      maxStorageBufferBindingSize,
      maxBufferSize,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupSizeY: 1,
      maxComputeWorkgroupSizeZ: 1,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupStorageSize: 16384,
      maxStorageBuffersPerShaderStage,
      maxUniformBufferBindingSize: 65536,
      maxComputeWorkgroupsPerDimension: 65535,
      minStorageBufferOffsetAlignment: 16,
    },
    createBuffer({ size, usage, label }) {
      return new FakeBuffer({ size, usage, label });
    },
    createBindGroup() {
      return {};
    },
    createShaderModule() {
      return {};
    },
    createCommandEncoder() {
      return {
        finish() {
          return {};
        },
      };
    },
  };
}

const name = 'model.embed_tokens.weight';
const buffers = [];
const device = createTinyStorageLimitDevice({ maxStorageBufferBindingSize: 64, maxBufferSize: 64 });
const create = device.createBuffer;
device.createBuffer = options => { assert.ok(options.size <= 64); const buffer = create(options); buffers.push(buffer); return buffer; };
const writes = [];
device.queue.writeBuffer = (buffer, offset, bytes) => writes.push(new Uint8Array(bytes.buffer ?? bytes, bytes.byteOffset ?? 0, bytes.byteLength).slice());
const bytes = Uint8Array.from({ length: 160 }, (_, i) => i);
const kernel = { kernel: 'gather_split4_packed_f16_vec4_f32_out.wgsl', entry: 'gather_vec4_f32_out' };
const location = { shape: [5, 16], dtype: 'F16', role: 'embedding', group: 'embed', layout: 'row', spans: [{ shardIndex: 0, offset: 0, size: 70 }, { shardIndex: 1, offset: 0, size: 90 }], size: 160 };
let fail = false, reads = 0;
const context = () => ({ tensorLocations: new Map([[name, location]]), gpuBuffers: new Set(), hostHasShaderF16: false,
  embeddingKernel: kernel, resolveWeightLayout: loc => loc.layout, shouldStreamLargeWeight: () => false,
  loadTensor() { throw new Error('Whole-table materialization is forbidden'); },
  async loadShardRange(shard, offset, size) { reads++; if (fail && reads === 2) throw new Error('Read failed'); const start = (shard === 1 ? 70 : 0) + offset; return bytes.slice(start, start + size).buffer; } });
try {
  destroyBufferPool(); setDevice(device, { platformConfig: null }); getBufferPool().configure({ enablePooling: false });
  setRuntimeConfig({ inference: { largeWeights: { enabled: true, safetyRatio: 1, gpuResidentOverrides: [name] } } });
  const ctx = context(); const value = await loadEmbeddings(ctx);
  assert.equal(value.metadata.packing, 'packed-f16-u32');
  assert.equal(value.metadata.splitGatherSectionCount, 4);
  assert.deepEqual(value.sections.map(s => [s.rowStart, s.rowCount]), [[0, 2], [2, 2], [4, 1]]);
  assert.equal(value.dtype, 'f16'); assert.equal(ctx.gpuBuffers.size, 3);
  assert.deepEqual(Buffer.concat(writes.map(x => Buffer.from(x))), Buffer.from(bytes));
  const manifest = { inference: { execution: { kernels: { embed: { kernel: 'gather_f16.wgsl', entry: 'main' } } }, largeWeights: { gpuResidentOverrides: [name] } }, tensors: { [name]: location } };
  const beforeManifest = JSON.stringify(manifest);
  const resolvedPath = { preLayer: [{ op: 'embed', ...kernel }] };
  const actualOwner = { manifest, tensorLocations: ctx.tensorLocations, gpuBuffers: new Set(), gpuCapabilities: { hasF16: false },
    representationPaths: [resolvedPath], shardCache: { loadRange: ctx.loadShardRange }, _loadTensor: ctx.loadTensor,
    _shouldStreamLargeWeight: ctx.shouldStreamLargeWeight, _resolveWeightLayout: ctx.resolveWeightLayout };
  await _loadEmbeddings.call(actualOwner, null);
  assert.equal(actualOwner.embeddings.metadata.packing, 'packed-f16-u32', 'Actual model executor uses resolved path');
  assert.equal(JSON.stringify(manifest), beforeManifest, 'Original manifest remains unchanged');
  assert.equal(resolveManifestGpuResidentEmbeddingLimitError(manifest, { resolvedKernelPaths: [resolvedPath] }), null);
  assert.deepEqual(resolveEmbeddingKernel(manifest, [{ decode: [] }]), manifest.inference.execution.kernels.embed);
  assert.throws(() => resolveEmbeddingKernel(manifest, [{ preLayer: [] }]), /exactly one/);
  assert.throws(() => resolveEmbeddingKernel(manifest, [resolvedPath, { decode: [] }]), /exactly one/);
  assert.throws(() => resolveEmbeddingKernel(manifest, [{ preLayer: [...resolvedPath.preLayer, ...resolvedPath.preLayer] }]), /exactly one/);
  assert.throws(() => resolveEmbeddingKernel(manifest, [resolvedPath, { preLayer: [{ op: 'embed', kernel: 'gather.wgsl', entry: 'main' }] }]), /disagree/);

  const firstFailureBuffer = buffers.length; reads = 0; fail = true;
  await assert.rejects(loadEmbeddings(context()), /Read failed/);
  assert.ok(buffers.slice(firstFailureBuffer).every(b => b.destroyed));
  fail = false;
  location.shape = [5, 15];
  await assert.rejects(loadEmbeddings(context()), /aligned width/);
  location.shape = [20, 16];
  await assert.rejects(loadEmbeddings(context()), /no materialization fallback/);
  location.shape = [5, 16]; device.limits.maxStorageBuffersPerShaderStage = 5;
  await assert.rejects(loadEmbeddings(context()), /no materialization fallback/);
} finally { destroyBufferPool(); resetRuntimeConfig(); setDevice(null); }
console.log('packed-f16-split-embedding: passed');
