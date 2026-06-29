import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';

globalThis.GPUBufferUsage = {
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
  constructor({ size, usage, label }) {
    this.size = size;
    this.usage = usage;
    this.label = label;
    this.bytes = new Uint8Array(size);
    this.destroyed = false;
  }

  destroy() {
    this.destroyed = true;
  }
}

globalThis.GPUBuffer = FakeBuffer;

const { setDevice } = await import('../../src/gpu/device.js');
const { destroyBufferPool, getBufferPool, releaseBuffer } = await import('../../src/memory/buffer-pool.js');
const { DEFAULT_LOADING_CONFIG } = await import('../../src/config/schema/loading.schema.js');
const { DopplerLoader } = await import('../../src/loader/doppler-loader.js');
const { loadTensorToGPU, loadTensorToCPU } = await import('../../src/loader/tensors/tensor-loader.js');

function createFakeDevice() {
  return {
    features: new Set(),
    limits: {
      maxBufferSize: 1 << 20,
      maxStorageBufferBindingSize: 1 << 20,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupStorageSize: 16384,
    },
    queue: {
      submit() {},
      onSubmittedWorkDone() {
        return Promise.resolve();
      },
      writeBuffer(buffer, offset, data) {
        const bytes = data instanceof Uint8Array
          ? data
          : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        buffer.bytes.set(bytes, offset);
      },
    },
    createBuffer(descriptor) {
      return new FakeBuffer(descriptor);
    },
  };
}

function resetRuntimeState(device) {
  destroyBufferPool();
  setDevice(device, { platformConfig: null });
  if (device) {
    getBufferPool().configure({ enablePooling: false });
  }
}

function u32(value) {
  const bytes = new Uint8Array(4);
  new DataView(bytes.buffer).setUint32(0, value, true);
  return bytes;
}

function i32Array(values) {
  return new Uint8Array(new Int32Array(values).buffer);
}

function f32Array(values) {
  return new Uint8Array(new Float32Array(values).buffer);
}

function concatBytes(chunks) {
  const total = chunks.reduce((sum, chunk) => sum + chunk.byteLength, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return out;
}

function sha256Digest(chunks) {
  const hash = createHash('sha256');
  for (const chunk of chunks) {
    hash.update(chunk);
  }
  return `sha256:${hash.digest('hex')}`;
}

function cloneLoadingConfig() {
  return JSON.parse(JSON.stringify(DEFAULT_LOADING_CONFIG));
}

function serializeZeroSiren() {
  return concatBytes([
    u32(1),
    u32(2),
    u32(1),
    f32Array([0, 0]),
    f32Array([0]),
  ]);
}

function serializeSparse(entries) {
  return concatBytes([
    u32(entries.length),
    i32Array(entries.map((entry) => entry.row)),
    i32Array(entries.map((entry) => entry.col)),
    f32Array(entries.map((entry) => entry.value)),
  ]);
}

const descriptorManifest = {
  schema_version: 'manifoldgguf.v0.1',
  slice_shape: [1, 2],
  crop_shape: [1, 2],
  padded_shape: [2, 2],
  storage_type: 'functional_descriptor',
  components: {
    prng_substrate: {
      algorithm: 'coord_hash_normal_v1',
      seed: 7,
      learned_scale: 0,
    },
    kronecker_sum: {
      shard_file: 'tiny.kron',
    },
    coordinate_inr: {
      type: 'siren',
      shard_file: 'tiny.siren',
    },
    sparse_outliers: {
      shard_file: 'tiny.sparse',
    },
  },
};

resetRuntimeState(createFakeDevice());

const shardData = new Uint8Array(0);
Object.defineProperty(shardData, 'descriptorShards', {
  value: new Map([
    ['tiny.kron', u32(0)],
    ['tiny.siren', serializeZeroSiren()],
    ['tiny.sparse', serializeSparse([
      { row: 0, col: 0, value: 1 },
      { row: 0, col: 1, value: 2 },
      { row: 1, col: 0, value: 3 },
      { row: 1, col: 1, value: 4 },
    ])],
  ]),
});

const result = await loadTensorToGPU(
  shardData,
  {
    dtype: 'FUNCTIONAL_DESCRIPTOR',
    role: 'matmul',
    shape: [1, 2],
    size: 0,
    descriptorManifest,
  },
  'model.layers.0.mlp.down_proj.weight',
  {
    useFusedQ4K: false,
    keepF32Weights: true,
    allowF32UpcastNonMatmul: false,
    q4kLayout: 'row',
    gpuCapabilities: { hasF16: false, hasSubgroups: false },
  }
);

assert.equal(result.data.dtype, 'f32');
assert.equal(result.data.layout, 'row');
assert.deepEqual(result.data.shape, [1, 2]);
assert.equal(result.allocatedBuffers.length, 1);
assert.equal(result.allocatedBuffers[0], result.data.buffer);
assert.deepEqual(
  Array.from(new Float32Array(result.data.buffer.bytes.buffer, 0, 2)),
  [1, 2]
);

assert.throws(
  () => loadTensorToCPU(new Uint8Array(0), { dtype: 'FUNCTIONAL_DESCRIPTOR' }),
  /FUNCTIONAL_DESCRIPTOR tensor/
);

releaseBuffer(result.allocatedBuffers[0]);
resetRuntimeState(null);

{
  const device = createFakeDevice();
  resetRuntimeState(device);

  const kronShard = u32(0);
  const sirenShard = serializeZeroSiren();
  const sparseShard = serializeSparse([
    { row: 0, col: 0, value: 9 },
    { row: 0, col: 1, value: 10 },
  ]);
  const descriptorBytes = kronShard.byteLength + sirenShard.byteLength + sparseShard.byteLength;
  const denseF16Bytes = 64 * 64 * 2;
  const loaderDescriptorManifest = {
    schema_version: 'manifoldgguf.v0.1',
    slice_shape: [64, 64],
    crop_shape: [64, 64],
    padded_shape: [64, 64],
    storage_type: 'functional_descriptor',
    descriptor_hash: sha256Digest([kronShard, sirenShard, sparseShard]),
    descriptor_bytes: descriptorBytes,
    dense_f16_bytes: denseF16Bytes,
    compression_ratio: denseF16Bytes / descriptorBytes,
    proof_status: 'passed',
    proof_status_gate: {
      sensitivity: 'passed',
      compression: 'passed',
      determinism: 'passed',
    },
    components: {
      prng_substrate: {
        algorithm: 'coord_hash_normal_v1',
        seed: 11,
        learned_scale: 0,
        learned_scale_frozen: true,
      },
      kronecker_sum: {
        shard_file: 'loader.kron',
      },
      coordinate_inr: {
        type: 'siren',
        shard_file: 'loader.siren',
      },
      sparse_outliers: {
        shard_file: 'loader.sparse',
        actual_nnz: 2,
      },
    },
  };

  const loader = new DopplerLoader(cloneLoadingConfig());
  loader.gpuCapabilities = { hasF16: false, hasSubgroups: false };
  loader.tensorLocations.set('model.layers.0.mlp.down_proj.weight', {
    dtype: 'FUNCTIONAL_DESCRIPTOR',
    role: 'matmul',
    shape: [64, 64],
    size: 0,
    descriptorManifest: loaderDescriptorManifest,
  });
  const auxiliaryFiles = new Map([
    ['loader.kron', kronShard],
    ['loader.siren', sirenShard],
    ['loader.sparse', sparseShard],
  ]);
  loader.setAuxiliaryFileLoader(async (path) => auxiliaryFiles.get(path) ?? null);

  const weight = await loader.loadTensor('model.layers.0.mlp.down_proj.weight', true, false);
  assert.equal(weight.dtype, 'f32');
  assert.equal(weight.metadata.storageType, 'functional_descriptor');
  assert.equal(weight.metadata.descriptorHash, loaderDescriptorManifest.descriptor_hash);
  assert.equal(weight.metadata.descriptorBytes, descriptorBytes);
  assert.equal(weight.metadata.denseF16Bytes, denseF16Bytes);
  assert.equal(weight.metadata.proofStatus, 'passed');
  assert.deepEqual(
    Array.from(new Float32Array(weight.buffer.bytes.buffer, 0, 2)),
    [9, 10]
  );
  assert.equal(loader.gpuBuffers.size, 1);

  await loader.unload();
  resetRuntimeState(null);
}

console.log('functional-descriptor-loader.test: ok');
