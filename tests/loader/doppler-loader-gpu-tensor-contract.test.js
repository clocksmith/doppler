import assert from 'node:assert/strict';

import { DopplerLoader } from '../../src/loader/doppler-loader.js';
import { tagBufferDtype } from '../../src/gpu/weight-buffer.js';

function createFakeBuffer(size) {
  return {
    __dopplerFakeGPUBuffer: true,
    size,
    usage: 0,
    destroy() {},
  };
}

const name = 'model.vision_tower.patch_embedder.position_embedding_table';

{
  const buffer = createFakeBuffer(96);
  tagBufferDtype(buffer, 'f16');
  const loader = Object.create(DopplerLoader.prototype);
  loader.tensorLocations = new Map([[name, {
    shape: [2, 3, 8],
    dtype: 'F16',
  }]]);
  loader._loadTensor = async () => buffer;

  const tensor = await loader.loadGpuTensor(name, true);
  assert.equal(tensor.buffer, buffer);
  assert.equal(tensor.dtype, 'f16');
  assert.deepEqual(tensor.shape, [2, 3, 8]);
  assert.equal(Object.isFrozen(tensor.shape), true);
}

{
  const buffer = createFakeBuffer(96);
  const loader = Object.create(DopplerLoader.prototype);
  loader.tensorLocations = new Map([[name, {
    shape: [2, 3, 8],
    dtype: 'F16',
  }]]);
  loader._loadTensor = async () => buffer;

  await assert.rejects(
    () => loader.loadGpuTensor(name, true),
    /requires f16 or f32 data/
  );
}

{
  const loader = Object.create(DopplerLoader.prototype);
  loader.tensorLocations = new Map();
  loader._loadTensor = async () => null;
  assert.equal(await loader.loadGpuTensor(name, true), null);
}

console.log('doppler-loader-gpu-tensor-contract.test: ok');
