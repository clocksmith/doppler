import assert from 'node:assert/strict';
import { createInitializedPipeline } from '../../src/inference/pipelines/factory.js';
import { createShaderSourceScope, bindStorageShaderSourceScope, getScopedShaderSource } from '../../src/gpu/kernels/shader-source-scope.js';
import { loadShaderSource } from '../../src/gpu/kernels/shader-cache.js';

const storage = {};
bindStorageShaderSourceScope(storage, createShaderSourceScope(new Map([['declared.wgsl', 'verified source']])));
let closed = 0;
class Pipeline {
  async initialize() { assert.equal(await loadShaderSource('declared.wgsl'), 'verified source'); }
  async loadModel(manifest) {
    if (manifest.fail) throw new Error('load failure');
    assert.equal(await loadShaderSource('declared.wgsl'), 'verified source');
  }
  async encodeSequence() { return loadShaderSource('declared.wgsl'); }
  async unload() { closed++; assert.equal(await loadShaderSource('declared.wgsl'), 'verified source'); }
}
const pipeline = await createInitializedPipeline(Pipeline, {}, { storage });
assert.equal(getScopedShaderSource('declared.wgsl'), null);
assert.equal(await pipeline.encodeSequence(), 'verified source');
await pipeline.unload();
await assert.rejects(createInitializedPipeline(Pipeline, { fail: true }, { storage }), /load failure/);
assert.equal(closed, 2);
class CleanupFailure extends Pipeline {
  async unload() { throw new Error('cleanup failure'); }
}
await assert.rejects(createInitializedPipeline(CleanupFailure, { fail: true }, { storage }), /load failure/);
assert.equal(getScopedShaderSource('declared.wgsl'), null);
console.log('pipeline-shader-scope.test: ok (contract doubles)');
