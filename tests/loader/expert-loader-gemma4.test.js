import assert from 'node:assert/strict';
import { loadExpert } from '../../src/loader/experts/expert-loader.js';

const gateUp = { __dopplerFakeGPUBuffer: true, label: 'gate_up', size: 128, usage: 0, destroy() {} };
const down = { __dopplerFakeGPUBuffer: true, label: 'down', size: 64, usage: 0, destroy() {} };
const loadedNames = [];
const loadedShards = [];
const cachedShards = new Set();

const tensors = new Map([
  ['model.decoder.layers.0.experts.gate_up_proj', gateUp],
  ['model.decoder.layers.0.experts.down_proj', down],
]);

const ctx = {
  manifest: {
    modelId: 'diffusiongemma-fixture',
    moeConfig: {
      numExperts: 128,
      numExpertsPerToken: 8,
      expertFormat: 'gemma4',
      expertIntermediateSize: 4,
    },
  },
  tensorLocations: new Map([
    [
      'model.decoder.layers.0.experts.gate_up_proj',
      {
        shardIndex: 9,
        offset: 0,
        size: 128,
        shape: [128, 8, 4],
        dtype: 'Q4_K_M',
        role: 'expert',
        group: 'layer.0',
        spans: [
          { shardIndex: 2, offset: 0, size: 64 },
          { shardIndex: 3, offset: 0, size: 64 },
        ],
      },
    ],
    [
      'model.decoder.layers.0.experts.down_proj',
      {
        shardIndex: 1,
        offset: 0,
        size: 64,
        shape: [128, 4, 4],
        dtype: 'Q4_K_M',
        role: 'expert',
        group: 'layer.0',
      },
    ],
  ]),
  async loadTensor(name) {
    loadedNames.push(name);
    return tensors.get(name) ?? null;
  },
  async loadShard(shardIndex) {
    loadedShards.push(shardIndex);
    cachedShards.add(shardIndex);
    return new ArrayBuffer(0);
  },
  shardCache: {
    has(shardIndex) {
      return cachedShards.has(shardIndex);
    },
  },
  expertCache: null,
  experts: new Map(),
  gpuBuffers: new Set(),
  keepF32Weights: false,
};

const first = await loadExpert(ctx, 0, 3);
assert.equal(first.expertFormat, 'gemma4');
assert.equal(first.expertIdx, 3);
assert.equal(first.numExperts, 128);
assert.equal(first.expertIntermediateSize, 4);
assert.equal(first.gateUp, gateUp);
assert.equal(first.down, down);
assert.deepEqual([...loadedShards].sort((a, b) => a - b), [1, 2, 3]);
assert.ok(loadedNames.includes('model.decoder.layers.0.experts.gate_up_proj'));
assert.ok(loadedNames.includes('model.decoder.layers.0.experts.down_proj'));

const countAfterFirst = loadedNames.length;
const shardCountAfterFirst = loadedShards.length;
const second = await loadExpert(ctx, 0, 4);
assert.equal(second.expertFormat, 'gemma4');
assert.equal(second.expertIdx, 4);
assert.equal(second.gateUp, gateUp);
assert.equal(second.down, down);
assert.equal(loadedNames.length, countAfterFirst, 'packed Gemma tensors should be reused per layer');
assert.equal(loadedShards.length, shardCountAfterFirst, 'packed Gemma shards should be reused per layer');

console.log('expert-loader-gemma4.test: ok');
