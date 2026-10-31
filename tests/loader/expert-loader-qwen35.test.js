import assert from 'node:assert/strict';
import { loadExpert } from '../../src/loader/experts/expert-loader.js';
const gateUp = { __dopplerFakeGPUBuffer: true, size: 128, usage: 0, destroy() {} };
const down = { __dopplerFakeGPUBuffer: true, size: 64, usage: 0, destroy() {} };
const entries = [['model.language_model.layers.3.mlp.experts.gate_up_proj', gateUp],
  ['model.language_model.layers.3.mlp.experts.down_proj', down]];
let loads = 0;
const cached = new Set();
const shards = [];
const ctx = { manifest: { modelId: 'qwen35-fixture', moeConfig: {
  numExperts: 256, numExpertsPerToken: 8, expertFormat: 'qwen3_5', expertIntermediateSize: 512,
  sharedExpertIntermediateSize: 512,
} }, experts: new Map(), expertCache: null, gpuBuffers: new Set(), keepF32Weights: true,
 tensorLocations: new Map(entries.map(([name], i) => [name, { shardIndex: i, spans: i === 0 ? [{ shardIndex: 0 }, { shardIndex: 2 }] : undefined }])),
 shardCache: { has: i => cached.has(i) },
 async loadShard(i) { shards.push(i); cached.add(i); return new ArrayBuffer(0); },
 async loadTensor(name) { loads++; return new Map(entries).get(name) ?? null; },
};
const first = await loadExpert(ctx, 3, 200);
assert.equal(first.expertFormat, 'qwen3_5');
assert.equal(first.expertIdx, 200);
assert.equal(first.gateUp, gateUp);
assert.equal(first.down, down);
assert.equal(first.expertIntermediateSize, 512);
assert.deepEqual(shards, [0, 2, 1]);
const count = loads;
cached.clear(); // Simulate bounded raw-shard eviction after bank preparation.
const shardLoads = shards.length;
assert.equal((await loadExpert(ctx, 3, 12)).gateUp, gateUp);
assert.equal(loads, count, 'all selected experts share one packed tensor load');
assert.equal(shards.length, shardLoads, 'resident packed banks do not reload evicted raw shards');
await assert.rejects(loadExpert(ctx, 3, 256), /outside/);
await assert.rejects(loadExpert(ctx, 3, -1), /outside/);
const broken = { ...ctx, experts: new Map(), async loadTensor() { return null; } };
await assert.rejects(loadExpert(broken, 3, 0), /missing tensors/);
assert.equal(broken.experts.size, 0, 'failed packed preparations are not cached');
console.log('expert-loader-qwen35.test: ok');
