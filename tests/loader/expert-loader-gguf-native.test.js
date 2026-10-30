import assert from 'node:assert/strict';
import { loadExpert } from '../../src/loader/experts/expert-loader.js';
const buffer = size => ({ __dopplerFakeGPUBuffer: true, size, usage: 0, destroy() {} });
const weight = (dtype, shape, size) => ({ buffer: buffer(size), dtype, layout: 'row', shape });
const gate = weight('q4_0', [2, 32, 32], 2 * 32 * 18), up = weight('q4_0', [2, 32, 32], 2 * 32 * 18);
const down = weight('q4_1', [2, 32, 32], 2 * 32 * 20);
const values = new Map([['blk.0.ffn_gate_exps.weight', gate], ['blk.0.ffn_up_exps.weight', up], ['blk.0.ffn_down_exps.weight', down]]);
const shards = [];
const ctx = { manifest: { modelId: 'native-fixture', architecture: { hiddenSize: 32 }, moeConfig: {
 numExperts: 2, numExpertsPerToken: 1, expertFormat: 'qwen3_5', expertIntermediateSize: 32, sharedExpertIntermediateSize: 32,
 } }, experts: new Map(), expertCache: null, gpuBuffers: new Set(), keepF32Weights: true,
 tensorLocations: new Map([...values].map(([name], index) => [name, { shardIndex: index }])), shardCache: { has: () => false },
 async loadShard(index) { shards.push(index); return new ArrayBuffer(0); },
 async loadTensor(name) { return values.get(name) ?? null; },
};
const loaded = await loadExpert(ctx, 0, 1);
assert.equal(loaded.projectionLayout, 'separate');
assert.equal(loaded.gate, gate); assert.equal(loaded.up, up); assert.equal(loaded.down, down);
assert.deepEqual(new Set(shards), new Set([0, 1, 2]));
const broken = { ...ctx, experts: new Map(), async loadTensor(name) { return name.endsWith('ffn_down_exps.weight') ? { ...down, shape: [1, 64, 32] } : values.get(name) ?? null; } };
await assert.rejects(loadExpert(broken, 0, 0), /geometry/);
assert.equal(broken.experts.size, 0);
console.log('expert-loader-gguf-native.test: ok');
