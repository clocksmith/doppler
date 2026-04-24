import assert from 'node:assert/strict';
import fs from 'node:fs';

import {
  projectAttentionQKV,
  resolveProjectionSliceOffsetBytes,
} from '../../src/inference/pipelines/text/attention/projections.js';
import { resolveAttentionNumKVHeads } from '../../src/inference/pipelines/text/layer.js';

function makeWeight(dtype, layout = 'row', shape = [2048, 1024]) {
  return {
    buffer: {},
    dtype,
    layout,
    shape,
  };
}

{
  const q4kWeight = makeWeight('q4k', 'row', [4096, 1024]);
  const offset = resolveProjectionSliceOffsetBytes(q4kWeight, 2048, 1024);
  // Q4K packs 256 logical elements into 144 bytes.
  const expected = 2048 * Math.ceil(1024 / 256) * 144;
  assert.equal(offset, expected);
}

{
  const f16Weight = makeWeight('f16', 'row', [4096, 1024]);
  const offset = resolveProjectionSliceOffsetBytes(f16Weight, 2048, 1024);
  assert.equal(offset, 2048 * 1024 * 2);
}

{
  const f32Weight = makeWeight('f32', 'row', [4096, 1024]);
  const offset = resolveProjectionSliceOffsetBytes(f32Weight, 2048, 1024);
  assert.equal(offset, 2048 * 1024 * 4);
}

{
  const q4kColWeight = makeWeight('q4k', 'column', [4096, 1024]);
  assert.throws(
    () => resolveProjectionSliceOffsetBytes(q4kColWeight, 2048, 1024),
    /unsupported q4k layout/i
  );
}

{
  const offset = resolveProjectionSliceOffsetBytes(makeWeight('q4k'), 0, 1024);
  assert.equal(offset, 0);
}

{
  const config = { hiddenSize: 5376, numKVHeads: 16, numGlobalKVHeads: 4 };
  assert.equal(
    resolveAttentionNumKVHeads(config, 'full_attention', { kProj: { shape: [2048, 5376] } }, 512),
    4
  );
  assert.equal(
    resolveAttentionNumKVHeads(config, 'sliding_attention', { kProj: { shape: [4096, 5376] } }, 256),
    16
  );
  assert.equal(resolveAttentionNumKVHeads(config, 'full_attention', {}, 512), 4);
}

{
  const planSource = fs.readFileSync('src/inference/pipelines/text/layer-plan-gpu.js', 'utf8');
  assert.match(
    planSource,
    /resolveAttentionNumKVHeads\(config, layerType, layerWeights, attentionHeadDim\)/,
    'execution-plan attention must resolve per-layer KV heads from layer weights'
  );
  assert.match(
    planSource,
    /numKVHeads:\s*attentionNumKVHeads/,
    'execution-plan attention must pass resolved per-layer KV heads'
  );
}

{
  await assert.rejects(
    () => projectAttentionQKV({
      recorder: null,
      normed: { buffer: { size: 16 }, dtype: 'f32', shape: [1, 4] },
      layerWeights: { kProj: {}, vProj: {} },
      numTokens: 1,
      numHeads: 1,
      numKVHeads: 1,
      headDim: 4,
      hiddenSize: 4,
      layerIdx: 0,
      kernelPath: null,
      matmulOutputDtype: 'f32',
      getWeightBuffer: null,
      lora: null,
      releaseTemporary() {},
    }),
    /Attention projection requires qProj/
  );
}

console.log('attention-projection-offsets.test: ok');
