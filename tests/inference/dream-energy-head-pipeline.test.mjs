import assert from 'node:assert/strict';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';

installNodeFileFetchShim();
if (!globalThis.GPUBufferUsage) {
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
}
if (!globalThis.GPUMapMode) {
  globalThis.GPUMapMode = { READ: 0x0001, WRITE: 0x0002 };
}

const { createPipeline } = await import('../../src/inference/pipelines/text.js');

{
  const manifest = {
    modelType: 'd1-to2-bridge-diffusion',
    modelId: 'dream-bridge-test',
    featureIds: ['f0', 'f1'],
    weights: [1, -1],
    bias: 0,
  };
  const pipeline = await createPipeline(manifest, {});
  const result = await pipeline.infer({
    backend: 'cpu',
    steps: 0,
    energyScale: 0,
    rows: [{ rowId: 'r0', features: [2, 1] }],
  });

  assert.equal(result.modelId, 'dream-bridge-test');
  assert.equal(result.rows.length, 1);
  assert.equal(result.rows[0].rowId, 'r0');
  assert.ok(Math.abs(result.rows[0].score - 0.7310585786) < 1e-6);
  assert.ok(Math.abs(result.rows[0].logit - 1) < 1e-9);
}

{
  const manifest = {
    modelType: 'synthesis-mixer-diffusion',
    modelId: 'dream-synth-test',
    featureIds: ['intent_density', 'seed_score'],
    weights: [0.5, 0.5],
    bias: 0.1,
  };
  const pipeline = await createPipeline(manifest, {});
  const result = await pipeline.infer({
    backend: 'cpu',
    steps: 0,
    energyScale: 0,
    rows: [{ candidateId: 'c0', features: { intent_density: 0.8, seed_score: 0.4 } }],
  });

  assert.equal(result.rows.length, 1);
  assert.equal(result.rows[0].rowId, 'c0');
  assert.ok(result.rows[0].score > 0.6);
}

{
  const manifest = {
    modelType: 'ebrm-diffusion',
    modelId: 'dream-ebrm-test',
    featureIds: ['n', 'l'],
    weights: [1, 1],
    bias: 0,
    treeHead: {
      featureIds: ['t0'],
      weights: [2],
      bias: 0.2,
      scale: 0.5,
    },
  };
  const pipeline = await createPipeline(manifest, {});
  const result = await pipeline.infer({
    backend: 'cpu',
    head: 'main',
    steps: 0,
    energyScale: 0,
    rows: [{ rowId: 'cg0', features: [1, 2] }],
  });

  assert.equal(result.activation, 'linear');
  assert.ok(Math.abs(result.rows[0].score - 3) < 1e-9);

  const treeResult = await pipeline.infer({
    backend: 'cpu',
    head: 'tree',
    steps: 0,
    energyScale: 0,
    rows: [{ rowId: 'cg0', features: { t0: 1.0 } }],
  });
  assert.equal(treeResult.rows.length, 1);
  assert.ok(treeResult.rows[0].score > 0.4);
}

{
  const manifest = {
    modelType: 'dream_energy_head',
    modelId: 'dream-energy-head-test',
    featureIds: ['f0', 'f1', 'f2'],
    weights: [0.2, 0.1, -0.3],
    bias: 0,
  };
  const pipeline = await createPipeline(manifest, {});
  await assert.rejects(
    () => pipeline.infer({
      backend: 'cpu',
      rows: [{ rowId: 'bad', features: [1, 2] }],
    }),
    /features length mismatch/
  );
}

