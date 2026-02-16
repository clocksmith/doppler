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

const {
  DreamStructuredPipeline,
  isDreamStructuredModelType,
} = await import('../../src/inference/pipelines/dream/pipeline.js');

const DREAM_MODEL_TYPES = [
  'dream_structured',
  'dream_intent_posterior_head',
  'dream_d1_to2_bridge',
  'dream_synthesis',
  'dream_energy_compose',
  'dream-intent-posterior-head',
  'dream-d1-to2-bridge',
  'dream-synthesis',
  'dream-energy-compose',
];

function createMockPipeline(chunks, manifest = {}) {
  const pipeline = new DreamStructuredPipeline();
  pipeline.manifest = {
    modelId: 'dream-test-model',
    modelHash: { alg: 'sha256', hex: 'a'.repeat(64) },
    ...manifest,
  };
  pipeline.runtimeConfig = { inference: { dream: { maxTokens: 32, temperature: 0 } } };
  pipeline.reset = () => {};
  pipeline.generate = async function* generate() {
    for (const chunk of chunks) {
      yield chunk;
    }
  };
  return pipeline;
}

for (const modelType of DREAM_MODEL_TYPES) {
  assert.equal(isDreamStructuredModelType(modelType), true, `${modelType} should be supported`);
}
assert.equal(isDreamStructuredModelType('transformer'), false);
assert.equal(isDreamStructuredModelType(''), false);

{
  const pipeline = createMockPipeline(['{"schemaVersion":1,"ok":true}']);
  const result = await pipeline.inferJSON({
    prompt: 'test prompt',
    nowIso: '2026-02-16T00:00:00.000Z',
    maxTokens: 16,
    temperature: 0,
  });

  assert.equal(result.output.schemaVersion, 1);
  assert.equal(result.output.ok, true);
  assert.equal(result.modelId, 'dream-test-model');
  assert.equal(result.promptHash.alg, 'sha256');
  assert.equal(result.promptHash.hex.length, 64);
}

{
  const pipeline = createMockPipeline([
    '```json\n',
    '{"schemaVersion":1,"cgCommitted":{"graphId":"cg_1"}}',
    '\n```',
  ]);
  const result = await pipeline.inferJSON({ prompt: 'compose' });
  assert.equal(result.output.schemaVersion, 1);
  assert.equal(result.output.cgCommitted.graphId, 'cg_1');
}

{
  const pipeline = createMockPipeline(['"not-object"']);
  await assert.rejects(
    () => pipeline.inferJSON({ prompt: 'bad' }),
    /output must be a JSON object/
  );
}
