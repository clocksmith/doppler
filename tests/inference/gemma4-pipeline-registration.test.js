import assert from 'node:assert/strict';

await import('../../src/inference/pipelines/text.js');
const { getPipelineFactory } = await import('../../src/inference/pipelines/registry.js');

assert.equal(typeof getPipelineFactory('transformer'), 'function');
assert.equal(typeof getPipelineFactory('gemma4'), 'function');
assert.equal(
  getPipelineFactory('gemma4'),
  getPipelineFactory('transformer'),
  'gemma4 should resolve through the transformer pipeline factory'
);

console.log('gemma4-pipeline-registration.test: ok');
