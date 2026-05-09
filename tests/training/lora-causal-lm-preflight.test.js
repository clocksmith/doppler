import assert from 'node:assert/strict';

import {
  preflightCausalLmLoraWorkload,
} from '../../src/experimental/training/lora-pipeline.js';

const datasetText = [
  JSON.stringify({
    rowId: 'columbo-row-1',
    source: 'system: Return JSON.\n\nuser: {"input":{"text":"alpha"}}',
    target: '{"findings":[]}',
  }),
  JSON.stringify({
    rowId: 'columbo-row-2',
    source: 'system: Return JSON.\n\nuser: {"input":{"text":"beta"}}',
    target: '{"findings":[{"category":"redaction"}]}',
  }),
].join('\n');

const workload = {
  schemaVersion: 1,
  kind: 'lora',
  id: 'columbo-causal-lm-preflight',
  baseModelId: 'gemma4-e2b-it',
  datasetPath: '/tmp/columbo-causal-lm-preflight.jsonl',
  pipeline: {
    datasetFormat: 'text-pairs',
    taskType: 'text_generation',
    adapter: {
      rank: 8,
      alpha: 16,
      targetModules: ['q_proj', 'v_proj'],
    },
  },
};

const preflight = await preflightCausalLmLoraWorkload(workload, {
  readFile: async () => datasetText,
});

assert.equal(preflight.runnerKey, 'gemma4-e2b-it::text-pairs::text_generation');
assert.equal(preflight.rowCount, 2);
assert.equal(preflight.firstRowId, 'columbo-row-1');
assert.equal(preflight.lastRowId, 'columbo-row-2');
assert.deepEqual(preflight.textPairFields, {
  prompt: 'source',
  completion: 'target',
});
assert.deepEqual(preflight.adapter, {
  rank: 8,
  alpha: 16,
  targetModules: ['q_proj', 'v_proj'],
});
assert.equal(preflight.supported, true);
assert.deepEqual(preflight.blockedReasons, []);

console.log('lora-causal-lm-preflight.test: ok');
