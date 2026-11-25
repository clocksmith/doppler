import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';

import { assertHfRerankResult } from '../../tools/compare-rerankers-hf.js';

// The HF Transformers reranker baseline runs outside Doppler (JavaScript-only);
// compare-rerankers-hf.js imports its JSON and must reject a mismatched run.
const help = spawnSync(process.execPath, ['tools/compare-rerankers-hf.js', '--help'], { encoding: 'utf8' });
assert.equal(help.status, 0, help.stderr);
assert.match(help.stdout, /--hf-result <path>/);
assert.match(help.stdout, /HF Transformers model/);
assert.doesNotMatch(help.stdout, /--hf-device|--hf-allow-download/);

const expected = {
  hfModelId: 'Qwen/Qwen3-Reranker-0.6B',
  query: 'Which document mentions WebGPU?',
  documents: ['WebGPU runs in the browser.', 'A note about bread.'],
  warmupRuns: 1,
  timedRuns: 3,
};
const result = {
  schemaVersion: 1,
  kind: 'hf-transformers-reranker-bench',
  task: 'rerank',
  modelId: expected.hfModelId,
  query: expected.query,
  documents: [...expected.documents],
  warmupRuns: 1,
  timedRuns: 3,
  batchSize: 1,
  environment: { runtime: { library: 'transformers', device: 'cpu', dtype: 'float32' } },
};
assertHfRerankResult(result, expected);
assert.throws(() => assertHfRerankResult({ ...result, kind: 'other' }, expected), /hf-transformers-reranker-bench/);
assert.throws(() => assertHfRerankResult({ ...result, modelId: 'other/model' }, expected), /expected Qwen/);
assert.throws(() => assertHfRerankResult({ ...result, documents: ['A note about bread.'] }, expected), /query or documents/);
assert.throws(() => assertHfRerankResult({ ...result, timedRuns: 5 }, expected), /run counts/);

const missing = spawnSync(process.execPath, ['tools/compare-rerankers-hf.js'], { encoding: 'utf8' });
assert.notEqual(missing.status, 0);
assert.match(missing.stderr, /--hf-result is required/);

console.log('hf-reranker-bench-contract.test: ok');
