import assert from 'node:assert/strict';

import {
  buildModelDetail,
  canRemoveModelStatus,
  modelLabel,
  selectDefaultStoredModel,
} from '../../demo/models.js';

assert.equal(canRemoveModelStatus('stored'), true);
assert.equal(canRemoveModelStatus('loaded'), true);
assert.equal(canRemoveModelStatus('available'), false);

assert.equal(buildModelDetail({ modelId: 'model-a', label: 'Model A' }), '', 'No size, no download line');
assert.equal(buildModelDetail(null), '');

// The picker shows the published catalog label, configuration included.
assert.equal(modelLabel({ modelId: 'e2b-ple', label: 'Gemma 4 E2B Instruct (Q4K/F32a/INT4 PLE)' }),
  'Gemma 4 E2B Instruct (Q4K/F32a/INT4 PLE)');
assert.equal(modelLabel({ modelId: 'bare' }), 'bare');

const catalog = [
  { modelId: 'model-a' },
  { modelId: 'model-b' },
];
const registered = [
  { modelId: 'model-b' },
];
assert.deepEqual(
  selectDefaultStoredModel(catalog, registered),
  { modelId: 'model-b' }
);
assert.deepEqual(
  selectDefaultStoredModel(catalog, registered, 'model-b'),
  { modelId: 'model-b' }
);
assert.equal(
  selectDefaultStoredModel(catalog, [{ modelId: 'not-visible' }]),
  null
);

console.log('demo-models.test: ok');
