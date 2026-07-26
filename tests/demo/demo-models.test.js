import assert from 'node:assert/strict';

import {
  buildModelCardDetail,
  canRemoveModelStatus,
  selectDefaultStoredModel,
} from '../../demo/models.js';

assert.equal(canRemoveModelStatus('stored'), true);
assert.equal(canRemoveModelStatus('loaded'), true);
assert.equal(canRemoveModelStatus('available'), false);

assert.equal(
  buildModelCardDetail({ modelId: 'model-a' }, 'stored'),
  'Downloaded · model-a'
);

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
