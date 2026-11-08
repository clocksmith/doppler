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

// The picker shows plain names; storage details stay in the catalog label.
const e2b = { modelId: 'e2b', label: 'Gemma 4 E2B Instruct (Q4K/F32a)' };
const e2bCompact = { modelId: 'e2b-ple', label: 'Gemma 4 E2B Instruct (Q4K/F32a/INT4 PLE)' };
const small = { modelId: 'small', label: 'Gemma 3 270M Instruct (Q4K/F32a)' };
assert.equal(modelLabel(small, [small, e2b, e2bCompact]), 'Gemma 3 270M Instruct');
assert.equal(modelLabel(e2b, [small, e2b, e2bCompact]), 'Gemma 4 E2B Instruct');
assert.equal(modelLabel(e2bCompact, [small, e2b, e2bCompact]), 'Gemma 4 E2B Instruct, compact');
const e2bF16 = { modelId: 'e2b-f16', label: 'Gemma 4 E2B Instruct (Q4K/F16a)' };
assert.equal(modelLabel(e2b, [e2b, e2bF16]), 'Gemma 4 E2B Instruct (Q4K/F32a)', 'Names that cannot be told apart keep the catalog label');
assert.equal(modelLabel({ modelId: 'bare' }, []), 'bare');

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
