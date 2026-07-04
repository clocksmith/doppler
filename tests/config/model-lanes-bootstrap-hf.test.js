import assert from 'node:assert/strict';

import { validateCatalog } from '../../tools/check-model-lanes.js';

const bootstrapEntry = {
  modelId: 'unit-bootstrap-model',
  sourceCheckpointId: 'org/unit',
  weightPackId: 'unit-bootstrap-wp-v1',
  manifestVariantId: 'unit-bootstrap-mv-v1',
  artifactCompleteness: 'complete',
  runtimePromotionState: 'manifest-owned',
  weightsRefAllowed: false,
  hf: {
    repoId: 'Clocksmith/rdrr',
    revision: null,
    path: 'models/unit-bootstrap-model',
  },
  lifecycle: {
    availability: {
      hf: false,
    },
    status: {
      runtime: 'active',
      tested: 'verified',
    },
    tested: {
      result: 'pass',
      contracts: {
        executionContractOk: true,
      },
    },
  },
};

assert.deepEqual(validateCatalog({ models: [bootstrapEntry] }), []);

assert.match(
  validateCatalog({
    models: [{
      ...bootstrapEntry,
      hf: {
        ...bootstrapEntry.hf,
        revision: 'abc123',
      },
    }],
  }).join('\n'),
  /hf metadata must be absent or a bootstrap repo\/path with no revision/
);

assert.match(
  validateCatalog({
    models: [{
      ...bootstrapEntry,
      lifecycle: {
        ...bootstrapEntry.lifecycle,
        status: {
          runtime: 'experimental',
          tested: 'verified',
        },
      },
    }],
  }).join('\n'),
  /bootstrap hf repo\/path requires active verified\/pass lifecycle/
);

console.log('model-lanes-bootstrap-hf.test: ok');
