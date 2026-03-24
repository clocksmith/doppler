import assert from 'node:assert/strict';

import {
  assertPromotionReady,
  buildArtifactUploadPlan,
  parseArgs,
} from '../../tools/publish-hf-registry-model.js';

{
  const args = parseArgs([
    '--model-id', 'translategemma-4b-it-q4k-ehf16-af32',
    '--catalog-file', 'models/catalog.json',
    '--dry-run',
  ]);
  assert.equal(args.modelId, 'translategemma-4b-it-q4k-ehf16-af32');
  assert.ok(args.catalogFile.endsWith('models/catalog.json'));
  assert.equal(args.dryRun, true);
  assert.throws(
    () => parseArgs(['--model-id', 'translategemma-4b-it-q4k-ehf16-af32', '--repo-id']),
    /Missing value for --repo-id/
  );
}

{
  const plan = buildArtifactUploadPlan({
    modelId: 'translategemma-4b-it-q4k-ehf16-af32',
    hf: {
      repoId: 'Clocksmith/rdrr',
      path: 'models/translategemma-4b-it-q4k-ehf16-af32',
    },
  }, {
    localDir: '/tmp/translategemma',
  });
  assert.equal(plan.repoId, 'Clocksmith/rdrr');
  assert.equal(plan.targetPath, 'models/translategemma-4b-it-q4k-ehf16-af32');
  assert.equal(plan.manifestDir, '/tmp/translategemma');
}

{
  assert.throws(
    () => buildArtifactUploadPlan({
      modelId: 'translategemma-4b-it-q4k-ehf16-af32',
      hf: {
        repoId: 'Clocksmith/rdrr',
        path: '',
      },
    }),
    /hf\.path is required to publish/
  );
}

{
  assert.doesNotThrow(() => assertPromotionReady({
    modelId: 'translategemma-4b-it-q4k-ehf16-af32',
    lifecycle: {
      availability: {
        hf: true,
      },
      status: {
        runtime: 'active',
        tested: 'verified',
      },
      tested: {
        contracts: {
          executionContractOk: true,
        },
      },
    },
  }));
}

{
  assert.throws(
    () => assertPromotionReady({
      modelId: 'translategemma-4b-it-q4k-ehf16-af32',
      lifecycle: {
        availability: {
          hf: false,
        },
        status: {
          runtime: 'active',
          tested: 'verified',
        },
        tested: {
          contracts: {
            executionContractOk: true,
          },
        },
      },
    }),
    /lifecycle\.availability\.hf must be true/
  );
  assert.throws(
    () => assertPromotionReady({
      modelId: 'translategemma-4b-it-q4k-ehf16-af32',
      lifecycle: {
        availability: {
          hf: true,
        },
        status: {
          runtime: 'blocked',
          tested: 'verified',
        },
        tested: {
          contracts: {
            executionContractOk: true,
          },
        },
      },
    }),
    /lifecycle\.status\.runtime must be "active"/
  );
  assert.throws(
    () => assertPromotionReady({
      modelId: 'translategemma-4b-it-q4k-ehf16-af32',
      lifecycle: {
        availability: {
          hf: true,
        },
        status: {
          runtime: 'active',
          tested: 'verified',
        },
        tested: {
          contracts: {},
        },
      },
    }),
    /execution contract gate must be explicitly true/
  );
}

console.log('publish-hf-registry-model.test: ok');
