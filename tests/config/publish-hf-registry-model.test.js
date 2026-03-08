import assert from 'node:assert/strict';

import {
  assertPromotionReady,
  buildArtifactUploadPlan,
  parseArgs,
} from '../../tools/publish-hf-registry-model.js';

{
  const args = parseArgs([
    '--model-id', 'translategemma-4b-it-q4k-ehf16-af32',
    '--dry-run',
  ]);
  assert.equal(args.modelId, 'translategemma-4b-it-q4k-ehf16-af32');
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
  assert.equal(plan.localDir, '/tmp/translategemma');
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
      status: {
        tested: 'verified',
      },
      tested: {
        contracts: {
          executionContractOk: true,
          executionV0GraphOk: true,
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
        status: {
          tested: 'verified',
        },
        tested: {
          contracts: {
            executionContractOk: true,
            executionV0GraphOk: false,
          },
        },
      },
    }),
    /execution-v0 graph gate must be explicitly true/
  );
  assert.throws(
    () => assertPromotionReady({
      modelId: 'translategemma-4b-it-q4k-ehf16-af32',
      lifecycle: {
        status: {
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
