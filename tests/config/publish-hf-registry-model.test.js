import assert from 'node:assert/strict';

import {
  assertPromotionReady,
  buildArtifactUploadPlan,
  parseArgs,
} from '../../tools/publish-hf-registry-model.js';

{
  const args = parseArgs([
    '--model-id', 'translategemma-4b-it-wq4k-ef16-hf16',
    '--dry-run',
  ]);
  assert.equal(args.modelId, 'translategemma-4b-it-wq4k-ef16-hf16');
  assert.equal(args.dryRun, true);
}

{
  const plan = buildArtifactUploadPlan({
    modelId: 'translategemma-4b-it-wq4k-ef16-hf16',
    hf: {
      repoId: 'Clocksmith/rdrr',
      path: 'models/translategemma-4b-it-wq4k-ef16-hf16',
    },
  }, {
    localDir: '/tmp/translategemma',
  });
  assert.equal(plan.repoId, 'Clocksmith/rdrr');
  assert.equal(plan.targetPath, 'models/translategemma-4b-it-wq4k-ef16-hf16');
  assert.equal(plan.localDir, '/tmp/translategemma');
}

{
  assert.doesNotThrow(() => assertPromotionReady({
    modelId: 'translategemma-4b-it-wq4k-ef16-hf16',
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
      modelId: 'translategemma-4b-it-wq4k-ef16-hf16',
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
    /execution-v0 graph gate failed/
  );
}

console.log('publish-hf-registry-model.test: ok');
