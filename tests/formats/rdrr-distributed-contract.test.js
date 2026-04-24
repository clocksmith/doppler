import assert from 'node:assert/strict';

import { parseDistributedPlan } from '../../src/formats/rdrr/distributed/parsing.js';

function buildPlan() {
  return {
    rdrd: 1,
    compatibility: {
      artifactIdentityHash: 'sha256:artifact',
      manifestHash: 'sha256:manifest',
      executionGraphDigest: 'sha256:execution',
      integrityExtensionsHash: null,
      synthesizerVersion: 'doppler-placement@1.0.0',
      synthesizedAt: '2026-04-23T00:00:00.000Z',
    },
    plans: [
      {
        id: 'lan-4peer-tp',
        topologyHash: 'sha256:topology',
      },
    ],
  };
}

{
  const parsed = parseDistributedPlan(JSON.stringify(buildPlan()), {
    expectedCompatibility: {
      artifactIdentityHash: 'sha256:artifact',
      manifestHash: 'sha256:manifest',
      executionGraphDigest: 'sha256:execution',
      integrityExtensionsHash: null,
    },
    expectedPlanId: 'lan-4peer-tp',
    expectedTopologyHash: 'sha256:topology',
  });
  assert.equal(parsed.plans[0].id, 'lan-4peer-tp');
}

{
  const stale = buildPlan();
  assert.throws(
    () => parseDistributedPlan(JSON.stringify(stale), {
      expectedCompatibility: {
        artifactIdentityHash: 'sha256:artifact',
        manifestHash: 'sha256:manifest-v2',
        executionGraphDigest: 'sha256:execution',
        integrityExtensionsHash: null,
      },
    }),
    /DOPPLER_DISTRIBUTED_PLAN_STALE/,
  );
}

{
  const artifactMismatch = buildPlan();
  assert.throws(
    () => parseDistributedPlan(JSON.stringify(artifactMismatch), {
      expectedCompatibility: {
        artifactIdentityHash: 'sha256:artifact-v2',
      },
    }),
    /DOPPLER_DISTRIBUTED_PLAN_ARTIFACT_MISMATCH/,
  );
}

{
  const malformed = buildPlan();
  malformed.plans = [];
  assert.throws(
    () => parseDistributedPlan(JSON.stringify(malformed)),
    /DOPPLER_DISTRIBUTED_PLAN_INVALID/,
  );
}

{
  const unsupported = buildPlan();
  unsupported.rdrd = 2;
  assert.throws(
    () => parseDistributedPlan(JSON.stringify(unsupported)),
    /DOPPLER_DISTRIBUTED_PLAN_UNSUPPORTED/,
  );
}

console.log('rdrr-distributed-contract.test: ok');

