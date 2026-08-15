import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

import {
  assertBundledAdapterAuthorized,
  assertResolutionNotRevoked,
  authorizeBundledAdapter,
  findResolutionRevocation,
  loadRevocationRegistry,
  validateRevocationRegistry,
} from '../../src/config/revocation-policy.js';
import { validateRevocationPropagation } from '../../tools/check-revocation-registry.js';
import { InferencePipeline } from '../../src/inference/pipelines/text.js';

const HASH = `sha256:${'a'.repeat(64)}`;
const ADAPTER_HASH = `sha256:${'b'.repeat(64)}`;
const EMPTY_TARGETS = Object.freeze({
  logicalModelIds: [],
  modelIds: [],
  sourceCheckpointIds: [],
  weightPackIds: [],
  manifestVariantIds: [],
  artifactVariantIds: [],
  adapterIds: [],
  adapterDigests: [],
});

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function registry() {
  return {
    $schema: 'schema/revocation-registry.schema.json',
    schemaVersion: 1,
    source: 'doppler',
    updatedAtUtc: '2026-08-15T00:00:00.000Z',
    trust: {
      distribution: 'bundled-package',
      signatureVerification: 'unavailable',
    },
    revocations: [
      {
        id: 'unsafe-model-v1',
        state: 'revoked',
        issuedAtUtc: '2026-08-15T00:00:00.000Z',
        severity: 'correctness',
        reason: 'Held-out correctness failure.',
        targets: {
          logicalModelIds: ['logical-model'],
          modelIds: ['resolved-model'],
          sourceCheckpointIds: ['source/model'],
          weightPackIds: ['weight-pack-v1'],
          manifestVariantIds: ['manifest-v1'],
          artifactVariantIds: [HASH],
          adapterIds: ['unsafe-adapter'],
          adapterDigests: [ADAPTER_HASH],
        },
        replacements: {
          ...EMPTY_TARGETS,
          modelIds: ['replacement-model'],
        },
        evidencePaths: ['docs/goals.md'],
      },
    ],
  };
}

{
  const bundled = await loadRevocationRegistry();
  assert.equal(bundled.revocations.length, 0);
  assert.equal(bundled.trust.signatureVerification, 'unavailable');
  assert.equal(Object.isFrozen(bundled), true);
}

{
  const policy = validateRevocationRegistry(registry());
  const identities = [
    { logicalModelId: 'logical-model' },
    { modelId: 'resolved-model' },
    { sourceCheckpointId: 'source/model' },
    { weightPackId: 'weight-pack-v1' },
    { manifestVariantId: 'manifest-v1' },
    { artifactVariantId: HASH.toUpperCase() },
    { adapterId: 'unsafe-adapter' },
    { adapterDigest: ADAPTER_HASH.toUpperCase() },
  ];
  for (const identity of identities) {
    assert.ok(findResolutionRevocation(identity, policy));
    assert.throws(
      () => assertResolutionNotRevoked(identity, policy),
      (error) => {
        assert.equal(error.name, 'DopplerRevocationError');
        assert.equal(error.code, 'DOPPLER_REVOKED');
        assert.equal(error.revocationId, 'unsafe-model-v1');
        assert.match(error.message, /No auto replacement/);
        assert.deepEqual(error.replacements.modelIds, ['replacement-model']);
        return true;
      }
    );
  }
  assert.doesNotThrow(() => assertResolutionNotRevoked({ modelId: 'unrelated-model' }, policy));
}

{
  const live = registry();
  live.trust = { distribution: 'signed-live', signatureVerification: 'verified' };
  const report = await validateRevocationPropagation(live, {}, { requireBundledTrust: true });
  assert.equal(report.ok, false);
  assert.ok(report.errors.includes('the package revocation registry must use bundled-package trust'));
}

{
  const adapter = {
    id: 'safe-adapter',
    identity: {
      schema: 'doppler.lora-execution-identity/v1',
      id: 'safe-adapter',
      digest: ADAPTER_HASH,
    },
  };
  assert.throws(() => assertBundledAdapterAuthorized(adapter), /revocation-checked/);
  assert.throws(
    () => InferencePipeline.prototype.setLoRAAdapter.call({}, adapter),
    /revocation-checked/
  );
  await authorizeBundledAdapter(adapter);
  assert.doesNotThrow(() => assertBundledAdapterAuthorized(adapter));
  const pipeline = {};
  InferencePipeline.prototype.setLoRAAdapter.call(pipeline, adapter);
  assert.equal(pipeline.lora, adapter);
  assert.doesNotThrow(() => assertBundledAdapterAuthorized(null));
}

{
  const invalid = registry();
  invalid.revocations[0].targets = clone(EMPTY_TARGETS);
  assert.throws(
    () => validateRevocationRegistry(invalid),
    /requires a revoked target/
  );
}

{
  const duplicate = registry();
  duplicate.revocations.push(clone(duplicate.revocations[0]));
  assert.throws(() => validateRevocationRegistry(duplicate), /ids must be unique/);
}

{
  const activeSurfaces = {
    catalog: {
      models: [{
        modelId: 'resolved-model',
        quickstart: true,
        demoVisible: true,
        lifecycle: { status: { runtime: 'active' } },
      }],
    },
    adapterCatalog: {
      artifacts: [{
        artifactId: 'unsafe-adapter',
        lifecycle: 'promoted',
        weights: { sha256: ADAPTER_HASH },
      }],
    },
    quickstart: {
      models: [{ modelId: 'resolved-model', aliases: [] }],
    },
    claimMatrix: {
      lanes: [{
        id: 'claim-lane',
        status: 'promoted',
        model: { dopplerModelId: 'resolved-model', sourceCheckpointId: 'source/model' },
        artifact: {
          weightPackId: 'weight-pack-v1',
          manifestVariantId: 'manifest-v1',
          manifestSha256: HASH,
        },
      }],
    },
    releaseMatrix: {
      localClaimLanes: [{
        laneId: 'claim-lane',
        status: 'promoted',
        claimReady: true,
        dopplerModelId: 'resolved-model',
      }],
    },
    productIntegrations: {
      integrations: [{
        id: 'application',
        lifecycle: 'active',
        claimAllowed: true,
        logicalModelId: 'logical-model',
        resolvedArtifactVariantId: HASH,
      }],
    },
    providerConformance: {
      suites: [{
        id: 'provider-suite',
        logicalModelId: 'logical-model',
        resolvedArtifactVariantId: HASH,
        claimAllowed: true,
        providers: [{
          laneId: 'browser-webgpu',
          logicalModelId: 'logical-model',
          resolvedArtifactVariantId: HASH,
          claimAllowed: true,
        }],
      }],
    },
    runtimeOwnership: {
      decisions: [{
        id: 'ownership-decision',
        logicalModelId: 'logical-model',
        resolvedArtifactVariantId: HASH,
        claimAllowed: true,
      }],
    },
  };
  const report = await validateRevocationPropagation(registry(), activeSurfaces, {
    repoRoot: process.cwd(),
  });
  assert.equal(report.ok, false);
  for (const label of [
    'catalog resolved-model',
    'adapter catalog unsafe-adapter',
    'quickstart resolved-model',
    'claim lane claim-lane',
    'release lane claim-lane',
    'product integration application',
    'provider suite provider-suite',
    'provider result provider-suite/browser-webgpu',
    'runtime ownership ownership-decision',
  ]) {
    assert.ok(report.errors.some((error) => error.startsWith(label)), report.errors.join('\n'));
  }

  const namedVariantSurfaces = clone(activeSurfaces);
  namedVariantSurfaces.providerConformance.suites[0].logicalModelId = 'unrelated-model';
  namedVariantSurfaces.providerConformance.suites[0].manifestVariantId = 'manifest-v1';
  namedVariantSurfaces.providerConformance.suites[0].resolvedArtifactVariantId = null;
  namedVariantSurfaces.providerConformance.suites[0].providers[0].logicalModelId = 'unrelated-model';
  namedVariantSurfaces.providerConformance.suites[0].providers[0].manifestVariantId = 'manifest-v1';
  namedVariantSurfaces.providerConformance.suites[0].providers[0].resolvedArtifactVariantId = null;
  namedVariantSurfaces.runtimeOwnership.decisions[0].logicalModelId = 'unrelated-model';
  namedVariantSurfaces.runtimeOwnership.decisions[0].manifestVariantId = 'manifest-v1';
  namedVariantSurfaces.runtimeOwnership.decisions[0].resolvedArtifactVariantId = null;
  const namedVariantReport = await validateRevocationPropagation(
    registry(),
    namedVariantSurfaces,
    { repoRoot: process.cwd() }
  );
  for (const label of [
    'provider suite provider-suite',
    'provider result provider-suite/browser-webgpu',
    'runtime ownership ownership-decision',
  ]) {
    assert.ok(
      namedVariantReport.errors.some((error) => error.startsWith(label)),
      namedVariantReport.errors.join('\n')
    );
  }

  const withdrawn = clone(activeSurfaces);
  withdrawn.catalog.models[0].quickstart = false;
  withdrawn.catalog.models[0].demoVisible = false;
  withdrawn.catalog.models[0].lifecycle.status.runtime = 'revoked';
  withdrawn.adapterCatalog.artifacts[0].lifecycle = 'revoked';
  withdrawn.quickstart.models = [];
  withdrawn.claimMatrix.lanes[0].status = 'revoked';
  withdrawn.releaseMatrix.localClaimLanes[0].status = 'revoked';
  withdrawn.releaseMatrix.localClaimLanes[0].claimReady = false;
  withdrawn.productIntegrations.integrations[0].lifecycle = 'revoked';
  withdrawn.productIntegrations.integrations[0].claimAllowed = false;
  withdrawn.providerConformance.suites[0].claimAllowed = false;
  withdrawn.providerConformance.suites[0].providers[0].claimAllowed = false;
  withdrawn.runtimeOwnership.decisions[0].claimAllowed = false;
  const withdrawnReport = await validateRevocationPropagation(registry(), withdrawn, {
    repoRoot: process.cwd(),
  });
  assert.deepEqual(withdrawnReport.errors, []);
  assert.equal(withdrawnReport.ok, true);
}

const sourceText = await fs.readFile(
  new URL('../../src/client/runtime/index.js', import.meta.url),
  'utf8'
);
assert.ok(
  sourceText.indexOf('assertBundledResolutionNotRevoked({') < sourceText.indexOf('await initDevice()'),
  'the root runtime must enforce bundled revocation before device initialization'
);

console.log('revocation-policy.test: ok');
