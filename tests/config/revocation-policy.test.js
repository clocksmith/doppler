import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { createHash } from 'node:crypto';

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
  weightCapsuleIds: [],
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
          weightCapsuleIds: ['weight-capsule-v1'],
          manifestVariantIds: ['manifest-v1'],
          artifactVariantIds: [HASH],
          adapterIds: ['unsafe-adapter'],
          adapterDigests: [ADAPTER_HASH],
        },
        replacements: {
          ...EMPTY_TARGETS,
          modelIds: ['replacement-model'],
        },
        evidencePaths: ['GOALS.md'],
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
    { weightCapsuleId: 'weight-capsule-v1' },
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
          weightCapsuleId: 'weight-capsule-v1',
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

// Exercise the real host and bundled-policy implementation with a controlled
// registry. Device acquisition stops at a double; this never opens a GPU.
const hostUrl = new URL('../../src/client/model-host/index.js?revocation-test', import.meta.url).href;
const policyUrl = new URL('../../src/config/revocation-policy.js?host-fixture', import.meta.url).href;
const sourceUrl = new URL('../../src/client/runtime/model-source.js', import.meta.url).href;
const sourceFixtureUrl = `${sourceUrl}?revocation-test`;
const scopeUrl = new URL('../../src/gpu/kernels/shader-source-scope.js', import.meta.url).href;
const artifactManifest = {
  version: 1, modelId: 'artifact-fixture', modelType: 'embedding', quantization: 'q4k',
  hashAlgorithm: 'sha256', inference: {}, totalSize: 1,
  tensors: { 'model.embed_tokens.weight': { role: 'embedding' } },
  shards: [{ index: 0, filename: 'shard_00000.bin', size: 1, hash: 'a'.repeat(64), offset: 0 }],
};
const artifactText = JSON.stringify(artifactManifest);
const artifactHash = createHash('sha256').update(artifactText).digest('hex');
const hostRegistry = registry();
hostRegistry.revocations[0].targets.artifactVariantIds = [`sha256:${artifactHash}`];
globalThis.revocationHostFixture = { registry: hostRegistry, events: [] };
const hostModules = {
  '../runtime/model-source.js': `export * from ${JSON.stringify(sourceUrl)};
    export const resolveModelSource = async model => model;`,
  '../../gpu/device.js': `export const initDevice = async () => {
    revocationHostFixture.events.push('device'); throw new Error('fixture device boundary');
  }; export const getKernelCapabilities = () => ({}); export const getPlatformConfig = () => null;`,
  '../../gpu/kernels/shader-source-scope.js': `import { runWithShaderSourceScope as run } from ${JSON.stringify(scopeUrl)};
    export const runWithShaderSourceScope = (scope, action) => {
      revocationHostFixture.events.push({ scope }); return run(scope, action);
    };`,
};
const hooks = registerHooks({ resolve(specifier, context, next) {
  if ([hostUrl, sourceFixtureUrl].includes(context.parentURL) && specifier === '../../config/revocation-policy.js') {
    return { shortCircuit: true, url: policyUrl };
  }
  const source = context.parentURL === hostUrl ? hostModules[specifier]
    : context.parentURL === policyUrl && specifier === '../formats/load-json.js'
      ? 'export const loadJson = async () => structuredClone(revocationHostFixture.registry);'
      : null;
  return source ? { shortCircuit: true, url: `data:text/javascript,${encodeURIComponent(source)}` }
    : next(specifier, context);
} });
try {
  const { createDopplerRuntimeService } = await import(hostUrl);
  const host = createDopplerRuntimeService({ ensureWebGPUAvailable: async () => {} });
  const revoked = error => {
    assert.equal(error.revocationId, 'unsafe-model-v1', error.message);
    return true;
  };
  for (const [field, value] of [
    ['modelId', 'resolved-model'],
    ['sourceCheckpointId', 'source/model'],
    ['weightCapsuleId', 'weight-capsule-v1'],
    ['manifestVariantId', 'manifest-v1'],
  ]) {
    const manifest = field === 'modelId' ? { modelId: value }
      : { modelId: 'allowed-fixture', artifactIdentity: { [field]: value } };
    await assert.rejects(host.load({ manifest }), revoked);
    assert.deepEqual(revocationHostFixture.events, [], 'revoked identities must fail before device acquisition');
  }
  await assert.rejects(host.load({ logicalModelId: 'logical-model', manifest: { modelId: 'allowed-fixture' } }), revoked);
  await assert.rejects(host.load({ manifest: artifactManifest }), revoked);
  assert.deepEqual(revocationHostFixture.events, [], 'logical and artifact denials must precede device acquisition');
  const { resolveModelSource } = await import(sourceFixtureUrl);
  await assert.rejects(resolveModelSource({ manifest: artifactManifest, manifestText: artifactText }), revoked);
  const allowedManifest = { ...artifactManifest, modelId: 'allowed-fixture' };
  const allowed = await resolveModelSource({ manifest: allowedManifest, manifestText: JSON.stringify(allowedManifest) });
  assert.equal(allowed.manifest.modelId, 'allowed-fixture', 'an unrelated denial must not block a valid manifest');
  await assert.rejects(host.load({ manifest: { modelId: 'allowed-fixture' } }), /fixture device boundary/);
  assert.deepEqual(revocationHostFixture.events, [{ scope: null }, 'device'],
    'allowed identities reach device acquisition in the host shader scope');
} finally {
  hooks.deregister();
  delete globalThis.revocationHostFixture;
}

console.log('revocation-policy.test: ok');
