import assert from 'node:assert/strict';
import { createDopplerRuntime } from '../../src/client/runtime/composition-root.js';
import {
  assertInitialExecutionIdentity,
  createInitialExecutionIdentity,
  createInitialExecutionIdentityV2,
  observeInitialExecutionIdentity,
  resolveProgramLoadRuntimeConfig,
} from '../../src/config/initial-execution-identity.js';
import {
  TEST_PACK_AUTHORITY,
  TEST_PACK_PUBLIC_KEY,
  createSignedPackFixture,
} from '../helpers/pack-v2-fixture.js';

const digest = (character) => `sha256:${character.repeat(64)}`;
const graphHash = `sha256:${'3'.repeat(64)}`;
const resolvedGraphHash = `sha256:${'4'.repeat(64)}`;
const executionPlanDigest = `sha256:${'5'.repeat(64)}`;
const fields = {
  executionGraphHash: graphHash,
  resolvedGraphHash,
  kernelClosure: [{ moduleId: 'main', file: 'main.wgsl', entry: 'main', digest: `sha256:${'6'.repeat(64)}` }],
  dtypeLane: { activation: 'f32', output: 'f32', kv: 'f32', math: 'f32', accumulation: 'f32' },
  fusionSet: [],
  kvLayout: { layout: 'contiguous', kvDtype: 'f32' },
  memoryPolicy: { kvcache: { layout: 'contiguous', kvDtype: 'f32' }, perLayerInputs: null, largeWeights: null },
  executionPlanDigest,
  runtimeEngine: { resolvedRuntimeSchema: 'doppler.resolved-runtime-session/v1', kernelPath: null },
};
const expectedIdentity = createInitialExecutionIdentity(fields);
const expectedIdentityV2 = createInitialExecutionIdentityV2({
  ...fields,
  programLoadPolicy: {
    schema: 'doppler.pack-program-load-policy/v2',
    runtimeConfig: {
      inference: {
        session: { decodeLoop: { batchSize: 1 } },
        compute: { outputDtype: 'f32' },
        generation: { disableMultiTokenDecode: true },
      },
    },
  },
});
const legacyPolicyIdentity = createInitialExecutionIdentityV2({
  ...fields,
  programLoadPolicy: {
    schema: 'doppler.pack-program-load-policy/v1',
    runtimeConfig: { inference: { session: {}, compute: {} } },
  },
});
assert.deepEqual(resolveProgramLoadRuntimeConfig(legacyPolicyIdentity), {
  inference: { session: {}, compute: {} },
});
assert.throws(
  () => createInitialExecutionIdentityV2({
    ...fields,
    programLoadPolicy: {
      schema: 'doppler.pack-program-load-policy/v2',
      runtimeConfig: { inference: { session: {}, compute: {} } },
    },
  }),
  /generation must be an object/,
  'identity v2 must bind every runtime input that changes execution-plan identity'
);
assert.throws(
  () => createInitialExecutionIdentityV2({
    ...fields,
    programLoadPolicy: {
      schema: 'doppler.pack-program-load-policy/v2',
      runtimeConfig: {
        inference: {
          session: {},
          compute: {},
          generation: { disableMultiTokenDecode: true, maxTokens: 128 },
        },
      },
    },
  }),
  /generation may contain only disableMultiTokenDecode/,
  'application generation policy must not leak into the signed program-load policy'
);
const fixture = await createSignedPackFixture({ initialExecutionIdentity: expectedIdentity });
const events = [];
let buffersCreated = 0;
const device = {
  getDevice() {
    return {
      limits: { maxBufferSize: 1024 },
      createBuffer() {
        buffersCreated += 1;
        return { destroy() {} };
      },
      createCommandEncoder() {},
      queue: { writeBuffer() {} },
    };
  },
  getProfile() {
    return { surface: 'test-webgpu', hasF16: false, hasSubgroups: false, maxBufferSize: 1024 };
  },
};
const baseProgram = {
  getInitialExecutionIdentity() { return expectedIdentityV2; },
  tokenize() { return [1]; }, decodeTokens() { return ''; }, getTokenContract() { return {}; }, reset() {},
  executePhase() { throw new Error('not reached'); }, releaseStepResult() {}, close() {},
};
const runtime = createDopplerRuntime({
  device,
  artifactStore: fixture.artifactStore,
  trustedSigners: { [TEST_PACK_AUTHORITY]: TEST_PACK_PUBLIC_KEY },
  observer: { observe(event) { events.push(event.type); } },
  async programFactory() { return baseProgram; },
});
const session = await runtime.openPack(fixture.pack);
assert.equal(session.observedInitialExecutionIdentity.digest, expectedIdentityV2.digest);
assertInitialExecutionIdentity(expectedIdentity, session.observedInitialExecutionIdentity);
assert.deepEqual(resolveProgramLoadRuntimeConfig(expectedIdentityV2), {
  inference: {
    session: { decodeLoop: { batchSize: 1 } },
    compute: { outputDtype: 'f32' },
    generation: { disableMultiTokenDecode: true },
  },
});
assert.equal(buffersCreated, 0, 'opening a bound session must not dispatch or allocate transient buffers');
assert.ok(events.includes('initial-execution-identity-bound'));
await session.close();

const mismatch = createInitialExecutionIdentityV2({
  ...fields,
  dtypeLane: { ...fields.dtypeLane, activation: 'f16' },
  programLoadPolicy: expectedIdentityV2.programLoadPolicy,
});
buffersCreated = 0;
let mismatchProgramClosed = false;
const mismatchedRuntime = createDopplerRuntime({
  device,
  artifactStore: fixture.artifactStore,
  trustedSigners: { [TEST_PACK_AUTHORITY]: TEST_PACK_PUBLIC_KEY },
  async programFactory() {
    return {
      ...baseProgram,
      getInitialExecutionIdentity() { return mismatch; },
      close() { mismatchProgramClosed = true; },
    };
  },
});
await assert.rejects(mismatchedRuntime.openPack(fixture.pack), /dtypeLane/);
assert.equal(buffersCreated, 0, 'identity mismatch must fail before resource binding or first dispatch');
assert.equal(mismatchProgramClosed, true, 'identity mismatch must close the loaded program');

for (const [field, replacement] of [
  ['dtypeLane', { ...fields.dtypeLane, activation: 'f16' }],
  ['fusionSet', [{ id: 'undeclared-fusion' }]],
  ['memoryPolicy', { ...fields.memoryPolicy, largeWeights: { residency: 'cpu' } }],
  ['executionGraphHash', digest('f')],
]) {
  let observedIdentity = expectedIdentityV2;
  const changing = createDopplerRuntime({
    device, artifactStore: fixture.artifactStore,
    trustedSigners: { [TEST_PACK_AUTHORITY]: TEST_PACK_PUBLIC_KEY },
    async programFactory() {
      return {
        ...baseProgram,
        getInitialExecutionIdentity() { return observedIdentity; },
        async encodeSequence() {
          observedIdentity = createInitialExecutionIdentityV2({ ...fields, [field]: replacement, programLoadPolicy: expectedIdentityV2.programLoadPolicy });
          return { pooledEmbedding: [1] };
        },
      };
    },
  });
  const changingSession = await changing.openPack(fixture.pack);
  await assert.rejects(changingSession.encodeSequence('MKT'), new RegExp(field));
  await changingSession.close();
}

const resolvedRuntimeSession = {
  schema: 'doppler.resolved-runtime-session/v1',
  id: digest('9'),
  manifestInference: {
    execution: {
      kernels: {
        main: { kernel: 'main.wgsl', entry: 'main', digest: digest('6') },
        recurrent: { kernel: 'gated_delta_recurrent.wgsl', entry: 'main', digest: digest('a') },
      },
      mechanismKernels: ['recurrent'],
      preLayer: [['embed', 'main']], prefill: [['attention', 'main']],
      decode: [['attention', 'main']], postLayer: [['sample', 'main']],
    },
  },
  runtime: {
    session: {
      kvcache: { layout: 'contiguous', kvDtype: 'f32' },
      perLayerInputs: { materialization: 'eager' },
      largeWeights: { residency: 'gpu' },
    },
    compute: { outputDtype: 'f32' },
  },
  execution: {
    primary: { id: 'primary', activationDtype: 'f32' },
    resolvedSteps: {
      prefill: [{ id: 'prefill.attention' }],
      decode: [{ id: 'decode.attention' }],
      all: [{ id: 'prefill.attention' }, { id: 'decode.attention' }],
    },
    resolvedStepsHash: digest('7'),
    appliedTransforms: [{ id: 'fuse-qkv' }],
  },
  dtypes: { activation: 'f32', output: 'f32', kv: 'f32', math: 'f32', accumulation: 'f32' },
  kernelPath: { id: 'portable', source: 'pack', hash: digest('8'), definition: {} },
  capabilityPolicy: { f16: false },
  laneIntegrity: { status: 'passed' },
};
const observed = observeInitialExecutionIdentity(resolvedRuntimeSession);
assert.equal(observed.kernelClosure[0].moduleId, 'main');
assert.equal(observed.kernelClosure[1].moduleId, 'recurrent');
assert.equal(observed.runtimeEngine.resolvedRuntimeSessionId, digest('9'));
assert.deepEqual(observed.fusionSet, [{ id: 'fuse-qkv' }]);
assert.deepEqual(observed.programLoadPolicy.runtimeConfig.inference.generation, {
  disableMultiTokenDecode: false,
});
for (const [field, mutate] of [
  ['executionGraphHash', (value) => { value.manifestInference.execution.decode.push(['extra', 'main']); }],
  ['resolvedGraphHash', (value) => { value.execution.resolvedSteps.all.push({ id: 'extra' }); }],
  ['dtypeLane', (value) => { value.dtypes.activation = 'f16'; }],
  ['fusionSet', (value) => { value.execution.appliedTransforms = []; }],
  ['kvLayout', (value) => { value.runtime.session.kvcache.layout = 'paged'; }],
  ['memoryPolicy', (value) => { value.runtime.session.perLayerInputs.materialization = 'lazy'; }],
  ['runtimeEngine', (value) => { value.kernelPath.id = 'different'; }],
  ['programLoadPolicy', (value) => { value.runtime.compute.outputDtype = 'f16'; }],
]) {
  const changedRuntime = structuredClone(resolvedRuntimeSession);
  mutate(changedRuntime);
  const changedIdentity = observeInitialExecutionIdentity(changedRuntime);
  assert.notEqual(changedIdentity.digest, observed.digest, `${field} must change observed identity`);
  assert.throws(() => assertInitialExecutionIdentity(observed, changedIdentity), new RegExp(field));
}

const legacyArraySteps = structuredClone(resolvedRuntimeSession);
legacyArraySteps.execution.resolvedSteps = [{ id: 'legacy' }];
assert.throws(
  () => observeInitialExecutionIdentity(legacyArraySteps),
  /missing compiled execution steps/,
  'observed identity must consume the canonical phase-partitioned execution-v1 shape'
);

console.log('✔ initial-execution-identity.test.js passed');
