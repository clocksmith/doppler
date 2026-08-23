import assert from 'node:assert/strict';
import { createDopplerRuntime } from '../../src/client/runtime/composition-root.js';
import {
  assertInitialExecutionIdentity,
  createInitialExecutionIdentity,
  observeInitialExecutionIdentity,
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
  getInitialExecutionIdentity() { return expectedIdentity; },
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
assert.equal(session.observedInitialExecutionIdentity.digest, expectedIdentity.digest);
assert.equal(buffersCreated, 0, 'opening a bound session must not dispatch or allocate transient buffers');
assert.ok(events.includes('initial-execution-identity-bound'));
await session.close();

const mismatch = createInitialExecutionIdentity({
  ...fields,
  dtypeLane: { ...fields.dtypeLane, activation: 'f16' },
});
buffersCreated = 0;
const mismatchedRuntime = createDopplerRuntime({
  device,
  artifactStore: fixture.artifactStore,
  trustedSigners: { [TEST_PACK_AUTHORITY]: TEST_PACK_PUBLIC_KEY },
  async programFactory() { return { ...baseProgram, getInitialExecutionIdentity() { return mismatch; } }; },
});
await assert.rejects(mismatchedRuntime.openPack(fixture.pack), /dtypeLane/);
assert.equal(buffersCreated, 0, 'identity mismatch must fail before resource binding or first dispatch');

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
  },
  execution: {
    primary: { id: 'primary', activationDtype: 'f32' },
    resolvedSteps: [{ id: 'prefill.attention' }, { id: 'decode.attention' }],
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
for (const [field, mutate] of [
  ['executionGraphHash', (value) => { value.manifestInference.execution.decode.push(['extra', 'main']); }],
  ['resolvedGraphHash', (value) => { value.execution.resolvedSteps.push({ id: 'extra' }); }],
  ['dtypeLane', (value) => { value.dtypes.activation = 'f16'; }],
  ['fusionSet', (value) => { value.execution.appliedTransforms = []; }],
  ['kvLayout', (value) => { value.runtime.session.kvcache.layout = 'paged'; }],
  ['memoryPolicy', (value) => { value.runtime.session.perLayerInputs.materialization = 'lazy'; }],
  ['runtimeEngine', (value) => { value.kernelPath.id = 'different'; }],
]) {
  const changedRuntime = structuredClone(resolvedRuntimeSession);
  mutate(changedRuntime);
  const changedIdentity = observeInitialExecutionIdentity(changedRuntime);
  assert.notEqual(changedIdentity.digest, observed.digest, `${field} must change observed identity`);
  assert.throws(() => assertInitialExecutionIdentity(observed, changedIdentity), new RegExp(field));
}

console.log('✔ initial-execution-identity.test.js passed');
