import assert from 'node:assert/strict';
import {
  createDopplerRuntimeService,
  resolvePackProgramLoadOptions,
} from '../../src/client/runtime/index.js';
import { createInitialExecutionIdentityV2 } from '../../src/config/initial-execution-identity.js';

let webGpuChecks = 0;
const service = createDopplerRuntimeService({
  async ensureWebGPUAvailable() { webGpuChecks += 1; },
  async resolvePackInput() { throw new Error('must not resolve a Pack after policy override rejection'); },
});
await assert.rejects(
  service.openPack({}, { modelLoadOptions: { runtimeConfig: { inference: {} } } }),
  /prohibits modelLoadOptions because signed TargetPlan policy is authoritative/,
);
assert.equal(webGpuChecks, 0);

const digest = (character) => `sha256:${character.repeat(64)}`;
const programLoadPolicy = {
  schema: 'doppler.pack-program-load-policy/v1',
  runtimeConfig: {
    inference: {
      session: { decodeLoop: { batchSize: 1, readbackInterval: 1 } },
      compute: { outputDtype: 'f32' },
    },
  },
};
const initialExecutionIdentity = createInitialExecutionIdentityV2({
  executionGraphHash: digest('1'),
  resolvedGraphHash: digest('2'),
  kernelClosure: [{ moduleId: 'main', file: 'main.wgsl', entry: 'main', digest: digest('3') }],
  dtypeLane: { activation: 'f32', kv: 'f16' },
  fusionSet: [],
  kvLayout: { layout: 'contiguous', kvDtype: 'f16' },
  memoryPolicy: { kvcache: { layout: 'contiguous', kvDtype: 'f16' } },
  executionPlanDigest: digest('4'),
  runtimeEngine: { schema: 'doppler.resolved-runtime-session/v1' },
  programLoadPolicy,
});
assert.deepEqual(
  resolvePackProgramLoadOptions({ initialExecutionIdentity }),
  { runtimeConfig: programLoadPolicy.runtimeConfig }
);
assert.deepEqual(resolvePackProgramLoadOptions({}), {});

console.log('✔ pack-load-options.test.js passed');
