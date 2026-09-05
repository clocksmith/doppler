import assert from 'node:assert/strict';
import { createCommandExecutor } from '../../src/client/runtime/command-executor.js';
import { readbackBuffer } from '../../src/gpu/readback-buffer.js';
import { executePackForecast } from '../../src/client/runtime/pack-forecast.js';
import { validatePackReleaseContract } from '../../src/config/pack-release-contract.js';
import { createPackReleaseFixture } from '../helpers/pack-v2-fixture.js';
import { validateTargetPlan } from '../../src/config/target-plan.js';

// Same shader/entry, different overrides: must create distinct pipelines.
const pipelines = [];
const device = {
  createShaderModule: value => value,
  async createComputePipelineAsync(descriptor) {
    pipelines.push(descriptor.compute.constants);
    return { getBindGroupLayout() { return {}; } };
  },
  createBindGroup() { return {}; },
  createCommandEncoder() {
    return { beginComputePass: () => ({ setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {} }), finish() {} };
  },
  queue: { submit() {} },
};
const executor = createCommandExecutor(device, { getSlot: () => ({ buffer: {} }) });
const modules = new Map([['probe', { id: 'probe', sourceHash: 'source', source: 'shader', entry: 'main' }]]);
const dispatch = constants => ({ kind: 'dispatch', moduleId: 'probe', constants, bindings: [], workgroups: [1] });
await executor.executePhase('forecast', [dispatch({ WIDTH: 4 }), dispatch({ WIDTH: 8 }), dispatch({ WIDTH: 4 })], { modules });
assert.deepEqual(pipelines, [{ WIDTH: 4 }, { WIDTH: 8 }]);

globalThis.GPUBufferUsage ??= { COPY_DST: 8, MAP_READ: 1 };
globalThis.GPUMapMode ??= { READ: 1 };
let destroyed = 0;
const failedReadback = { createBuffer: () => ({ mapAsync: async () => { throw new Error('map failed'); }, destroy() { destroyed++; } }),
  createCommandEncoder: () => ({ copyBufferToBuffer() {}, finish() {} }), queue: { submit() {} } };
await assert.rejects(readbackBuffer(failedReadback, { size: 4 }, 4), /map failed/);
assert.equal(destroyed, 1);

const release = createPackReleaseFixture();
release.lifecycle.supersedes = null;
release.lifecycle.migration = null;
release.lifecycle.failedUpgrade.previousPackId = null;
release.lifecycle.failedUpgrade.previousSemanticRoot = null;
assert.equal(validatePackReleaseContract(release).ok, true);
release.lifecycle.failedUpgrade.previousPackId = 'invented';
assert.equal(validatePackReleaseContract(release).ok, false);

const application = { applicationId: 'forecast-test', applicationRevision: '1' };
const base = { identity: {}, release: { application }, targetPlan: { targetId: 'test', qualification: [{ operation: 'forecast' }] },
  targetPlanDigest: 'test', artifactReceipts: [], releaseEventDigest: null,
  program: { async forecast(request) { return { horizon: request.horizon, layout: 'time-quantile', quantileLevels: [0.5], values: [7] }; } } };
const request = { application, context: [1, 2, 3], horizon: 1, assignmentHash: null };
const result = await executePackForecast({ ...base, request });
assert.equal(result.receipt.operation, 'forecast');
assert.match(result.receipt.inputHash, /^sha256:/);
await assert.rejects(executePackForecast({ ...base, request: { ...request, application: { ...application, applicationRevision: '2' } } }), /application identity/);
await assert.rejects(executePackForecast({ ...base, request: { ...request, assignmentHash: undefined } }), /Invalid/);
await assert.rejects(executePackForecast({ ...base, request, program: { async forecast() { return { ...result, values: [NaN] }; } } }), /Malformed/);
const cancelled = new AbortController(); cancelled.abort(new Error('cancelled'));
await assert.rejects(executePackForecast({ ...base, request: { ...request, signal: cancelled.signal } }), /cancelled/);
assert.equal(validateTargetPlan({ phases: { forecast: [] }, qualification: [{ operation: 'generate' }] }).ok, false);
console.log('✔ pack-forecast.test.js passed');
