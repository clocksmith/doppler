import assert from 'node:assert/strict';
import { createDopplerRuntimeService } from '../../src/client/runtime/index.js';

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

console.log('✔ pack-load-options.test.js passed');
