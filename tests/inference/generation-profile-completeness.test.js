import assert from 'node:assert/strict';
import { submitDecodeRecorderProfile } from '../../src/inference/pipelines/text/generator/decode.js';

// Exercise the generation consumer, independently of recorder coverage tests.
for (const enabled of [true, false]) {
  const state = { stats: {}, decodeStepCount: 1 };
  let submitted = false;
  const recorder = {
    async submitAndWait() { submitted = true; },
    isProfilingEnabled() { return enabled; },
    getStats() { return { profiling: { untimestampedPassCount: 1 } }; },
    async resolveProfileTimings(options) {
      if (options?.requireComplete) {
        throw new Error(enabled ? 'incomplete timestamp coverage' : 'timestamp profiling unavailable');
      }
      return enabled ? { projection: 1 } : null;
    },
  };
  await assert.rejects(submitDecodeRecorderProfile(state, { profile: true, executionObserver: true }, recorder, ''),
    enabled ? /incomplete timestamp coverage/ : /timestamp profiling unavailable/);
  assert.equal(submitted, true);
  assert.equal(state.stats.gpuTimeDecodeMs, undefined, 'failed observations cannot become phase totals');
}
const measured = { stats: {}, decodeStepCount: 1 };
await submitDecodeRecorderProfile(measured, { profile: true, executionObserver: true }, {
  async submitAndWait() {},
  async resolveProfileTimings(options) {
    assert.equal(options.requireComplete, true);
    return { projection: 5, attention: 2 };
  },
  getStats() { return { profiling: { untimestampedPassCount: 0 } }; },
}, '');
assert.equal(measured.stats.gpuTimeDecodeMs, 7);
assert.equal(measured.stats.decodeProfileSteps[0].totalMs, 7);
const ordinary = { async submitAndWait() {}, resolveProfileTimings() { throw new Error('ordinary readback'); } };
await submitDecodeRecorderProfile({ stats: {} }, { profile: false }, ordinary, '');
console.log('generation-profile-completeness.test: ok');
