import assert from 'node:assert/strict';

import { createConversionRunTiming } from '../../src/tooling/conversion-run-timing.js';

const dates = [
  new Date('2026-08-23T12:00:00.000Z'),
  new Date('2026-08-23T12:00:01.250Z'),
];
const monotonicValues = [100, 1350];
const timing = createConversionRunTiming({
  now: () => dates.shift(),
  monotonicNow: () => monotonicValues.shift(),
});
const receipt = timing.complete();

assert.deepEqual(receipt, {
  startedAtUtc: '2026-08-23T12:00:00.000Z',
  completedAtUtc: '2026-08-23T12:00:01.250Z',
  durationMs: 1250,
});
assert.equal(timing.complete(), receipt);

console.log('conversion-run-timing.test: ok');
