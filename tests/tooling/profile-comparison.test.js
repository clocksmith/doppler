import assert from 'node:assert/strict';
import { compareProfileRuns, validateProfilePairs } from '../../tools/lib/profile-comparison.js';

const output = { text: 'hello', tokenIds: [1, 2] };
const run = (instrumented, elapsedMs, firstMs) => ({
  instrumented, elapsedMs, tokens: [{ elapsedMs: firstMs }], completed: { output: structuredClone(output) },
});
const runs = [
  run(false, 500, 250), // cold first request cannot be the overhead control
  run(false, 100, 20), run(true, 120, 24),
  run(true, 90, 18), run(false, 100, 20),
];
const pairs = [{ ordinaryRun: 1, instrumentedRun: 2 }, { ordinaryRun: 4, instrumentedRun: 3 }];
const comparison = compareProfileRuns(runs, pairs);
assert.deepEqual(comparison.pairs[0].completeMs,
  { ordinary: 100, instrumented: 120, deltaMs: 20, deltaPercent: 20 });
assert.equal(comparison.pairs[0].firstOutputMs.deltaMs, 4);
assert.equal(comparison.pairs[1].completeMs.deltaMs, -10, 'retain favorable as well as unfavorable observations');
assert.equal(comparison.pairs[1].order, 'instrumented,ordinary');
assert.match(comparison.scope, /not independent opening samples/);
for (const invalid of [
  [], [{ ordinaryRun: 0, instrumentedRun: 1 }],
  [{ ordinaryRun: 1, instrumentedRun: 3 }],
  [{ ordinaryRun: 2, instrumentedRun: 1 }],
  [{ ordinaryRun: 4, instrumentedRun: 5 }],
  [pairs[0], pairs[0]],
]) assert.throws(() => validateProfilePairs(runs, invalid));
const changed = structuredClone(runs);
changed[2].completed.output.tokenIds = [9];
assert.throws(() => compareProfileRuns(changed, pairs), /changed the output/);
for (const badTiming of [NaN, Infinity, -1]) {
  const invalid = structuredClone(runs);
  invalid[2].elapsedMs = badTiming;
  assert.throws(() => compareProfileRuns(invalid, pairs), /invalid profiling timing/);
}
const missingDelivery = structuredClone(runs);
missingDelivery[2].tokens = [];
assert.equal(compareProfileRuns(missingDelivery, pairs).pairs[0].firstOutputMs, null);
const zero = structuredClone(runs);
zero[1].tokens[0].elapsedMs = 0;
assert.equal(compareProfileRuns(zero, pairs).pairs[0].firstOutputMs.deltaPercent, null);
console.log('profile-comparison.test: ok');
