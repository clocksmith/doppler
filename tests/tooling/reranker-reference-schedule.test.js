import assert from 'node:assert/strict';
import { assertRerankerRunCoverage, buildRerankerReferenceSchedule } from '../../tools/reranker-reference-schedule.js';

const reference = buildRerankerReferenceSchedule(null);
assert(reference.every(run => run.phase === 'reference'), 'Qualification cannot silently become timed evidence.');
for (const invalid of [undefined, {}, { warmupRuns: 1 }, { warmupRuns: 0, timedRuns: 0 },
  { warmupRuns: -1, timedRuns: 3 }, { warmupRuns: 1, timedRuns: 2.5 },
  { warmupRuns: 1, timedRuns: 3, discardFailures: true }]) {
  assert.throws(() => buildRerankerReferenceSchedule(invalid));
}
const schedule = buildRerankerReferenceSchedule({ warmupRuns: 1, timedRuns: 3 });
const complete = [
  { phase: 'warmup', iteration: 0, referenceIndex: 0 }, { phase: 'warmup', iteration: 0, referenceIndex: 1 },
  { phase: 'timed', iteration: 0, referenceIndex: 0 }, { phase: 'timed', iteration: 0, referenceIndex: 1 },
  { phase: 'timed', iteration: 1, referenceIndex: 0 }, { phase: 'timed', iteration: 1, referenceIndex: 1 },
  { phase: 'timed', iteration: 2, referenceIndex: 0 }, { phase: 'timed', iteration: 2, referenceIndex: 1 },
];
assertRerankerRunCoverage(complete, schedule, 2);
assert.throws(() => assertRerankerRunCoverage([], schedule, 2), /exact declared/);
assert.throws(() => assertRerankerRunCoverage(complete.slice(1), schedule, 2), /exact declared/);
const relabeled = structuredClone(complete); relabeled[0].phase = 'timed';
assert.throws(() => assertRerankerRunCoverage(relabeled, schedule, 2), /exact declared/);
const duplicated = structuredClone(complete); duplicated[1].referenceIndex = 0;
assert.throws(() => assertRerankerRunCoverage(duplicated, schedule, 2), /exact declared/);
console.log('reranker-reference-schedule.test: ok');
