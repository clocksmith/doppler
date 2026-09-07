import assert from 'node:assert/strict';

// Qualification has no performance population. Calibration requires an explicit
// policy; both browser engines consume the same expanded run schedule.
export function buildRerankerReferenceSchedule(sampling) {
  if (sampling === null) return [{ phase: 'reference', iteration: 0 }];
  assert(sampling && typeof sampling === 'object' && !Array.isArray(sampling));
  assert.deepEqual(Object.keys(sampling).sort(), ['timedRuns', 'warmupRuns']);
  assert(Number.isSafeInteger(sampling.warmupRuns) && sampling.warmupRuns >= 0);
  assert(Number.isSafeInteger(sampling.timedRuns) && sampling.timedRuns > 0);
  return [
    ...Array.from({ length: sampling.warmupRuns }, (_, iteration) => ({ phase: 'warmup', iteration })),
    ...Array.from({ length: sampling.timedRuns }, (_, iteration) => ({ phase: 'timed', iteration })),
  ];
}

export function assertRerankerRunCoverage(runs, schedule, referenceCount) {
  assert(Number.isSafeInteger(referenceCount) && referenceCount > 0);
  const expected = schedule.flatMap(sample => Array.from({ length: referenceCount }, (_, referenceIndex) => ({ ...sample, referenceIndex })));
  assert.deepEqual(runs.map(({ phase, iteration, referenceIndex }) => ({ phase, iteration, referenceIndex })), expected,
    'Observed reranking must cover the exact declared run schedule and reference inputs.');
}
