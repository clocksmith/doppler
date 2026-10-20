import assert from 'node:assert/strict';

/**
 * @typedef {{ ordinaryRun: number, instrumentedRun: number }} ProfilePair
 * @typedef {{ instrumented: boolean, elapsedMs?: number,
 *   tokens?: Array<{ elapsedMs: number }>, completed?: { output: unknown } }} ProfileRun
 */

// Pair selection is declared before execution, never chosen from favorable timings.
/** @param {ProfileRun[]} runs @param {ProfilePair[]} pairs */
export function validateProfilePairs(runs, pairs) {
  assert(Array.isArray(pairs) && pairs.length > 0, 'overheadPairs must be nonempty');
  assert.equal(runs[0]?.instrumented, false, 'first request must be ordinary warmup');
  const used = new Set();
  for (const { ordinaryRun, instrumentedRun } of pairs) {
    for (const index of [ordinaryRun, instrumentedRun]) {
      assert(Number.isSafeInteger(index) && index > 0 && index < runs.length,
        'overhead pair indices must exclude the first request and exist in runs');
      assert(!used.has(index), 'overhead pairs must not reuse a request');
      used.add(index);
    }
    assert.equal(Math.abs(ordinaryRun - instrumentedRun), 1,
      'overhead comparisons require adjacent requests');
    assert.equal(runs[ordinaryRun].instrumented, false, 'ordinary arm must be uninstrumented');
    assert.equal(runs[instrumentedRun].instrumented, true, 'instrumented arm must be instrumented');
  }
}

/** @param {number | undefined} ordinary @param {number | undefined} instrumented */
function compareTiming(ordinary, instrumented) {
  assert(typeof ordinary === 'number' && Number.isFinite(ordinary) && ordinary >= 0
    && typeof instrumented === 'number' && Number.isFinite(instrumented) && instrumented >= 0,
  'invalid profiling timing');
  return {
    ordinary,
    instrumented,
    deltaMs: instrumented - ordinary,
    deltaPercent: ordinary > 0 ? 100 * (instrumented - ordinary) / ordinary : null,
  };
}

/** @param {ProfileRun[]} runs @param {ProfilePair[]} pairs */
export function compareProfileRuns(runs, pairs) {
  validateProfilePairs(runs, pairs);
  return {
    scope: 'Within-session diagnostic overhead, including CPU sampling and API observation. '
      + 'First request excluded. No outlier removal, GPU-kernel attribution, or speedup claim. '
      + 'Repeated requests share one opening and are not independent opening samples.',
    pairs: pairs.map(({ ordinaryRun, instrumentedRun }) => {
      const ordinary = runs[ordinaryRun];
      const instrumented = runs[instrumentedRun];
      assert(ordinary.completed?.output && instrumented.completed?.output,
        'both overhead arms must complete');
      assert.deepEqual(instrumented.completed.output, ordinary.completed.output,
        'instrumentation changed the output');
      const ordinaryFirst = ordinary.tokens?.[0]?.elapsedMs;
      const instrumentedFirst = instrumented.tokens?.[0]?.elapsedMs;
      return {
        ordinaryRun,
        instrumentedRun,
        order: ordinaryRun < instrumentedRun ? 'ordinary,instrumented' : 'instrumented,ordinary',
        completeMs: compareTiming(ordinary.elapsedMs, instrumented.elapsedMs),
        firstOutputMs: ordinaryFirst == null || instrumentedFirst == null
          ? null : compareTiming(ordinaryFirst, instrumentedFirst),
      };
    }),
  };
}
