import assert from 'node:assert/strict';

const median = values => {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
};

function tokens(run) { return run.evidence?.tokenIds ?? run.tokenIds; }

export function compare(left, right) {
  assert.equal(left.complete, true, 'Left engine did not complete');
  assert.equal(right.complete, true, 'Right engine did not complete');
  assert.deepEqual(left.policy.source, right.policy.source, 'Source identity differs');
  assert.deepEqual(left.policy.shared, right.policy.shared, 'Shared workload differs');
  const policy = left.policy.shared;
  const count = policy.promptTokenCounts.length * (policy.warmupRuns + policy.timedRuns);
  assert.equal(left.runs.length, count, 'Left run count differs');
  assert.equal(right.runs.length, count, 'Right run count differs');
  for (let i = 0; i < count; i++) {
    const a = left.runs[i], b = right.runs[i];
    assert.equal(a.warmup, b.warmup, 'Warmup ordering differs');
    assert.deepEqual(a.promptTokenIds, b.promptTokenIds, 'Prompt IDs differ');
    assert.equal(tokens(a).length, policy.maxTokens, 'Left output length differs');
    assert.equal(tokens(b).length, policy.maxTokens, 'Right output length differs');
    assert.deepEqual(tokens(a), tokens(b), `Generated IDs differ at run ${i}`);
    for (const run of [a, b]) {
      assert.equal(run.tokenTimes.length, policy.maxTokens, 'Token delivery count differs');
      assert(run.tokenTimes.every((ms, n) => Number.isFinite(ms) && ms > 0
        && (n === 0 || ms >= run.tokenTimes[n - 1])), 'Invalid delivery timing');
      assert(Number.isFinite(run.totalRunMs) && run.totalRunMs >= run.tokenTimes.at(-1), 'Invalid wall timing');
      assert(run.tokenTimes.at(-1) > run.tokenTimes[0], 'Invalid decode duration');
    }
  }
  return {
    exactTokenAgreement: true,
    claim: `Sequential diagnostic, ${policy.timedRuns} timed repetitions per length; not an interleaved performance ranking.`,
    timingBoundary: 'Prompt-to-first delivered token; decode uses 127 delivery intervals for 128 tokens. Includes orchestration; not pure GPU time.',
    measurements: policy.promptTokenCounts.map(length => ({
      promptTokens: length,
      engines: [left, right].map(result => {
        const runs = result.runs.filter(run => !run.warmup && run.promptTokenIds.length === length);
        assert.equal(runs.length, policy.timedRuns);
        return {
          engine: result.engine,
          totalRunMsMedian: median(runs.map(run => run.totalRunMs)),
          promptToFirstTokenMsMedian: median(runs.map(run => run.tokenTimes[0])),
          decodeTokensPerSecondMedian: median(runs.map(run =>
            1000 * (run.tokenTimes.length - 1) / (run.tokenTimes.at(-1) - run.tokenTimes[0]))),
        };
      }),
    })),
  };
}
