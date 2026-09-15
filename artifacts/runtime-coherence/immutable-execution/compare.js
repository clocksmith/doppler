import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { computeSampleStats } from '../../../src/debug/stats.js';

const before = JSON.parse(await fs.readFile(new URL('../capsule-profile/receipt.json', import.meta.url)));
const candidate = JSON.parse(await fs.readFile(new URL('./receipt.json', import.meta.url)));
const after = JSON.parse(await fs.readFile(new URL('./baseline-after.json', import.meta.url)));
assert(before.passed && candidate.passed && after.passed);
assert.equal(before.package.sha256, after.package.sha256);
const output = before.runs[0].completed.output;
const summarize = receipt => {
  assert.deepEqual(receipt.config.model, before.config.model);
  assert.deepEqual(receipt.hardware, before.hardware);
  assert.equal(receipt.browserVersion, before.browserVersion);
  return receipt.runs.map(run => {
    assert.deepEqual(run.completed.output, output);
    const intervals = run.tokens.slice(1).map((token, index) => token.elapsedMs - run.tokens[index].elapsedMs);
    const timing = computeSampleStats(intervals, { outlierPolicy: 'none' });
    return { label: run.label, instrumented: run.instrumented, elapsedMs: run.elapsedMs,
      firstTokenMs: run.tokens[0].elapsedMs, decodeMedianMs: timing.median, decodeMeanMs: timing.mean,
      tokenCount: run.completed.output.tokenIds.length };
  });
};
const cohorts = { before: summarize(before), candidate: summarize(candidate), after: summarize(after) };
const controls = [...cohorts.before, ...cohorts.after].filter(run => !run.instrumented).map(run => run.decodeMedianMs);
const changed = cohorts.candidate.filter(run => !run.instrumented).map(run => run.decodeMedianMs);
const receipt = { schema: 'doppler.immutable-metadata-comparison/v1', passed: true,
  baselineArchiveSha256: before.package.sha256, candidateArchiveSha256: candidate.package.sha256,
  outputSha256: createHash('sha256').update(JSON.stringify(output)).digest('hex'),
  cohorts,
  baselineMedianRangeMs: [Math.min(...controls), Math.max(...controls)],
  candidateMedianRangeMs: [Math.min(...changed), Math.max(...changed)],
  consistentLocalReduction: Math.max(...changed) < Math.min(...controls),
  scope: 'Sequential baseline/candidate/baseline on one host, identical frozen requests and outputs. First and repeated uninstrumented requests are retained; instrumented runs are separate. This diagnoses removed CPU work, not a randomized throughput tournament or universal speedup claim.' };
await fs.writeFile(new URL('./comparison.json', import.meta.url), JSON.stringify(receipt, null, 2) + '\n');
console.log(JSON.stringify(receipt, null, 2));
