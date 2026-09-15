import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { computeSampleStats } from '../../../src/debug/stats.js';

const receiptPath = path.resolve(process.argv[2]);
const receiptBytes = await fs.readFile(receiptPath);
const receipt = JSON.parse(receiptBytes);
assert(receipt.passed);
const stats = values => computeSampleStats(values, { outlierPolicy: 'none' });
const unionMs = rows => {
  const intervals = rows.filter(row => row.kind.endsWith('-wait'))
    .map(row => [row.startMs, row.startMs + row.durationMs]).sort((a, b) => a[0] - b[0]);
  let end = -Infinity, total = 0;
  for (const [start, stop] of intervals) { total += Math.max(0, stop - Math.max(start, end)); end = Math.max(end, stop); }
  return total;
};
const results = [];
for (const run of receipt.runs) {
  const result = { label: run.label, instrumented: run.instrumented,
    tokenCount: run.completed.output.tokenIds.length, elapsedMs: run.elapsedMs,
    firstTokenMs: run.tokens[0].elapsedMs,
    decodeTokenLatencyMs: stats(run.tokens.slice(1).map((token, index) => token.elapsedMs - run.tokens[index].elapsedMs)),
    stopReason: run.completed.output.completion.stopReason };
  if (run.gpu) {
    const firstRows = run.tokens[0].gpu.rows;
    const summarize = rows => {
      const readbacks = {};
      for (const row of rows.filter(row => row.kind === 'readback-copy')) {
        const value = readbacks[row.source] ??= { calls: 0, bytes: 0 };
        value.calls++; value.bytes += row.bytes;
      }
      return { readbacks, fenceUnionMs: unionMs(rows),
        queueWaitMs: rows.filter(row => row.kind === 'queue-wait').reduce((sum, row) => sum + row.durationMs, 0),
        mapWaitMs: rows.filter(row => row.kind === 'map-wait').reduce((sum, row) => sum + row.durationMs, 0) };
    };
    result.firstTokenGpu = { ...summarize(run.gpu.rows.slice(0, firstRows)), counts: run.tokens[0].gpu.counts };
    result.decodeGpu = { ...summarize(run.gpu.rows.slice(firstRows)), counts: Object.fromEntries(
      Object.entries(run.gpu.counts).map(([key, value]) => [key, value - (run.tokens[0].gpu.counts[key] ?? 0)])) };
    const profile = JSON.parse(await fs.readFile(path.join(path.dirname(receiptPath), run.cpuProfile)));
    const nodes = new Map(profile.nodes.map(node => [node.id, node]));
    const parents = new Map(profile.nodes.flatMap(node => (node.children ?? []).map(child => [child, node.id])));
    const totals = new Map();
    const inclusive = { sampleCapsuleLogits: 0, sample: 0, applyRepetitionPenalty: 0, applyPresencePenalty: 0 };
    assert.equal(profile.samples.length, profile.timeDeltas.length);
    for (const [index, id] of profile.samples.entries()) {
      const ms = profile.timeDeltas[index] / 1000;
      const frame = nodes.get(id).callFrame;
      const key = `${frame.functionName} (${frame.url}:${frame.lineNumber + 1})`;
      totals.set(key, (totals.get(key) ?? 0) + ms);
      const matched = new Set();
      for (let ancestor = id; ancestor !== undefined; ancestor = parents.get(ancestor)) {
        const name = nodes.get(ancestor).callFrame.functionName;
        if (Object.hasOwn(inclusive, name) && !matched.has(name)) { inclusive[name] += ms; matched.add(name); }
      }
    }
    result.cpuProfile = { sampleCount: profile.samples.length,
      durationMs: (profile.endTime - profile.startTime) / 1000,
      inclusiveEstimatedMs: inclusive,
      largestSelfSamples: [...totals].sort((a, b) => b[1] - a[1]).slice(0, 25).map(([frame, estimatedMs]) => ({ frame, estimatedMs })),
      caveat: 'Statistical stack samples including observation overhead, not exact timers or independent additive costs.' };
  }
  results.push(result);
}
const summary = { schema: 'doppler.installed-capsule-profile-summary/v1', passed: true,
  receiptSha256: createHash('sha256').update(receiptBytes).digest('hex'),
  runtimeArchiveSha256: receipt.package.sha256, hardware: receipt.hardware,
  load: receipt.load, runs: results,
  scope: 'One local diagnostic population, cold first request and warm repeated requests. No baseline/candidate speedup comparison. Readback bytes count API copies, not measured bus throughput. Queue and map waits overlap; use fence union, not their sum.' };
await fs.writeFile(new URL('./summary.json', import.meta.url), JSON.stringify(summary, null, 2) + '\n');
console.log(JSON.stringify(summary, null, 2));
