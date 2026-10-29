import assert from 'node:assert/strict';
import { buildDecodeBottleneckDiagnostics } from '../../src/tooling/browser-harness/decode-diagnostics.js';

// A CPU recorder active for 80 ms can overlap 88 ms of GPU work in a 100 ms
// window. These counters cannot partition that window or establish saved time.
const result = buildDecodeBottleneckDiagnostics({
  decodeMs: 100,
  gpu: {
    decodeRecordMs: 80,
    decodeSubmitWaitMs: 30,
    decodeReadbackWaitMs: 32,
    decodeReadbackMapWaitMs: 30,
    decodeReadbackCleanupMs: 1,
    decodeReadbackCopyMs: 1,
    decodeOrchestrationMs: 2,
    decodeMs: 88,
  },
});
assert.equal(result.schemaVersion, 2);
assert.equal(result.dominant.id, 'gpu_compute');
assert.equal(result.bottleneckClass, null);
assert.equal(result.attribution.status, 'unresolved');
assert.equal(result.componentsMs.commandRecordMs, 80);
assert.equal(result.componentsMs.gpuTimestampMs, 88);
assert.equal(result.componentsMs.effectiveSubmitReadbackWaitMs, 32);
for (const field of ['submitReadbackSlackMs', 'readbackUnattributedMs', 'residualMs']) {
  assert.equal(result.componentsMs[field], null);
}
assert.equal(result.shares.commandRecord, 0.8);
assert.equal(result.shares.gpuTimestamp, 0.88);
assert.match(result.semantics.shares, /not additive/);

// Partial timestamp totals do not establish a host-only remainder either.
const partial = buildDecodeBottleneckDiagnostics({ decodeMs: 100, gpu: { decodeRecordMs: 80, decodeReadbackWaitMs: 32, decodeMs: 5 } });
assert.equal(partial.dominant.id, 'command_record');
assert.equal(partial.bottleneckClass, null);
assert.equal(partial.componentsMs.submitReadbackSlackMs, null);
assert.equal(partial.componentsMs.residualMs, null);
const unprofiled = buildDecodeBottleneckDiagnostics({ decodeMs: 100, gpu: { decodeRecordMs: 80 } });
assert.equal(unprofiled.componentsMs.gpuTimestampMs, null);
assert.equal(unprofiled.bottleneckClass, null);
assert.equal(buildDecodeBottleneckDiagnostics({ decodeMs: 0 }), null);
console.log('decode-diagnostics-overlap.test: ok');

const { printMetricsSummary } = await import('../../src/cli/cli-output.js');
const lines = [], previousLog = console.log;
try {
  console.log = (...args) => lines.push(args.join(' '));
  printMetricsSummary({ suite: 'bench', metrics: { gpu: {}, decodeBottleneck: result } });
} finally {
  console.log = previousLog;
}
assert.ok(lines.some((line) => line.includes('largest_observed_interval=') && line.includes('causal bottleneck unknown')));
assert.ok(lines.every((line) => !line.includes('[gpu] bottleneck=')));
