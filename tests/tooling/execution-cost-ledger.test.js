import assert from 'node:assert/strict';
import {
  buildTokenCostLedger,
  classifyTokenCostLedger,
  isExecutionObservationRequested,
} from '../../src/tooling/execution-cost-ledger.js';

const metrics = {
  prefillMs: 12,
  decodeMs: 8,
  prefillProfileSteps: [{
    timings: { matmul_q4: 7, attention_prefill: 2 },
    recorderStats: {
      opLabelCounts: { matmul_q4: 2, attention_prefill: 1 },
      dispatches: [
        { label: 'matmul_q4', kind: 'direct', workgroups: [2, 4, 1] },
        { label: 'matmul_q4', kind: 'direct', workgroups: [2, 4, 1] },
        { label: 'attention_prefill', kind: 'direct', workgroups: [1, 8, 1] },
      ],
    },
  }],
  decodeProfileSteps: [{
    timings: { lm_head_argmax: 3 },
    recorderStats: {
      opLabelCounts: { lm_head_argmax: 1 },
      dispatches: [{ label: 'lm_head_argmax', kind: 'indirect', workgroups: null }],
    },
  }],
  kernelPathId: 'test-path',
};
const identity = Object.fromEntries(
  ['artifactDigest', 'manifestDigest', 'executionGraphDigest'].map(
    (key, index) => [key, `sha256:${String(index + 1).repeat(64)}`]
  )
);
const ledger = buildTokenCostLedger({
  metrics,
  identity,
  device: { vendor: 'test' },
  browser: { version: 'test' },
});
assert.equal(ledger.schema, 'doppler.token-cost-ledger/v1');
assert.equal(ledger.phases[0].measurementSource, 'gpu-timestamp-query');
assert.equal(ledger.phases[0].attributedGpuMs, 9);
assert.equal(ledger.phases[0].unattributedWallMs, 3);
assert.equal(ledger.phases[0].dispatches, 3);
assert.equal(ledger.phases[0].operations[0].workgroups, 16);
assert.match(ledger.phases[0].overlapSemantics, /not asserted to equal wall time/);
assert.equal(ledger.phases[0].estimatedBytesMoved.semantics, 'estimated-not-measured');

const classification = classifyTokenCostLedger(ledger, {
  schema: 'doppler.token-cost-classifier-policy/v1',
  walls: [
    { id: 'projection', patterns: ['matmul', 'lm_head'], experiments: ['tiling'] },
    { id: 'attention', patterns: ['attention'], experiments: ['kv-layout'] },
  ],
});
assert.equal(classification.dominantWall, 'projection');
assert.deepEqual(classification.prescribedExperiments, ['tiling']);

const cpuEstimated = buildTokenCostLedger({
  metrics: {
    prefillMs: 5,
    decodeMs: 0,
    prefillProfileSteps: [{
      timings: null,
      recorderStats: {
        opLabelCounts: { matmul_q4: 1 },
        dispatches: [
          { label: 'matmul_q4', kind: 'direct', workgroups: [2, 2, 1] },
        ],
      },
    }],
  },
});
assert.equal(cpuEstimated.phases[0].measurementSource, 'cpu-wall-estimate');
assert.equal(cpuEstimated.phases[0].attributedGpuMs, null);
assert.equal(cpuEstimated.phases[0].dispatches, 1);
assert.equal(cpuEstimated.phases[0].operations[0].gpuMs, null);

assert.equal(isExecutionObservationRequested({
  shared: { benchmark: { run: { executionObserver: { enabled: true } } } },
}), true);
assert.equal(isExecutionObservationRequested({
  shared: { debug: { profiler: { enabled: true } } },
}), true);
assert.equal(isExecutionObservationRequested({ shared: {} }), false);

console.log('execution-cost-ledger.test: ok');
