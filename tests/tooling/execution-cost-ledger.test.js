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
        { label: 'matmul_q4', kind: 'direct', workgroups: [2, 4, 1], count: 2 },
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
  decodeRecordMs: 2,
  decodeSubmitWaitMs: 5,
  decodeReadbackWaitMs: 4,
  decodeReadbackMapWaitMs: 3,
  decodeReadbackCleanupMs: 0.5,
  decodeReadbackCopyMs: 0.25,
};
const identity = Object.fromEntries(
  ['artifactDigest', 'manifestDigest', 'executionGraphDigest'].map(
    (key, index) => [key, `sha256:${String(index + 1).repeat(64)}`]
  )
);
const ledger = buildTokenCostLedger({
  metrics,
  identity,
  device: {
    hasF16: true,
    submitProbeMs: 0.5,
    adapterInfo: { vendor: 'test', architecture: 'test-arch' },
  },
  browser: { version: 'test' },
});
assert.equal(ledger.schema, 'doppler.token-cost-ledger/v1');
assert.equal(ledger.phases[0].measurementSource, 'gpu-timestamp-query');
assert.equal(ledger.phases[0].attributedGpuMs, 9);
assert.equal(ledger.phases[0].unattributedWallMs, 3);
assert.equal(ledger.phases[0].dispatches, 3);
assert.equal(ledger.phases[0].unattributedDispatches, 0);
assert.equal(ledger.phases[0].operations[0].workgroups, 16);
assert.match(ledger.phases[0].overlapSemantics, /not asserted to equal wall time/);
assert.equal(ledger.phases[0].estimatedBytesMoved.semantics, 'estimated-not-measured');
assert.equal(ledger.phases[1].hostCosts.fenceWaitMs, 5);
assert.equal(ledger.phases[1].hostCosts.observedSerialMs, 7);
assert.equal(ledger.phases[1].hostCosts.dominantWall, 'submit-readback-fence');
assert.match(ledger.phases[1].hostCosts.overlapSemantics, /not added again/);
assert.deepEqual(ledger.dominantObservedWall, {
  phase: 'prefill',
  wall: 'gpu-operation',
  ms: 9,
});

const sameAdapterDifferentProbe = buildTokenCostLedger({
  metrics,
  identity,
  device: {
    hasF16: true,
    submitProbeMs: 9.5,
    adapterInfo: { vendor: 'test', architecture: 'test-arch' },
  },
  browser: { version: 'test' },
});
assert.equal(
  ledger.identity.adapterDigest,
  sameAdapterDifferentProbe.identity.adapterDigest,
  'adapter identity must exclude volatile submit-probe latency'
);

const classification = classifyTokenCostLedger(ledger, {
  schema: 'doppler.token-cost-classifier-policy/v1',
  walls: [
    { id: 'projection', patterns: ['matmul', 'lm_head'], experiments: ['tiling'] },
    { id: 'attention', patterns: ['attention'], experiments: ['kv-layout'] },
  ],
});
assert.equal(classification.dominantWall, 'projection');
assert.deepEqual(classification.prescribedExperiments, ['tiling']);
assert.equal(classification.classifiedHostMs[0].wall, 'submit-readback-fence');

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

const lightweightEstimated = buildTokenCostLedger({
  metrics: {
    decodeMs: 6,
    decodeRecordMs: 2,
    gpu: {
      decodeRecordOps: { median: 25, mean: 25 },
      decodeSubmitWaitMs: { median: 3, mean: 3 },
      decodeRecordTopOps: [
        { label: 'fused_ffn', count: 12, shareOfOps: 0.6 },
        { label: 'attention', count: 8, shareOfOps: 0.4 },
      ],
    },
  },
});
assert.equal(lightweightEstimated.phases[1].measurementSource, 'cpu-wall-estimate');
assert.equal(lightweightEstimated.phases[1].dispatches, 25);
assert.equal(lightweightEstimated.phases[1].unattributedDispatches, 5);
assert.equal(lightweightEstimated.phases[1].dispatchGeometryCoverage, 0);
assert.equal(lightweightEstimated.phases[1].hostCosts.recordMs, 2);
assert.equal(lightweightEstimated.phases[1].hostCosts.submitWaitMs, 3);
assert.deepEqual(
  lightweightEstimated.phases[1].operations.map(({ label, dispatches }) => ({ label, dispatches })),
  [
    { label: 'fused_ffn', dispatches: 12 },
    { label: 'attention', dispatches: 8 },
  ]
);

const fenceDominated = buildTokenCostLedger({
  metrics: {
    decodeMs: 12,
    decodeRecordMs: 2,
    decodeSubmitWaitMs: 9,
    decodeReadbackWaitMs: 8,
  },
});
const fenceClassification = classifyTokenCostLedger(fenceDominated, {
  schema: 'doppler.token-cost-classifier-policy/v1',
  walls: [
    { id: 'projection', patterns: ['matmul'], experiments: ['tiling'] },
  ],
});
assert.equal(fenceClassification.dominantWall, 'submit-readback-fence');
assert.ok(fenceClassification.prescribedExperiments.includes('deepen-readback-ring'));

assert.equal(isExecutionObservationRequested({
  shared: { benchmark: { run: { executionObserver: { enabled: true } } } },
}), true);
assert.equal(isExecutionObservationRequested({
  shared: { debug: { profiler: { enabled: true } } },
}), true);
assert.equal(isExecutionObservationRequested({ shared: {} }), false);

console.log('execution-cost-ledger.test: ok');
