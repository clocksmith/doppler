import assert from 'node:assert/strict';
import {
  compareRefactorReceipts,
  createRefactorReceipt,
  verifyRefactorReceipt,
} from '../../src/inference/refactor-receipt.js';
import { captureAttentionRefactorReceipt } from '../../src/inference/pipelines/text/attention/receipt.js';

const input = {
  commandContext: {
    workload: 'inference',
    command: 'verify',
    intent: 'verify',
  },
  resolvedSession: {
    kernelPathId: 'fixture',
    activationDtype: 'f16',
  },
  observationContext: {
    diagnostics: 'on_failure',
  },
  operationPlan: {
    contract: {
      phase: 'decode',
      numTokens: 1,
    },
  },
  operations: [
    { id: 'q_proj', order: 0 },
    { id: 'attention', order: 1 },
  ],
  dtypeTransitions: [
    { from: 'f16', to: 'f32', declaredBy: 'step_precision' },
  ],
  resourceEvents: [
    { resourceId: 'q', event: 'acquire', owner: 'scope' },
    { resourceId: 'q', event: 'release', owner: 'scope' },
  ],
  failure: null,
};

const receipt = createRefactorReceipt(input);
assert.equal(verifyRefactorReceipt(receipt), true);
assert.equal(Object.isFrozen(receipt), true);
assert.equal(Object.isFrozen(receipt.operations), true);

const reordered = createRefactorReceipt({
  ...input,
  commandContext: {
    intent: 'verify',
    command: 'verify',
    workload: 'inference',
  },
});
assert.equal(receipt.receiptHash, reordered.receiptHash);
assert.deepEqual(compareRefactorReceipts(receipt, reordered), {
  matches: true,
  semanticMatches: true,
  executionMatches: true,
  differences: [],
});

const observationChange = createRefactorReceipt({
  ...input,
  observationContext: {
    diagnostics: 'always',
  },
});
const observationComparison = compareRefactorReceipts(receipt, observationChange);
assert.equal(observationComparison.matches, false);
assert.equal(observationComparison.semanticMatches, true);
assert.equal(observationComparison.executionMatches, true);
assert.deepEqual(observationComparison.differences, ['observationContext']);

const intentChange = createRefactorReceipt({
  ...input,
  commandContext: {
    command: 'bench',
    workload: 'inference',
    intent: 'calibrate',
  },
});
const intentComparison = compareRefactorReceipts(receipt, intentChange);
assert.equal(intentComparison.matches, false);
assert.equal(intentComparison.semanticMatches, true);
assert.equal(intentComparison.executionMatches, true);
assert.deepEqual(intentComparison.differences, ['commandContext']);

const operationChange = createRefactorReceipt({
  ...input,
  operations: [
    { id: 'q_proj', order: 0 },
    { id: 'attention_changed', order: 1 },
  ],
});
const operationComparison = compareRefactorReceipts(receipt, operationChange);
assert.equal(operationComparison.semanticMatches, true);
assert.equal(operationComparison.executionMatches, false);
assert.deepEqual(operationComparison.differences, ['executionHash', 'operations']);

assert.throws(
  () => createRefactorReceipt({ operationPlan: { data: new Float32Array([1]) } }),
  /embedding binary data/
);

const tampered = {
  ...receipt,
  operations: [{ id: 'tampered', order: 0 }],
};
assert.throws(() => verifyRefactorReceipt(tampered), /executionHash mismatch/);

console.log('refactor-receipt.test: ok');

{
  const state = {
    resolvedRuntimeSession: {
      schema: 'doppler.resolved-runtime-session/v1',
      id: 'session',
    },
    observationContext: {
      schema: 'doppler.observation-context/v1',
      commandContext: {
        schemaVersion: 1,
        command: 'verify',
        workload: 'inference',
        intent: 'verify',
      },
      diagnostics: 'on_failure',
      probes: [],
      tracing: {},
      receiptPolicy: 'on_failure',
    },
    stats: {},
  };
  const error = new Error('planned failure');
  const receipt = captureAttentionRefactorReceipt({
    state,
    plan: {
      schema: 'doppler.attention-plan/v1',
      id: 'plan',
      transitions: { inputCast: 'f16->f32' },
      stages: ['attention'],
    },
    resourceEvents: [{
      sequence: 0,
      action: 'release',
      label: 'q',
      ownership: 'scopeOwned',
      detail: null,
    }],
    error,
    failureBoundary: 'attention',
  });
  assert.equal(receipt.failure.boundary, 'attention');
  assert.equal(receipt.failure.cleanup, 'completed');
  assert.equal(error.details.refactorReceipt, receipt);
  assert.equal(state.stats.refactorReceipts.length, 1);
}
