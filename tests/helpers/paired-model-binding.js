import assert from 'node:assert/strict';
import { compareReferenceModel } from '../../../reploid/tests/fixtures/distributed-reference-model.js';

// The explicitly approved program/representation changes do not authorize a
// weight, tokenizer, source equation, source epsilon, or F16 KV policy change.
export function bindPairedModel(model, baseline, execution) {
  assert.deepEqual(model.inference.execution, execution, 'Paired graph differs from the frozen declared program.');
  const sourceFacts = structuredClone(model);
  sourceFacts.inference.execution = structuredClone(baseline.inference.execution);
  compareReferenceModel(sourceFacts, baseline);
  return { weightsTokenizerAndSourceFactsUnchanged: true, executionGraphChanged: true,
    precisionAuthorization: 'User-approved interleaved two-F32 intermediate and persistent state chain',
    publicLogits: 'f32', kvStorage: 'f16', numericalReference: 'unchanged approved independent F64', tolerance: 0.001 };
}
