import assert from 'node:assert/strict';
import { shouldDisablePrefillCommandBatching } from '../../src/inference/pipelines/text/generator/prefill-policy.js';
import { isDecodeRecorderEnabled } from '../../src/inference/pipelines/text/execution-plan.js';

const call = { startPos: 0, returnHidden: false };
const opts = { disableCommandBatching: false, debug: false };
for (const enabled of [false, true]) {
  const state = { dopplerLoader: { getRoutedExpertPolicy: () => ({ enabled }) } };
  assert.equal(shouldDisablePrefillCommandBatching(state, opts, null, call), enabled);
}
assert.equal(shouldDisablePrefillCommandBatching({}, opts, null, call), false);
assert.equal(shouldDisablePrefillCommandBatching({}, { ...opts, disableCommandBatching: true }, null, call), true);
assert.equal(isDecodeRecorderEnabled({ hasDevice: true, debug: false, disableCommandBatching: false, kvLayout: 'contiguous' }), true);
console.log('routed prefill immediate, single-token decode recording policy passed');
