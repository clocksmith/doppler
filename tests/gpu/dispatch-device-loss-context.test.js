import assert from 'node:assert/strict';

import { dispatch, dispatchIndirect } from '../../src/gpu/kernels/dispatch.js';

assert.throws(
  () => dispatch(null, {}, {}, 1, 'vision_attention'),
  /vision_attention dispatch requires an active GPU device; lastDeviceLoss=/
);
assert.throws(
  () => dispatchIndirect(null, {}, {}, {}, 0, 'vision_attention_indirect'),
  /vision_attention_indirect dispatch requires an active GPU device; lastDeviceLoss=/
);

console.log('dispatch-device-loss-context.test: ok');
