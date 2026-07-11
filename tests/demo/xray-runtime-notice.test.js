import assert from 'node:assert/strict';

import { getXrayRuntimeNoticeText } from '../../demo/ui/xray/index.js';

assert.equal(
  getXrayRuntimeNoticeText({
    tokenPressEnabled: true,
    profilingEnabled: true,
    traceEnabled: false,
  }),
  'Token logits runs stepwise; X-Ray profiling is on. Batch view is unavailable.'
);

assert.equal(
  getXrayRuntimeNoticeText({
    tokenPressEnabled: true,
    profilingEnabled: false,
    traceEnabled: false,
  }),
  'Token logits runs stepwise. Batch view is unavailable.'
);

assert.equal(
  getXrayRuntimeNoticeText({
    tokenPressEnabled: false,
    profilingEnabled: true,
    traceEnabled: true,
  }),
  'X-Ray profiling is on. Trace logging is separate.'
);

assert.equal(
  getXrayRuntimeNoticeText({
    tokenPressEnabled: false,
    profilingEnabled: false,
    traceEnabled: true,
  }),
  'Trace logging is on; X-Ray selection is unchanged.'
);

assert.equal(
  getXrayRuntimeNoticeText({
    tokenPressEnabled: false,
    profilingEnabled: false,
    traceEnabled: false,
  }),
  null
);

console.log('xray-runtime-notice.test: ok');
