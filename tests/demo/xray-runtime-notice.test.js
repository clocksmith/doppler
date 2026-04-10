import assert from 'node:assert/strict';

import { getXrayRuntimeNoticeText } from '../../demo/ui/xray/index.js';

assert.equal(
  getXrayRuntimeNoticeText({
    tokenPressEnabled: true,
    profilingEnabled: true,
    traceEnabled: false,
  }),
  'Token Press is active. Decode, Kernel, and GPU X-Ray panels add per-step profiling; Batch stays unavailable.'
);

assert.equal(
  getXrayRuntimeNoticeText({
    tokenPressEnabled: true,
    profilingEnabled: false,
    traceEnabled: false,
  }),
  'Token Press is a separate generation mode. It runs step-by-step decode, so Batch stats stay unavailable.'
);

assert.equal(
  getXrayRuntimeNoticeText({
    tokenPressEnabled: false,
    profilingEnabled: true,
    traceEnabled: true,
  }),
  'Decode, Kernel, and GPU X-Ray panels request extra profiling. Trace logging is separate.'
);

assert.equal(
  getXrayRuntimeNoticeText({
    tokenPressEnabled: false,
    profilingEnabled: false,
    traceEnabled: true,
  }),
  'Trace logging is active. It affects runtime logs, not X-Ray panel selection.'
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
