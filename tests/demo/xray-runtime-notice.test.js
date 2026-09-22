import assert from 'node:assert/strict';

import { getXrayRuntimeNoticeText } from '../../demo/ui/xray/index.js';

assert.equal(
  getXrayRuntimeNoticeText({
    wordQualityEnabled: true,
    profilingEnabled: true,
    traceEnabled: false,
  }),
  'X-Ray captures GPU timestamps and changes execution. Timings are diagnostic, not a throughput benchmark.'
);

assert.equal(
  getXrayRuntimeNoticeText({
    wordQualityEnabled: true,
    profilingEnabled: false,
    traceEnabled: false,
  }),
  'Token inspection changes execution. Compare quality only with matching comparison fingerprints.'
);

assert.equal(
  getXrayRuntimeNoticeText({
    wordQualityEnabled: false,
    profilingEnabled: true,
    traceEnabled: true,
  }),
  'X-Ray captures GPU timestamps and changes execution. Timings are diagnostic, not a throughput benchmark.'
);

assert.equal(
  getXrayRuntimeNoticeText({
    wordQualityEnabled: false,
    profilingEnabled: false,
    traceEnabled: true,
  }),
  'Always-on evidence records existing wall timing without GPU timestamp queries. This is the performance-representative observation tier.'
);

assert.equal(
  getXrayRuntimeNoticeText({
    wordQualityEnabled: false,
    profilingEnabled: false,
    traceEnabled: false,
  }),
  'Always-on evidence records existing wall timing without GPU timestamp queries. This is the performance-representative observation tier.'
);

console.log('xray-runtime-notice.test: ok');

assert.equal(
  getXrayRuntimeNoticeText({ tokenInspectorActive: true, wordQualityEnabled: false, profilingEnabled: false }),
  'Token inspection changes execution. Compare quality only with matching comparison fingerprints.'
);
