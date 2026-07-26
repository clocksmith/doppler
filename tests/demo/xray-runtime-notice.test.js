import assert from 'node:assert/strict';

import { getXrayRuntimeNoticeText } from '../../demo/ui/xray/index.js';

assert.equal(
  getXrayRuntimeNoticeText({
    wordQualityEnabled: true,
    profilingEnabled: true,
    traceEnabled: false,
  }),
  'Deep X-Ray modifies execution and enables GPU timestamp queries. Its timings are diagnostic, not representative throughput.'
);

assert.equal(
  getXrayRuntimeNoticeText({
    wordQualityEnabled: true,
    profilingEnabled: false,
    traceEnabled: false,
  }),
  'Guided quality inspection captures token probabilities and changes execution. Compare quality only when the canonical fingerprint matches.'
);

assert.equal(
  getXrayRuntimeNoticeText({
    wordQualityEnabled: false,
    profilingEnabled: true,
    traceEnabled: true,
  }),
  'Deep X-Ray modifies execution and enables GPU timestamp queries. Its timings are diagnostic, not representative throughput.'
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
