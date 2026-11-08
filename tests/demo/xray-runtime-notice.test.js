import assert from 'node:assert/strict';

import { getXrayRuntimeNoticeText } from '../../demo/ui/xray/index.js';

assert.equal(
  getXrayRuntimeNoticeText({
    wordQualityEnabled: true,
    profilingEnabled: true,
    traceEnabled: false,
  }),
  'The next answer measures GPU time for each step. Measuring slows the model, so its times are not a speed benchmark.'
);

assert.equal(
  getXrayRuntimeNoticeText({
    wordQualityEnabled: true,
    profilingEnabled: false,
    traceEnabled: false,
  }),
  'Every answer saves the alternatives for each token, which slows the model. Times shown are not a speed benchmark.'
);

assert.equal(
  getXrayRuntimeNoticeText({
    wordQualityEnabled: false,
    profilingEnabled: true,
    traceEnabled: true,
  }),
  'The next answer measures GPU time for each step. Measuring slows the model, so its times are not a speed benchmark.'
);

assert.equal(
  getXrayRuntimeNoticeText({
    wordQualityEnabled: false,
    profilingEnabled: false,
    traceEnabled: true,
  }),
  'Times are recorded without slowing the model.'
);

assert.equal(
  getXrayRuntimeNoticeText({
    wordQualityEnabled: false,
    profilingEnabled: false,
    traceEnabled: false,
  }),
  'Times are recorded without slowing the model.'
);

assert.equal(
  getXrayRuntimeNoticeText({ activationRecording: true, profilingEnabled: true, alternativesRecorded: true }),
  'The next answer records every layer. Recording slows the model, so its times are not a speed benchmark.',
  'The layer grid takes precedence over GPU timing, as in the run policy'
);

console.log('xray-runtime-notice.test: ok');

assert.equal(
  getXrayRuntimeNoticeText({ alternativesRecorded: true, wordQualityEnabled: false, profilingEnabled: false }),
  'Every answer saves the alternatives for each token, which slows the model. Times shown are not a speed benchmark.'
);
