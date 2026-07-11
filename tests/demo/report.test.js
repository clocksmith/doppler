import assert from 'node:assert/strict';

import { getReportOutput, validateImportedReport } from '../../demo/report.js';

assert.equal(
  getReportOutput({ tokens: [{ text: 'hello' }, { text: ' world' }] }),
  'hello world'
);
assert.equal(getReportOutput({ output: 'preferred', tokens: [{ text: 'ignored' }] }), 'preferred');
assert.equal(
  validateImportedReport({
    schema: 'doppler.demo-report/v1',
    output: 'valid',
  }).output,
  'valid'
);
assert.throws(() => validateImportedReport([]), /must be an object/);
assert.throws(
  () => validateImportedReport({ schema: 'doppler.unknown/v1', output: 'x' }),
  /Unsupported report schema/
);
assert.throws(() => validateImportedReport({}), /no output or generation stats/);

console.log('report.test: ok');
