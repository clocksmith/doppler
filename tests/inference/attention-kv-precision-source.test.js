import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const runSource = readFileSync(new URL('../../src/inference/pipelines/text/attention/run.js', import.meta.url), 'utf8');
const recordSource = readFileSync(new URL('../../src/inference/pipelines/text/attention/record.js', import.meta.url), 'utf8');

assert.match(
  runSource,
  /resolveAttentionPrecisionContract\(config, state\)/,
  'run.js must resolve the explicit attention precision contract before KV-cache narrowing'
);
assert.match(
  runSource,
  /isAttentionKvDtypeExplicit\(attentionPrecisionContract, 'f16'\)/,
  'run.js must treat explicit f16 KV-cache narrowing as manifest-owned'
);
assert.match(
  recordSource,
  /resolveAttentionPrecisionContract\(config, state\)/,
  'record.js must resolve the explicit attention precision contract before KV-cache narrowing'
);
assert.match(
  recordSource,
  /isAttentionKvDtypeExplicit\(attentionPrecisionContract, 'f16'\)/,
  'record.js must treat explicit f16 KV-cache narrowing as manifest-owned'
);

console.log('attention-kv-precision-source.test: ok');
