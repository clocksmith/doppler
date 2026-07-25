import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const runSource = readFileSync(new URL('../../src/inference/pipelines/text/attention/executor-immediate.js', import.meta.url), 'utf8');
const interpreterSource = readFileSync(new URL('../../src/inference/pipelines/text/attention/interpreter.js', import.meta.url), 'utf8');

assert.match(
  runSource,
  /interpretAttentionWithRecorder\(/,
  'the immediate adapter must delegate precision and KV-cache decisions to the canonical interpreter'
);
assert.doesNotMatch(
  runSource,
  /resolveAttentionPrecisionContract|isAttentionKvDtypeExplicit/,
  'the immediate adapter must not duplicate attention precision decisions'
);
assert.match(
  interpreterSource,
  /resolveAttentionPrecisionContract\(config, state\)/,
  'the canonical executor must resolve the explicit attention precision contract before KV-cache narrowing'
);
assert.match(
  interpreterSource,
  /isAttentionKvDtypeExplicit\(attentionPrecisionContract, 'f16'\)/,
  'the canonical executor must treat explicit f16 KV-cache narrowing as manifest-owned'
);

console.log('attention-kv-precision-source.test: ok');
