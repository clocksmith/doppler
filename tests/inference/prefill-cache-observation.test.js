import assert from 'node:assert/strict';
import { log } from '../../src/debug/index.js';
import { logPrefillCacheState } from '../../src/inference/pipelines/text/generator/prefill-observation.js';

const messages = [];
const debug = log.debug, warn = log.warn;
log.debug = (category, message) => messages.push({ level: 'debug', category, message });
log.warn = (category, message) => messages.push({ level: 'warn', category, message });
try {
  // Hybrid models keep KV only for full-attention layers (first index may be 3).
  logPrefillCacheState({ currentSeqLen: 28, kvCache: {
    hasGPUCache: () => true,
    getKeyCache() { throw new RangeError('Layer 0 has no allocated KV cache'); },
  } });
  assert.deepEqual(messages.pop(), {
    level: 'debug', category: 'Pipeline', message: 'KV cache active after prefill: seqLen=28',
  });
  for (const kvCache of [null, {}, { hasGPUCache: () => false }]) {
    logPrefillCacheState({ currentSeqLen: 28, kvCache });
    assert.equal(messages.pop().level, 'warn');
  }
} finally {
  log.debug = debug;
  log.warn = warn;
}
console.log('prefill-cache-observation.test: ok');
