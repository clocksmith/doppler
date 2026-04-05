import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const generatorSource = readFileSync(
  new URL('../../src/inference/pipelines/text/generator.js', import.meta.url),
  'utf8'
);

assert.match(
  generatorSource,
  /createKVCache\(/,
  'Generator finiteness fallback must rebuild the KV cache when the fallback plan changes kvDtype.'
);

assert.match(
  generatorSource,
  /rollbackSeqLen !== 0/,
  'KV-dtype-changing finiteness fallback must fail closed unless it can restart from a fresh prefill.'
);

assert.match(
  generatorSource,
  /_retryWithPersistentFinitenessFallback\(/,
  'Prefill-time finiteness recovery must keep the fallback plan active past the retry.'
);

assert.match(
  generatorSource,
  /'prefill-sample'[\s\S]*_retryWithPersistentFinitenessFallback\(/,
  'Prefill-sample recovery must reuse the persistent finiteness fallback path.'
);

console.log('gemma4-finiteness-fallback-kvcache-contract.test: ok');
