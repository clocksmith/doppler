import assert from 'node:assert/strict';
import { parseFinitenessStatusWords } from '../../src/inference/pipelines/text/finiteness-guard-status.js';

{
  const status = parseFinitenessStatusWords(new Uint32Array([0, 7, 9]), 0);
  assert.equal(status.triggered, false);
  assert.equal(status.layer, 0);
  assert.equal(status.step, 0);
  assert.equal(status.metadata, '');
}

{
  const status = parseFinitenessStatusWords(new Uint32Array([1, 8, 12]), 0);
  assert.equal(status.triggered, true);
  assert.equal(status.layer, 8);
  assert.equal(status.step, 12);
  assert.equal(status.metadata, ' (layer 8, step 12)');
}

{
  // Sample staging payload layout: [token, status, layer, step, pad]
  const status = parseFinitenessStatusWords(new Uint32Array([42, 1, 3, 15, 0]), 1);
  assert.equal(status.triggered, true);
  assert.equal(status.layer, 3);
  assert.equal(status.step, 15);
  assert.equal(status.metadata, ' (layer 3, step 15)');
}

console.log('finiteness-guard-status.test: ok');
