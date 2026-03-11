import assert from 'node:assert/strict';

import { sample } from '../../src/inference/pipelines/text/sampling.js';

{
  const logits = new Float32Array([123, Number.NaN, 5]);
  const token = sample(logits, {
    temperature: 0,
    topP: 1,
    topK: 1,
    padTokenId: 0,
  });
  assert.equal(token, 2);
}

{
  const logits = new Float32Array([123, Number.NaN, Number.NaN]);
  assert.throws(
    () => sample(logits, {
      temperature: 0,
      topP: 1,
      topK: 1,
      padTokenId: 0,
    }),
    /no finite candidate logits after masking the pad token/i
  );
}

{
  const logits = new Float32Array([123, Number.NaN, Number.NaN]);
  assert.throws(
    () => sample(logits, {
      temperature: 0.8,
      topP: 1,
      topK: 3,
      padTokenId: 0,
    }),
    /no finite candidate logits after masking the pad token/i
  );
}

console.log('sampling-nan-guard.test: ok');
