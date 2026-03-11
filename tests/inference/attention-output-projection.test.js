import assert from 'node:assert/strict';

import { prepareAttentionProjectionInput } from '../../src/inference/pipelines/text/attention/output-projection.js';

{
  const attnForProjection = { dtype: 'f32', label: 'gated_attn' };
  let castSource = null;
  const castedTensor = { dtype: 'f16', label: 'casted_gated_attn' };

  const result = await prepareAttentionProjectionInput(
    attnForProjection,
    'f16',
    async (tensor) => {
      castSource = tensor;
      return castedTensor;
    }
  );

  assert.equal(castSource, attnForProjection);
  assert.equal(result.oProjInput, castedTensor);
  assert.equal(result.oProjInputTemp, castedTensor);
}

{
  const attnForProjection = { dtype: 'f16', label: 'already_f16' };
  let castCalls = 0;

  const result = await prepareAttentionProjectionInput(
    attnForProjection,
    'f16',
    async () => {
      castCalls += 1;
      return { dtype: 'f16', label: 'unexpected_cast' };
    }
  );

  assert.equal(castCalls, 0);
  assert.equal(result.oProjInput, attnForProjection);
  assert.equal(result.oProjInputTemp, null);
}

console.log('attention-output-projection.test: ok');
