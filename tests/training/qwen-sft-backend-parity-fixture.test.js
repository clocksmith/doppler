import assert from 'node:assert/strict';

import {
  createQwenSftBackendParityFixture,
} from '../../src/experimental/training/qwen-sft-backend-parity-fixture.js';

const first = createQwenSftBackendParityFixture();
const second = createQwenSftBackendParityFixture();
assert.deepEqual(first, second);
assert.equal(first.model.activeTokenCount, 2);
assert.deepEqual(first.targets, [-100, 4, 5]);
assert.equal(first.precisionContract.adapterDropout, 0);
assert.equal(
  first.architectureContract.decoderOrder,
  'input_norm_token_mixer_residual_post_attention_norm_mlp_residual'
);
assert.equal(first.architectureContract.ropePairing, 'split_half_within_partial_rotary_prefix');
assert.equal(first.architectureContract.partialRotaryFactor, 0.25);
assert.match(first.architectureContract.transformersDecoderSourceSha256, /^[a-f0-9]{64}$/);
assert.equal(Object.keys(first.adapters).length, 7);
for (const adapter of Object.values(first.adapters)) {
  assert.equal(adapter.rank, 32);
  assert.equal(adapter.alpha, 64);
  assert.equal(adapter.A.shape[1], 32);
  assert.equal(adapter.B.shape[0], 32);
  assert.equal(adapter.A.data.length, adapter.A.shape[0] * adapter.A.shape[1]);
  assert.equal(adapter.B.data.length, adapter.B.shape[0] * adapter.B.shape[1]);
}
assert.equal(first.frozen.qWeight.shape[0], 64);
assert.ok(first.frozen.embedding.data.every(Number.isFinite));
assert.throws(
  () => createQwenSftBackendParityFixture({ rank: 0 }),
  /positive rank/
);

console.log('qwen-sft-backend-parity-fixture.test: ok');
