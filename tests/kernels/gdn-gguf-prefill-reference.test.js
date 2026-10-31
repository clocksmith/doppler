import assert from 'node:assert/strict';
import { gdnReference, makeGdnFixture, sliceGdnFixture, packGdnInputs } from '../helpers/gdn-gguf-prefill-fixture.js';

// Hand-computed zero-Q/K fixture: state decays, output is zero, history shifts.
const f = makeGdnFixture({ T: 2, V: 3, window: 1 });
f.qkv.fill(0); f.convState.fill(0); f.a.fill(0); f.b.fill(0); f.bias.fill(0); f.aLog.fill(-1);
const result = gdnReference(f);
assert(result.output.every(x => x === 0));
for (let i = 0; i < f.state.length; i++) assert.equal(result.state[i], Math.fround(f.state[i] * 0.25));
assert(result.convState.every(x => x === 0));

// One-hot heads provide a closed-form oracle for head % NK, GGUF A, epsilon
// placement and norm-before-gate. For NK=2/NV=4, heads 1 and 2 expose /qRep bugs.
const oneHot = makeGdnFixture({ T: 1, V: 1, window: 1, eps: 0.25 });
oneHot.rmsEps = 0.3;
for (const key of ['qkv', 'state', 'convState', 'a', 'b', 'bias']) oneHot[key].fill(0);
oneHot.convWeight.fill(1); oneHot.aLog.fill(-1); oneHot.z.fill(1);
oneHot.qkv[0] = 1; oneHot.qkv[129] = 2;
oneHot.qkv[oneHot.Q] = 3; oneHot.qkv[oneHot.Q + 129] = 4;
for (let h = 0; h < 4; h++) oneHot.state[h * 128 + h % 2] = h + 1;
const analytic = gdnReference(oneHot), silu = x => x / (1 + Math.exp(-x));
for (let h = 0; h < 4; h++) {
  const q = silu(1 + h % 2), k = silu(3 + h % 2);
  const kn = k / Math.sqrt(k * k + 0.25);
  const expectedState = (h + 1) * 0.5 * (1 - 0.5 * kn * kn);
  const raw = expectedState * q / Math.sqrt(128 * (q * q + 0.25));
  const expectedOutput = raw / Math.sqrt(raw * raw + 0.3) * oneHot.norm[h] * silu(1);
  assert(Math.abs(analytic.state[h * 128 + h % 2] - expectedState) < 1e-6);
  assert(Math.abs(analytic.output[h] - expectedOutput) < 1e-6);
}

// Chunk continuation must use both returned state tensors, including decode tails.
const continuation = makeGdnFixture();
const whole = gdnReference(continuation);
let state = continuation.state, convState = continuation.convState, start = 0;
const outputs = [];
for (const count of [2, 3, 1, 3]) {
  const next = gdnReference({ ...sliceGdnFixture(continuation, start, count), state, convState });
  outputs.push(...next.output); ({ state, convState } = next); start += count;
}
assert.deepEqual(Float32Array.from(outputs), whole.output);
assert.deepEqual(state, whole.state); assert.deepEqual(convState, whole.convState);

// Layout fixtures preserve the distinct head/token sentinels and offset gap.
for (const ab of [false, true]) for (const qz of [false, true]) {
  const packed = packGdnInputs(continuation, ab, qz);
  for (let t = 0; t < continuation.T; t++) {
    const stride = continuation.C + (qz ? continuation.D : 0);
    assert.deepEqual(packed.qkv.slice(t * stride, t * stride + continuation.C), continuation.qkv.slice(t * continuation.C, (t + 1) * continuation.C));
    if (qz) assert.deepEqual(packed.z.slice(t * stride + continuation.C, (t + 1) * stride), continuation.z.slice(t * continuation.D, (t + 1) * continuation.D));
  }
  assert.deepEqual(packed.b.slice(packed.bOffset, packed.bOffset + continuation.b.length), continuation.b);
}
console.log('gdn-gguf-prefill-reference: analytic decay, nonzero state continuation and packed fixture addressing passed (CPU only)');
