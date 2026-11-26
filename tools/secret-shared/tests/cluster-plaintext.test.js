// node --test tools/secret-shared/tests/*.test.js   (plain CPU; opens no GPU device and no socket)
// The plaintext control (sharing off): the same plan, encoding, rounds and wire format as the secret-shared run, with
// each stage's input sent unshared to one operator. It must compute the same W x as the shared path, contact only the
// chosen operator of each stage, and send that operator exactly encode(x).
import assert from 'node:assert/strict';
import test from 'node:test';
import { SharedCluster } from '../requester/cluster.js';
import { encode } from '../requester/ring.js';
import { fakeOperator, plan } from './fake-operator.js';

const M64 = (1n << 64n) - 1n;
// One fixed 3 x 4 weight matrix per stage, encoded at fw as every operator encodes its weights.
const W = [[0.5, -0.25, 0.125, 1], [-1, 0.75, 0, 0.0625], [0.3, 0.2, -0.1, -0.4]];
const Wq = W.map(row => row.map(w => BigInt(Math.round(w * 2 ** plan.fw))));

/** An exact ring operator: y = W~ x (mod 2^64) on whatever vector it receives, returned in bits shift..63. */
const ringCompute = (header, payload, state) => {
  const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
  const shift = state.options?.returnShift ?? 0, width = (64 - shift) / 8;
  const out = new Uint8Array(header.n * 3 * width);
  for (let c = 0; c < header.n; c++) for (let r = 0; r < 3; r++) {
    let acc = 0n;
    for (let k = 0; k < 4; k++) acc += Wq[r][k] * view.getBigUint64(8 * (c * 4 + k), true);
    const y = acc & M64;
    for (let b = 0; b < width; b++) out[(c * 3 + r) * width + b] = Number((y >> BigInt(shift + 8 * b)) & 0xffn);
  }
  return out;
};

function cluster() {
  const operators = [0, 1].map(stage => [0, 1].map(share => fakeOperator(stage, share, { compute: ringCompute, caps: ['returnShift'] })));
  const env = { connect: async entry => operators[entry.stage][entry.share].stream, crypto: globalThis.crypto, shake256: null };
  const ops = { token: 't', stages: [0, 1].map(stage => [0, 1].map(share => ({ stage, share }))) };
  return { operators, open: options => SharedCluster.open(plan, ops, env, { seeded: false, ...options }) };
}

const x = new Float32Array([0.75, -1.5, 0.3125, 2, -0.5, 0.25, 1.125, -0.875]);  // two columns
const expected = [0, 1].flatMap(c => W.map(row => row.reduce((s, w, k) => s + w * x[c * 4 + k], 0)));

for (const returnShift of [0, 24]) {
  test(`sharing off equals sharing on and W x (return shift ${returnShift})`, async () => {
    const tolerance = returnShift ? 2 ** -19 : 2 ** -18;  // the input encoding error times the row's |w| sum, plus the truncation
    const shared = await cluster().open({ returnShift });
    const yShared = await shared.run('g1', x, 2);
    const { operators, open } = cluster();
    const plain = await open({ returnShift, sharing: false });
    const yPlain = await plain.run('g1', x, 2);
    for (let i = 0; i < expected.length; i++) {
      assert.ok(Math.abs(yPlain[i] - expected[i]) < tolerance, `plain ${i}: ${yPlain[i]} vs ${expected[i]}`);
      assert.ok(Math.abs(yPlain[i] - yShared[i]) < tolerance, `plain vs shared ${i}`);
    }
    // Only share 0 of each stage was contacted, and stage 1's operator received exactly encode(x): the plaintext.
    assert.deepEqual(operators.map(pair => pair.map(op => op.state.seen.length > 0)), [[true, false], [true, false]]);
    assert.deepEqual([...operators[1][0].state.payloads[0]], [...new Uint8Array(encode(x, plan.fx).buffer)]);
    assert.deepEqual(plain.describe().map(row => [row.sharing, row.returnShift, row.prg, row.operatorShare]), [['off', returnShift, 'none', 0], ['off', returnShift, 'none', 0]]);
    // One operator per stage: the same upload as share B alone, and half the shared download.
    const roundShared = shared.rounds.at(-1), roundPlain = plain.rounds.at(-1);
    assert.equal(roundPlain.up[0], roundShared.up[1]);
    assert.equal(roundPlain.down[0], roundShared.down[1]);
    assert.equal(roundPlain.up[1] + roundPlain.down[1], 0);
    await plain.close(); await shared.close();
  });
}

test('sharing off can use the other entry of each stage', async () => {
  const operators = [0, 1].map(stage => [0, 1].map(share => fakeOperator(stage, share, { compute: ringCompute })));
  const env = { connect: async entry => operators[entry.stage][entry.share].stream, crypto: globalThis.crypto };
  const ops = { token: 't', stages: [0, 1].map(stage => [0, 1].map(share => ({ stage, share }))) };
  const c = await SharedCluster.open(plan, ops, env, { sharing: false, plaintextShare: 1 });
  assert.deepEqual(operators.map(pair => pair.map(op => op.state.seen.length > 0)), [[false, true], [false, true]]);
  const y = await c.run('g1', x, 2);
  assert.ok(Math.abs(y[0] - expected[0]) < 2 ** -18);
  await c.close();
});

test('sharing off refuses an operator that cannot honour a strict return shift', async () => {
  const operators = [0, 1].map(stage => [0, 1].map(share => fakeOperator(stage, share)));
  const env = { connect: async entry => operators[entry.stage][entry.share].stream, crypto: globalThis.crypto };
  const ops = { token: 't', stages: [0, 1].map(stage => [0, 1].map(share => ({ stage, share }))) };
  await assert.rejects(SharedCluster.open(plan, ops, env, { sharing: false, returnShift: 24, strict: true }), /lacks a requested capability/);
});
