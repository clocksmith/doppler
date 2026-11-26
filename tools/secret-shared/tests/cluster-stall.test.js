// node --test tools/secret-shared/tests/   (plain CPU; opens no GPU device and no socket)
// A stalled share operator: every reply the requester waits for has a deadline, the stall names the stage and share
// that missed it, and the whole session is cancelled (every operator connection closed, nothing more sent).
import assert from 'node:assert/strict';
import test from 'node:test';
import { DEFAULT_ROUND_TIMEOUT_MS, OperatorStallError, SharedCluster } from '../requester/cluster.js';

const MAGIC = 0x31535352;
const plan = { fx: 20, fw: 24, stages: [0, 1].map(index => ({ index, groups: [{ id: `g${index}`, in: 4, tensors: [{ shape: [3, 4] }] }] })) };

function frame(header, payload = new Uint8Array(0)) {
  const js = new TextEncoder().encode(JSON.stringify(header));
  const out = new Uint8Array(16 + js.length + payload.length), view = new DataView(out.buffer);
  view.setUint32(0, MAGIC, true); view.setUint32(4, js.length, true); out.set(js, 8);
  view.setUint32(8 + js.length, payload.length, true); out.set(payload, 16 + js.length);
  return out;
}

/** An in-memory operator for stage/share; `stallOn(header)` true makes it never answer that message (or anything after). */
function fakeOperator(stage, share, { stallOn = () => false } = {}) {
  let inbox = new Uint8Array(0), outbox = [], waiter = null, stalled = false;
  const seen = [], state = { closed: false, seen };
  const pump = () => {
    if (!waiter) return;
    if (state.closed) { const w = waiter; waiter = null; w.reject(Error('operator closed the connection')); return; }
    const have = outbox.reduce((s, c) => s + c.length, 0);
    if (have < waiter.n) return;
    const all = new Uint8Array(have); let o = 0; for (const c of outbox) { all.set(c, o); o += c.length; }
    const { n, resolve } = waiter; waiter = null;
    outbox = [all.subarray(n)];
    resolve(all.slice(0, n));
  };
  const handle = header => {
    seen.push(header.op);
    if (stalled || stallOn(header)) { stalled = true; return; }
    if (header.op === 'hello') return frame({ stage, share, fx: plan.fx, fw: plan.fw, caps: [] });
    if (header.op === 'linear') return frame({ round: header.round, computeNs: 1000 }, new Uint8Array(header.n * 3 * 8));
    return frame({ ok: true });
  };
  const stream = {
    write(bytes) {
      const next = new Uint8Array(inbox.length + bytes.length); next.set(inbox); next.set(bytes, inbox.length); inbox = next;
      for (;;) {
        if (inbox.length < 8) return;
        const view = new DataView(inbox.buffer, inbox.byteOffset, inbox.length), jsLen = view.getUint32(4, true);
        if (inbox.length < 16 + jsLen) return;
        const len = view.getUint32(8 + jsLen, true) + view.getUint32(12 + jsLen, true) * 4294967296;
        if (inbox.length < 16 + jsLen + len) return;
        const header = JSON.parse(new TextDecoder().decode(inbox.subarray(8, 8 + jsLen)));
        inbox = inbox.slice(16 + jsLen + len);
        const reply = handle(header);
        if (reply) { outbox.push(reply); queueMicrotask(pump); }
      }
    },
    read: n => new Promise((resolve, reject) => { waiter = { n, resolve, reject }; pump(); }),
    close() { state.closed = true; pump(); }
  };
  return { stream, state };
}

function cluster(stallAt) {
  const operators = [0, 1].map(stage => [0, 1].map(share => fakeOperator(stage, share, { stallOn: stallAt(stage, share) })));
  const env = { connect: async entry => operators[entry.stage][entry.share].stream, crypto: globalThis.crypto, shake256: (seed, round, size) => new Uint8Array(size) };
  const ops = { token: 't', stages: [0, 1].map(stage => [0, 1].map(share => ({ stage, share }))) };
  return { operators, open: options => SharedCluster.open(plan, ops, env, { seeded: false, ...options }) };
}
const never = () => () => false;

test('the default deadline is unchanged for existing callers', () => {
  assert.equal(DEFAULT_ROUND_TIMEOUT_MS, 300000);
});

test('a share that stops answering mid-answer fails the round within the deadline, named, and cancels every connection', async () => {
  let rounds = 0;
  const { operators, open } = cluster((stage, share) => stage === 1 && share === 1 ? header => header.op === 'linear' && ++rounds === 3 : () => false);
  const c = await open({ roundTimeoutMs: 150 });
  const x = new Float32Array(4).fill(0.5);
  for (let i = 0; i < 2; i++) { await c.run('g0', x, 1); await c.run('g1', x, 1); }
  const t0 = performance.now();
  await assert.rejects(c.run('g1', x, 1), error => {
    assert.ok(error instanceof OperatorStallError);
    assert.equal(error.code, 'operator-stalled');
    assert.deepEqual(error.stalled, [{ stage: 1, share: 1 }]);
    assert.equal(error.timeoutMs, 150);
    return true;
  });
  const waited = performance.now() - t0;
  assert.ok(waited >= 140 && waited < 1000, `timed out after ${waited} ms`);
  // Cancellation: every operator's connection is closed and nothing further is sent.
  assert.ok(operators.flat().every(op => op.state.closed));
  assert.deepEqual(c.cancelled.stalled, [{ stage: 1, share: 1 }]);
  const sentBefore = operators.flat().map(op => op.state.seen.length);
  await assert.rejects(c.run('g0', x, 1), /cancelled/);
  await assert.rejects(c.endStep(5), /cancelled/);
  const closing = await c.close();
  assert.deepEqual(closing.flat().map(row => row.cancelled), [true, true, true, true]);
  assert.deepEqual(operators.flat().map(op => op.state.seen.length), sentBefore);
});

test('both shares silent: the stall names both', async () => {
  const { open } = cluster(stage => stage === 0 ? header => header.op === 'linear' : () => false);
  const c = await open({ roundTimeoutMs: 100 });
  await assert.rejects(c.run('g0', new Float32Array(4), 1), error => {
    assert.deepEqual(error.stalled, [{ stage: 0, share: 0 }, { stage: 0, share: 1 }]);
    return true;
  });
});

test('an operator that never answers hello fails the open within the deadline and every opened connection is closed', async () => {
  const { operators, open } = cluster((stage, share) => stage === 0 && share === 1 ? header => header.op === 'hello' : () => false);
  const t0 = performance.now();
  await assert.rejects(open({ roundTimeoutMs: 120 }), error => error.code === 'operator-stalled' && error.stalled[0].stage === 0 && error.stalled[0].share === 1);
  assert.ok(performance.now() - t0 < 1000);
  assert.ok(operators.flat().every(op => op.state.closed));
});

test('a stalled end-of-step reply is a stall too', async () => {
  const { open } = cluster((stage, share) => stage === 0 && share === 0 ? header => header.op === 'step' : () => false);
  const c = await open({ roundTimeoutMs: 100 });
  await c.run('g0', new Float32Array(4), 1);
  await assert.rejects(c.endStep(0), error => error.code === 'operator-stalled' && error.what === 'step');
  assert.ok(c.cancelled);
});

test('a close that gets no reply is bounded and reported, not thrown', async () => {
  const { operators, open } = cluster((stage, share) => stage === 1 && share === 0 ? header => header.op === 'close' : () => false);
  const c = await open({ roundTimeoutMs: 1000, closeTimeoutMs: 80 });
  const t0 = performance.now();
  const closing = await c.close();
  assert.ok(performance.now() - t0 < 1000);
  assert.deepEqual(closing[1][0], { timedOut: true, stalled: { stage: 1, share: 0 }, timeoutMs: 80 });
  assert.equal(closing[0][0].ok, true);
  assert.ok(operators.flat().every(op => op.state.closed));
});

test('an operator that drops its connection is attributed to its stage and share and cancels the rest', async () => {
  const { operators, open } = cluster(never);
  const c = await open({ roundTimeoutMs: 1000 });
  operators[0][1].stream.close();
  await assert.rejects(c.run('g0', new Float32Array(4), 1), error => {
    assert.deepEqual(error.operator, { stage: 0, share: 1 });
    return true;
  });
  assert.ok(operators.flat().every(op => op.state.closed));
});
