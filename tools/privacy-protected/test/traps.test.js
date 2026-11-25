import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync } from 'node:fs';
import { request as httpRequest } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { approxLogits, circuitDigest, loadModel, newPlan } from '../plan.js';
import { createTrapOracle, main as oracleMain, parseDuration } from '../trap-oracle.js';
import { main as trapsetMain } from '../trapset.js';
import {
  TRAP_DISTRIBUTION, b64, checkTrapLogits, circuitPin, createTrapRequest, ed25519FromSeed, issueTrapBatch, openTrapBatch, rawPublicKey,
  round4, trapDetectionProbability, unb64, verifyTrapSet,
} from '../traps.js';

const { model, id } = loadModel();
const pl = newPlan(model);
const circuit = circuitDigest(pl, id);
const pin = circuitPin(circuit, id);
const privateKey = ed25519FromSeed(Buffer.alloc(32, 7));
const oracle = createTrapOracle({ pl, modelId: id, privateKey, ttlMs: 60000 });
const oracleKeys = [b64(rawPublicKey(privateKey))];

function fresh(count = 3, promptLength = 3, generate = 4) {
  const { request, secret } = createTrapRequest({ circuit, promptLength, generate, count });
  return { request, secret, env: oracle.issue(request) };
}

const open = (env, request, secret, extra = {}) => openTrapBatch(env, { request, secret, oracleKeys, circuit: pin, maxTolerance: 1, ...extra });

test('oracle batches open, are bound to the request and carry the pinned reference', () => {
  const seen = new Set();
  for (let round = 0; round < 4; round++) {
    const { request, secret, env } = fresh();
    const batch = open(env, request, secret);
    assert.equal(batch.traps.length, 3);
    assert.equal(batch.distribution, TRAP_DISTRIBUTION);
    for (const t of batch.traps) {
      seen.add(t.prompt.join(','));
      const toks = [...t.prompt, ...t.continuation];
      t.expectedLogits.forEach((row, i) => assert.deepEqual(row, round4(approxLogits(pl, toks.slice(0, 3 + i)))));
      assert.ok(checkTrapLogits(t.expectedLogits[0], approxLogits(pl, t.prompt), 0.75).ok);
    }
  }
  // Twelve draws from a mixture over thousands of prompts: a fixed list would repeat.
  assert.ok(seen.size >= 6, `only ${seen.size} distinct trap prompts in 12 draws`);
  assert.throws(() => oracle.issue({ ...fresh().request, circuit: `sha256:${'0'.repeat(64)}` }), /not for this circuit/);
});

test('tampered, replayed, foreign and expired batches are refused', () => {
  const { request, secret, env } = fresh(2);
  const flip = (s) => {
    const b = unb64(s);
    b[b.length >> 1] ^= 1;
    return b64(b);
  };
  assert.throws(() => open({ ...env, signature: flip(env.signature) }, request, secret), /signature/);
  assert.throws(() => open({ ...env, ciphertext: flip(env.ciphertext) }, request, secret), /digest/);
  assert.throws(() => open({ ...env, header: env.header.replace('"issuedAt"', '"issuedAt" ') }, request, secret), /signature/);
  const other = createTrapRequest({ circuit, promptLength: 3, generate: 4, count: 2 });
  assert.throws(() => open(env, other.request, other.secret), /another request/);
  assert.throws(() => open(env, request, { ...secret, privateKey: other.secret.privateKey }), /decrypt/);
  assert.throws(() => openTrapBatch(env, { request, secret, oracleKeys, circuit: { ...pin, identity: `sha256:${'1'.repeat(64)}` }, maxTolerance: 1 }),
    /pinned circuit/);
  assert.throws(() => open(env, request, secret, { now: () => Date.now() + 120000 }), /expired/);
  assert.throws(() => open(env, request, secret, { oracleKeys: [b64(Buffer.alloc(32))] }), /not pinned/);
  assert.throws(() => open(env, request, secret, { maxTolerance: 0.5 }), /tolerance/);
  const used = new Map();
  open(env, request, secret, { usedBatches: used });
  assert.throws(() => open(env, request, secret, { usedBatches: used }), /already used/);
});

test('verifier helpers', () => {
  assert.equal(trapDetectionProbability({ lanes: 4, trapLanes: 3 }), 0.75);
  assert.equal(trapDetectionProbability({ lanes: 4, trapLanes: 3, tampered: 2 }), 1);
  const row = [0, 1, 5, 2];
  assert.deepEqual(checkTrapLogits(row, [0.1, 1, 5.5, 2], 0.75), { ok: true, maxAbsError: 0.5, argmaxOk: true });
  assert.equal(checkTrapLogits(row, [0, 6, 5, 2], 0.75).argmaxOk, false);
  assert.equal(checkTrapLogits(row, [NaN, 1, 5, 2], 0.75).ok, false);
  assert.equal(parseDuration('10m'), 600000);
  assert.equal(parseDuration('1h30m'), 5400000);
  assert.throws(() => parseDuration('10x'));
});

test('trapset signs and verifies; the oracle serves over HTTP', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'trapset-'));
  const out = join(dir, 't.json');
  const env = await trapsetMain(['--key', join(dir, 'k'), '--new-key', '--out', out, '--per-length', '2']);
  const set = verifyTrapSet(JSON.parse(readFileSync(out, 'utf8')), { publisherKeys: [env.publicKey], circuit: pin, maxTolerance: 1 });
  assert.ok(set.traps.length > 4);
  for (const t of set.traps.slice(0, 3)) assert.deepEqual(t.expectedLogits[0], round4(approxLogits(pl, t.prompt)));
  assert.throws(() => verifyTrapSet({ ...env, body: `${env.body} ` }, { publisherKeys: [env.publicKey], circuit: pin }), /signature/);
  assert.equal(readFileSync(join(dir, 'k'), 'utf8').length, 64);

  const server = await oracleMain(['--key', join(dir, 'o'), '--new-key', '--listen', '127.0.0.1:0']);
  try {
    const { port } = server.address();
    const call = (path, method, body) => new Promise((resolve, reject) => {
      const req = httpRequest({ host: '127.0.0.1', port, path, method }, (res) => {
        const chunks = [];
        res.on('data', (c) => chunks.push(c));
        res.on('end', () => resolve({ status: res.statusCode, body: Buffer.concat(chunks).toString('utf8') }));
      });
      req.on('error', reject);
      req.end(body);
    });
    const desc = JSON.parse((await call('/v1/descriptor', 'GET')).body);
    assert.equal(desc.circuit, circuit);
    const { request, secret } = createTrapRequest({ circuit, promptLength: 2, generate: 3, count: 2 });
    const res = await call('/v1/trap-batch', 'POST', JSON.stringify(request));
    assert.equal(res.status, 200);
    const batch = openTrapBatch(JSON.parse(res.body), { request, secret, oracleKeys: [desc.oracleKey], circuit: pin, maxTolerance: 1 });
    assert.equal(batch.traps.length, 2);
    assert.equal((await call('/v1/trap-batch', 'POST', 'x'.repeat(5000))).status, 400);
    assert.equal((await call('/v1/trap-batch', 'GET')).status, 405);
  } finally {
    server.close();
  }
});

test('issueTrapBatch refuses malformed requests', () => {
  const { request } = createTrapRequest({ circuit, promptLength: 3, generate: 4, count: 2 });
  const ctx = { pl, modelId: id, circuit, privateKey, tolerance: 0.75, minMargin: 2.5, ttlMs: 1000 };
  for (const bad of [{ count: 9 }, { promptLength: 8, generate: 2 }, { requesterNonce: 'ab' }, { schema: 'x' }]) {
    assert.throws(() => issueTrapBatch(ctx, { ...request, ...bad }));
  }
});
