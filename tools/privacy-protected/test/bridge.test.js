import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { PassThrough } from 'node:stream';
import { test } from 'node:test';
import { BRIDGE_PROTOCOL, FrameReader, encodeFrame, runBridge } from '../bridge.js';
import { lookupProfile } from '../ckks/params.js';
import { SecretKey, decrypt } from '../ckks/keys.js';
import { digest } from '../webgpu/wire.js';

const ANY = new Proxy({}, { has: () => true, get: () => 1 << 30 });

function host(opts) {
  const toBridge = new PassThrough();
  const fromBridge = new PassThrough();
  const done = runBridge({ input: toBridge, output: fromBridge, profile: 'dev-insecure-n12-u32', allowInsecure: true, ...opts });
  const reader = new FrameReader(fromBridge);
  return {
    done,
    send: (kind, fields, blobs = []) => toBridge.write(encodeFrame(BRIDGE_PROTOCOL, { kind, fields }, blobs)),
    recv: () => reader.recv(BRIDGE_PROTOCOL, ANY),
  };
}

test('the bridge refuses the insecure profile unless allowed', async () => {
  await assert.rejects(runBridge({ input: new PassThrough(), output: new PassThrough(), profile: 'dev-insecure-n12-u32' }), /insecure/);
});

test('bridge: keys, encryption, decryption, epochs', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'bridge-'));
  const h = host({ stateDir: dir });
  const opened = (await h.recv()).header;
  assert.equal(opened.kind, 'opened');
  assert.equal(opened.fields.reused, false);
  assert.equal(statSync(join(dir, 'key-epoch.sk')).mode & 0o777, 0o600);

  h.send('keygen');
  const keys = [];
  for (let i = 0; i < 7; i++) {
    const { header, blobs } = await h.recv();
    assert.equal(header.kind, 'key');
    assert.equal(header.seq, i);
    keys.push(blobs[0]);
  }
  const done = (await h.recv()).header;
  assert.equal(done.kind, 'keys-done');
  assert.equal(done.fields.keysDigest, digest(...keys.map((k) => digest(k))));
  assert.equal(keys[0].readUInt32LE(4), 0);
  assert.equal(keys[1].readUInt32LE(4), 1);

  h.send('commit-keys', { keysDigest: done.fields.keysDigest });
  assert.equal((await h.recv()).header.kind, 'committed');

  h.send('encrypt', { tokens: [5, 30] });
  const { header: ch, blobs: cts } = await h.recv();
  assert.equal(ch.kind, 'ciphertexts');
  assert.equal(cts.length, 2);
  // Decrypt one inside the bridge, the other with the stored secret (the same key).
  h.send('decrypt', {}, [cts[0]]);
  const { header: ph, blobs: pv } = await h.recv();
  assert.equal(ph.kind, 'plaintext');
  const vals = Array.from({ length: 32 }, (_, i) => pv[0].readDoubleLE(8 * i));
  vals.forEach((v, i) => assert.ok(Math.abs(v - (i === 5 ? 1 : 0)) < 1e-4, `slot ${i}: ${v}`));
  const p = lookupProfile('dev-insecure-n12-u32');
  const sk = SecretKey.fromBytes(p, (await import('node:fs')).readFileSync(join(dir, 'key-epoch.sk')));
  const other = decrypt(sk, cts[1]);
  assert.ok(Math.abs(other[30] - 1) < 1e-4 && Math.abs(other[30 + 32] - 1) < 1e-4);

  h.send('decrypt', {}, [Buffer.alloc(40)]);
  assert.equal((await h.recv()).header.kind, 'refused');
  h.send('close');
  const { peakRssBytes, ...counts } = (await h.recv()).header.fields;
  assert.deepEqual(counts, { encryptions: 2, decryptions: 1 });
  assert.ok(Number.isSafeInteger(peakRssBytes) && peakRssBytes > 0);
  await h.done;

  // The next bridge reuses the epoch and its committed keys digest; retiring deletes it.
  const h2 = host({ stateDir: dir });
  const o2 = (await h2.recv()).header.fields;
  assert.equal(o2.reused, true);
  assert.equal(o2.keyEpoch, opened.fields.keyEpoch);
  assert.equal(o2.keysDigest, done.fields.keysDigest);
  h2.send('retire-epoch');
  assert.equal((await h2.recv()).header.kind, 'retired');
  assert.equal(existsSync(join(dir, 'key-epoch.sk')), false);
  h2.send('close');
  await h2.recv();
  await h2.done;
});

test('the bridge ends on an unknown kind', async () => {
  const h = host({});
  await h.recv();
  h.send('evaluate');
  await assert.rejects(h.done, /does not accept/);
});
