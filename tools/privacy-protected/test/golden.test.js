// The JavaScript port against the values the removed Go/Lattigo tools produced (fixtures/go-golden.json):
// profiles, Chebyshev approximations, plaintext references, plaintext encoding, decryption of a Lattigo ciphertext,
// every RNS operation (including key switching), and whole compiled programs, word for word.

import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { BigFloat } from '../ckks/bigfloat.js';
import { encodeCompact } from '../ckks/encoder.js';
import { SecretKey, decrypt } from '../ckks/keys.js';
import { lookupProfile } from '../ckks/params.js';
import { RnsExec } from '../ckks/rns-exec.js';
import { compileProgram } from '../compiler.js';
import { approxLogits, approximations, circuitDigest, exactLogits, greedy, loadModel, newPlan, tokenize } from '../plan.js';
import { digest } from '../webgpu/wire.js';
import { SplitMix64 } from './splitmix.js';

const G = JSON.parse(readFileSync(new URL('../fixtures/go-golden.json', import.meta.url), 'utf8'));
const sha = (b) => `sha256:${createHash('sha256').update(b).digest('hex')}`;

test('profiles: moduli, roots, Galois elements and digests', () => {
  for (const [id, g] of Object.entries(G.profiles)) {
    const p = lookupProfile(id);
    assert.deepEqual(p.q, g.q);
    assert.deepEqual(p.p, g.p);
    assert.deepEqual(p.psi, g.psi);
    assert.deepEqual(p.galois, g.galois);
    assert.equal(p.digest, g.digest);
    assert.equal(p.insecure, g.insecure);
  }
});

test('plan: model identity, circuit digest and every Chebyshev coefficient at 128 bits', () => {
  const { model, id } = loadModel();
  assert.equal(id, G.model);
  const pl = newPlan(model);
  assert.equal(circuitDigest(pl, id), G.circuit);
  approximations(pl).forEach((a, i) => {
    const g = G.approximations[i];
    assert.equal(a.name, g.name);
    assert.equal(a.A.toGoText(), g.intervalA);
    assert.equal(a.B.toGoText(), g.intervalB);
    assert.equal(a.scalar, g.scalar);
    assert.equal(a.offset, g.offset);
    assert.deepEqual(a.coeffs.map((c) => c.re.toGoText()), g.coeffs, `${a.name} coefficients`);
  });
});

test('references: tokens equal, logits within 1e-9', () => {
  const { model } = loadModel();
  const pl = newPlan(model);
  for (const r of G.references) {
    const toks = tokenize(r.prompt);
    const ap = greedy(toks, r.generate, (t) => approxLogits(pl, t));
    const ex = greedy(toks, r.generate, (t) => exactLogits(model, t));
    assert.deepEqual(ap.tokens, r.approxTokens, `${r.prompt} approximate`);
    assert.deepEqual(ex.tokens, r.exactTokens, `${r.prompt} exact`);
    ap.logits.forEach((row, i) => row.forEach((v, j) => {
      assert.ok(Math.abs(v - r.approxLogits[i][j]) < 1e-9);
      assert.ok(Math.abs(ex.logits[i][j] - r.exactLogits[i][j]) < 1e-9);
    }));
  }
});

test('encoder: compact NTT plaintexts word for word (Lattigo, 64-bit big-float FFT)', () => {
  for (const c of G.encoder) {
    const w = encodeCompact(lookupProfile(c.profile), c.values, BigFloat.fromGoText(c.scale), c.level);
    assert.equal(sha(Buffer.from(w.buffer)), c.digest, `${c.profile} level ${c.level} scale ${c.scale}`);
  }
});

test('decryption of a Lattigo ciphertext with its exported secret key', () => {
  const d = G.decrypt;
  const p = lookupProfile(d.profile);
  const sk = new SecretKey(p, Int8Array.from([...d.secretTernary], (c) => (c === '+' ? 1 : c === '-' ? -1 : 0)));
  const got = decrypt(sk, Buffer.from(d.ciphertext, 'base64'));
  got.forEach((v, i) => {
    assert.ok(Math.abs(v - d.values[i]) < 1e-6);
    assert.ok(Math.abs(v - d.decodedByLattigo[i]) < 1e-9);
  });
});

test('RNS operations equal Lattigo word for word (levels 13, 12, 3, 1; set PROTECTED_ALL_LEVELS=1 for all)', () => {
  const p = lookupProfile(G.rns.profile);
  const n = p.n;
  const rng = new SplitMix64(0x5eed);
  const mods = [...p.q, ...p.p];
  const x = new RnsExec(p);
  const key = () => {
    const b = new Uint32Array(p.digits * 2 * mods.length * n);
    let o = 0;
    for (let d = 0; d < p.digits; d++) for (let c = 0; c < 2; c++) for (const q of mods) for (let j = 0; j < n; j++) b[o++] = rng.below(q);
    return b;
  };
  x.loadKey(0, 0, key());
  for (const gal of p.galois) x.loadKey(1, gal, key());
  const words = (level, per) => Uint32Array.from({ length: (level + 1) * per }, (_, i) => rng.below(p.q[Math.floor(i / per)]));
  const randCt = (deg, level) => {
    const ct = x.newCt(deg, level);
    for (let i = 0; i < ct.body.length; i++) ct.body[i] = rng.below(mods[Math.floor(i / n) % (level + 1)]);
    return ct;
  };
  const all = process.env.PROTECTED_ALL_LEVELS === '1';
  let ci = 0;
  let checked = 0;
  for (const l of [36, 25, 26, 13, 12, 3, 1]) {
    const blobs = { smul: words(l, 1), sadd: words(l, 1), smac: words(l, 1), pmul: words(l, 128), padd: words(l, 128) };
    const names = ['add', 'sub', 'smul', 'sadd', 'smac', 'pmul', 'padd', 'tensor', 'relin', 'rescale', 'rescale', ...p.galois.map(() => 'rotate')];
    for (const name of names) {
      const c = G.rns.cases[ci++];
      const ins = c.inputs.map(([d, lv]) => randCt(d, lv));
      if (!all && l > 13) continue;
      const out = x.run(c.op, ins[0], ins[1], blobs[name] ?? null);
      assert.equal(out.deg, c.outDegree);
      assert.equal(out.level, c.outLevel);
      assert.equal(digest(Buffer.from(out.body.buffer)), c.digest, c.name);
      checked++;
    }
  }
  assert.ok(checked >= 68);
});

test('compiler: programs equal the Go recorder\'s and the receipts\' byte for byte', () => {
  for (const g of G.programs) {
    const { json, blob } = compileProgram({ profile: g.profile, prompt: g.prompt, decodeThrough: g.decodeThrough });
    const program = JSON.parse(json);
    assert.equal(program.blobDigest, g.blobDigest, `${g.name} constants`);
    assert.equal(sha(Buffer.from(blob.buffer)), g.blobDigest);
    assert.equal(program.ops.length, g.ops);
    assert.equal(program.regs.length, g.regs);
    if (!g.note) assert.equal(sha(Buffer.from(json)), g.jsonSha256, `${g.name} program JSON`);
  }
});
