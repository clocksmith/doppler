// bridge: the requester's key holder for a host process (Reploid) that runs the session protocol itself. The secret
// key is generated (or loaded from the key-epoch store) in this process and never leaves it. The bridge holds no
// weights and loads no model. It speaks over stdin/stdout to its parent only, never to an evaluator, and counts every
// encryption and decryption so the host can report them.
//
// Framing: u32 header length | header JSON | blobs (the session protocol's). Decrypted values return as one blob of
// little-endian float64s, so non-finite values survive (JSON cannot carry them) and the host decides what to do.
// Profiles: the u32 (raw-u32/v0 wire) profiles only; keys and ciphertexts are in that format.
//
//   node tools/privacy-protected/bridge.js --profile research-n16-u32 [--allow-insecure-dev] [--state-dir d] [--key-ttl 24h]

import { randomBytes } from 'node:crypto';
import { mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import { lookupProfile } from './ckks/params.js';
import { SecretKey, decrypt, evaluationKeys, sealValues } from './ckks/keys.js';
import { CONTEXT, SLOTS, VOCAB, replicate } from './plan.js';
import { parseDuration } from './trap-oracle.js';
import { digest } from './webgpu/wire.js';

export const BRIDGE_PROTOCOL = 'doppler-protected-bridge/v1';
const MAX_HEADER = 64 << 10;
const CEILING = { keygen: 0, 'commit-keys': 0, 'retire-epoch': 0, encrypt: 0, decrypt: 32 << 20, close: 0 };

// --- key epochs (requester.go's store: key-epoch.json and the secret in key-epoch.sk, 0600) ---

export function loadOrCreateEpoch(profile, stateDir, ttlSeconds) {
  const now = Math.floor(Date.now() / 1000);
  if (stateDir) {
    try {
      const ep = JSON.parse(readFileSync(join(stateDir, 'key-epoch.json'), 'utf8'));
      const sk = SecretKey.fromBytes(profile, readFileSync(join(stateDir, 'key-epoch.sk')));
      if (ep.profileDigest === profile.digest && ep.expires > now + 300) return { epoch: ep, sk, reused: true };
    } catch { /* no usable epoch: make one */ }
  }
  const epoch = { epoch: randomBytes(16).toString('hex'), profileDigest: profile.digest, created: now, expires: now + ttlSeconds, keysDigest: '' };
  const sk = SecretKey.generate(profile);
  if (stateDir) {
    mkdirSync(stateDir, { recursive: true, mode: 0o700 });
    writeFileSync(join(stateDir, 'key-epoch.sk'), sk.toBytes(), { mode: 0o600 });
    saveEpoch(stateDir, epoch);
  }
  return { epoch, sk, reused: false };
}

export function saveEpoch(stateDir, epoch) {
  if (stateDir) writeFileSync(join(stateDir, 'key-epoch.json'), JSON.stringify(epoch), { mode: 0o600 });
}

// --- stdio framing ---

export class FrameReader {
  constructor(stream) {
    this.chunks = [];
    this.have = 0;
    this.ended = null;
    this.waiter = null;
    stream.on('data', (d) => {
      this.chunks.push(d);
      this.have += d.length;
      this.#wake();
    });
    stream.on('end', () => {
      this.ended = new Error('input closed');
      this.#wake();
    });
    stream.on('error', (e) => {
      this.ended = e;
      this.#wake();
    });
  }

  #wake() {
    const w = this.waiter;
    this.waiter = null;
    w?.();
  }

  async read(n) {
    while (this.have < n) {
      if (this.ended) throw this.ended;
      await new Promise((resolve) => { this.waiter = resolve; });
    }
    const buf = Buffer.concat(this.chunks);
    this.chunks = [buf.subarray(n)];
    this.have -= n;
    return buf.subarray(0, n);
  }

  // One frame; kinds outside `ceilings` or another protocol are errors before any blob byte is read.
  async recv(protocol, ceilings) {
    const n = (await this.read(4)).readUInt32LE(0);
    if (n === 0 || n > MAX_HEADER) throw new Error(`header length ${n} outside (0, ${MAX_HEADER}]`);
    let h;
    try {
      h = JSON.parse((await this.read(n)).toString('utf8'));
    } catch (e) {
      throw new Error(`malformed header: ${e.message}`);
    }
    if (!(h.kind in ceilings) || h.protocol !== protocol) throw new Error(`bridge does not accept ${JSON.stringify(h.kind)}`);
    const sizes = h.blobs ?? [];
    let total = 0;
    for (const b of sizes) {
      if (!Number.isInteger(b) || b < 0) throw new Error('negative blob length');
      total += b;
    }
    if (total > ceilings[h.kind]) throw new Error(`${h.kind} carries ${total} bytes, above its ${ceilings[h.kind]} ceiling`);
    const blobs = [];
    for (const b of sizes) blobs.push(Buffer.from(await this.read(b)));
    return { header: h, blobs };
  }
}

// The Go header layout: kind, protocol, session, seq, blobs (null when none), fields (omitted when empty).
export function encodeFrame(protocol, { kind, session = '', seq = 0, fields }, blobs = []) {
  const h = { kind, protocol, session, seq, blobs: blobs.length ? blobs.map((b) => b.length) : null };
  if (fields && Object.keys(fields).length) h.fields = fields;
  const raw = Buffer.from(JSON.stringify(h));
  const pre = Buffer.alloc(4);
  pre.writeUInt32LE(raw.length);
  return Buffer.concat([pre, raw, ...blobs]);
}

function writeAll(out, buf) {
  return new Promise((resolve, reject) => {
    if (out.write(buf)) resolve();
    else {
      out.once('drain', resolve);
      out.once('error', reject);
    }
  });
}

function oneHot(tok) {
  const v = new Array(VOCAB).fill(0);
  v[tok] = 1;
  return replicate(v, VOCAB);
}

// runBridge: serve one host over (input, output) until `close` or end of input.
export async function runBridge({ input, output, profile: profileId, allowInsecure = false, stateDir = null, keyTtlSeconds = 86400 }) {
  const p = lookupProfile(profileId);
  if (p.insecure && !allowInsecure) throw new Error(`profile ${p.id} is insecure and only allowed with --allow-insecure-dev`);
  const { epoch, sk, reused } = loadOrCreateEpoch(p, stateDir, keyTtlSeconds);
  const reader = new FrameReader(input);
  const send = (h, ...blobs) => writeAll(output, encodeFrame(BRIDGE_PROTOCOL, h, blobs));
  await send({ kind: 'opened', fields: { profile: p.id, profileDigest: p.digest, insecureDevelopment: p.insecure, keyEpoch: epoch.epoch,
    keyEpochExpires: epoch.expires, reused, keysDigest: epoch.keysDigest, vocab: VOCAB, context: CONTEXT, inputLevel: p.maxLevel } });
  let encryptions = 0;
  let decryptions = 0;
  for (;;) {
    const { header: h, blobs } = await reader.recv(BRIDGE_PROTOCOL, CEILING);
    switch (h.kind) {
      case 'keygen': {
        // Each evaluation key is generated, sent to the host and dropped; the host forwards it to the evaluator.
        const kd = [];
        let seq = 0;
        for (const key of evaluationKeys(sk)) {
          kd.push(digest(key));
          await send({ kind: 'key', seq: seq++ }, key);
        }
        await send({ kind: 'keys-done', fields: { keysDigest: digest(...kd) } });
        break;
      }
      case 'commit-keys':
        // The evaluator acknowledged these keys for this epoch: later sessions may reuse them without upload.
        epoch.keysDigest = typeof h.fields?.keysDigest === 'string' ? h.fields.keysDigest : '';
        saveEpoch(stateDir, epoch);
        await send({ kind: 'committed' });
        break;
      case 'retire-epoch':
        // After a failed verification the host retires this key epoch: the stored secret and record are deleted, so
        // the next session makes a new key and uploads new evaluation keys.
        if (stateDir) {
          rmSync(join(stateDir, 'key-epoch.sk'), { force: true });
          rmSync(join(stateDir, 'key-epoch.json'), { force: true });
        }
        epoch.keysDigest = '';
        await send({ kind: 'retired', fields: { keyEpoch: epoch.epoch } });
        break;
      case 'encrypt': {
        const toks = h.fields?.tokens;
        if (!Array.isArray(toks) || toks.length === 0 || toks.length > 8 * CONTEXT) throw new Error(`encrypt needs 1 to ${8 * CONTEXT} tokens`);
        if (!toks.every((t) => Number.isInteger(t) && t >= 0 && t < VOCAB)) throw new Error('token outside the vocabulary');
        const out = toks.map((t) => sealValues(sk, oneHot(t)));
        encryptions += out.length;
        await send({ kind: 'ciphertexts' }, ...out);
        break;
      }
      case 'decrypt': {
        if (blobs.length !== 1) throw new Error('decrypt takes one ciphertext');
        let vals;
        let level;
        try {
          level = blobs[0].readUInt32LE(8);
          if (blobs[0].readUInt32LE(4) !== 1) throw new Error('degree');
          vals = decrypt(sk, blobs[0]);
        } catch {
          // Shape failures answer without decrypting; the host decides whether the session survives.
          await send({ kind: 'refused', fields: { reason: 'result does not have the pinned shape' } });
          break;
        }
        decryptions++;
        const out = Buffer.alloc(8 * VOCAB);
        for (let i = 0; i < VOCAB && i < SLOTS; i++) out.writeDoubleLE(vals[i], 8 * i);
        await send({ kind: 'plaintext', fields: { level } }, out);
        break;
      }
      case 'close':
        await send({ kind: 'closed', fields: { encryptions, decryptions, peakRssBytes: process.resourceUsage().maxRSS * 1024 } });
        return { encryptions, decryptions };
      default:
        throw new Error(`bridge does not accept ${JSON.stringify(h.kind)}`);
    }
  }
}

export async function main(argv = process.argv.slice(2)) {
  const { values } = parseArgs({
    args: argv,
    options: {
      profile: { type: 'string', default: 'research-n16-u32' },
      'allow-insecure-dev': { type: 'boolean', default: false },
      'state-dir': { type: 'string' },
      'key-ttl': { type: 'string', default: '24h' },
    },
  });
  return runBridge({ input: process.stdin, output: process.stdout, profile: values.profile, allowInsecure: values['allow-insecure-dev'],
    stateDir: values['state-dir'] ?? null, keyTtlSeconds: Math.floor(parseDuration(values['key-ttl']) / 1000) });
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((e) => {
    if (e.message !== 'input closed') console.error(`error: ${e.message}`);
    process.exit(1);
  });
}
