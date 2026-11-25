// Known-answer traps for the protected route: the trap oracle's secret fresh batches (v2), the publisher-signed public
// trap set (v1, kept for comparison tests: a perturbation that vanishes on every published trap defeats it), and the
// requester-side verifier. Formats match the removed Go tools (oracle.go, trapset.go) and Reploid's verifier
// (packages/reploid/src/mesh/protected-generation.js) byte for byte: JSON field order as the Go structs marshal,
// base64url without padding, Ed25519 over "<schema>\0" || exact bytes.

import {
  createCipheriv, createDecipheriv, createHash, createPrivateKey, createPublicKey, diffieHellman, generateKeyPairSync,
  hkdfSync, randomBytes, randomInt, sign, verify,
} from 'node:crypto';
import { SplitMix64 } from './test/splitmix.js';
import { ALPHABET, CONTEXT, PROTOCOL_ID, VOCAB, approxLogits, greedy, margin, newTrace } from './plan.js';

export const TRAP_REQUEST_SCHEMA = 'doppler.protected-trap-request/v1';
export const TRAP_BATCH_SCHEMA = 'doppler.protected-trap-batch/v1';
export const TRAP_BATCH_ENVELOPE_SCHEMA = 'doppler.protected-trap-batch-envelope/v1';
export const TRAP_SET_SCHEMA = 'doppler.protected-trapset/v1';
export const TRAP_SET_ENVELOPE_SCHEMA = 'doppler.protected-trapset-envelope/v1';
export const TRAP_DISTRIBUTION = 'doppler.fixture-prompt-mixture/v1';
export const MAX_TRAPS_PER_BATCH = 8;
export const TRAP_REQUEST_LIMIT = 4 << 10;

// --- encodings and keys ---

export const b64 = (bytes) => Buffer.from(bytes).toString('base64url');
export const unb64 = (s) => {
  if (typeof s !== 'string' || !/^[A-Za-z0-9_-]*$/.test(s)) throw new Error('not base64url');
  return Buffer.from(s, 'base64url');
};
export const sha256 = (bytes) => `sha256:${createHash('sha256').update(bytes).digest('hex')}`;

const ED25519_PKCS8 = Buffer.from('302e020100300506032b657004220420', 'hex');

// Ed25519 private key from a 32-byte seed (Go's ed25519.NewKeyFromSeed).
export function ed25519FromSeed(seed) {
  if (seed.length !== 32) throw new Error('an Ed25519 seed is 32 bytes');
  return createPrivateKey({ key: Buffer.concat([ED25519_PKCS8, seed]), format: 'der', type: 'pkcs8' });
}

export function rawPublicKey(key) {
  const pub = key.type === 'public' ? key : createPublicKey(key);
  return unb64(pub.export({ format: 'jwk' }).x);
}

function importRaw(crv, raw) {
  if (raw.length !== 32) throw new Error(`${crv} public key is not 32 bytes`);
  return createPublicKey({ key: { kty: 'OKP', crv, x: b64(raw) }, format: 'jwk' });
}

export function signingBytes(schema, bytes) {
  return Buffer.concat([Buffer.from(`${schema}\x00`), Buffer.from(bytes)]);
}

// --- sampling (round4 expectations from the pinned approximate circuit) ---

// Go's math.Round(x*1e4)/1e4: half away from zero (Math.round rounds half up).
function goRound4(x) {
  const y = x * 1e4;
  const r = Math.sign(y) * Math.round(Math.abs(y));
  return (Object.is(r, -0) ? 0 : r) / 1e4;
}

export function round4(v) {
  return v.map(goRound4);
}

// A trap: its prompt, the approximate circuit's own greedy continuation (an honest requester feeds these back as the
// decode inputs), and the expected logits at every generated position; null unless the whole trajectory stays inside
// every approximation interval with a top-1 margin of at least minMargin.
function makeTrap(pl, id, prompt, generate, minMargin) {
  const tr = newTrace();
  const { tokens, logits } = greedy(prompt, generate, (t) => approxLogits(pl, t, tr));
  let ok = Object.keys(tr.outOfRange).length === 0;
  for (const row of logits) ok = ok && row.every(Number.isFinite) && margin(row) >= minMargin;
  if (!ok) return null;
  return { id, prompt, continuation: tokens.slice(0, generate - 1), expectedLogits: logits.map(round4) };
}

// Uniform in [0, 1) with 53 CSPRNG bits (oracle.go cryptoFloat).
const cryptoUnit = () => (randomInt(2 ** 26) * 2 ** 27 + randomInt(2 ** 27)) / 2 ** 53;

// samplePrompt (oracle.go): a third letter progressions (the fixture's training task), a third sequences the model
// generates from a random start (its softmax at temperature 1), a third uniformly random letters. CSPRNG throughout.
export function samplePrompt(pl, L) {
  const p = new Array(L).fill(0);
  switch (randomInt(3)) {
    case 0: {
      const a = randomInt(26);
      const stride = 1 + randomInt(3);
      for (let i = 0; i < L; i++) p[i] = (a + i * stride) % 26;
      break;
    }
    case 1: {
      p[0] = randomInt(26);
      for (let i = 1; i < L; i++) {
        const l = approxLogits(pl, p.slice(0, i));
        const mx = Math.max(...l.slice(0, 26));
        const w = l.slice(0, 26).map((x) => Math.exp(x - mx));
        const z = w.reduce((a, b) => a + b, 0);
        const u = cryptoUnit() * z;
        let acc = 0;
        p[i] = 25;
        for (let j = 0; j < 26; j++) {
          acc += w[j];
          if (u < acc) {
            p[i] = j;
            break;
          }
        }
      }
      break;
    }
    default:
      for (let i = 0; i < L; i++) p[i] = randomInt(26);
  }
  return p;
}

// freshTraps: `count` traps of prompt length L and `generate` outputs, sampled until each passes makeTrap's checks.
export function freshTraps(pl, L, generate, count, minMargin) {
  const out = [];
  for (let attempt = 0; out.length < count && attempt < 400 * count; attempt++) {
    const t = makeTrap(pl, `t${out.length}`, samplePrompt(pl, L), generate, minMargin);
    if (t) out.push(t);
  }
  if (out.length < count) throw new Error(`could not sample ${count} traps of prompt length ${L}`);
  return out;
}

// makeTraps (trapset.go): per prompt length, alternately letter progressions and random letters, deterministic in
// the seed. The Go tools drew from Go's math/rand, which is not reproduced here; this uses SplitMix64 (recorded in
// the body as the generator), so a JavaScript set differs from a Go set with the same seed.
export const TRAP_SET_GENERATOR = 'splitmix64';

export function makeTraps(pl, perLength, seed, minMargin) {
  const rng = new SplitMix64(seed);
  const intn = (n) => rng.below(n);
  const seen = new Set();
  const out = [];
  for (let L = 1; L <= CONTEXT; L++) {
    let kept = 0;
    for (let attempt = 0; kept < perLength && attempt < 200 * perLength; attempt++) {
      const prompt = new Array(L);
      if (attempt % 2 === 0) {
        const a = intn(26);
        const stride = 1 + intn(3);
        for (let i = 0; i < L; i++) prompt[i] = (a + i * stride) % 26;
      } else {
        for (let i = 0; i < L; i++) prompt[i] = intn(26);
      }
      const key = prompt.join(',');
      if (seen.has(key)) continue;
      seen.add(key);
      const t = makeTrap(pl, `L${L}-${String(kept).padStart(3, '0')}`, prompt, CONTEXT - L + 1, minMargin);
      if (!t) continue;
      out.push(t);
      kept++;
    }
    // A length with no trap is not covered: a requester refuses such prompts before sending anything.
  }
  if (!out.length) throw new Error('no trap meets the interval and margin requirements');
  return out;
}

// --- the oracle's side ---

const HEX64 = /^[a-f0-9]{64}$/;

// issueTrapBatch: one fresh batch for one request, encrypted to the requester's one-time X25519 key (ECDH with a fresh
// oracle key, HKDF-SHA256 with the nonce as salt and the schema as info, AES-256-GCM with the batch id as associated
// data), the header signed with the oracle's Ed25519 key.
export function issueTrapBatch({ pl, modelId, circuit, privateKey, tolerance, minMargin, ttlMs }, req, now = Date.now()) {
  if (req?.schema !== TRAP_REQUEST_SCHEMA || req.circuit !== circuit || !HEX64.test(req.requesterNonce ?? '')
    || !Number.isInteger(req.promptLength) || !Number.isInteger(req.generate) || !Number.isInteger(req.count)
    || req.promptLength < 1 || req.generate < 1 || req.promptLength + req.generate - 1 > CONTEXT
    || req.count < 1 || req.count > MAX_TRAPS_PER_BATCH) {
    throw new Error('trap request is invalid or not for this circuit');
  }
  const requesterKey = importRaw('X25519', unb64(req.requesterKey));
  const traps = freshTraps(pl, req.promptLength, req.generate, req.count, minMargin);
  const batchId = randomBytes(16).toString('hex');
  const body = {
    schema: TRAP_BATCH_SCHEMA, batchId, circuit, model: modelId, requesterNonce: req.requesterNonce, promptLength: req.promptLength,
    generate: req.generate, distribution: TRAP_DISTRIBUTION,
    tolerance: { maxAbsLogitError: tolerance, minReferenceMargin: minMargin, requireArgmax: true }, traps,
  };
  const eph = generateKeyPairSync('x25519');
  const shared = diffieHellman({ privateKey: eph.privateKey, publicKey: requesterKey });
  const key = Buffer.from(hkdfSync('sha256', shared, Buffer.from(req.requesterNonce, 'hex'), TRAP_BATCH_SCHEMA, 32));
  const iv = randomBytes(12);
  const c = createCipheriv('aes-256-gcm', key, iv);
  c.setAAD(Buffer.from(batchId));
  const sealed = Buffer.concat([c.update(JSON.stringify(body)), c.final(), c.getAuthTag()]);
  const header = JSON.stringify({
    schema: TRAP_BATCH_SCHEMA, batchId, circuit, requesterNonce: req.requesterNonce, requesterKey: req.requesterKey,
    oracleEphemeralKey: b64(rawPublicKey(eph.publicKey)), iv: b64(iv), ciphertextDigest: sha256(sealed),
    issuedAt: now, expiresAt: now + ttlMs,
  });
  return {
    schema: TRAP_BATCH_ENVELOPE_SCHEMA, header, ciphertext: b64(sealed), oracleKey: b64(rawPublicKey(privateKey)),
    signature: b64(sign(null, signingBytes(TRAP_BATCH_SCHEMA, header), privateKey)),
  };
}

// signTrapSet: the publisher's envelope over the exact body bytes.
export function signTrapSet({ pl, modelId, circuit, privateKey, perLength, seed, tolerance, minMargin, publishedAt }) {
  const body = JSON.stringify({
    schema: TRAP_SET_SCHEMA, protocol: PROTOCOL_ID, model: modelId, circuit, tokenizer: { alphabet: ALPHABET }, vocab: VOCAB,
    context: CONTEXT, tolerance: { maxAbsLogitError: tolerance, minReferenceMargin: minMargin, requireArgmax: true },
    reference: 'approximate-circuit-plaintext', seed, generator: TRAP_SET_GENERATOR, publishedAt,
    traps: makeTraps(pl, perLength, seed, minMargin),
  });
  return {
    schema: TRAP_SET_ENVELOPE_SCHEMA, body, bodyDigest: sha256(Buffer.from(body)), publicKey: b64(rawPublicKey(privateKey)),
    signature: b64(sign(null, signingBytes(TRAP_SET_SCHEMA, body), privateKey)),
  };
}

// --- the requester's side (the verifier) ---

// A circuit pin: { identity, modelIdentity, vocab, context, tokenizer: { alphabet } } (Reploid's shape).
export function circuitPin(circuit, modelId) {
  return { identity: circuit, modelIdentity: modelId, vocab: VOCAB, context: CONTEXT, tokenizer: { alphabet: ALPHABET } };
}

const exact = (obj, keys) => obj && typeof obj === 'object' && !Array.isArray(obj)
  && Object.keys(obj).length === keys.length && keys.every((k) => k in obj);
const label = (v) => typeof v === 'string' && /^[A-Za-z0-9._-]{1,64}$/.test(v);

// createTrapRequest: a fresh 256-bit nonce and a one-time X25519 key. `secret` never leaves the requester.
export function createTrapRequest({ circuit, promptLength, generate, count }) {
  const pair = generateKeyPairSync('x25519');
  const requesterNonce = randomBytes(32).toString('hex');
  const request = {
    schema: TRAP_REQUEST_SCHEMA, circuit, promptLength, generate, count, requesterNonce,
    requesterKey: b64(rawPublicKey(pair.publicKey)),
  };
  return { request, secret: { privateKey: pair.privateKey, requesterNonce } };
}

// openTrapBatch: verify (pinned oracle key, signature over the exact header, binding to this request's nonce and key,
// expiry, unused batch id, ciphertext digest), decrypt, and check every trap's shape. Any failure throws.
export function openTrapBatch(envelope, { request, secret, oracleKeys, circuit, maxTolerance, now = Date.now, usedBatches = null }) {
  const refused = (m) => new Error(`Trap batch refused: ${m}. Nothing was sent.`);
  if (!exact(envelope, ['schema', 'header', 'ciphertext', 'oracleKey', 'signature']) || envelope.schema !== TRAP_BATCH_ENVELOPE_SCHEMA
    || typeof envelope.header !== 'string' || typeof envelope.ciphertext !== 'string') throw refused('malformed envelope');
  if (!oracleKeys.includes(envelope.oracleKey)) throw refused('oracle key is not pinned');
  let valid = false;
  try {
    valid = verify(null, signingBytes(TRAP_BATCH_SCHEMA, envelope.header), importRaw('Ed25519', unb64(envelope.oracleKey)), unb64(envelope.signature));
  } catch {
    valid = false;
  }
  if (!valid) throw refused('oracle signature does not verify');
  let header;
  try {
    header = JSON.parse(envelope.header);
  } catch {
    throw refused('header is not JSON');
  }
  const t = now();
  if (header?.schema !== TRAP_BATCH_SCHEMA || header.circuit !== circuit.identity) throw refused('it is not for the pinned circuit');
  if (header.requesterNonce !== secret.requesterNonce || header.requesterKey !== request.requesterKey) {
    throw refused('it answers another request (stale or replayed)');
  }
  if (!(Number.isSafeInteger(header.expiresAt) && header.expiresAt > t && header.issuedAt <= t + 60000)) throw refused('it has expired');
  if (!/^[a-f0-9]{32}$/.test(header.batchId ?? '') || usedBatches?.has(header.batchId)) throw refused('its batch id was already used');
  const sealed = unb64(envelope.ciphertext);
  if (sha256(sealed) !== header.ciphertextDigest) throw refused('ciphertext digest differs');
  let body;
  try {
    const shared = diffieHellman({ privateKey: secret.privateKey, publicKey: importRaw('X25519', unb64(header.oracleEphemeralKey)) });
    const key = Buffer.from(hkdfSync('sha256', shared, Buffer.from(secret.requesterNonce, 'hex'), TRAP_BATCH_SCHEMA, 32));
    const d = createDecipheriv('aes-256-gcm', key, unb64(header.iv));
    d.setAAD(Buffer.from(header.batchId));
    d.setAuthTag(sealed.subarray(sealed.length - 16));
    body = JSON.parse(Buffer.concat([d.update(sealed.subarray(0, sealed.length - 16)), d.final()]).toString('utf8'));
  } catch {
    throw refused('it does not decrypt under this request\'s key');
  }
  const tol = body?.tolerance;
  if (body?.schema !== TRAP_BATCH_SCHEMA || body.batchId !== header.batchId || body.requesterNonce !== secret.requesterNonce
    || body.circuit !== circuit.identity || body.model !== circuit.modelIdentity || body.promptLength !== request.promptLength
    || body.generate !== request.generate) throw refused('its contents do not match its header and request');
  if (!(typeof tol?.maxAbsLogitError === 'number' && tol.maxAbsLogitError > 0 && tol.maxAbsLogitError <= maxTolerance
    && tol.requireArgmax === true)) throw refused('its tolerance exceeds this requester\'s limit');
  const traps = Array.isArray(body.traps) ? body.traps : [];
  const token = (v) => Number.isSafeInteger(v) && v >= 0 && v < circuit.vocab;
  if (traps.length !== request.count || new Set(traps.map((x) => x?.id)).size !== traps.length || !traps.every((x) => label(x?.id)
    && Array.isArray(x.prompt) && x.prompt.length === request.promptLength && x.prompt.every(token)
    && Array.isArray(x.continuation) && x.continuation.length === request.generate - 1 && x.continuation.every(token)
    && Array.isArray(x.expectedLogits) && x.expectedLogits.length === request.generate
    && x.expectedLogits.every((row) => Array.isArray(row) && row.length === circuit.vocab && row.every(Number.isFinite)))) {
    throw refused('a trap is malformed');
  }
  usedBatches?.set(header.batchId, header.expiresAt);
  return {
    source: 'oracle', digest: header.ciphertextDigest, batchId: header.batchId, oracleKey: envelope.oracleKey,
    distribution: typeof body.distribution === 'string' ? body.distribution : null, tolerance: { maxAbsLogitError: tol.maxAbsLogitError },
    traps, circuit: circuit.identity, issuedAt: header.issuedAt,
  };
}

// verifyTrapSet: a publisher-signed public set (v1). Not a guarantee: see the module comment.
export function verifyTrapSet(envelope, { publisherKeys, circuit, maxTolerance = Infinity }) {
  const bad = (m) => new Error(`Trap set: ${m}. Nothing was sent.`);
  if (!exact(envelope, ['schema', 'body', 'bodyDigest', 'publicKey', 'signature']) || envelope.schema !== TRAP_SET_ENVELOPE_SCHEMA
    || typeof envelope.body !== 'string') throw bad('malformed envelope');
  if (!publisherKeys.includes(envelope.publicKey)) throw bad('publisher key is not pinned');
  let valid = false;
  try {
    valid = verify(null, signingBytes(TRAP_SET_SCHEMA, envelope.body), importRaw('Ed25519', unb64(envelope.publicKey)), unb64(envelope.signature));
  } catch {
    valid = false;
  }
  if (!valid) throw bad('signature does not verify');
  const bodyDigest = sha256(Buffer.from(envelope.body));
  if (bodyDigest !== envelope.bodyDigest) throw bad('body digest differs');
  let body;
  try {
    body = JSON.parse(envelope.body);
  } catch {
    throw bad('body is not JSON');
  }
  const tol = body?.tolerance;
  if (body?.schema !== TRAP_SET_SCHEMA || body.circuit !== circuit.identity || body.model !== circuit.modelIdentity
    || body.vocab !== circuit.vocab || body.context !== circuit.context || body.tokenizer?.alphabet !== circuit.tokenizer.alphabet) {
    throw bad('it is not for the pinned circuit');
  }
  if (!(typeof tol?.maxAbsLogitError === 'number' && tol.maxAbsLogitError > 0 && tol.maxAbsLogitError <= maxTolerance
    && tol.requireArgmax === true)) throw bad('its tolerance exceeds this requester\'s limit');
  const traps = Array.isArray(body.traps) ? body.traps : [];
  const token = (v) => Number.isSafeInteger(v) && v >= 0 && v < circuit.vocab;
  for (const x of traps) {
    const n = x?.prompt?.length;
    if (!label(x?.id) || !Array.isArray(x.prompt) || !x.prompt.every(token) || !Array.isArray(x.continuation)
      || !x.continuation.every(token) || n < 1 || n + x.continuation.length !== circuit.context
      || !Array.isArray(x.expectedLogits) || x.expectedLogits.length !== circuit.context - n + 1
      || !x.expectedLogits.every((row) => Array.isArray(row) && row.length === circuit.vocab && row.every(Number.isFinite))) {
      throw bad('a trap is malformed');
    }
  }
  if (!traps.length) throw bad('it has no traps');
  return { source: 'public-set', digest: bodyDigest, publisherKey: envelope.publicKey, tolerance: { maxAbsLogitError: tol.maxAbsLogitError }, traps,
    circuit: circuit.identity };
}

// Argmax that cannot fail: NaN counts as -Infinity, ties take the lowest index.
export function totalArgmax(values) {
  let best = 0;
  let bestValue = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = Number.isNaN(values[i]) ? -Infinity : values[i];
    if (v > bestValue) {
      best = i;
      bestValue = v;
    }
  }
  return best;
}

// checkTrapLogits: one decrypted trap result against its expected row. Non-finite values fail.
export function checkTrapLogits(expected, logits, tolerance) {
  let maxAbsError = 0;
  for (let i = 0; i < expected.length; i++) {
    const e = Math.abs(logits[i] - expected[i]);
    maxAbsError = Number.isFinite(e) ? Math.max(maxAbsError, e) : Infinity;
  }
  const argmaxOk = totalArgmax(logits) === totalArgmax(expected);
  const limit = typeof tolerance === 'number' ? tolerance : tolerance.maxAbsLogitError;
  return { ok: maxAbsError <= limit && argmaxOk, maxAbsError, argmaxOk };
}

// Probability that one session catches an evaluator tampering `tampered` of its lanes: 1 - C(K - T, m) / C(K, m).
export function trapDetectionProbability({ lanes, trapLanes, tampered = 1 }) {
  const choose = (n, k) => {
    if (k < 0 || k > n) return 0;
    let r = 1;
    for (let i = 1; i <= k; i++) r = (r * (n - k + i)) / i;
    return r;
  };
  return 1 - choose(lanes - trapLanes, tampered) / choose(lanes, tampered);
}

// Publisher / oracle key files: a hex Ed25519 seed (as the Go tools wrote them), mode 0600.
export async function loadOrCreateSeedKey(path, create) {
  const { readFile, writeFile } = await import('node:fs/promises');
  let raw;
  try {
    raw = await readFile(path, 'utf8');
  } catch (e) {
    if (e.code !== 'ENOENT' || !create) throw e;
    const seed = randomBytes(32);
    await writeFile(path, seed.toString('hex'), { mode: 0o600 });
    return ed25519FromSeed(seed);
  }
  const seed = Buffer.from(raw.trim(), 'hex');
  if (seed.length !== 32 || seed.toString('hex') !== raw.trim().toLowerCase()) throw new Error('key file must hold a hex Ed25519 seed');
  return ed25519FromSeed(seed);
}
