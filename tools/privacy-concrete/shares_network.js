/**
 * Authenticated, encrypted TCP transport and enrolled-operator identity verifier for the two-server shares route.
 *
 * Wraps any pipe-speaking shares evaluator (shares_evaluator.py or shares_evaluator_gpu.js) without changing its
 * framing. Each session proves the provider's enrolled Ed25519 identity and ownerId over fresh nonces, the pinned
 * policy, plan, model, role ('a' or 'b'), per-session shares-manifest.json digest and ephemeral X25519 keys, then
 * encrypts every stream chunk and EOF marker with ChaCha20-Poly1305 under directional HKDF-SHA256 keys.
 */
import { spawn, spawnSync } from 'node:child_process';
import {
  createCipheriv,
  createDecipheriv,
  createHash,
  createPrivateKey,
  createPublicKey,
  diffieHellman,
  generateKeyPairSync,
  hkdfSync,
  randomBytes,
  sign,
  timingSafeEqual,
  verify,
} from 'node:crypto';
import { chmodSync, lstatSync, readFileSync, realpathSync, unlinkSync, writeFileSync } from 'node:fs';
import { createConnection, createServer } from 'node:net';
import { join, resolve as resolvePath } from 'node:path';
import { pathToFileURL } from 'node:url';

const ENROLLMENT_SCHEMA = 'doppler.two-server-shares-enrollment/v1';
const PRIVATE_ID_SCHEMA = 'doppler.two-server-shares-provider-private-key/v1';
const CLIENT_HELLO_SCHEMA = 'doppler.shares-net-client-hello/v1';
const SERVER_HELLO_SCHEMA = 'doppler.shares-net-server-hello/v1';
const TRANSCRIPT_SCHEMA = 'doppler.shares-net-transcript/v1';
const AUTH_DOMAIN = Buffer.from('doppler-shares-provider-auth/v1\0', 'utf8');
const KEY_DOMAIN = Buffer.from('doppler-shares-net-keys/v1\0', 'utf8');

const DIR_C2S = 0x01;
const DIR_S2C = 0x02;
const REC_READY = 0x01;
const REC_DATA = 0x02;
const REC_EOF = 0x03;

const MAX_HELLO_BYTES = 8192;
const MAX_PLAINTEXT_CHUNK = 1024 * 1024;
const MAX_FRAME_BYTES = MAX_PLAINTEXT_CHUNK + 64;
const HIGH_WATER_BUFFER_BYTES = 2 * MAX_FRAME_BYTES;
const HARD_BUFFER_CAP_BYTES = 8 * MAX_FRAME_BYTES;
const HANDSHAKE_TIMEOUT_MS = 15_000;
const HEX64 = /^[0-9a-f]{64}$/;
const HEX128 = /^[0-9a-f]{128}$/;
const DIGITS = /^\d+$/;
const ZERO32 = Buffer.alloc(32, 0);

const X25519_SPKI_PREFIX = Buffer.from('302a300506032b656e032100', 'hex');

// RFC 8032 Edwards25519 parameters for canonical encoding and prime-order subgroup verification.
const ED25519_P = (1n << 255n) - 19n;
const ED25519_L = (1n << 252n) + 27742317777372353535851937790883648493n;

function modField(a, m = ED25519_P) {
  const r = a % m;
  return r >= 0n ? r : r + m;
}

function modPow(base, exp, m = ED25519_P) {
  let result = 1n;
  let b = modField(base, m);
  let e = exp;
  while (e > 0n) {
    if (e & 1n) result = (result * b) % m;
    b = (b * b) % m;
    e >>= 1n;
  }
  return result;
}

const ED25519_D = modField(-121665n * modPow(121666n, ED25519_P - 2n));
const ED25519_SQRT_M1 = modPow(2n, (ED25519_P - 1n) / 4n);

function decodeLittleEndian(buf) {
  let acc = 0n;
  for (let i = buf.length - 1; i >= 0; i--) {
    acc = (acc << 8n) | BigInt(buf[i]);
  }
  return acc;
}

function edwardsDouble([X, Y, Z]) {
  const A = modField(X * X);
  const B = modField(Y * Y);
  const C = modField(2n * modField(Z * Z));
  const H = modField(A + B);
  const sumXY = modField(X + Y);
  const E = modField(H - modField(sumXY * sumXY));
  const G = modField(A - B);
  const F = modField(C + G);
  return [modField(E * F), modField(G * H), modField(F * G), modField(E * H)];
}

function edwardsAdd([X1, Y1, Z1, T1], [X2, Y2, Z2, T2]) {
  const A = modField(modField(Y1 - X1) * modField(Y2 - X2));
  const B = modField(modField(Y1 + X1) * modField(Y2 + X2));
  const C = modField(modField(2n * ED25519_D) * modField(T1 * T2));
  const D = modField(2n * modField(Z1 * Z2));
  const E = modField(B - A);
  const F = modField(D - C);
  const G = modField(D + C);
  const H = modField(B + A);
  return [modField(E * F), modField(G * H), modField(F * G), modField(E * H)];
}

function edwardsScalarMul(point, scalar) {
  let acc = [0n, 1n, 1n, 0n];
  let addend = point;
  let k = scalar;
  while (k > 0n) {
    if (k & 1n) acc = edwardsAdd(acc, addend);
    addend = edwardsDouble(addend);
    k >>= 1n;
  }
  return acc;
}

export function isCanonicalEd25519Point(raw, { requirePrimeOrder = true } = {}) {
  if (!Buffer.isBuffer(raw) || raw.length !== 32) return false;
  const copy = Buffer.from(raw);
  const x0 = (copy[31] >>> 7) & 1;
  copy[31] &= 0x7f;
  const y = decodeLittleEndian(copy);
  if (y >= ED25519_P) return false;
  const y2 = modField(y * y);
  const u = modField(y2 - 1n);
  const v = modField(ED25519_D * y2 + 1n);
  const v3 = modField(v * modField(v * v));
  const v7 = modField(v3 * modField(v3 * v));
  let x = modField(modField(u * v3) * modPow(modField(u * v7), (ED25519_P - 5n) / 8n));
  const vx2 = modField(v * modField(x * x));
  if (vx2 !== u) {
    if (vx2 === modField(-u)) {
      x = modField(x * ED25519_SQRT_M1);
    } else {
      return false;
    }
  }
  if (x === 0n && x0 === 1) return false;
  if (Number(x & 1n) !== x0) x = ED25519_P - x;
  if (requirePrimeOrder) {
    if (x === 0n && y === 1n) return false;
    const [XL, YL, ZL] = edwardsScalarMul([x, y, 1n, modField(x * y)], ED25519_L);
    if (XL !== 0n || YL !== ZL) return false;
  }
  return true;
}

function verifyStrictEd25519Signature(edPublicKey, messageBuf, signatureHex) {
  if (typeof signatureHex !== 'string' || !HEX128.test(signatureHex)) return false;
  const sigBuf = Buffer.from(signatureHex, 'hex');
  if (!isCanonicalEd25519Point(sigBuf.subarray(0, 32), { requirePrimeOrder: false })) return false;
  const s = decodeLittleEndian(sigBuf.subarray(32, 64));
  if (s >= ED25519_L) return false;
  return verify(null, messageBuf, edPublicKey, sigBuf);
}

const policyBytes = readFileSync(new URL('shares-policy.json', import.meta.url));
const POLICY = JSON.parse(policyBytes.toString('utf8'));
const POLICY_ID = createHash('sha256').update(policyBytes).digest('hex');

function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  if (value && typeof value === 'object') {
    const keys = Object.keys(value).filter(key => value[key] !== undefined).sort();
    return `{${keys.map(key => `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(',')}}`;
  }
  return JSON.stringify(value);
}

function sha256Hex(bufferOrString) {
  return createHash('sha256').update(bufferOrString).digest('hex');
}

function sha256Buf(bufferOrString) {
  return createHash('sha256').update(bufferOrString).digest();
}

function isHex64(value) {
  return typeof value === 'string' && HEX64.test(value);
}

function nonEmptyText(value) {
  return typeof value === 'string' && value.trim().length > 0 && value.trim() === value && !value.startsWith('--');
}

function readRegularFileBytes(filePath, label) {
  let st;
  try {
    st = lstatSync(filePath);
  } catch {
    throw new Error(`Missing ${label}: ${filePath}`);
  }
  if (!st.isFile()) {
    throw new Error(`Regular non-symlink file required for ${label}`);
  }
  return readFileSync(filePath);
}

function loadPinnedEnrollment(filePath, expectedSha256 = null) {
  const raw = readRegularFileBytes(filePath, 'enrollment');
  if (expectedSha256 !== null) {
    if (!isHex64(expectedSha256)) throw new Error('Invalid --enrollment-sha256');
    const actual = sha256Hex(raw);
    if (actual !== expectedSha256) {
      throw new Error(`Enrollment digest mismatch: expected ${expectedSha256}, got ${actual}`);
    }
  }
  return JSON.parse(raw.toString('utf8'));
}

export function validEd25519PublicJwk(jwk) {
  if (!jwk || typeof jwk !== 'object' || Array.isArray(jwk)) return false;
  if (jwk.kty !== 'OKP' || jwk.crv !== 'Ed25519' || jwk.d !== undefined || typeof jwk.x !== 'string') return false;
  try {
    const raw = Buffer.from(jwk.x, 'base64url');
    if (raw.length !== 32 || raw.toString('base64url') !== jwk.x) return false;
    if (!isCanonicalEd25519Point(raw, { requirePrimeOrder: true })) return false;
    createPublicKey({ key: { kty: 'OKP', crv: 'Ed25519', x: jwk.x }, format: 'jwk' });
    return true;
  } catch {
    return false;
  }
}

export function validateEnrollment(enrollment) {
  if (!enrollment || typeof enrollment !== 'object' || enrollment.schema !== ENROLLMENT_SCHEMA) {
    throw new Error('Versioned shares enrollment required');
  }
  if (!Array.isArray(enrollment.computers) || enrollment.computers.length < 2) {
    throw new Error('At least two enrolled computers required');
  }
  if (typeof enrollment.operatorIndependenceAttested !== 'boolean') {
    throw new Error('Explicit boolean operatorIndependenceAttested required');
  }
  if (enrollment.requesterOwnerId !== undefined && enrollment.requesterOwnerId !== null
    && !nonEmptyText(enrollment.requesterOwnerId)) {
    throw new Error('Invalid requesterOwnerId in enrollment');
  }
  const seenComputers = new Set();
  const seenParticipants = new Set();
  const seenKeys = new Set();
  const byComputer = new Map();
  for (const row of enrollment.computers) {
    if (!row || typeof row !== 'object'
      || !nonEmptyText(row.computerId)
      || !nonEmptyText(row.participantId)
      || !nonEmptyText(row.ownerId)
      || !validEd25519PublicJwk(row.publicJwk)) {
      throw new Error('Invalid computer entry in shares enrollment');
    }
    if (seenComputers.has(row.computerId)) throw new Error(`Duplicate enrolled computerId: ${row.computerId}`);
    if (seenParticipants.has(row.participantId)) throw new Error(`Duplicate enrolled participantId: ${row.participantId}`);
    if (seenKeys.has(row.publicJwk.x)) throw new Error(`Duplicate enrolled Ed25519 public key on ${row.computerId}`);
    seenComputers.add(row.computerId);
    seenParticipants.add(row.participantId);
    seenKeys.add(row.publicJwk.x);
    byComputer.set(row.computerId, row);
  }
  return {
    byComputer,
    requesterOwnerId: enrollment.requesterOwnerId ?? null,
    operatorIndependenceAttested: enrollment.operatorIndependenceAttested,
  };
}

export function verifyDistinctEnrolledProviders(enrollment, computerIdA, computerIdB, requesterOwnerOverride = null) {
  const { byComputer, requesterOwnerId, operatorIndependenceAttested } = validateEnrollment(enrollment);
  if (!nonEmptyText(computerIdA) || !nonEmptyText(computerIdB)) {
    throw new Error('Two target computerIds required');
  }
  if (computerIdA === computerIdB) {
    throw new Error('Two distinct enrolled computers required');
  }
  const a = byComputer.get(computerIdA);
  const b = byComputer.get(computerIdB);
  if (!a || !b) {
    throw new Error('Target provider computerId is not enrolled');
  }
  if (a.ownerId === b.ownerId) {
    throw new Error(`Providers ${a.computerId} and ${b.computerId} share enrolled ownerId ${a.ownerId}`);
  }
  if (a.publicJwk.x === b.publicJwk.x) {
    throw new Error('Providers share an Ed25519 public key');
  }
  if (requesterOwnerOverride !== null && requesterOwnerId !== null && requesterOwnerOverride !== requesterOwnerId) {
    throw new Error('Conflicting requesterOwnerId between CLI and enrollment');
  }
  const activeRequesterOwner = requesterOwnerOverride ?? requesterOwnerId;
  if (activeRequesterOwner !== null) {
    if (!nonEmptyText(activeRequesterOwner)) throw new Error('Invalid requesterOwnerId');
    if (a.ownerId === activeRequesterOwner || b.ownerId === activeRequesterOwner) {
      throw new Error('Provider ownerId matches requesterOwnerId');
    }
  }
  return {
    distinctEnrolledOwnersVerified: true,
    nonCollusionEstablished: operatorIndependenceAttested === true,
    requesterOwnerExcluded: activeRequesterOwner !== null,
    providers: [
      { role: 'a', computerId: a.computerId, participantId: a.participantId, ownerId: a.ownerId },
      { role: 'b', computerId: b.computerId, participantId: b.participantId, ownerId: b.ownerId },
    ],
  };
}

function readRegularManifestDigest(cacheDir) {
  const raw = readRegularFileBytes(join(cacheDir, 'shares-manifest.json'), 'shares-manifest.json');
  const parsed = JSON.parse(raw.toString('utf8'));
  if (parsed?.schema !== 'doppler.two-server-shares-manifest/v1' || !Array.isArray(parsed.projections)) {
    throw new Error('Invalid shares-manifest.json');
  }
  return sha256Hex(raw);
}

function generateEphemeralX25519() {
  const { publicKey, privateKey } = generateKeyPairSync('x25519');
  const spki = publicKey.export({ format: 'der', type: 'spki' });
  const rawPub = spki.subarray(spki.length - 32);
  if (rawPub.length !== 32) throw new Error('Unexpected X25519 SPKI length');
  return { privateKey, rawPublicHex: rawPub.toString('hex') };
}

function importPeerX25519(rawHex) {
  if (!isHex64(rawHex)) throw new Error('Invalid peer X25519 public key');
  const raw = Buffer.from(rawHex, 'hex');
  if (timingSafeEqual(raw, ZERO32)) throw new Error('Low-order X25519 public key rejected');
  return createPublicKey({
    key: Buffer.concat([X25519_SPKI_PREFIX, raw]),
    format: 'der',
    type: 'spki',
  });
}

function deriveSessionKeys(myPrivateX25519, peerPublicX25519, clientNonceHex, serverNonceHex, transcriptHash) {
  if (!isHex64(clientNonceHex) || !isHex64(serverNonceHex)) {
    throw new Error('Invalid session nonce');
  }
  const shared = diffieHellman({ privateKey: myPrivateX25519, publicKey: peerPublicX25519 });
  if (shared.length !== 32 || timingSafeEqual(shared, ZERO32)) {
    throw new Error('Invalid X25519 shared secret');
  }
  const salt = Buffer.concat([Buffer.from(clientNonceHex, 'hex'), Buffer.from(serverNonceHex, 'hex')]);
  const info = Buffer.concat([KEY_DOMAIN, transcriptHash]);
  const material = Buffer.from(hkdfSync('sha256', shared, salt, info, 64));
  return {
    c2sKey: material.subarray(0, 32),
    s2cKey: material.subarray(32, 64),
  };
}

function makeNonce(direction, seq) {
  const nonce = Buffer.alloc(12, 0);
  nonce[0] = direction;
  nonce.writeBigUInt64LE(seq, 4);
  return nonce;
}

function makeAad(transcriptHash, direction, seq, recordType) {
  const aad = Buffer.alloc(32 + 1 + 8 + 1);
  transcriptHash.copy(aad, 0);
  aad[32] = direction;
  aad.writeBigUInt64LE(seq, 33);
  aad[41] = recordType;
  return aad;
}

function encryptRecord(key, transcriptHash, direction, seq, recordType, plaintext) {
  if (plaintext.length > MAX_PLAINTEXT_CHUNK) throw new Error('Plaintext chunk exceeds frame bound');
  const nonce = makeNonce(direction, seq);
  const aad = makeAad(transcriptHash, direction, seq, recordType);
  const cipher = createCipheriv('chacha20-poly1305', key, nonce, { authTagLength: 16 });
  cipher.setAAD(aad, { plaintextLength: plaintext.length });
  const enc = Buffer.concat([cipher.update(plaintext), cipher.final()]);
  const tag = cipher.getAuthTag();
  const body = Buffer.allocUnsafe(1 + enc.length + 16);
  body[0] = recordType;
  enc.copy(body, 1);
  tag.copy(body, 1 + enc.length);
  return wrapFrame(body);
}

function decryptRecord(key, transcriptHash, direction, seq, frame) {
  if (frame.length < 17) throw new Error('Truncated encrypted frame');
  const recordType = frame[0];
  const ciphertext = frame.subarray(1, frame.length - 16);
  const tag = frame.subarray(frame.length - 16);
  if (ciphertext.length > MAX_PLAINTEXT_CHUNK) throw new Error('Ciphertext chunk exceeds frame bound');
  const nonce = makeNonce(direction, seq);
  const aad = makeAad(transcriptHash, direction, seq, recordType);
  const decipher = createDecipheriv('chacha20-poly1305', key, nonce, { authTagLength: 16 });
  decipher.setAAD(aad, { plaintextLength: ciphertext.length });
  decipher.setAuthTag(tag);
  const plaintext = Buffer.concat([decipher.update(ciphertext), decipher.final()]);
  return { recordType, plaintext };
}

function wrapFrame(body) {
  if (body.length > MAX_FRAME_BYTES) throw new Error('Frame exceeds maximum length');
  const out = Buffer.allocUnsafe(4 + body.length);
  out.writeUInt32LE(body.length, 0);
  body.copy(out, 4);
  return out;
}

class FrameReader {
  constructor(socket, maxBytes = MAX_FRAME_BYTES) {
    this.socket = socket;
    this.maxBytes = maxBytes;
    this.chunks = [];
    this.buffered = 0;
    this.ended = false;
    this.error = null;
    this.waiter = null;
    this.paused = false;

    socket.on('data', chunk => {
      if (this.buffered + chunk.length > HARD_BUFFER_CAP_BYTES) {
        this.error = new Error('FrameReader buffer ceiling exceeded');
        socket.destroy(this.error);
        this.#wake();
        return;
      }
      this.chunks.push(chunk);
      this.buffered += chunk.length;
      if (!this.paused && this.buffered > HIGH_WATER_BUFFER_BYTES) {
        this.paused = true;
        socket.pause();
      }
      this.#wake();
    });
    socket.on('end', () => {
      this.ended = true;
      this.#wake();
    });
    socket.on('close', () => {
      this.ended = true;
      this.#wake();
    });
    socket.on('error', err => {
      this.error = err;
      this.#wake();
    });
  }

  #wake() {
    if (this.waiter) {
      const resolve = this.waiter;
      this.waiter = null;
      resolve();
    }
  }

  #take(count) {
    const out = Buffer.allocUnsafe(count);
    let offset = 0;
    while (offset < count) {
      const head = this.chunks[0];
      const need = count - offset;
      if (head.length <= need) {
        head.copy(out, offset);
        offset += head.length;
        this.chunks.shift();
      } else {
        head.copy(out, offset, 0, need);
        this.chunks[0] = head.subarray(need);
        offset += need;
      }
    }
    this.buffered -= count;
    if (this.paused && this.buffered <= MAX_FRAME_BYTES && !this.ended && !this.socket.destroyed) {
      this.paused = false;
      this.socket.resume();
    }
    return out;
  }

  async readFrame(limit = this.maxBytes) {
    while (this.buffered < 4) {
      if (this.error) throw this.error;
      if (this.ended) {
        if (this.buffered > 0) throw new Error('Truncated frame header');
        return null;
      }
      await new Promise(resolve => { this.waiter = resolve; });
    }
    const header = this.#take(4);
    const length = header.readUInt32LE(0);
    if (length < 1 || length > limit) {
      throw new Error(`Invalid frame length ${length}`);
    }
    while (this.buffered < length) {
      if (this.error) throw this.error;
      if (this.ended) throw new Error('Truncated frame before completion');
      await new Promise(resolve => { this.waiter = resolve; });
    }
    return this.#take(length);
  }
}

function writeSocket(socket, buffer) {
  return new Promise((resolve, reject) => {
    if (socket.destroyed || !socket.writable) {
      reject(new Error('Socket not writable'));
      return;
    }
    socket.write(buffer, err => (err ? reject(err) : resolve()));
  });
}

function flushSocketEnd(socket) {
  return new Promise(resolve => {
    if (socket.destroyed || !socket.writable) {
      resolve();
      return;
    }
    let settled = false;
    const done = () => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve();
    };
    const timer = setTimeout(done, 5000);
    timer.unref();
    socket.once('close', done);
    socket.end(done);
  });
}

function writeStream(stream, buffer) {
  return new Promise((resolve, reject) => {
    if (stream.destroyed || !stream.writable) {
      reject(new Error('Stream not writable'));
      return;
    }
    stream.write(buffer, err => (err ? reject(err) : resolve()));
  });
}

export function parseEndpoint(spec, { allowZeroPort = false } = {}) {
  if (typeof spec !== 'string' || !spec.trim()) throw new Error('Endpoint required');
  let host;
  let portStr;
  if (spec.startsWith('[')) {
    const closeIdx = spec.indexOf(']:');
    if (closeIdx <= 1) throw new Error(`Invalid bracketed IPv6 host:port: ${spec}`);
    host = spec.slice(1, closeIdx);
    portStr = spec.slice(closeIdx + 2);
  } else {
    const idx = spec.lastIndexOf(':');
    if (idx <= 0) throw new Error(`Expected host:port, got ${spec}`);
    host = spec.slice(0, idx);
    if (host.includes(':')) throw new Error(`Bracket IPv6 literals in endpoint: ${spec}`);
    portStr = spec.slice(idx + 1);
  }
  if (!host || !DIGITS.test(portStr)) throw new Error(`Invalid host:port: ${spec}`);
  const port = Number(portStr);
  const minPort = allowZeroPort ? 0 : 1;
  if (!Number.isSafeInteger(port) || port < minPort || port > 65535) {
    throw new Error(`Port out of range in ${spec}`);
  }
  return { host, port };
}

export function parseFlags(args, { required = [], optional = {} } = {}) {
  const allowed = new Set([...required, ...Object.keys(optional)]);
  const parsed = new Map();
  for (let i = 0; i < args.length; i += 1) {
    const token = args[i];
    if (!allowed.has(token)) {
      throw new Error(`Unknown or unexpected argument: ${token}`);
    }
    if (parsed.has(token)) {
      throw new Error(`Duplicate flag: ${token}`);
    }
    if (i + 1 >= args.length) {
      throw new Error(`Missing value for ${token}`);
    }
    const val = args[i + 1];
    if (typeof val !== 'string' || val.length === 0 || val.startsWith('--')) {
      throw new Error(`Invalid value for ${token}`);
    }
    parsed.set(token, val);
    i += 1;
  }
  const result = {};
  for (const req of required) {
    if (!parsed.has(req)) throw new Error(`Missing ${req}`);
    result[req] = parsed.get(req);
  }
  for (const [opt, fallback] of Object.entries(optional)) {
    result[opt] = parsed.has(opt) ? parsed.get(opt) : fallback;
  }
  return result;
}

function buildTranscriptJson({
  role,
  computerId,
  participantId,
  ownerId,
  planId,
  modelId,
  integrityPolicyId,
  manifestSha256,
  clientNonce,
  serverNonce,
  clientEphemeralX25519,
  serverEphemeralX25519,
}) {
  return canonicalJson({
    clientEphemeralX25519,
    clientNonce,
    computerId,
    integrityPolicyId,
    manifestSha256,
    modelId,
    ownerId,
    participantId,
    planId,
    role,
    schema: TRANSCRIPT_SCHEMA,
    serverEphemeralX25519,
    serverNonce,
  });
}

function killChildGroup(child) {
  if (!child || typeof child.pid !== 'number') return;
  if (process.platform === 'win32') {
    try {
      spawnSync('taskkill', ['/PID', String(child.pid), '/T', '/F'], { stdio: 'ignore', timeout: 5000 });
    } catch {}
    try { child.kill('SIGKILL'); } catch {}
    return;
  }
  try {
    process.kill(-child.pid, 'SIGKILL');
  } catch {}
  try { child.kill('SIGKILL'); } catch {}
}

function writeSafeNewFile(filePath, content, mode) {
  let st = null;
  try {
    st = lstatSync(filePath);
  } catch (err) {
    if (err?.code !== 'ENOENT') throw err;
  }
  if (st !== null) {
    if (!st.isFile()) throw new Error(`Refusing to overwrite non-regular file ${filePath}`);
    unlinkSync(filePath);
  }
  writeFileSync(filePath, content, { flag: 'wx', mode });
  if (process.platform !== 'win32') {
    chmodSync(filePath, mode);
  }
}

async function runKeygen(args) {
  const flags = parseFlags(args, {
    required: ['--computer-id', '--participant-id', '--owner-id', '--out-private', '--out-public'],
  });
  const computerId = flags['--computer-id'];
  const participantId = flags['--participant-id'];
  const ownerId = flags['--owner-id'];
  const outPrivate = resolvePath(flags['--out-private']);
  const outPublic = resolvePath(flags['--out-public']);
  if (outPrivate === outPublic) throw new Error('--out-private and --out-public must differ');
  if (!nonEmptyText(computerId) || !nonEmptyText(participantId) || !nonEmptyText(ownerId)) {
    throw new Error('Non-empty computerId, participantId and ownerId required');
  }
  const { publicKey, privateKey } = generateKeyPairSync('ed25519');
  const rawPub = publicKey.export({ format: 'jwk' });
  const rawPriv = privateKey.export({ format: 'jwk' });
  const publicJwk = { kty: 'OKP', crv: 'Ed25519', x: rawPub.x };
  const privateJwk = { kty: 'OKP', crv: 'Ed25519', x: rawPriv.x, d: rawPriv.d };
  if (!validEd25519PublicJwk(publicJwk)) throw new Error('Generated Ed25519 key failed prime-order check');
  writeSafeNewFile(outPrivate, JSON.stringify({
    schema: PRIVATE_ID_SCHEMA,
    computerId,
    participantId,
    ownerId,
    privateJwk,
  }, null, 2) + '\n', 0o600);
  writeSafeNewFile(outPublic, JSON.stringify({
    computerId,
    participantId,
    ownerId,
    publicJwk,
  }, null, 2) + '\n', 0o644);
}

async function runVerifyEnrollment(args) {
  const flags = parseFlags(args, {
    required: ['--enrollment', '--provider-a', '--provider-b'],
    optional: { '--requester-owner': null, '--enrollment-sha256': null },
  });
  const enrollment = loadPinnedEnrollment(flags['--enrollment'], flags['--enrollment-sha256']);
  const summary = verifyDistinctEnrolledProviders(
    enrollment,
    flags['--provider-a'],
    flags['--provider-b'],
    flags['--requester-owner'],
  );
  process.stdout.write(JSON.stringify(summary) + '\n');
}

async function handleProviderSession(socket, identity, edPrivateKey, cacheDir, evaluatorCmd, onChildSpawned) {
  socket.setNoDelay(true);
  socket.setTimeout(POLICY.operationTimeoutSeconds * 1000);
  socket.on('timeout', () => socket.destroy(new Error('Provider socket idle timeout')));

  const reader = new FrameReader(socket);
  let child = null;
  let handshakeAdmitted = false;
  const handshakeTimer = setTimeout(() => socket.destroy(new Error('Handshake timeout')), HANDSHAKE_TIMEOUT_MS);
  const sessionTimer = setTimeout(
    () => socket.destroy(new Error('Provider attempt deadline exceeded')),
    POLICY.attemptTimeoutSeconds * 1000,
  );
  try {
    const helloFrame = await reader.readFrame(MAX_HELLO_BYTES);
    if (!helloFrame) throw new Error('Empty client hello');
    const hello = JSON.parse(helloFrame.toString('utf8'));
    if (hello?.schema !== CLIENT_HELLO_SCHEMA
      || (hello.role !== 'a' && hello.role !== 'b')
      || hello.expectedComputerId !== identity.computerId
      || hello.integrityPolicyId !== POLICY_ID
      || !isHex64(hello.planId)
      || !isHex64(hello.modelId)
      || !isHex64(hello.clientNonce)
      || !isHex64(hello.clientEphemeralX25519)) {
      throw new Error('Rejected client hello');
    }
    const manifestSha256 = readRegularManifestDigest(cacheDir);
    const clientPubX25519 = importPeerX25519(hello.clientEphemeralX25519);
    const serverEph = generateEphemeralX25519();
    const serverNonce = randomBytes(32).toString('hex');
    const transcript = buildTranscriptJson({
      role: hello.role,
      computerId: identity.computerId,
      participantId: identity.participantId,
      ownerId: identity.ownerId,
      planId: hello.planId,
      modelId: hello.modelId,
      integrityPolicyId: hello.integrityPolicyId,
      manifestSha256,
      clientNonce: hello.clientNonce,
      serverNonce,
      clientEphemeralX25519: hello.clientEphemeralX25519,
      serverEphemeralX25519: serverEph.rawPublicHex,
    });
    const transcriptHash = sha256Buf(transcript);
    const sig = sign(null, Buffer.concat([AUTH_DOMAIN, Buffer.from(transcript, 'utf8')]), edPrivateKey).toString('hex');
    const serverHello = Buffer.from(JSON.stringify({
      schema: SERVER_HELLO_SCHEMA,
      computerId: identity.computerId,
      participantId: identity.participantId,
      ownerId: identity.ownerId,
      manifestSha256,
      serverNonce,
      serverEphemeralX25519: serverEph.rawPublicHex,
      signature: sig,
    }), 'utf8');
    await writeSocket(socket, wrapFrame(serverHello));

    const { c2sKey, s2cKey } = deriveSessionKeys(
      serverEph.privateKey,
      clientPubX25519,
      hello.clientNonce,
      serverNonce,
      transcriptHash,
    );

    let c2sSeq = 0n;
    let s2cSeq = 0n;
    const readyFrame = await reader.readFrame(MAX_HELLO_BYTES);
    if (!readyFrame) throw new Error('Missing client ready confirmation');
    const ready = decryptRecord(c2sKey, transcriptHash, DIR_C2S, c2sSeq++, readyFrame);
    if (ready.recordType !== REC_READY || ready.plaintext.length !== 32 || !timingSafeEqual(ready.plaintext, transcriptHash)) {
      throw new Error('Invalid client ready confirmation');
    }
    handshakeAdmitted = true;
    clearTimeout(handshakeTimer);

    child = spawn(evaluatorCmd[0], evaluatorCmd.slice(1), {
      stdio: ['pipe', 'pipe', 'inherit'],
      detached: process.platform !== 'win32',
    });
    onChildSpawned(child);
    child.stdin.on('error', () => {});
    child.stdout.on('error', () => {});
    const childExit = new Promise((resolve, reject) => {
      child.once('error', reject);
      child.once('exit', code => resolve(code));
    });
    childExit.catch(() => {});

    const pumpClientToEvaluator = (async () => {
      try {
        for (;;) {
          const frame = await reader.readFrame();
          if (frame === null) {
            throw new Error('Unauthenticated client stream truncation');
          }
          const { recordType, plaintext } = decryptRecord(c2sKey, transcriptHash, DIR_C2S, c2sSeq++, frame);
          if (recordType === REC_DATA) {
            if (plaintext.length > 0) await writeStream(child.stdin, plaintext);
          } else if (recordType === REC_EOF) {
            if (plaintext.length !== 0) throw new Error('Invalid EOF payload');
            child.stdin.end();
            return;
          } else {
            throw new Error('Unexpected client record type');
          }
        }
      } catch (err) {
        killChildGroup(child);
        socket.destroy();
        throw err;
      }
    })();

    const pumpEvaluatorToClient = (async () => {
      try {
        for await (const chunk of child.stdout) {
          for (let off = 0; off < chunk.length; off += MAX_PLAINTEXT_CHUNK) {
            const slice = chunk.subarray(off, Math.min(chunk.length, off + MAX_PLAINTEXT_CHUNK));
            await writeSocket(socket, encryptRecord(s2cKey, transcriptHash, DIR_S2C, s2cSeq++, REC_DATA, slice));
          }
        }
        const exitCode = await childExit;
        if (exitCode !== 0) {
          throw new Error(`Evaluator exited with status ${exitCode}`);
        }
        await writeSocket(socket, encryptRecord(s2cKey, transcriptHash, DIR_S2C, s2cSeq++, REC_EOF, Buffer.alloc(0)));
        await flushSocketEnd(socket);
      } catch (err) {
        killChildGroup(child);
        socket.destroy();
        throw err;
      }
    })();

    await Promise.all([pumpClientToEvaluator, pumpEvaluatorToClient]);
    return { handshakeAdmitted };
  } finally {
    clearTimeout(handshakeTimer);
    clearTimeout(sessionTimer);
    killChildGroup(child);
    onChildSpawned(null);
    socket.destroy();
  }
}

async function runServe(args) {
  const sepIndex = args.indexOf('--');
  if (sepIndex < 0 || sepIndex + 1 >= args.length) {
    throw new Error('Expected -- followed by evaluator command');
  }
  const flags = parseFlags(args.slice(0, sepIndex), {
    required: ['--listen', '--identity', '--cache-dir'],
    optional: { '--max-sessions': '0' },
  });
  const evaluatorCmd = args.slice(sepIndex + 1);
  const { host, port } = parseEndpoint(flags['--listen'], { allowZeroPort: true });
  const maxSessionsStr = flags['--max-sessions'];
  if (!DIGITS.test(maxSessionsStr)) throw new Error('Invalid --max-sessions');
  const maxSessions = Number(maxSessionsStr);
  if (!Number.isSafeInteger(maxSessions) || maxSessions < 0) throw new Error('Invalid --max-sessions');

  const identityRaw = readRegularFileBytes(flags['--identity'], 'identity');
  const identity = JSON.parse(identityRaw.toString('utf8'));
  if (identity?.schema !== PRIVATE_ID_SCHEMA
    || !nonEmptyText(identity.computerId)
    || !nonEmptyText(identity.participantId)
    || !nonEmptyText(identity.ownerId)
    || identity.privateJwk?.kty !== 'OKP'
    || identity.privateJwk?.crv !== 'Ed25519'
    || typeof identity.privateJwk?.d !== 'string'
    || !validEd25519PublicJwk({ kty: 'OKP', crv: 'Ed25519', x: identity.privateJwk.x })) {
    throw new Error('Invalid provider private identity file');
  }
  const edPrivateKey = createPrivateKey({
    key: { kty: 'OKP', crv: 'Ed25519', x: identity.privateJwk.x, d: identity.privateJwk.d },
    format: 'jwk',
  });
  if (createPublicKey(edPrivateKey).export({ format: 'jwk' }).x !== identity.privateJwk.x) {
    throw new Error('Provider privateJwk x does not match private scalar d');
  }
  const initialManifestSha256 = readRegularManifestDigest(flags['--cache-dir']);

  let active = false;
  let activeChild = null;
  let completed = 0;
  const server = createServer(socket => {
    socket.on('error', () => {});
    if (active) {
      socket.destroy();
      return;
    }
    active = true;
    let admitted = false;
    handleProviderSession(
      socket,
      identity,
      edPrivateKey,
      flags['--cache-dir'],
      evaluatorCmd,
      spawned => { activeChild = spawned; },
    )
      .then(res => { admitted = Boolean(res?.handshakeAdmitted); })
      .catch(err => {
        process.stderr.write(`shares_network serve session error: ${err.message}\n`);
      })
      .finally(() => {
        active = false;
        if (admitted) {
          completed += 1;
          if (maxSessions > 0 && completed >= maxSessions) {
            server.close();
          }
        }
      });
  });

  const shutdown = () => {
    killChildGroup(activeChild);
    server.close(() => process.exit(0));
    setTimeout(() => process.exit(0), 1000).unref();
  };
  process.once('SIGINT', shutdown);
  process.once('SIGTERM', shutdown);

  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(port, host, () => resolve());
  });
  const addr = server.address();
  process.stdout.write(JSON.stringify({
    event: 'listening',
    computerId: identity.computerId,
    participantId: identity.participantId,
    ownerId: identity.ownerId,
    host: addr.address,
    port: addr.port,
    manifestSha256: initialManifestSha256,
  }) + '\n');
}

async function runConnect(args) {
  process.stdout.on('error', () => {});
  const flags = parseFlags(args, {
    required: [
      '--endpoint',
      '--enrollment',
      '--computer-id',
      '--role',
      '--plan-id',
      '--model-id',
      '--policy-id',
      '--manifest-sha256',
    ],
    optional: {
      '--enrollment-sha256': null,
      '--peer-computer-id': null,
      '--requester-owner': null,
    },
  });
  const { host, port } = parseEndpoint(flags['--endpoint'], { allowZeroPort: false });
  const expectedComputerId = flags['--computer-id'];
  const role = flags['--role'];
  const planId = flags['--plan-id'];
  const modelId = flags['--model-id'];
  const integrityPolicyId = flags['--policy-id'];
  const expectedManifestSha256 = flags['--manifest-sha256'];

  if ((role !== 'a' && role !== 'b')
    || !isHex64(planId)
    || !isHex64(modelId)
    || integrityPolicyId !== POLICY_ID
    || !isHex64(expectedManifestSha256)) {
    throw new Error('Invalid connect binding parameters');
  }
  const enrollment = loadPinnedEnrollment(flags['--enrollment'], flags['--enrollment-sha256']);
  const { byComputer } = validateEnrollment(enrollment);
  if (flags['--peer-computer-id'] !== null) {
    const [compA, compB] = role === 'a'
      ? [expectedComputerId, flags['--peer-computer-id']]
      : [flags['--peer-computer-id'], expectedComputerId];
    verifyDistinctEnrolledProviders(enrollment, compA, compB, flags['--requester-owner']);
  }
  const enrolled = byComputer.get(expectedComputerId);
  if (!enrolled) throw new Error(`Computer ${expectedComputerId} is not in enrollment`);
  const edPublicKey = createPublicKey({
    key: { kty: 'OKP', crv: 'Ed25519', x: enrolled.publicJwk.x },
    format: 'jwk',
  });

  const socket = await new Promise((resolve, reject) => {
    const conn = createConnection({ host, port });
    const timer = setTimeout(() => {
      conn.destroy();
      reject(new Error('Connect timeout'));
    }, HANDSHAKE_TIMEOUT_MS);
    conn.once('error', err => {
      clearTimeout(timer);
      reject(err);
    });
    conn.once('connect', () => {
      clearTimeout(timer);
      resolve(conn);
    });
  });
  socket.on('error', () => {});
  socket.setNoDelay(true);
  socket.setTimeout(POLICY.operationTimeoutSeconds * 1000);
  socket.on('timeout', () => socket.destroy(new Error('Client socket idle timeout')));

  const reader = new FrameReader(socket);
  const handshakeTimer = setTimeout(() => socket.destroy(new Error('Handshake timeout')), HANDSHAKE_TIMEOUT_MS);
  const sessionTimer = setTimeout(
    () => socket.destroy(new Error('Client attempt deadline exceeded')),
    POLICY.attemptTimeoutSeconds * 1000,
  );
  try {
    const clientEph = generateEphemeralX25519();
    const clientNonce = randomBytes(32).toString('hex');
    const clientHello = Buffer.from(JSON.stringify({
      schema: CLIENT_HELLO_SCHEMA,
      role,
      expectedComputerId,
      planId,
      modelId,
      integrityPolicyId,
      clientNonce,
      clientEphemeralX25519: clientEph.rawPublicHex,
    }), 'utf8');
    await writeSocket(socket, wrapFrame(clientHello));

    const serverFrame = await reader.readFrame(MAX_HELLO_BYTES);
    if (!serverFrame) throw new Error('Provider closed before ServerHello');
    const serverHello = JSON.parse(serverFrame.toString('utf8'));
    if (serverHello?.schema !== SERVER_HELLO_SCHEMA
      || serverHello.computerId !== enrolled.computerId
      || serverHello.participantId !== enrolled.participantId
      || serverHello.ownerId !== enrolled.ownerId
      || serverHello.manifestSha256 !== expectedManifestSha256
      || !isHex64(serverHello.serverNonce)
      || !isHex64(serverHello.serverEphemeralX25519)) {
      throw new Error('ServerHello failed identity, owner or manifest binding');
    }
    const serverPubX25519 = importPeerX25519(serverHello.serverEphemeralX25519);
    const transcript = buildTranscriptJson({
      role,
      computerId: enrolled.computerId,
      participantId: enrolled.participantId,
      ownerId: enrolled.ownerId,
      planId,
      modelId,
      integrityPolicyId,
      manifestSha256: expectedManifestSha256,
      clientNonce,
      serverNonce: serverHello.serverNonce,
      clientEphemeralX25519: clientEph.rawPublicHex,
      serverEphemeralX25519: serverHello.serverEphemeralX25519,
    });
    const signedPayload = Buffer.concat([AUTH_DOMAIN, Buffer.from(transcript, 'utf8')]);
    if (!verifyStrictEd25519Signature(edPublicKey, signedPayload, serverHello.signature)) {
      throw new Error('Ed25519 provider signature verification failed');
    }

    const transcriptHash = sha256Buf(transcript);
    const { c2sKey, s2cKey } = deriveSessionKeys(
      clientEph.privateKey,
      serverPubX25519,
      clientNonce,
      serverHello.serverNonce,
      transcriptHash,
    );

    let c2sSeq = 0n;
    let s2cSeq = 0n;
    await writeSocket(socket, encryptRecord(c2sKey, transcriptHash, DIR_C2S, c2sSeq++, REC_READY, transcriptHash));
    clearTimeout(handshakeTimer);

    const pumpStdinToServer = (async () => {
      try {
        for await (const chunk of process.stdin) {
          for (let off = 0; off < chunk.length; off += MAX_PLAINTEXT_CHUNK) {
            const slice = chunk.subarray(off, Math.min(chunk.length, off + MAX_PLAINTEXT_CHUNK));
            await writeSocket(socket, encryptRecord(c2sKey, transcriptHash, DIR_C2S, c2sSeq++, REC_DATA, slice));
          }
        }
        await writeSocket(socket, encryptRecord(c2sKey, transcriptHash, DIR_C2S, c2sSeq++, REC_EOF, Buffer.alloc(0)));
      } catch (err) {
        socket.destroy();
        throw err;
      }
    })();

    const pumpServerToStdout = (async () => {
      try {
        for (;;) {
          const frame = await reader.readFrame();
          if (frame === null) {
            throw new Error('Unauthenticated provider stream truncation');
          }
          const { recordType, plaintext } = decryptRecord(s2cKey, transcriptHash, DIR_S2C, s2cSeq++, frame);
          if (recordType === REC_DATA) {
            if (plaintext.length > 0) await writeStream(process.stdout, plaintext);
          } else if (recordType === REC_EOF) {
            if (plaintext.length !== 0) throw new Error('Invalid provider EOF payload');
            return;
          } else {
            throw new Error('Unexpected provider record type');
          }
        }
      } catch (err) {
        socket.destroy();
        throw err;
      }
    })();

    await Promise.all([pumpStdinToServer, pumpServerToStdout]);
    await flushSocketEnd(socket);
  } finally {
    clearTimeout(handshakeTimer);
    clearTimeout(sessionTimer);
    socket.destroy();
  }
}

function isExecutedDirectly() {
  if (!process.argv[1]) return false;
  try {
    return import.meta.url === pathToFileURL(realpathSync(process.argv[1])).href;
  } catch {
    return false;
  }
}

if (isExecutedDirectly()) {
  const mode = process.argv[2];
  const rest = process.argv.slice(3);
  const dispatch = {
    keygen: runKeygen,
    'verify-enrollment': runVerifyEnrollment,
    serve: runServe,
    connect: runConnect,
  }[mode];
  if (!dispatch) {
    process.stderr.write('Usage: node shares_network.js <keygen|verify-enrollment|serve|connect> ...\n');
    process.exit(2);
  }
  dispatch(rest).catch(err => {
    process.stderr.write(`shares_network ${mode}: ${err.message}\n`);
    process.exit(1);
  });
}
