/** Session binding, key schedule and authenticated frames for `doppler-attested-session/v1`. */
import { createCipheriv, createDecipheriv, createHash, createPublicKey, diffieHellman, generateKeyPairSync, hkdfSync } from 'node:crypto';
import { readFileSync } from 'node:fs';

export const PROTOCOL = 'doppler-attested-session/v1';
export const CLIENT_TO_PROVIDER = 1;
export const PROVIDER_TO_CLIENT = 2;
const X25519_SPKI_PREFIX = Buffer.from('302a300506032b656e032100', 'hex');
const TAG_BYTES = 16;

export function sessionError(code, message) {
  return Object.assign(new Error(message), { code });
}

function check(ok, code, message) {
  if (!ok) throw sessionError(code, message);
}

const digestBytes = (value, label) => {
  check(typeof value === 'string' && /^sha256:[a-f0-9]{64}$/.test(value), 'session-binding', `Invalid ${label} digest.`);
  return Buffer.from(value.slice(7), 'hex');
};

export function loadSessionPolicy(url = new URL('session-policy.json', import.meta.url)) {
  const bytes = readFileSync(url);
  const policy = JSON.parse(bytes.toString('utf8'));
  check(policy.schema === 'doppler.attested.session-policy/v1' && policy.protocol === PROTOCOL, 'session-policy', 'Versioned session policy required.');
  for (const key of ['requestPaddingBytes', 'responsePaddingBytes', 'maximumRequestBytes', 'maximumResponseChunkBytes', 'maximumResponseFrameBytes', 'maximumResponseFrames']) {
    check(Number.isSafeInteger(policy[key]) && policy[key] > 0, 'session-policy', `Invalid ${key}.`);
  }
  return Object.freeze({ ...policy, id: `sha256:${createHash('sha256').update(bytes).digest('hex')}` });
}

export function generateSessionKey() {
  const { publicKey, privateKey } = generateKeyPairSync('x25519');
  return { privateKey, publicBytes: publicKey.export({ type: 'spki', format: 'der' }).subarray(X25519_SPKI_PREFIX.length) };
}

/**
 * The 64 bytes the provider asks its hardware to sign. Every field is fixed length, so the encoding is unambiguous.
 * It names the attempt (client nonce), the key the client will encrypt to, the exact model, the state namespace and the session policy.
 */
export function bindingReportData({ clientNonce, providerKey, modelIdentity, stateNamespace, sessionPolicyId }) {
  check(clientNonce.length === 32 && providerKey.length === 32, 'session-binding', 'Nonce and provider key must be 32 bytes.');
  return createHash('sha512').update(`${PROTOCOL}\0report-data\0`).update(clientNonce).update(providerKey)
    .update(digestBytes(modelIdentity, 'model')).update(digestBytes(stateNamespace, 'state namespace'))
    .update(digestBytes(sessionPolicyId, 'session policy')).digest();
}

/** Direction keys from the X25519 secret, salted by a transcript of the signed binding, the report and the client's key. */
export function deriveSessionKeys({ privateKey, peerKey, reportData, reportDigest, clientKey }) {
  check(peerKey.length === 32 && clientKey.length === 32 && reportData.length === 64 && reportDigest.length === 32, 'session-binding', 'Malformed key schedule input.');
  let secret;
  try {
    secret = diffieHellman({ privateKey, publicKey: createPublicKey({ key: Buffer.concat([X25519_SPKI_PREFIX, peerKey]), format: 'der', type: 'spki' }) });
  } catch { throw sessionError('session-key', 'Key agreement failed.'); }
  check(secret.some(byte => byte !== 0), 'session-key', 'Key agreement produced the identity.');
  const transcript = createHash('sha256').update(`${PROTOCOL}\0transcript\0`).update(reportData).update(reportDigest).update(clientKey).digest();
  const keys = Buffer.from(hkdfSync('sha256', secret, transcript, `${PROTOCOL} keys`, 64));
  secret.fill(0);
  return { transcript, request: keys.subarray(0, 32), response: keys.subarray(32, 64) };
}

function associated(transcript, direction, counter, final) {
  const header = Buffer.alloc(10);
  header[0] = direction;
  header.writeBigUInt64BE(counter, 1);
  header[9] = final ? 1 : 0;
  return Buffer.concat([transcript, header]);
}

function nonce(counter) {
  const bytes = Buffer.alloc(12);
  bytes.writeBigUInt64BE(counter, 4);
  return bytes;
}

function pad(plaintext, bucket) {
  const padded = Buffer.alloc(Math.ceil((plaintext.length + 4) / bucket) * bucket);
  padded.writeUInt32BE(plaintext.length, 0);
  plaintext.copy(padded, 4);
  return padded;
}

function unpad(padded) {
  check(padded.length >= 4, 'session-frame', 'Frame too short.');
  const length = padded.readUInt32BE(0);
  check(length <= padded.length - 4 && padded.subarray(4 + length).every(byte => byte === 0), 'session-frame', 'Malformed frame padding.');
  return padded.subarray(4, 4 + length);
}

/** Frames carry an implicit counter. Order, direction, the final marker and the transcript are all authenticated. */
export function createSealer(key, transcript, direction, paddingBytes) {
  let counter = 0n;
  let closed = false;
  return function seal(plaintext, final) {
    check(!closed, 'session-frame', 'Stream already finished.');
    const cipher = createCipheriv('aes-256-gcm', key, nonce(counter));
    cipher.setAAD(associated(transcript, direction, counter, final));
    const body = Buffer.concat([cipher.update(pad(Buffer.from(plaintext), paddingBytes)), cipher.final()]);
    counter += 1n;
    closed = final;
    return Buffer.concat([Buffer.from([final ? 1 : 0]), body, cipher.getAuthTag()]);
  };
}

export function createOpener(key, transcript, direction, maximumFrameBytes) {
  let counter = 0n;
  let closed = false;
  const open = frame => {
    check(!closed, 'session-frame', 'Frame after the final frame.');
    check(frame.length > 1 + TAG_BYTES && frame.length <= maximumFrameBytes && frame[0] <= 1, 'session-frame', 'Malformed frame.');
    const final = frame[0] === 1;
    const decipher = createDecipheriv('aes-256-gcm', key, nonce(counter));
    decipher.setAAD(associated(transcript, direction, counter, final));
    decipher.setAuthTag(frame.subarray(frame.length - TAG_BYTES));
    let padded;
    try {
      padded = Buffer.concat([decipher.update(frame.subarray(1, frame.length - TAG_BYTES)), decipher.final()]);
    } catch { throw sessionError('session-frame', 'Frame failed authentication.'); }
    counter += 1n;
    closed = final;
    return { plaintext: unpad(padded), final };
  };
  open.finished = () => closed;
  return open;
}
