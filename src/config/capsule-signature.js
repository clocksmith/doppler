import { hashCapsuleV2PublicKey } from './capsule-v2.js';

export function validateCapsuleSignature(signature, digest) {
  const errors = [];
  if (!signature || typeof signature !== 'object' || Array.isArray(signature)) {
    return ['Capsule signature is required.'];
  }
  const fields = ['authority', 'algorithm', 'publicKeyDigest', 'signatureHex', 'signedDigest'];
  if (Object.keys(signature).some((key) => !fields.includes(key))) errors.push('Unknown Capsule signature field.');
  if (typeof signature.authority !== 'string' || !signature.authority.trim()) errors.push('Signing authority is required.');
  if (signature.algorithm !== 'Ed25519') errors.push('Capsule signature algorithm must be Ed25519.');
  if (!/^sha256:[0-9a-f]{64}$/.test(signature.publicKeyDigest)) errors.push('Invalid signing key digest.');
  if (!/^[0-9a-f]{128}$/.test(signature.signatureHex)) errors.push('Invalid Ed25519 signature.');
  if (signature.signedDigest !== digest) errors.push('Signature does not bind the declared digest.');
  return errors;
}

export async function signCapsuleDigest(digest, signer) {
  if (!/^sha256:[0-9a-f]{64}$/.test(digest)) throw new Error('Invalid Capsule signing digest.');
  if (typeof signer?.authority !== 'string' || !signer.authority.trim()) throw new Error('Capsule signer authority is required.');
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) throw new Error('Capsule signing requires WebCrypto.');
  const key = await subtle.importKey('jwk', signer.privateKeyJwk, 'Ed25519', false, ['sign']);
  const bytes = new Uint8Array(await subtle.sign('Ed25519', key, new TextEncoder().encode(digest)));
  const signature = {
    authority: signer.authority,
    algorithm: 'Ed25519',
    publicKeyDigest: hashCapsuleV2PublicKey(signer.publicKeyJwk),
    signatureHex: Array.from(bytes, (value) => value.toString(16).padStart(2, '0')).join(''),
    signedDigest: digest,
  };
  await verifyCapsuleDigest(signature, digest, signer.publicKeyJwk);
  return signature;
}

export async function verifyCapsuleDigest(signature, digest, publicKeyJwk) {
  const errors = validateCapsuleSignature(signature, digest);
  if (errors.length) throw new Error(errors.join('; '));
  if (!publicKeyJwk || hashCapsuleV2PublicKey(publicKeyJwk) !== signature.publicKeyDigest) {
    throw new Error('Untrusted Capsule signing key.');
  }
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) throw new Error('Capsule verification requires WebCrypto.');
  const key = await subtle.importKey('jwk', publicKeyJwk, 'Ed25519', false, ['verify']);
  const bytes = Uint8Array.from(signature.signatureHex.match(/.{2}/g), (value) => Number.parseInt(value, 16));
  if (!await subtle.verify('Ed25519', key, bytes, new TextEncoder().encode(digest))) {
    throw new Error('Capsule signature verification failed.');
  }
  return true;
}
