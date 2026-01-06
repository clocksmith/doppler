/**
 * Hot-Swap Manifest Utilities
 *
 * Verifies signed hot-swap bundles for JS/WGSL/JSON artifacts.
 *
 * @module hotswap/manifest
 */

import type { HashAlgorithm } from '../config/schema/index.js';
import type { HotSwapConfigSchema, HotSwapSignerSchema } from '../config/schema/hotswap.schema.js';
import { log } from '../debug/index.js';

// =============================================================================
// Types
// =============================================================================

export interface HotSwapArtifact {
  path: string;
  hash: string;
  hashAlgorithm?: HashAlgorithm;
}

export interface HotSwapManifest {
  bundleId: string;
  version: string;
  artifacts: HotSwapArtifact[];
  signerId?: string;
  signature?: string;
  createdAt?: string;
  metadata?: Record<string, string>;
}

export interface HotSwapVerificationResult {
  ok: boolean;
  reason: string;
  signerId?: string;
}

// =============================================================================
// Fetch + Verification
// =============================================================================

export async function fetchHotSwapManifest(url: string): Promise<HotSwapManifest> {
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`Failed to fetch hot-swap manifest: ${response.status}`);
  }
  return response.json() as Promise<HotSwapManifest>;
}

export async function verifyHotSwapManifest(
  manifest: HotSwapManifest,
  policy: HotSwapConfigSchema
): Promise<HotSwapVerificationResult> {
  if (!policy.enabled) {
    return { ok: false, reason: 'Hot-swap disabled' };
  }

  if (!manifest.signature) {
    if (policy.localOnly && policy.allowUnsignedLocal) {
      return { ok: true, reason: 'Local-only unsigned manifest accepted' };
    }
    return { ok: false, reason: 'Signature required' };
  }

  if (!manifest.signerId) {
    return { ok: false, reason: 'Missing signerId' };
  }

  const signer = policy.trustedSigners.find((entry) => entry.id === manifest.signerId);
  if (!signer) {
    return { ok: false, reason: `Signer not trusted: ${manifest.signerId}`, signerId: manifest.signerId };
  }

  const subtle = globalThis.crypto?.subtle;
  if (!subtle) {
    return { ok: false, reason: 'WebCrypto unavailable', signerId: manifest.signerId };
  }

  try {
    const payloadBytes = new TextEncoder().encode(serializeHotSwapManifest(manifest));
    const payload = payloadBytes.buffer.slice(payloadBytes.byteOffset, payloadBytes.byteOffset + payloadBytes.byteLength);
    const signatureBytes = decodeBase64ToArrayBuffer(manifest.signature);
    const key = await importSignerKey(signer);
    const ok = await subtle.verify(
      { name: 'ECDSA', hash: 'SHA-256' },
      key,
      signatureBytes,
      payload
    );
    return ok
      ? { ok: true, reason: 'Signature verified', signerId: manifest.signerId }
      : { ok: false, reason: 'Signature mismatch', signerId: manifest.signerId };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    log.warn('HotSwap', `Signature verification failed: ${message}`);
    return { ok: false, reason: 'Signature verification failed', signerId: manifest.signerId };
  }
}

export function serializeHotSwapManifest(manifest: HotSwapManifest): string {
  const { signature, ...payload } = manifest;
  return stableStringify(payload);
}

// =============================================================================
// Helpers
// =============================================================================

async function importSignerKey(signer: HotSwapSignerSchema): Promise<CryptoKey> {
  return globalThis.crypto.subtle.importKey(
    'jwk',
    signer.publicKeyJwk,
    { name: 'ECDSA', namedCurve: 'P-256' },
    false,
    ['verify']
  );
}

function decodeBase64ToArrayBuffer(value: string): ArrayBuffer {
  if (typeof atob === 'function') {
    const raw = atob(value);
    const bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i += 1) {
      bytes[i] = raw.charCodeAt(i);
    }
    return bytes.buffer;
  }

  if (typeof Buffer !== 'undefined') {
    const buffer = Buffer.from(value, 'base64');
    return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
  }

  throw new Error('No base64 decoder available');
}

function stableStringify(value: unknown): string {
  if (value === null || typeof value !== 'object') {
    return JSON.stringify(value);
  }

  if (Array.isArray(value)) {
    return `[${value.map((item) => stableStringify(item)).join(',')}]`;
  }

  const record = value as Record<string, unknown>;
  const keys = Object.keys(record).sort();
  const entries = keys.map((key) => `${JSON.stringify(key)}:${stableStringify(record[key])}`);
  return `{${entries.join(',')}}`;
}
