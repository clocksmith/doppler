import { sha256Hex } from '../../utils/sha256.js';
import { KERNEL_REF_CONTENT_DIGESTS } from './kernel-ref-digests.js';

export const KERNEL_REF_VERSION = '1.0.0';

function normalizeKernelRefIdToken(value) {
  return String(value ?? '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '.')
    .replace(/^\.+|\.+$/g, '') || 'kernel';
}

function normalizeKernel(value) {
  const kernel = String(value ?? '').trim();
  if (kernel.length === 0) {
    throw new Error('kernel must be a non-empty string');
  }
  return kernel;
}

function normalizeEntry(value) {
  const entry = String(value ?? 'main').trim() || 'main';
  return entry;
}

export function getKernelRefContentDigest(kernel, entry = 'main') {
  const normalizedKernel = normalizeKernel(kernel);
  const normalizedEntry = normalizeEntry(entry);
  const key = `${normalizedKernel}#${normalizedEntry}`;
  const digest = KERNEL_REF_CONTENT_DIGESTS[key];
  if (typeof digest !== 'string' || digest.length !== 64) {
    throw new Error(`No kernel content digest registered for "${key}"`);
  }
  return digest;
}

export function buildKernelRefFromKernelEntry(kernel, entry = 'main') {
  const normalizedKernel = normalizeKernel(kernel);
  const normalizedEntry = normalizeEntry(entry);
  const kernelToken = normalizeKernelRefIdToken(normalizedKernel.replace(/\.wgsl$/i, ''));
  const entryToken = normalizeKernelRefIdToken(normalizedEntry);
  const digest = getKernelRefContentDigest(normalizedKernel, normalizedEntry);
  return {
    id: `${kernelToken}.${entryToken}`,
    version: KERNEL_REF_VERSION,
    digest: `sha256:${digest}`,
  };
}

export function buildLegacyKernelRefFromKernelEntry(kernel, entry = 'main') {
  const normalizedKernel = normalizeKernel(kernel);
  const normalizedEntry = normalizeEntry(entry);
  const kernelToken = normalizeKernelRefIdToken(normalizedKernel.replace(/\.wgsl$/i, ''));
  const entryToken = normalizeKernelRefIdToken(normalizedEntry);
  const digest = sha256Hex(`${normalizedKernel}#${normalizedEntry}`);
  return {
    id: `${kernelToken}.${entryToken}`,
    version: KERNEL_REF_VERSION,
    digest: `sha256:${digest}`,
  };
}

export function isKernelRefBoundToKernel(kernelRef, kernel, entry = 'main') {
  if (!kernelRef || typeof kernelRef !== 'object' || Array.isArray(kernelRef)) {
    return false;
  }
  const expected = buildKernelRefFromKernelEntry(kernel, entry);
  return (
    kernelRef.id === expected.id
    && kernelRef.version === expected.version
    && kernelRef.digest === expected.digest
  );
}

