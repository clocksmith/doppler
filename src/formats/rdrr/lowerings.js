
export const DOPPLER_LOWERING_MISSING = 'DOPPLER_LOWERING_MISSING';
export const DOPPLER_LOWERING_REJECTED = 'DOPPLER_LOWERING_REJECTED';

function getEntries(manifest) {
  const entries = manifest?.integrityExtensions?.lowerings?.entries;
  return Array.isArray(entries) ? entries : [];
}

export function findLowering(manifest, kernelRef, backend) {
  if (typeof kernelRef !== 'string' || kernelRef.length === 0) {
    throw new Error('findLowering: kernelRef must be a non-empty string');
  }
  if (typeof backend !== 'string' || backend.length === 0) {
    throw new Error('findLowering: backend must be a non-empty string');
  }
  for (const entry of getEntries(manifest)) {
    if (entry.kernelRef === kernelRef && entry.backend === backend) {
      return entry;
    }
  }
  return null;
}

export function isRejectionEntry(entry) {
  return !!(entry
    && Array.isArray(entry.rejectionReasons)
    && entry.rejectionReasons.length > 0);
}

export function findLoweringOrThrow(manifest, kernelRef, backend) {
  const entry = findLowering(manifest, kernelRef, backend);
  if (entry === null) {
    const error = new Error(
      `No lowering entry for kernelRef="${kernelRef}" backend="${backend}"`
    );
    error.code = DOPPLER_LOWERING_MISSING;
    error.kernelRef = kernelRef;
    error.backend = backend;
    throw error;
  }
  if (isRejectionEntry(entry)) {
    const error = new Error(
      `Backend "${backend}" refused kernel "${kernelRef}": ${entry.rejectionReasons.join(', ')}`
    );
    error.code = DOPPLER_LOWERING_REJECTED;
    error.kernelRef = kernelRef;
    error.backend = backend;
    error.rejectionReasons = entry.rejectionReasons.slice();
    throw error;
  }
  return entry;
}

export function listSupportedBackends(manifest, kernelRefs) {
  if (!Array.isArray(kernelRefs) || kernelRefs.length === 0) {
    return [];
  }
  const required = new Set(kernelRefs);
  const byBackend = new Map();
  for (const entry of getEntries(manifest)) {
    if (!required.has(entry.kernelRef)) continue;
    let state = byBackend.get(entry.backend);
    if (!state) {
      state = { covered: new Set(), rejected: false };
      byBackend.set(entry.backend, state);
    }
    if (isRejectionEntry(entry)) {
      state.rejected = true;
    } else {
      state.covered.add(entry.kernelRef);
    }
  }
  const result = [];
  for (const [backend, state] of byBackend.entries()) {
    if (state.rejected) continue;
    if (state.covered.size === required.size) {
      result.push(backend);
    }
  }
  result.sort();
  return result;
}
