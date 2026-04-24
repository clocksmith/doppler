/**
 * Lookup helpers for integrityExtensions.lowerings entries.
 *
 * validateManifest checks the *shape* of the lowerings section. This module
 * provides the runtime-side lookups that a backend-selection pass calls when
 * it needs a specific (kernelRef, backend) pair. A caller that requires a
 * specific backend must call findLoweringOrThrow; a caller that wants to
 * probe capability uses findLowering.
 */

export const DOPPLER_LOWERING_MISSING = 'DOPPLER_LOWERING_MISSING';
export const DOPPLER_LOWERING_REJECTED = 'DOPPLER_LOWERING_REJECTED';

function getEntries(manifest) {
  const entries = manifest?.integrityExtensions?.lowerings?.entries;
  return Array.isArray(entries) ? entries : [];
}

/**
 * Return the lowering entry for (kernelRef, backend), or null if absent.
 * Does not distinguish success from rejection — caller inspects the entry.
 */
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

/**
 * Return the lowering entry for (kernelRef, backend), or throw with
 * DOPPLER_LOWERING_MISSING / DOPPLER_LOWERING_REJECTED.
 *
 * Runtime loaders pick a backend first, then call this per execution-graph
 * step. A missing entry fails fast here instead of at first-token during
 * weight load.
 */
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

/**
 * List the backends for which this manifest carries a successful (non-rejected)
 * lowering for every kernelRef in the provided set. A backend that has a
 * rejection for any kernelRef is excluded.
 *
 * kernelRefs must be the complete set a runtime loader intends to execute —
 * the caller owns that enumeration from the execution graph.
 */
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
