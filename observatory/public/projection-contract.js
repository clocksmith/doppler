// Replaying or projecting a saved run requires its actual weights, not merely
// another artifact published under the same human-readable model name.
//
// A run records the hash as "sha256:<hex>"; a loaded model reports the same hash
// as bare hex. Both spellings name one identity, so they are compared in one form.
const identity = (hash) => (typeof hash === 'string' && hash.trim() ? hash.trim().toLowerCase().replace(/^sha256:/, '') : null);
export function assertRunModel(run, manifestHash) {
  if (!identity(run.model.manifestHash) || identity(run.model.manifestHash) !== identity(manifestHash)) {
    throw new Error('The loaded model files do not have the saved run’s verified manifest identity.');
  }
}

export function validateProjection(result, layers) {
  if (result?.projection?.kind !== 'projected' || !Array.isArray(result.rows) || result.rows.length !== layers
    || result.rows.some((row) => !Array.isArray(row?.candidates) || !row.candidates.length
      || row.candidates.some((candidate) => !Number.isSafeInteger(candidate?.tokenId) || candidate.tokenId < 0
        || typeof candidate.text !== 'string' || !Number.isFinite(candidate.probability)
        || candidate.probability < 0 || candidate.probability > 1))) {
    throw new Error('The runtime must return one valid candidate projection per recorded layer.');
  }
  return result;
}
