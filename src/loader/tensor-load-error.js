import { getTensorShardIndices } from './tensor-shard-indices.js';

export function annotateWeightLoadError(error, details = {}) {
  if (!error || typeof error !== 'object') {
    return error;
  }
  const existingDetails = error.details && typeof error.details === 'object'
    ? error.details
    : {};
  error.details = {
    ...existingDetails,
    weightLoadFailure: {
      ...(existingDetails.weightLoadFailure && typeof existingDetails.weightLoadFailure === 'object'
        ? existingDetails.weightLoadFailure
        : {}),
      ...details,
    },
  };
  return error;
}


export function annotateTensorLoadError(error, name, location, details = {}) {
  return annotateWeightLoadError(error, {
    tensorName: name,
    tensorRole: location?.role ?? null,
    tensorDtype: location?.dtype ?? null,
    tensorShape: Array.isArray(location?.shape) ? [...location.shape] : null,
    tensorSizeBytes: Number.isFinite(location?.size) ? location.size : null,
    tensorShardIndices: getTensorShardIndices(location),
    ...details,
  });
}
