import { annotateWeightLoadError } from '../inference/pipelines/text/load-errors.js';
import { getTensorShardIndices } from './tensor-shard-indices.js';

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
