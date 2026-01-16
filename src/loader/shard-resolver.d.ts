import type { RDRRManifest } from '../storage/rdrr-format.js';
import type { TensorLocation } from './loader-types.js';

export interface BuildTensorLocationsOptions {
  hasCustomLoader?: boolean;
  tensorsJsonUrl?: string | null;
}

export function buildTensorLocations(
  manifest: RDRRManifest,
  options?: BuildTensorLocationsOptions
): Promise<Map<string, TensorLocation>>;
