import type { RDRRManifest } from '../formats/rdrr/index.js';
import type { TensorLocation } from './loader-types.js';

export interface BuildTensorLocationsOptions {
  hasCustomLoader?: boolean;
  tensorsJsonUrl?: string | null;
  loadTensorsJson?: (() => Promise<string | Record<string, unknown> | null | undefined>) | null;
}

export function buildTensorLocations(
  manifest: RDRRManifest,
  options?: BuildTensorLocationsOptions
): Promise<Map<string, TensorLocation>>;
