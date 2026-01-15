import type { RDRRManifest } from '../storage/rdrr-format.js';
import type { TensorLocation } from './loader-types.js';

export interface BuildTensorLocationsOptions {
  hasCustomLoader?: boolean;
  tensorsJsonUrl?: string;
}

export function buildTensorLocations(
  manifest: RDRRManifest,
  options?: BuildTensorLocationsOptions
): Promise<Map<string, TensorLocation>>;

export function isEmbeddingTensor(name: string): boolean;
export function isLMHeadTensor(name: string): boolean;
export function isNormTensor(name: string): boolean;
export function isMatmulTensor(name: string): boolean;
