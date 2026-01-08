/**
 * Shard Resolver - Tensor location mapping and resolution.
 *
 * Pure functions for building tensor location maps from manifests.
 * Used by DopplerLoader to resolve tensor names to shard locations.
 *
 * @module loader/shard-resolver
 */

import type { RDRRManifest } from '../storage/rdrr-format.js';
import type { TensorLocation } from './loader-types.js';

export interface BuildTensorLocationsOptions {
  /** URL to fetch tensors.json from (for HTTP-based loading) */
  tensorsJsonUrl?: string | null;
  /** Whether using custom shard loader (skip OPFS) */
  hasCustomLoader?: boolean;
}

/**
 * Build tensor location map from manifest.
 *
 * Supports two formats:
 * - v1: External tensors.json file (referenced by manifest.tensorsFile)
 * - Legacy: Inline tensors in manifest.tensors
 *
 * @param manifest - Model manifest
 * @param options - Build options
 * @returns Map of tensor names to locations
 */
export declare function buildTensorLocations(
  manifest: RDRRManifest,
  options?: BuildTensorLocationsOptions
): Promise<Map<string, TensorLocation>>;

/**
 * Check if tensor name indicates an embedding weight.
 */
export declare function isEmbeddingTensor(name: string): boolean;

/**
 * Check if tensor name indicates an LM head weight.
 */
export declare function isLMHeadTensor(name: string): boolean;

/**
 * Check if tensor name indicates a norm weight.
 */
export declare function isNormTensor(name: string): boolean;

/**
 * Check if tensor name indicates a matmul weight (projection).
 */
export declare function isMatmulTensor(name: string): boolean;
