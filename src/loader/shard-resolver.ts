/**
 * Shard Resolver - Tensor location mapping and resolution.
 *
 * Pure functions for building tensor location maps from manifests.
 * Used by DopplerLoader to resolve tensor names to shard locations.
 *
 * @module loader/shard-resolver
 */

import { loadTensorsFromOPFS } from '../storage/shard-manager.js';
import {
  parseTensorMap,
  type RDRRManifest,
  type TensorLocation as RDRRTensorLocation,
} from '../storage/rdrr-format.js';
import type { TensorLocation } from './loader-types.js';
import { log, trace as debugTrace } from '../debug/index.js';

// ============================================================================
// Tensor Location Building
// ============================================================================

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
export async function buildTensorLocations(
  manifest: RDRRManifest,
  options: BuildTensorLocationsOptions = {}
): Promise<Map<string, TensorLocation>> {
  const locations = new Map<string, TensorLocation>();

  // v1 format: load external tensors.json
  if (manifest.tensorsFile) {
    debugTrace.loader(`Loading external tensor map: ${manifest.tensorsFile}`);

    let tensorsJsonRaw: string | null = null;

    // Try OPFS first (for downloaded models)
    if (!options.hasCustomLoader) {
      tensorsJsonRaw = await loadTensorsFromOPFS();
    }

    // Try HTTP if we have a tensors URL set (for HTTP-based testing)
    if (!tensorsJsonRaw && options.tensorsJsonUrl) {
      try {
        const resp = await fetch(options.tensorsJsonUrl);
        if (resp.ok) {
          tensorsJsonRaw = await resp.text();
          debugTrace.loader(`Loaded tensors.json via HTTP: ${options.tensorsJsonUrl}`);
        }
      } catch (e) {
        log.warn('Loader', `Failed to load tensors.json from ${options.tensorsJsonUrl}: ${(e as Error).message}`);
      }
    }

    if (tensorsJsonRaw) {
      const tensorsJson = parseTensorMap(tensorsJsonRaw);
      for (const [name, rdrrInfo] of Object.entries(tensorsJson)) {
        const info = rdrrInfo as RDRRTensorLocation;
        locations.set(name, {
          shardIndex: info.shard,
          offset: info.offset,
          size: info.size,
          shape: info.shape,
          dtype: info.dtype,
          spans: info.spans,
          layout: info.layout,
          originalShape: info.originalShape,
        });
      }
      debugTrace.loader(`Loaded ${locations.size} tensors from tensors.json`);
      return locations;
    }
  }

  // Legacy format: inline tensors in manifest
  if (!manifest.tensors) {
    log.warn('Loader', 'No tensor locations in manifest');
    return locations;
  }

  for (const [name, info] of Object.entries(manifest.tensors)) {
    const tensorInfo = info as {
      shard?: number;
      shardIndex?: number;
      offset: number;
      size: number;
      shape: number[];
      dtype: string;
      spans?: Array<{ shardIndex: number; offset: number; size: number }>;
      layout?: 'row' | 'column';
      originalShape?: number[];
    };
    locations.set(name, {
      shardIndex: tensorInfo.shardIndex ?? tensorInfo.shard ?? 0,
      offset: tensorInfo.offset,
      size: tensorInfo.size,
      shape: tensorInfo.shape,
      dtype: tensorInfo.dtype,
      spans: tensorInfo.spans,
      layout: tensorInfo.layout,
      originalShape: tensorInfo.originalShape,
    });
  }
  debugTrace.loader(`Tensor map: ${locations.size} tensors (inline)`);
  return locations;
}

// ============================================================================
// Tensor Predicates
// ============================================================================

/**
 * Check if tensor name indicates an embedding weight.
 */
export function isEmbeddingTensor(name: string): boolean {
  const lower = name.toLowerCase();
  return (
    lower.includes('embd') ||
    lower.includes('embed') ||
    lower.includes('wte')
  );
}

/**
 * Check if tensor name indicates an LM head weight.
 */
export function isLMHeadTensor(name: string): boolean {
  const lower = name.toLowerCase();
  return (
    lower.includes('lm_head') ||
    lower.includes('output.weight')
  );
}

/**
 * Check if tensor name indicates a norm weight.
 */
export function isNormTensor(name: string): boolean {
  const lower = name.toLowerCase();
  return (
    lower.includes('norm') ||
    lower.includes('ln_') ||
    lower.includes('layernorm')
  );
}

/**
 * Check if tensor name indicates a matmul weight (projection).
 */
export function isMatmulTensor(name: string): boolean {
  const lower = name.toLowerCase();
  return (
    lower.includes('proj') ||
    lower.includes('gate') ||
    lower.includes('up') ||
    lower.includes('down') ||
    lower.includes('wq') ||
    lower.includes('wk') ||
    lower.includes('wv') ||
    lower.includes('wo')
  );
}
