/**
 * LoRA adapter loader.
 *
 * Supports JSON manifest with inline tensor data (array or base64),
 * OPFS storage, and URL-based loading for runtime weight deltas.
 *
 * @module adapters/lora-loader
 */

import type { LoRAAdapter, LoRAModuleName } from '../inference/pipeline/lora.js';
import type { AdapterManifest, AdapterTensorSpec } from './adapter-manifest.js';

/**
 * @deprecated Use AdapterTensorSpec from adapter-manifest.ts
 */
export interface LoRATensorSpec {
  name: string;
  shape: [number, number];
  dtype?: 'f32';
  data?: number[];
  base64?: string;
  opfsPath?: string;
  url?: string;
}

/**
 * @deprecated Use AdapterManifest from adapter-manifest.ts
 */
export interface LoRAManifest {
  name: string;
  version?: string;
  baseModel?: string;
  rank: number;
  alpha: number;
  targetModules?: LoRAModuleName[];
  tensors: LoRATensorSpec[];
}

/**
 * Options for loading LoRA weights.
 */
export interface LoRALoadOptions {
  /** Function to read from OPFS storage */
  readOPFS?: (path: string) => Promise<ArrayBuffer>;
  /** Function to write to OPFS storage */
  writeOPFS?: (path: string, data: ArrayBuffer) => Promise<void>;
  /** Function to fetch from URL */
  fetchUrl?: (url: string) => Promise<ArrayBuffer>;
  /** Skip checksum verification */
  skipVerify?: boolean;
  /** Progress callback */
  onProgress?: (loaded: number, total: number) => void;
}

/**
 * Result of loading LoRA weights.
 */
export interface LoRAWeightsResult {
  adapter: LoRAAdapter;
  manifest: AdapterManifest;
  loadedFromCache: boolean;
  checksumValid?: boolean;
}

/**
 * Loads LoRA weights from a file path (OPFS or URL).
 */
export declare function loadLoRAWeights(
  path: string,
  options?: LoRALoadOptions
): Promise<LoRAWeightsResult>;

/**
 * Loads LoRA adapter from parsed manifest.
 */
export declare function loadLoRAFromManifest(
  manifest: LoRAManifest,
  options?: LoRALoadOptions
): Promise<LoRAAdapter>;

/**
 * Loads LoRA adapter from URL.
 */
export declare function loadLoRAFromUrl(
  url: string,
  options?: LoRALoadOptions
): Promise<LoRAAdapter>;

/**
 * Applies delta weights to base model weights at runtime.
 */
export declare function applyDeltaWeights(
  baseWeight: Float32Array,
  loraA: Float32Array,
  loraB: Float32Array,
  scale: number
): Float32Array;

/**
 * Loads LoRA weights from safetensors format.
 */
export declare function loadLoRAFromSafetensors(
  data: ArrayBuffer,
  manifest: AdapterManifest
): Promise<LoRAAdapter>;
