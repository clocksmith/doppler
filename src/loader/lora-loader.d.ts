/**
 * LoRA Loader - Loads LoRA weights from an adapter manifest.
 *
 * @module loader/lora-loader
 */

import type { RDRRManifest } from '../storage/rdrr-format.js';
import type { LoRAAdapter } from '../inference/pipelines/text/lora.js';
import type { WeightBuffer, CpuWeightBuffer } from '../gpu/weight-buffer.js';

export interface LoRALoaderContext {
  manifest: RDRRManifest;
  loadTensor: (name: string, toGPU: boolean, silent: boolean) => Promise<GPUBuffer | WeightBuffer | CpuWeightBuffer | Float32Array | Uint8Array | null>;
}

/**
 * Load LoRA weights from an adapter manifest.
 */
export declare function loadLoRAWeights(
  manifest: RDRRManifest,
  loadTensor: (name: string, toGPU: boolean, silent: boolean) => Promise<GPUBuffer | WeightBuffer | CpuWeightBuffer | Float32Array | Uint8Array | null>
): Promise<LoRAAdapter>;
