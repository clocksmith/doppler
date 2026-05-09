import type { AdapterManifest } from '../adapters/adapter-manifest.js';
import type { LoRAModuleName } from '../../inference/pipelines/text/lora-types.js';
import type { Tensor } from '../../gpu/tensor.js';

export interface LoRAExportTensor {
  name: string;
  tensor: Tensor | Float32Array | GPUBuffer;
  shape?: [number, number];
  dtype?: 'f16' | 'f32';
}

export interface LoRAExportOptions {
  id: string;
  name: string;
  baseModel: string;
  rank: number;
  alpha: number;
  targetModules: LoRAModuleName[];
  version?: string;
  description?: string;
  metadata?: Record<string, unknown>;
  tensors: LoRAExportTensor[];
  format?: 'base64' | 'array';
  weightsFormat?: 'json' | 'safetensors';
  weightsPath?: string;
  pretty?: boolean;
}

export interface LoRAExportResult {
  manifest: AdapterManifest;
  json: string;
  weights?: ArrayBuffer;
  weightsSha256?: string;
  weightsPath?: string;
}

export declare function serializeLoRASafetensors(
  tensors: Array<{ name: string; shape: [number, number]; data: Float32Array | number[] }>
): ArrayBuffer;

export declare function exportLoRAAdapter(options: LoRAExportOptions): Promise<LoRAExportResult>;
