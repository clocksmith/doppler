import type { Tensor } from '../../gpu/tensor.js';
import type { QwenHybridLayer } from './qwen-hybrid-decoder-training-module.js';

export type QwenPeftProjection =
  | 'q_proj'
  | 'k_proj'
  | 'v_proj'
  | 'o_proj'
  | 'gate_proj'
  | 'up_proj'
  | 'down_proj';

export interface QwenPeftAdapterTensor {
  sourceName: string;
  canonicalName: string;
  layerIndex: number;
  branch: 'self_attn' | 'mlp';
  projection: QwenPeftProjection;
  kind: 'a' | 'b';
  sourceDtype: 'F32' | 'F16' | 'BF16';
  sourceShape: [number, number];
  shape: [number, number];
  data: Float32Array;
}

export interface QwenPeftAdapterImport {
  rank: number;
  alpha: number;
  scale: number;
  targetModules: QwenPeftProjection[];
  layerTypes: Array<'linear_attention' | 'full_attention'>;
  tensors: QwenPeftAdapterTensor[];
  tensorCount: number;
  pairCount: number;
  elementCount: number;
}

export interface QwenPeftAdapterImportOptions {
  rank?: number;
  r?: number;
  alpha?: number;
  lora_alpha?: number;
  targetModules?: QwenPeftProjection[];
  target_modules?: QwenPeftProjection[];
  layerTypes: Array<'linear_attention' | 'full_attention'>;
}

export interface QwenPeftAdapterUploadPlanEntry {
  sourceName: string;
  canonicalName: string;
  tensor: Tensor;
  data: Float32Array;
}

export declare function parseQwenPeftAdapterSafetensors(
  data: ArrayBuffer | ArrayBufferView,
  options: QwenPeftAdapterImportOptions
): QwenPeftAdapterImport;

export declare function buildQwenPeftAdapterUploadPlan(
  layers: QwenHybridLayer[],
  adapter: QwenPeftAdapterImport
): QwenPeftAdapterUploadPlanEntry[];

export declare function uploadQwenPeftAdapterToLayers(
  layers: QwenHybridLayer[],
  adapter: QwenPeftAdapterImport
): { tensorCount: number; elementCount: number; canonicalNames: string[] };
