import type { QwenHybridLayer } from './qwen-hybrid-decoder-training-module.js';
import type { QwenPeftProjection } from './qwen-peft-adapter-import.js';

export interface QwenPeftAdapterExportOptions {
  rank: number;
  alpha: number;
  dropout: number;
  baseModel: string;
  targetModules: QwenPeftProjection[];
  layerTypes: Array<'linear_attention' | 'full_attention'>;
}

export interface QwenPeftAdapterExportResult {
  weights: ArrayBuffer;
  adapterConfig: Record<string, unknown>;
  adapterConfigJson: string;
  tensorCount: number;
  pairCount: number;
  elementCount: number;
  tensorNames: string[];
}

export declare function exportQwenPeftAdapterFromLayers(
  layers: QwenHybridLayer[],
  options: QwenPeftAdapterExportOptions
): Promise<QwenPeftAdapterExportResult>;
