/**
 * LoRA adapter type definitions for pipeline modules.
 *
 * @module inference/pipelines/text/lora-types
 */

import type { MaybeGPUBuffer } from './buffer-types.js';

export type LoRAModuleName =
  | 'q_proj'
  | 'k_proj'
  | 'v_proj'
  | 'o_proj'
  | 'gate_proj'
  | 'up_proj'
  | 'down_proj'
  | 'gate_up_proj';

export interface LoRAModuleWeights {
  a: MaybeGPUBuffer;
  b: MaybeGPUBuffer;
  rank: number;
  alpha: number;
  scale: number;
}

export type LoRALayerMap = Record<string, LoRAModuleWeights>;

export interface LoRAAdapter {
  name: string;
  version?: string;
  baseModel?: string;
  rank: number;
  alpha: number;
  targetModules?: LoRAModuleName[];
  layers: Map<number, LoRALayerMap>;
}

export const LORA_MODULE_ALIASES: Record<string, LoRAModuleName>;
