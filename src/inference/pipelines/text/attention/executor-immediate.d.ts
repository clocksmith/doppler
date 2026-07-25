import type { Tensor } from '../../../../gpu/tensor.js';
import type {
  CpuWeightBuffer,
  WeightBuffer,
} from '../../../../gpu/weight-buffer.js';
import type { LoRAAdapter } from '../lora.js';
import type { LayerWeights } from '../types.js';
import type {
  AttentionConfig,
  AttentionDebugFlags,
  AttentionResult,
  AttentionState,
} from './types.js';

export function runLayerAttentionGPU(
  input: Tensor,
  layerWeights: LayerWeights | null,
  config: AttentionConfig,
  state: AttentionState,
  debug?: boolean,
  debugFlags?: AttentionDebugFlags,
  getWeightBuffer?: (
    weight: GPUBuffer | WeightBuffer | Float32Array | ArrayBuffer | CpuWeightBuffer,
    label: string
  ) => GPUBuffer | WeightBuffer,
  getNormWeightBuffer?: (
    weight: GPUBuffer | Float32Array | ArrayBuffer | CpuWeightBuffer,
    label: string
  ) => GPUBuffer,
  debugCheckBuffer?: (
    buffer: GPUBuffer,
    label: string,
    numTokens: number,
    expectedDim?: number
  ) => Promise<void>,
  lora?: LoRAAdapter | null
): Promise<AttentionResult>;
