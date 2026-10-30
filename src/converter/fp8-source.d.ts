import type { FP8Policy } from '../config/schema/fp8.schema.js';
import type { FP8StorageDescriptor } from '../formats/rdrr/fp8-storage.js';
export interface FP8SourceTensor { name: string; dtype: string; shape: number[]; layout?: string; }
export interface FP8SourcePlan { tensors: FP8SourceTensor[]; companions: Map<string, FP8SourceTensor>; }
export function fp8SourceEncoding(dtype: string): 'e4m3fn' | 'e5m2' | null;
export function prepareFP8SourceTensors(tensors: FP8SourceTensor[], policy: FP8Policy | null | undefined,
  modelConfig: { quantization_config?: { quant_method?: string; fmt?: string; scale_fmt?: string; weight_block_size?: number[] } } | null): FP8SourcePlan;
export function readFP8CompanionScales(tensor: FP8SourceTensor, companion: FP8SourceTensor | undefined,
  policy: FP8Policy | null | undefined, read: (tensor: FP8SourceTensor) => Promise<ArrayBuffer | Uint8Array>): Promise<Float32Array | null>;
export function packFP8Source(bytes: Uint8Array, shape: number[], policy: FP8Policy,
  scales: Float32Array | null): { quantized: Uint8Array; storage: FP8StorageDescriptor };
export function materializeFP8Source(bytes: Uint8Array, shape: number[], policy: FP8Policy,
  scales: Float32Array | null): Float32Array;
