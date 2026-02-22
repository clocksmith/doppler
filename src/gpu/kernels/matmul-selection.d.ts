import type { KernelConfig } from './utils.js';
import type { TensorDtype } from '../tensor.js';
import type { WeightBuffer } from '../weight-buffer.js';
import type { KernelPathSchema } from '../../config/schema/index.js';

export declare function resolveMatmulPhase(M: number): string;

export declare function resolveMatmulConstants(
  options: {
    constants?: Record<string, number | boolean>;
    role?: string;
    layerIdx?: number;
    kernelPath?: KernelPathSchema | null;
  },
  phase: string
): Record<string, number | boolean> | null;

export declare function getMatmulConfig(
  variant: string,
  constants: Record<string, number | boolean> | null
): KernelConfig;

export declare function isFusedQ4KDisabled(options?: {
  kernelPath?: KernelPathSchema | null;
}): boolean;

export declare function toMatmulDtype(dtype: string | null | undefined): 'f16' | 'f32' | 'q4k';

export declare function selectMatmulKernel(options?: {
  preferF16?: boolean;
  useVec4?: boolean;
  outputDtype?: TensorDtype | 'f16' | 'f32';
  aDtype?: 'f16' | 'f32' | null;
  bDtype?: 'f16' | 'f32' | 'q4k' | null;
  isPrefill?: boolean;
  prefillRows?: number;
  transposeB?: boolean;
}): string;

export declare function resolveTransposeB(
  B: GPUBuffer | WeightBuffer,
  transposeBOption: boolean | 'auto'
): boolean;

export declare function validateMatmulDimensions(label: string, M: number, N: number, K: number): void;

export declare function validateMatmulOffsets(label: string, aOffset: number, bOffset: number, cOffset: number): void;

export declare function getMatmulBindingSizes(
  label: string,
  A: GPUBuffer,
  B: GPUBuffer,
  M: number,
  N: number,
  K: number,
  aDtype: 'f16' | 'f32',
  bDtype: 'f16' | 'f32' | 'q4k',
  transposeB: boolean,
  aOffset: number,
  bOffset: number
): { aBindingSize: number; bBindingSize: number };

export declare function requiresF32Input(variant: string): boolean;

export declare function selectMatmulVariantAndFlags(
  mode: string,
  M: number,
  N: number,
  K: number,
  aDtype: 'f16' | 'f32',
  bDtype: 'f16' | 'f32' | 'q4k',
  transposeB: boolean,
  requestedOutputDtype: TensorDtype | 'f16' | 'f32',
  options: { role?: string; layerIdx?: number; kernelPath?: KernelPathSchema | null; [key: string]: unknown }
): { variant: string; useQ4KFused: boolean; useGemv: boolean };

export declare function resolveMatmulOutput(
  variant: string,
  M: number,
  N: number,
  outputBuffer?: GPUBuffer | null
): {
  output: GPUBuffer;
  outputSize: number;
  cBindingSize: number;
  actualOutputDtype: TensorDtype;
};
