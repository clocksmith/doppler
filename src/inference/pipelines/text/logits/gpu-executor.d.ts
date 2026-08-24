import type { CommandRecorder } from '../../../../gpu/command-recorder.js';
import type { Tensor } from '../../../../gpu/tensor.js';
import type { LogitsConfig, LogitsWeights } from './types.js';

export declare function resolvePrecisionFieldDtype(
  precision: Record<string, unknown> | null | undefined,
  fallback: string | null | undefined,
  field: string
): string | null | undefined;
export declare function resolveMatmulStepDtype(
  role: string,
  phase: string,
  kernelPath: unknown,
  fallback: string | null | undefined,
  field: string
): string | null | undefined;
export declare function resolvePostLayerStepDtype(
  op: string,
  phase: string,
  kernelPath: unknown,
  fallback: string | null | undefined,
  field: string
): string | null | undefined;
export declare function resolveLmHeadMatmulRole(phase: string): 'lm_head_prefill' | 'lm_head';
export declare function coerceTensorDtype(
  tensor: Tensor,
  targetDtype: string | null | undefined,
  recorder?: CommandRecorder | null,
  options?: Record<string, unknown>
): Promise<Tensor>;
export declare function resolveFinalNormGpuBuffer(
  finalNorm: unknown,
  queue: GPUQueue,
  label: string
): { buffer: GPUBuffer; owned: boolean };
export declare function recordLogitsGPU(
  recorder: CommandRecorder,
  hiddenStates: GPUBuffer,
  numTokens: number,
  weights: LogitsWeights,
  config: LogitsConfig,
  operatorDiagnostics?: unknown
): Promise<{ logitsBuffer: GPUBuffer; vocabSize: number; logitsDtype: 'f16' | 'f32' }>;
export declare function recordGreedyLmHeadArgmaxGPU(
  recorder: CommandRecorder,
  hiddenStates: GPUBuffer,
  numTokens: number,
  weights: LogitsWeights,
  config: LogitsConfig,
  options: {
    padTokenId: number | null;
    logitSoftcap: number;
    outputBuffer: GPUBuffer;
    outputIndex: number;
  },
  operatorDiagnostics?: unknown
): Promise<GPUBuffer>;

