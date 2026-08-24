import type { Tensor } from '../../../../gpu/tensor.js';
import type { DiffusionModelConfig, DiffusionRuntimeConfig } from '../types.js';
import type { DiffusionWeightEntry } from '../weights.js';
import type { CommandRecorder } from '../../../../gpu/command-recorder.js';

export interface SD3ExecutionBindings {
  applyAdaLayerNorm: (...args: unknown[]) => Promise<unknown>;
  applyGate: (...args: unknown[]) => Promise<unknown>;
  applyQKNorm: (...args: unknown[]) => Promise<unknown>;
  buildModulation: (...args: unknown[]) => Promise<unknown>;
  concatKV: (...args: unknown[]) => Promise<unknown>;
  createBiasTensorWithDtype: (...args: unknown[]) => unknown;
  createKernelOps: (...args: unknown[]) => Record<string, unknown>;
  createVectorBuffer: (...args: unknown[]) => GPUBuffer;
  runFFN: (...args: unknown[]) => Promise<unknown>;
  runMatmulResolved: (...args: unknown[]) => Promise<unknown>;
  runQKV: (...args: unknown[]) => Promise<unknown>;
}

export declare function executeSD3Transformer(
  latents: Tensor,
  context: Tensor,
  timeText: Tensor,
  weightsEntry: DiffusionWeightEntry,
  modelConfig: DiffusionModelConfig,
  runtime: DiffusionRuntimeConfig,
  options: { recorder?: CommandRecorder | null } | undefined,
  bindings: SD3ExecutionBindings
): Promise<Tensor>;

