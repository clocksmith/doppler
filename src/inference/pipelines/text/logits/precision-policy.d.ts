import type { KernelPathSchema } from '../../../../config/schema/index.js';
import type { LogitsConfig } from './types.js';

export function shouldForceStableF32Logits(
  config: Pick<LogitsConfig, 'finalLogitSoftcapping' | 'rmsNormWeightOffset' | 'hiddenSize'>,
  inputDtype: 'f16' | 'f32'
): boolean;

export function createStableF32LogitsKernelPath(
  kernelPath: KernelPathSchema | null | undefined
): KernelPathSchema | null | undefined;
