import type { Tensor } from '../../../../gpu/tensor.js';

export function projectSeparateAttentionGate(options: {
  runMatmul: (...args: unknown[]) => Promise<Tensor>;
  projectionInput: Tensor;
  gateWeight: unknown;
  numTokens: number;
  outputSize: number;
  hiddenSize: number;
  layerIdx: number;
  kernelPath?: unknown;
  outputDtype?: string | null;
  matmulDebug?: unknown;
  executionPolicies?: unknown;
  fusedNormWeight?: unknown;
  fusedNormEps?: number | null;
  fusedNormOffset?: boolean;
}): Promise<Tensor | null>;
