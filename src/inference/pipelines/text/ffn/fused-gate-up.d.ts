import type { DenseGateUpInputs } from './types.js';
import type { Tensor } from '../../../../gpu/tensor.js';
export function runFusedGateUpFFNGPU(inputs: DenseGateUpInputs & {
  defaultMatmulOutputDtype: 'f16' | 'f32';
  fusedGateUpWeights: ReturnType<typeof import('./dense-plan.js').resolveFusedGateUpWeights>;
  gateDtype: string | null;
}): Promise<Tensor>;
