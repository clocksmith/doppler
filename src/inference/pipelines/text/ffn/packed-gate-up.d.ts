import type { DenseGateUpInputs } from './types.js';
import type { Tensor } from '../../../../gpu/tensor.js';
export function runPackedGateUpFFNGPU(inputs: DenseGateUpInputs & { lastTokenIdx: number }): Promise<Tensor>;
