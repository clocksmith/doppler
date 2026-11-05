import { doMatmul } from '../ops.js';
import { isWeightBuffer } from '../../../../gpu/weight-buffer.js';
import { getKernelPathMatmulVariant } from '../../../../config/kernel-path-loader.js';

/** @type {typeof import('./projection.js').createHyperConnectionProjection} */
export function createHyperConnectionProjection({ recorder, kernelPath, phase, layerIdx }) {
  if (!kernelPath || (phase !== 'prefill' && phase !== 'decode')) {
    throw new Error('[HyperConnection] Projections require a declared kernel path and execution phase.');
  }
  return async (role, input, weight, outFeatures, inFeatures) => {
    if (!getKernelPathMatmulVariant(role, phase, layerIdx, kernelPath)) {
      throw new Error(`[HyperConnection] Missing declared ${phase} projection ${role} at layer ${layerIdx}.`);
    }
    if (input.dtype !== 'f32' || input.precision || !isWeightBuffer(weight)) {
      throw new Error(`[HyperConnection] ${role} requires ordinary f32 activations and declared projection weights.`);
    }
    return doMatmul(input, weight, input.shape[0], outFeatures, inFeatures, {
      transposeB: true, outputDtype: input.dtype, kernelPath, phaseOverride: phase, layerIdx, role,
    }, recorder ?? undefined);
  };
}
