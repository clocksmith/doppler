import { doMatmul } from "../execution/ops.js";
import { isWeightBuffer } from '../../../../gpu/weight-buffer.js';
import { KERNEL_REF_CONTENT_DIGESTS } from '../../../../config/kernels/kernel-ref-digests.js';
import { getKernelConfigs } from '../../../../config/kernel-registry-contract.js';
import { getKernelPathMatmulVariant, getLayerSteps, getKernelPathStepConfig } from '../../../../config/kernel-path-loader.js';

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

/** @type {typeof import('./projection.js').resolveHyperConnectionMixBranch} */
export function resolveHyperConnectionMixBranch({ kernelPath, phase, layerIdx }) {
  if (kernelPath == null) return false;
  if ((phase !== 'prefill' && phase !== 'decode') || !Number.isSafeInteger(layerIdx) || layerIdx < 0) {
    throw new Error('[HyperConnection] Mix-branch selection requires explicit phase and layer index.');
  }
  const steps = getLayerSteps(kernelPath, layerIdx, phase).filter(step => step.op === 'hc_mix_branch');
  if (steps.length === 0) return false;
  const config = steps.length === 1 ? getKernelPathStepConfig('hyper_connection_mix_branch', steps[0], phase) : null;
  const digest = config ? KERNEL_REF_CONTENT_DIGESTS[`${config.shaderFile}#${config.entryPoint}`] : null;
  const constants = steps[0]?.constants ?? {};
  const registered = config?.wgslOverrides ?? {};
  if (!config || config.variant !== 'f32' || !digest || steps[0].digest !== `sha256:${digest}`
    || Object.keys(constants).some(key => constants[key] !== registered[key])) {
    throw new Error('[HyperConnection] hc_mix_branch requires one registered refined F32 kernel with its compiled entry digest and model-bound geometry.');
  }
  return true;
}

/** @type {typeof import('./projection.js').resolveHyperConnectionUpMixBranch} */
export function resolveHyperConnectionUpMixBranch({ kernelPath, phase, layerIdx, role }) {
  if (role !== 'attn_hc' && role !== 'mlp_hc') {
    throw new Error('[HyperConnection] Up-mix fusion admits only explicit attention/MLP roles; the root mixer retains its declared program.');
  }
  if (kernelPath == null) return false;
  if ((phase !== 'prefill' && phase !== 'decode') || !Number.isSafeInteger(layerIdx) || layerIdx < 0) {
    throw new Error('[HyperConnection] Up-mix selection requires explicit phase and layer index.');
  }
  const steps = getLayerSteps(kernelPath, layerIdx, phase).filter(step => step.op === `${role}_up_mix_branch`);
  if (steps.length === 0) return false;
  const config = steps.length === 1 ? getKernelPathStepConfig('hyper_connection_up_mix_branch', steps[0], phase) : null;
  const digest = config ? KERNEL_REF_CONTENT_DIGESTS[`${config.shaderFile}#${config.entryPoint}`] : null;
  const constants = steps[0]?.constants ?? {};
  const registered = config?.wgslOverrides ?? {};
  const sourceVariant = getKernelPathMatmulVariant(`${role}_up`, phase, layerIdx, kernelPath);
  const source = sourceVariant ? getKernelConfigs().matmul?.[sourceVariant] : null;
  if (phase !== 'decode' || !config || config.variant !== 'q4_0_gemv32_f32'
    || !digest || steps[0].digest !== `sha256:${digest}`
    || Object.keys(constants).some(key => constants[key] !== registered[key])
    || sourceVariant !== 'q4_0_gemv' || source?.shaderFile !== 'matmul_q4_0.wgsl' || source.entryPoint !== 'gemv'
    || source.wgslOverrides?.WORKGROUP_SIZE !== 128 || source.wgslOverrides?.THREADS_PER_COL_GEMV !== 32
    || source.wgslOverrides?.COLS_PER_WG !== 4 || source.wgslOverrides?.QUANT_FORMAT !== 0
    || source.wgslOverrides?.BLOCK_VALUES !== 32 || source.wgslOverrides?.BLOCK_BYTES !== 18) {
    throw new Error('[HyperConnection] Up-mix requires one pinned Q4_0 gemv32 fusion in decode and its original q4_0_gemv role.');
  }
  return true;
}
