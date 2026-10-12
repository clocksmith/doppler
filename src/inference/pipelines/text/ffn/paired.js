import { doMatmul, doSiLU, releaseOrTrack } from '../ops.js';
import { getWeightBuffer } from '../weights.js';
import { resolveLayerIntermediateSize } from '../config.js';
import { getLayerSteps, getKernelPathMatmulVariant } from '../../../../config/kernel-path-loader.js';
import { getKernelConfigs } from '../../../../config/kernel-registry-contract.js';
import { isGpuBufferInstance, isWeightBuffer } from '../../../../gpu/weight-buffer.js';
import { tensorBytes } from '../../../../gpu/tensor.js';
import { runProbes } from '../probes.js';

/** @type {typeof import('./paired.js').validatePairedFFN} */
export function validatePairedFFN(layerIdx, input, numTokens, context, weights) {
  const { config, kernelPath, phase } = context;
  const intermediateSize = resolveLayerIntermediateSize(config, layerIdx);
  if (input?.dtype !== 'f32' || input.precision !== 'two-f32'
    || input.components !== 2 || input.elementStrideBytes !== 8
    || !Number.isSafeInteger(layerIdx) || layerIdx < 0
    || !Number.isSafeInteger(numTokens) || numTokens <= 0
    || input.shape?.length !== 2 || input.shape[0] !== numTokens
    || input.shape[1] !== config.hiddenSize
    || !Number.isSafeInteger(input.buffer?.size)
    || input.buffer.size < tensorBytes(input.shape, input.dtype, input)) {
    throw new Error('[PairedFFN] Explicit two-F32 input with logical [tokens, hidden] shape is required.');
  }
  if (config.hiddenActivation !== 'silu' || config.gatedActivation === false
    || config.swigluLimit !== null || config.useDoubleWideMlp === true
    || context.lora || weights?.gateUp || !weights?.gate || !weights?.up || !weights?.down
    || ('gateBias' in weights && weights.gateBias) || ('upBias' in weights && weights.upBias)
    || ('downBias' in weights && weights.downBias)) {
    throw new Error('[PairedFFN] Requires separate gate/up/down, unclipped gated SiLU and no undeclared adapter or bias.');
  }
  if (!kernelPath || (phase !== 'prefill' && phase !== 'decode')) {
    throw new Error('[PairedFFN] Explicit phase and split kernel path are required.');
  }
  const steps = getLayerSteps(kernelPath, layerIdx, phase);
  const registry = getKernelConfigs();
  const projectionIndices = [];
  for (const [role, alias] of [['ffn_gate', 'gate_proj'], ['ffn_up', 'up_proj'], ['ffn_down', 'down_proj']]) {
    const matches = steps.filter(step => step.op === role || step.op === alias);
    if (matches.length !== 1) {
      throw new Error(`[PairedFFN] Exactly one declared ${role} projection is required.`);
    }
    projectionIndices.push(steps.indexOf(matches[0]));
    for (const dtype of [matches[0].precision?.inputDtype, matches[0].precision?.outputDtype]) {
      if (dtype != null && dtype !== 'f32') {
        throw new Error(`[PairedFFN] ${role} component dtype must remain F32.`);
      }
    }
    const variant = getKernelPathMatmulVariant(role, phase, layerIdx, kernelPath);
    const declared = variant === null ? null : registry.matmul?.[variant];
    if (declared?.variantMetadata?.inputPrecision !== 'two-f32'
      || declared?.variantMetadata?.outputPrecision !== 'two-f32') {
      throw new Error(`[PairedFFN] ${role} must preserve declared two-F32 precision.`);
    }
  }
  const activation = steps.filter(step => step.op === 'activation' || step.op === 'ffn');
  const silu = registry.paired_pointwise?.silu;
  if (activation.length !== 1 || activation[0].kernel !== silu?.shaderFile
    || activation[0].entry !== silu?.entryPoint
    || steps.some(step => step.op === 'ffn_gate_up')) {
    throw new Error('[PairedFFN] Declare paired SiLU between separate projections; scalar fused FFN is unsupported.');
  }
  const activationIndex = steps.indexOf(activation[0]);
  if (activation[0].constants?.HAS_GATE === false || activation[0].constants?.INPUT_USE_IDENTITY === false) {
    throw new Error('[PairedFFN] Activation must declare SiLU(gate) multiplied by the identity up projection.');
  }
  if (!(projectionIndices[0] < projectionIndices[1] && projectionIndices[1] < activationIndex
    && activationIndex < projectionIndices[2])) {
    throw new Error('[PairedFFN] Declared order must be gate, up, paired SiLU, down.');
  }
  return { hiddenSize: config.hiddenSize, intermediateSize };
}

/** @type {typeof import('./paired.js').runPairedFFNGPU} */
export async function runPairedFFNGPU(layerIdx, input, numTokens, context, weights) {
  const { hiddenSize, intermediateSize } = validatePairedFFN(layerIdx, input, numTokens, context, weights);
  const recorder = context.recorder;
  const owned = new Set();
  /** @param {'gate' | 'up' | 'down'} name */
  const bindWeight = (name) => {
    const source = weights[name];
    if (!source) throw new Error(`[PairedFFN] Missing ${name} weight.`);
    const weight = getWeightBuffer(source, `ffn_${name}`);
    if (!isGpuBufferInstance(weights[name]) && !isWeightBuffer(weights[name])) {
      owned.add(isWeightBuffer(weight) ? weight.buffer : weight);
    }
    return weight;
  };
  /**
   * @param {import('../../../../gpu/tensor.js').Tensor} tensor
   * @param {GPUBuffer | import('../../../../gpu/weight-buffer.js').WeightBuffer} weight
   * @param {number} N
   * @param {number} K
   * @param {string} role
   */
  const projection = async (tensor, weight, N, K, role) => {
    const result = await doMatmul(tensor, weight, numTokens, N, K, {
      transposeB: 'auto', outputDtype: 'f32', outputPrecision: 'two-f32',
      role, label: `L${layerIdx}.${role}`, layerIdx,
      kernelPath: context.kernelPath, phaseOverride: context.phase,
      executionPolicies: context.executionPolicies ?? null,
    }, recorder);
    owned.add(result.buffer);
    if (result.precision !== 'two-f32' || result.components !== 2 || result.elementStrideBytes !== 8
      || result.shape.length !== 2 || result.shape[0] !== numTokens || result.shape[1] !== N) {
      throw new Error(`[PairedFFN] ${role} lost its logical shape or paired representation.`);
    }
    return result;
  };
  /** @param {import('../../../../config/schema/index.js').ProbeStage} stage @param {import('../../../../gpu/tensor.js').Tensor} tensor */
  const probe = (stage, tensor) => runProbes(stage, tensor.buffer, {
    layerIdx, numTokens, hiddenSize: intermediateSize,
    probes: context.debugProbes, recorder, operatorDiagnostics: context.operatorDiagnostics,
    dtype: tensor.dtype, precision: tensor.precision,
    components: tensor.components, elementStrideBytes: tensor.elementStrideBytes,
  });
  try {
    const gateWeight = bindWeight('gate');
    const upWeight = bindWeight('up');
    const downWeight = bindWeight('down');
    const gate = await projection(input, gateWeight, intermediateSize, hiddenSize, 'ffn_gate');
    const up = await projection(input, upWeight, intermediateSize, hiddenSize, 'ffn_up');
    await probe('ffn_gate', gate);
    await probe('ffn_up', up);
    const activation = await doSiLU(up, {
      size: numTokens * intermediateSize, gate, gateActivation: 'silu', inputActivation: 'identity',
      swigluLimit: context.config.swigluLimit, label: `L${layerIdx}.ffn_activation`, layerIdx,
    }, recorder);
    owned.add(activation.buffer);
    if (activation.precision !== 'two-f32' || activation.components !== 2 || activation.elementStrideBytes !== 8
      || activation.shape.length !== 2 || activation.shape[0] !== numTokens || activation.shape[1] !== intermediateSize) {
      throw new Error('[PairedFFN] SiLU lost its logical shape or paired representation.');
    }
    await probe('ffn_act', activation);
    const capture = context.trainingCapture;
    if (capture?.layerIdx === layerIdx && capture.stage === 'ffn_act') {
      await capture.capture({ layerIdx, stage: 'ffn_act', tensor: activation, numTokens, hiddenSize: intermediateSize, recorder: recorder ?? null });
    }
    const output = await projection(activation, downWeight, hiddenSize, intermediateSize, 'ffn_down');
    owned.delete(output.buffer);
    return output;
  } finally {
    for (const buffer of owned) releaseOrTrack(recorder, buffer, context.decodeBuffers);
  }
}
