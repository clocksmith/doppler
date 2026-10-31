import { isWeightBuffer } from '../../../../gpu/weight-buffer.js';
import { isNativeGpuBuffer } from '../../../../gpu/buffer-identity.js';
import { runMatmul, recordMatmul } from '../../../../gpu/kernels/matmul.js';
import { runSiLU, recordSiLU } from '../../../../gpu/kernels/silu.js';
import { runRepeatChannels, recordRepeatChannels } from '../../../../gpu/kernels/repeat_channels.js';
import { runResidualAdd, recordResidualAdd } from '../../../../gpu/kernels/residual.js';
import { releaseOrTrack } from '../ops.js';
import { getKernelConfigs } from '../../../../config/kernel-registry-contract.js';
import { getLayerSteps, getKernelPathStepConfig } from '../../../../config/kernel-path-loader.js';
import { runSigmoidGatedAdd, recordSigmoidGatedAdd } from '../../../../gpu/kernels/sigmoid-gated-add.js';

const combinePlans = new WeakMap();

/** @type {typeof import('./qwen-shared-expert.js').resolveQwenSharedSlot} */
export function resolveQwenSharedSlot(options) {
  if (options.kernelPath == null) return false;
  if ((options.phase !== 'decode' && options.phase !== 'prefill')
    || !Number.isSafeInteger(options.layerIdx) || options.layerIdx < 0) {
    throw new Error('[MoE] Shared slot lookup requires explicit phase and layer index.');
  }
  const steps = getLayerSteps(options.kernelPath, options.layerIdx, options.phase);
  const declared = steps.filter(step => step.op === 'moe_shared_slot_gu' || step.op === 'moe_shared_slot_down');
  if (declared.length === 0) return false;
  for (const operation of ['moe_shared_slot_gu', 'moe_shared_slot_down']) {
    const matches = declared.filter(step => step.op === operation);
    if (matches.length !== 1 || !getKernelPathStepConfig(operation, matches[0], options.phase)
      || (matches[0].constants && Object.keys(matches[0].constants).length)) {
      throw new Error('[MoE] Shared slot requires one registered GU and down step with model-bound geometry.');
    }
  }
  if (steps.some(step => step.op === 'moe_shared_combine')) {
    throw new Error('[MoE] Shared slot already combines routed/shared output; a second combine is invalid.');
  }
  return true;
}

/** @type {typeof import('./qwen-shared-expert.js').resolveQwenSharedExpertCombine} */
export function resolveQwenSharedExpertCombine(options) {
  if (options.kernelPath == null) return false;
  if ((options.phase !== 'decode' && options.phase !== 'prefill')
    || typeof options.layerIdx !== 'number' || !Number.isSafeInteger(options.layerIdx) || options.layerIdx < 0) {
    throw new Error('[MoE] Shared combine lookup requires an explicit phase and layer index.');
  }
  let registries = combinePlans.get(options.kernelPath);
  if (!registries) { registries = new WeakMap(); combinePlans.set(options.kernelPath, registries); }
  const configs = getKernelConfigs();
  let plans = registries.get(configs);
  if (!plans) { plans = new Map(); registries.set(configs, plans); }
  const key = `${options.phase}:${options.layerIdx}`;
  if (plans.has(key)) return plans.get(key);
  const steps = getLayerSteps(options.kernelPath, options.layerIdx, options.phase)
    .filter(step => step.op === 'moe_shared_combine');
  if (steps.length === 0) { plans.set(key, false); return false; }
  if (steps.length !== 1 || !getKernelPathStepConfig('sigmoid_gated_add', steps[0], options.phase)) {
    throw new Error('[MoE] Shared combine requires exactly one registered sigmoid_gated_add kernel step.');
  }
  if (steps[0].constants && Object.keys(steps[0].constants).length !== 0) {
    throw new Error('[MoE] Shared combine geometry comes from resolved model dimensions, not step overrides.');
  }
  plans.set(key, true);
  return true;
}

/** @type {typeof import('./qwen-shared-expert.js').validateQwenSharedExpert} */
export function validateQwenSharedExpert(input, routed, weights, options) {
  for (const key of /** @type {const} */ (['numTokens', 'hiddenSize', 'intermediateSize'])) {
    if (!Number.isSafeInteger(options?.[key]) || options[key] < 1) {
      throw new Error(`[MoE] Qwen shared expert requires positive ${key}.`);
    }
  }
  if (input?.dtype !== 'f32' || routed?.dtype !== 'f32') {
    throw new Error('[MoE] Qwen shared expert currently requires explicit f32 activations.');
  }
  const shape = [options.numTokens, options.hiddenSize];
  for (const tensor of [input, routed]) {
    if (tensor.shape.length !== 2 || tensor.shape.some((n, i) => n !== shape[i])) {
      throw new Error('[MoE] Qwen shared expert input and routed output must match token/hidden geometry.');
    }
  }
  for (const key of /** @type {const} */ (['sharedExpertGate', 'sharedExpertUp', 'sharedExpertDown', 'sharedExpertRouter'])) {
    if (!weights?.[key]) throw new Error(`[MoE] Qwen shared expert missing ${key}.`);
    if (!isWeightBuffer(weights[key]) && !isNativeGpuBuffer(weights[key])) {
      throw new Error(`[MoE] Qwen shared expert ${key} must be a materialized GPU weight.`);
    }
  }
  if (options.swigluLimit === undefined) throw new Error('[MoE] Qwen shared expert requires explicit swigluLimit.');
}

/** @type {typeof import('./qwen-shared-expert.js').runQwenSharedExpert} */
export async function runQwenSharedExpert(input, routed, weights, options) {
  validateQwenSharedExpert(input, routed, weights, options);
  const fusedCombine = resolveQwenSharedExpertCombine(options);
  const { numTokens, hiddenSize, intermediateSize, kernelPath, swigluLimit } = options;
  const recorder = options.recorder ?? null;
  const workspace = recorder?.decodeWorkspace && numTokens === 1 ? recorder.decodeWorkspace : null;
  if (workspace && (typeof options.layerIdx !== 'number' || !Number.isInteger(options.layerIdx) || options.layerIdx < 0)) {
    throw new Error('[MoE] Prepared shared execution requires an explicit layerIdx.');
  }
  /** @param {string} role @param {number} count */
  const output = (role, count) => workspace?.getBuffer(`moe:${options.layerIdx}:shared-${role}`, count * 4);
  const matmul = recorder ? recordMatmul.bind(null, recorder) : runMatmul;
  const activate = recorder ? recordSiLU.bind(null, recorder) : runSiLU;
  const repeat = recorder ? recordRepeatChannels.bind(null, recorder) : runRepeatChannels;
  const add = recorder ? recordResidualAdd.bind(null, recorder) : runResidualAdd;
  /** @type {GPUBuffer[]} */
  const owned = [];
  /** @param {import('../../../../gpu/tensor.js').Tensor} tensor */
  const keep = (tensor) => { owned.push(tensor.buffer); return tensor; };
  try {
    /** @type {import('../../../../gpu/kernels/matmul.js').MatmulOptions} */
    const projectionOptions = { transposeB: true, outputDtype: input.dtype, kernelPath, layerIdx: options.layerIdx };
    const gate = keep(await matmul(input, weights.sharedExpertGate,
      numTokens, intermediateSize, hiddenSize, { ...projectionOptions, role: 'ffn_gate', outputBuffer: output('gate', intermediateSize) }));
    const up = keep(await matmul(input, weights.sharedExpertUp,
      numTokens, intermediateSize, hiddenSize, { ...projectionOptions, role: 'ffn_up', outputBuffer: output('up', intermediateSize) }));
    const activated = keep(await activate(up, { size: numTokens * intermediateSize,
      gate, inputActivation: 'identity', gateActivation: 'silu', swigluLimit, outputBuffer: output('activated', intermediateSize) }));
    const down = keep(await matmul(activated, weights.sharedExpertDown,
      numTokens, hiddenSize, intermediateSize, { ...projectionOptions, role: 'ffn_down', outputBuffer: output('down', hiddenSize) }));
    const scalarGate = keep(await matmul(input, weights.sharedExpertRouter,
      numTokens, 1, hiddenSize, { ...projectionOptions, role: 'moe_shared_router', outputBuffer: output('router', 1) }));
    if (fusedCombine) {
      const combine = recorder ? recordSigmoidGatedAdd.bind(null, recorder) : runSigmoidGatedAdd;
      return await combine(down, scalarGate, routed, { numTokens, hiddenSize, swigluLimit,
        outputBuffer: output('output', hiddenSize) });
    }
    const expandedGate = keep(await repeat(scalarGate,
      { inChannels: numTokens, height: 1, width: 1, repeats: hiddenSize, outputBuffer: output('expanded', hiddenSize) }));
    const gated = keep(await activate(down, { size: numTokens * hiddenSize,
      gate: expandedGate, inputActivation: 'identity', gateActivation: 'sigmoid', swigluLimit, outputBuffer: output('gated', hiddenSize) }));
    return await add(routed, gated, numTokens * hiddenSize, { outputBuffer: output('output', hiddenSize) });
  } finally {
    for (const buffer of owned) releaseOrTrack(recorder, buffer);
  }
}
