import { isWeightBuffer } from '../../../../gpu/weight-buffer.js';
import { isNativeGpuBuffer } from '../../../../gpu/buffer-identity.js';
import { runMatmul, recordMatmul } from '../../../../gpu/kernels/matmul.js';
import { runSiLU, recordSiLU } from '../../../../gpu/kernels/silu.js';
import { runRepeatChannels, recordRepeatChannels } from '../../../../gpu/kernels/repeat_channels.js';
import { runResidualAdd, recordResidualAdd } from '../../../../gpu/kernels/residual.js';
import { releaseOrTrack } from '../ops.js';

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
    const expandedGate = keep(await repeat(scalarGate,
      { inChannels: numTokens, height: 1, width: 1, repeats: hiddenSize, outputBuffer: output('expanded', hiddenSize) }));
    const gated = keep(await activate(down, { size: numTokens * hiddenSize,
      gate: expandedGate, inputActivation: 'identity', gateActivation: 'sigmoid', swigluLimit, outputBuffer: output('gated', hiddenSize) }));
    return await add(routed, gated, numTokens * hiddenSize, { outputBuffer: output('output', hiddenSize) });
  } finally {
    for (const buffer of owned) releaseOrTrack(recorder, buffer);
  }
}
