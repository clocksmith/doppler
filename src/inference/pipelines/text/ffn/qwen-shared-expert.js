import { isWeightBuffer } from '../../../../gpu/weight-buffer.js';
import { isNativeGpuBuffer } from '../../../../gpu/buffer-identity.js';
import { runMatmul } from '../../../../gpu/kernels/matmul.js';
import { runSiLU } from '../../../../gpu/kernels/silu.js';
import { runRepeatChannels } from '../../../../gpu/kernels/repeat_channels.js';
import { runResidualAdd } from '../../../../gpu/kernels/residual.js';
import { releaseBuffer } from '../../../../memory/buffer-pool.js';

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
  /** @type {GPUBuffer[]} */
  const owned = [];
  /** @param {import('../../../../gpu/tensor.js').Tensor} tensor */
  const keep = (tensor) => { owned.push(tensor.buffer); return tensor; };
  try {
    /** @type {import('../../../../gpu/kernels/matmul.js').MatmulOptions} */
    const projectionOptions = { transposeB: true, outputDtype: input.dtype, kernelPath };
    const gate = keep(await runMatmul(input, weights.sharedExpertGate,
      numTokens, intermediateSize, hiddenSize, { ...projectionOptions, role: 'ffn_gate' }));
    const up = keep(await runMatmul(input, weights.sharedExpertUp,
      numTokens, intermediateSize, hiddenSize, { ...projectionOptions, role: 'ffn_up' }));
    const activated = keep(await runSiLU(up, { size: numTokens * intermediateSize,
      gate, inputActivation: 'identity', gateActivation: 'silu', swigluLimit }));
    const down = keep(await runMatmul(activated, weights.sharedExpertDown,
      numTokens, hiddenSize, intermediateSize, { ...projectionOptions, role: 'ffn_down' }));
    const scalarGate = keep(await runMatmul(input, weights.sharedExpertRouter,
      numTokens, 1, hiddenSize, { ...projectionOptions, role: 'moe_shared_router' }));
    const expandedGate = keep(await runRepeatChannels(scalarGate,
      { inChannels: numTokens, height: 1, width: 1, repeats: hiddenSize }));
    const gated = keep(await runSiLU(down, { size: numTokens * hiddenSize,
      gate: expandedGate, inputActivation: 'identity', gateActivation: 'sigmoid', swigluLimit }));
    return await runResidualAdd(routed, gated, numTokens * hiddenSize);
  } finally {
    for (const buffer of owned) releaseBuffer(buffer);
  }
}
