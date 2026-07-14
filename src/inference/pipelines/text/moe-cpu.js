import { getDevice } from '../../../gpu/device.js';
import { getWeightDtype, isWeightBuffer } from '../../../gpu/weight-buffer.js';
import { acquireBuffer, releaseBuffer, readBuffer } from '../../../memory/buffer-pool.js';
import { createTensor } from '../../../gpu/tensor.js';
import { runMatmul, runSiLU, runGeLU } from '../../../gpu/kernel-selector.js';
import { createExpertExecutionPlan, combineExpertOutputs } from '../../moe-router.js';
import { log } from '../../../debug/index.js';
import { ensureExpertLoaded, gatherTokens } from './moe-helpers.js';
import { selectRuleValue } from '../../../rules/rule-registry.js';
import { runGptOssExpertCPU } from './moe-cpu-gptoss.js';

function requireMixtralExpertWeightDtype(weights, role, layerIdx, expertIdx) {
  const tensor = weights?.[role];
  if (isWeightBuffer(tensor)) {
    return getWeightDtype(tensor);
  }
  const declared = weights?.[`${role}Dtype`] ?? weights?.weightsDtype ?? tensor?.dtype;
  if (declared === 'f16' || declared === 'f32' || declared === 'q4k') {
    return declared;
  }
  throw new Error(
    `[MoE] Expert layer_${layerIdx}_expert_${expertIdx}.${role} requires explicit weight dtype.`
  );
}

export async function moeFeedForwardCPU(
  hiddenStates,
  numTokens,
  config,
  moeRouter,
  expertWeights,
  expertLoader,
  layerIdx
) {
  if (config.expertFormat !== 'mixtral' && config.expertFormat !== 'gpt-oss') {
    throw new Error(`[MoE] CPU fallback only supports mixtral/gpt-oss experts, got ${config.expertFormat ?? 'unknown'}.`);
  }
  const selections = moeRouter.route(hiddenStates, numTokens);
  const plan = createExpertExecutionPlan(selections, config.numExperts);
  const expertOutputs = new Map();

  for (const [expertIdx, data] of plan) {
    if (data.tokenIndices.length === 0) continue;

    await ensureExpertLoaded(layerIdx, expertIdx, expertWeights, expertLoader);
    const expertInput = gatherTokens(hiddenStates, data.tokenIndices, config.hiddenSize);

    const expertOutput = await runExpertCPU(
      layerIdx,
      expertIdx,
      expertInput,
      config,
      expertWeights
    );
    expertOutputs.set(expertIdx, expertOutput);
  }

  const combined = combineExpertOutputs(
    expertOutputs,
    selections,
    numTokens,
    config.hiddenSize
  );

  return combined;
}

async function runExpertCPU(layerIdx, expertIdx, input, config, expertWeights) {
  if (config.expertFormat === 'gpt-oss') {
    return runGptOssExpertCPU(layerIdx, expertIdx, input, config, expertWeights);
  }

  const key = `layer_${layerIdx}_expert_${expertIdx}`;
  const weights = expertWeights.get(key);

  if (!weights || !weights.gate || !weights.up || !weights.down) {
    log.warn('MoE', `Expert ${expertIdx} weights not available for layer ${layerIdx}`);
    return new Float32Array(input.length);
  }

  const device = getDevice();
  const { hiddenSize, intermediateSize, hiddenActivation, swigluLimit } = config;
  const kernelPath = config.kernelPath ?? null;
  const numTokens = input.length / hiddenSize;

  if (!device) {
    return new Float32Array(input.length);
  }

  const gateDtype = requireMixtralExpertWeightDtype(weights, 'gate', layerIdx, expertIdx);
  const upDtype = requireMixtralExpertWeightDtype(weights, 'up', layerIdx, expertIdx);
  const downDtype = requireMixtralExpertWeightDtype(weights, 'down', layerIdx, expertIdx);

  const inputBuffer = acquireBuffer(input.byteLength, undefined, 'expert_input');
  device.queue.writeBuffer(inputBuffer, 0, input);
  const inputTensor = createTensor(inputBuffer, 'f32', [numTokens, hiddenSize], 'expert_input');
  let gateOutput = null;
  let upOutput = null;
  let activatedOutput = null;
  let output = null;
  try {
    gateOutput = await runMatmul(inputTensor, weights.gate, numTokens, intermediateSize, hiddenSize, {
      transposeB: 'auto',
      role: 'moe_gate',
      kernelPath,
      bDtype: gateDtype,
      outputDtype: 'f32',
    });

    upOutput = await runMatmul(inputTensor, weights.up, numTokens, intermediateSize, hiddenSize, {
      transposeB: 'auto',
      role: 'moe_up',
      kernelPath,
      bDtype: upDtype,
      outputDtype: 'f32',
    });

    const activationFn = {
      gelu: runGeLU,
      silu: runSiLU,
    }[selectRuleValue('inference', 'ffn', 'activationOp', { hiddenActivation })];
    activatedOutput = await activationFn(upOutput, {
      size: numTokens * intermediateSize,
      gate: gateOutput,
      inputActivation: 'identity',
      swigluLimit,
    });

    output = await runMatmul(activatedOutput, weights.down, numTokens, hiddenSize, intermediateSize, {
      transposeB: 'auto',
      role: 'moe_down',
      kernelPath,
      bDtype: downDtype,
      outputDtype: 'f32',
    });

    const outputData = await readBuffer(output.buffer, input.byteLength);
    return new Float32Array(outputData);
  } finally {
    releaseBuffer(inputBuffer);
    if (gateOutput) releaseBuffer(gateOutput.buffer);
    if (upOutput) releaseBuffer(upOutput.buffer);
    if (activatedOutput) releaseBuffer(activatedOutput.buffer);
    if (output) releaseBuffer(output.buffer);
  }
}
