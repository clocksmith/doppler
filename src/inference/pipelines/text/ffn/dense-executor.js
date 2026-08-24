import {
  doBiasAdd, doMatmul, doSiLU, doGeLU, doSiLURowSplit, doMatmulRMSNormFused,
  releaseOrTrack
} from '../ops.js';
import { createTensor } from '../../../../gpu/tensor.js';
import {
  getWeightDtype,
  isGpuBufferInstance,
  isWeightBuffer,
  resolveWeightBufferMaterialization,
} from '../../../../gpu/weight-buffer.js';
import { getDevice, getKernelCapabilities } from '../../../../gpu/device.js';
import { getRuntimeConfig } from '../../../../config/runtime.js';
import { acquireBuffer, releaseBuffer, readBuffer } from '../../../../memory/buffer-pool.js';
import {
  runFusedFFN,
  recordFusedFFN,
  runFusedFFNFromRMSNormStats,
  recordFusedFFNFromRMSNormStats,
  castF16ToF32,
  castF32ToF16,
  recordCastF16ToF32,
  recordCastF32ToF16,
  isFusedQ4KDisabled
} from '../../../../gpu/kernel-selector.js';
import { log, trace, isTraceEnabled } from '../../../../debug/index.js';
import { isKernelDebugEnabled, dumpTokenVector, decodeReadback, getLogitsHealth, shouldDebugLayerOutput } from '../debug-utils.js';
import { applyLoRA } from '../lora-apply.js';
import { getLoRAModule } from '../lora.js';
import { getWeightBuffer, getNormWeightBuffer, getVectorTensor } from '../weights.js';
import { runProbes } from '../probes.js';
import { selectRuleValue } from '../../../../rules/rule-registry.js';
import {
  getKernelPathMatmulConstants,
  getKernelPathMatmulPrecision,
  getKernelPathMatmulVariant,
} from '../../../../config/kernel-path-loader.js';
import { resolveLayerIntermediateSize } from '../config.js';
import { assertImplicitDtypeTransitionAllowed } from '../dtype-contract.js';
import { canFuseSplitPrefillF16GateUpPath, hasQ4KMaterialization, resolveDenseFFNFusedPathDtypes, resolveDenseFFNMatmulStepDtype, resolveFusedGateUpPipelineConstants, resolveFusedGateUpVariant, resolveFusedGateUpWeights, resolveGateUpPathMode } from './dense-plan.js';

export const ACTIVATION_FN_MAP = {
  gelu: doGeLU,
  silu: doSiLU,
};

export function resolveActivationOp(hiddenActivation) {
  return selectRuleValue('inference', 'ffn', 'activationOp', { hiddenActivation });
}

export function resolveMatmulStepDtype(role, phase, layerIdx, kernelPath, fallback, field, ffnStepPrecision = null) {
  return resolveDenseFFNMatmulStepDtype({
    role,
    phase,
    layerIdx,
    kernelPath,
    fallback,
    field,
    ffnStepPrecision,
  });
}

export function canUseNativeF16FusedGateUp(options = {}) {
  if (options.inputDtype !== 'f16' || options.hasF16 !== true) {
    return false;
  }
  return options.gateDtype === 'f16' || options.gateDtype === 'q4k';
}

export async function coerceTensorDtype(tensor, targetDtype, recorder, options = {}) {
  if (!targetDtype || tensor.dtype === targetDtype) {
    return tensor;
  }
  assertImplicitDtypeTransitionAllowed({
    executionPolicies: options.executionPolicies ?? null,
    fromDtype: tensor.dtype,
    toDtype: targetDtype,
    op: options.op ?? 'ffn',
    detail: 'The execution graph must declare this cast explicitly.',
    transitionDeclaredBy: options.transitionDeclaredBy ?? null,
  });
  if (tensor.dtype === 'f32' && targetDtype === 'f16') {
    return recorder ? await recordCastF32ToF16(recorder, tensor) : await castF32ToF16(tensor);
  }
  if (tensor.dtype === 'f16' && targetDtype === 'f32') {
    return recorder ? await recordCastF16ToF32(recorder, tensor) : await castF16ToF32(tensor);
  }
  throw new Error(`Unsupported FFN matmul dtype coercion: ${tensor.dtype} -> ${targetDtype}`);
}

export function requireFusedWeightDtype(dtype, label) {
  if (dtype !== 'bf16' && dtype !== 'f16' && dtype !== 'f32' && dtype !== 'q4k') {
    throw new Error(`[FFN] ${label} dtype metadata is required for fused gate/up planning.`);
  }
  return dtype;
}

export async function dispatchActivation(hiddenActivation, input, options, recorder) {
  const op = resolveActivationOp(hiddenActivation);
  const fn = ACTIVATION_FN_MAP[op];
  if (!fn) {
    throw new Error(`Unsupported FFN activation op "${op}".`);
  }
  return fn(input, options, recorder);
}

export async function dispatchFusedGateUp({
  inputTensor,
  gateWeight,
  upWeight,
  gateDtype,
  hiddenSize,
  intermediateSize,
  numTokens,
  hiddenActivation,
  swigluLimit,
  recorder,
  executionPolicies = null,
  normStats = null,
  pipelineConstants = null,
  variant = null,
}) {
  const useNativeF16Fused = canUseNativeF16FusedGateUp({
    inputDtype: inputTensor.dtype,
    gateDtype,
    hasF16: getKernelCapabilities().hasF16,
  });
  let fusedInput = inputTensor;
  if (!useNativeF16Fused && inputTensor.dtype === 'f16') {
    assertImplicitDtypeTransitionAllowed({
      executionPolicies,
      fromDtype: 'f16',
      toDtype: 'f32',
      op: 'ffn_gate_up',
      detail: 'The fused FFN kernel would widen activations internally.',
    });
    fusedInput = recorder
      ? await recordCastF16ToF32(recorder, inputTensor)
      : await castF16ToF32(inputTensor);
  }

  if (recorder && fusedInput !== inputTensor) {
    recorder.trackTemporaryBuffer(fusedInput.buffer);
  }

  const activation = resolveActivationOp(hiddenActivation);
  if (normStats) {
    if (variant != null) {
      throw new Error(`[FFN] explicit fused gate/up variant "${variant}" is not supported by the norm-stats path.`);
    }
    if (inputTensor.dtype !== 'f32' || gateDtype !== 'q4k' || numTokens !== 1) {
      throw new Error(
        `Fused post-attention norm gate/up requires f32 Q4_K decode input; ` +
        `got input=${inputTensor.dtype}, gate=${gateDtype}, tokens=${numTokens}.`
      );
    }
    const fusedNormedOutput = recorder
      ? await recordFusedFFNFromRMSNormStats(
        recorder,
        inputTensor,
        normStats.invRmsBuffer,
        normStats.normWeight,
        gateWeight,
        upWeight,
        hiddenSize,
        intermediateSize,
        {
          batchSize: numTokens,
          activation,
          swigluLimit,
          rmsNormWeightOffset: normStats.rmsNormWeightOffset === true,
          pipelineConstants,
        }
      )
      : await runFusedFFNFromRMSNormStats(
        inputTensor,
        normStats.invRmsBuffer,
        normStats.normWeight,
        gateWeight,
        upWeight,
        hiddenSize,
        intermediateSize,
        {
          batchSize: numTokens,
          activation,
          swigluLimit,
          rmsNormWeightOffset: normStats.rmsNormWeightOffset === true,
          pipelineConstants,
        }
      );
    return fusedNormedOutput;
  }
  const fusedOutput = recorder
    ? await recordFusedFFN(
      recorder, fusedInput, gateWeight, upWeight,
      hiddenSize, intermediateSize,
      { batchSize: numTokens, activation, swigluLimit, pipelineConstants, variant }
    )
    : await runFusedFFN(
      fusedInput, gateWeight, upWeight,
      hiddenSize, intermediateSize,
      { batchSize: numTokens, activation, swigluLimit, pipelineConstants, variant }
    );

  if (!recorder && fusedInput !== inputTensor) {
    releaseBuffer(fusedInput.buffer);
  }

  return fusedOutput;
}

export async function applyDenseProjectionBias(
  tensor,
  biasWeight,
  label,
  numTokens,
  dim,
  context,
  layerIdx
) {
  if (!biasWeight) return tensor;
  const { tensor: biasTensor, owned } = getVectorTensor(
    biasWeight,
    label,
    dim,
    context.weightConfig,
    context.debugFlags
  );
  try {
    return await doBiasAdd(tensor, biasTensor, numTokens, dim, {
      label,
      layerIdx,
      executionPolicies: context.executionPolicies ?? null,
    }, context.recorder);
  } finally {
    if (owned) releaseOrTrack(context.recorder, biasTensor.buffer, context.decodeBuffers);
  }
}

export async function runDenseUngatedFFNGPU(layerIdx, inputTensor, numTokens, context, layerWeights) {
  const { config, recorder } = context;
  const { hiddenSize, hiddenActivation } = config;
  const intermediateSize = resolveLayerIntermediateSize(config, layerIdx);
  if (layerWeights.gate || layerWeights.gateUp) {
    throw new Error(`Layer ${layerIdx} declares an ungated FFN but contains gate weights.`);
  }
  if (!layerWeights.up || !layerWeights.down) {
    throw new Error(`Layer ${layerIdx} ungated FFN requires up_proj and down_proj weights.`);
  }

  const kernelPath = context.kernelPath ?? null;
  const phase = context.phase ?? (numTokens === 1 ? 'decode' : 'prefill');
  const ffnStepPrecision = context.ffnStepPrecision ?? null;
  const upInputDtype = resolveMatmulStepDtype(
    'ffn_up', phase, layerIdx, kernelPath, inputTensor.dtype, 'inputDtype', ffnStepPrecision
  );
  const upOutputDtype = resolveMatmulStepDtype(
    'ffn_up', phase, layerIdx, kernelPath, inputTensor.dtype, 'outputDtype', ffnStepPrecision
  );
  const downInputDtype = resolveMatmulStepDtype(
    'ffn_down', phase, layerIdx, kernelPath, upOutputDtype, 'inputDtype', ffnStepPrecision
  );
  const downOutputDtype = resolveMatmulStepDtype(
    'ffn_down', phase, layerIdx, kernelPath, inputTensor.dtype, 'outputDtype', ffnStepPrecision
  );

  let upInput = inputTensor;
  let upInputOwned = false;
  if (upInputDtype && upInputDtype !== inputTensor.dtype) {
    upInput = await coerceTensorDtype(inputTensor, upInputDtype, recorder, {
      executionPolicies: context.executionPolicies ?? null,
      op: 'ffn_up_input',
      transitionDeclaredBy: 'step_precision',
    });
    upInputOwned = upInput !== inputTensor;
  }

  const upWeight = getWeightBuffer(layerWeights.up, 'ffn_up');
  let upOutput = await doMatmul(
    upInput,
    upWeight,
    numTokens,
    intermediateSize,
    hiddenSize,
    {
      transposeB: 'auto',
      label: `L${layerIdx}.ffn_up`,
      layerIdx,
      kernelPath,
      outputDtype: upOutputDtype,
      role: 'ffn_up',
      executionPolicies: context.executionPolicies ?? null,
    },
    recorder
  );
  const loraUp = getLoRAModule(context.lora ?? null, layerIdx, 'up_proj');
  if (loraUp) {
    const combined = await applyLoRA(
      upInput,
      upOutput,
      loraUp,
      { M: numTokens, N: intermediateSize, K: hiddenSize },
      getWeightBuffer,
      recorder,
      { kernelPath }
    );
    if (combined.buffer !== upOutput.buffer) {
      releaseOrTrack(recorder, upOutput.buffer, context.decodeBuffers);
      upOutput = combined;
    }
  }
  upOutput = await applyDenseProjectionBias(
    upOutput,
    layerWeights.upBias,
    `L${layerIdx}.ffn_up_bias`,
    numTokens,
    intermediateSize,
    context,
    layerIdx
  );
  const activated = await dispatchActivation(hiddenActivation, upOutput, {
    size: numTokens * intermediateSize,
    gate: null,
    label: `L${layerIdx}.ffn_activation`,
    layerIdx,
  }, recorder);

  let downInput = activated;
  let downInputOwned = false;
  if (downInputDtype && activated.dtype !== downInputDtype) {
    downInput = await coerceTensorDtype(activated, downInputDtype, recorder, {
      executionPolicies: context.executionPolicies ?? null,
      op: 'ffn_down_input',
      transitionDeclaredBy: 'step_precision',
    });
    downInputOwned = downInput !== activated;
  }
  const downWeight = getWeightBuffer(layerWeights.down, 'ffn_down');
  let output = await doMatmul(
    downInput,
    downWeight,
    numTokens,
    hiddenSize,
    intermediateSize,
    {
      transposeB: 'auto',
      label: `L${layerIdx}.ffn_down`,
      layerIdx,
      kernelPath,
      outputDtype: downOutputDtype,
      role: 'ffn_down',
      executionPolicies: context.executionPolicies ?? null,
    },
    recorder
  );
  const loraDown = getLoRAModule(context.lora ?? null, layerIdx, 'down_proj');
  if (loraDown) {
    const combined = await applyLoRA(
      downInput,
      output,
      loraDown,
      { M: numTokens, N: hiddenSize, K: intermediateSize },
      getWeightBuffer,
      recorder,
      { kernelPath }
    );
    if (combined.buffer !== output.buffer) {
      releaseOrTrack(recorder, output.buffer, context.decodeBuffers);
      output = combined;
    }
  }
  output = await applyDenseProjectionBias(
    output,
    layerWeights.downBias,
    `L${layerIdx}.ffn_down_bias`,
    numTokens,
    hiddenSize,
    context,
    layerIdx
  );

  if (!isGpuBufferInstance(layerWeights.up) && !isWeightBuffer(layerWeights.up)) {
    releaseOrTrack(recorder, isWeightBuffer(upWeight) ? upWeight.buffer : upWeight, context.decodeBuffers);
  }
  if (!isGpuBufferInstance(layerWeights.down) && !isWeightBuffer(layerWeights.down)) {
    releaseOrTrack(recorder, isWeightBuffer(downWeight) ? downWeight.buffer : downWeight, context.decodeBuffers);
  }
  if (upInputOwned) releaseOrTrack(recorder, upInput.buffer, context.decodeBuffers);
  if (downInputOwned) releaseOrTrack(recorder, downInput.buffer, context.decodeBuffers);
  releaseOrTrack(recorder, upOutput.buffer, context.decodeBuffers);
  releaseOrTrack(recorder, activated.buffer, context.decodeBuffers);
  return output;
}

export async function runDenseFFNWithFusedPostNormGPU(
  layerIdx,
  inputTensor,
  numTokens,
  context,
  layerWeights,
  residualTensor,
  eps,
  transposeB,
  outputBuffer
) {
  const device = getDevice();
  if (!device) throw new Error('No GPU device');

  const { config, weightConfig, debugFlags, recorder } = context;
  const { hiddenSize, hiddenActivation, swigluLimit, useDoubleWideMlp } = config;
  const intermediateSize = resolveLayerIntermediateSize(config, layerIdx);
  const lora = context.lora || null;
  const ffnStepPrecision = context.ffnStepPrecision ?? null;
  const kernelPath = context.kernelPath ?? null;
  const phase = context.phase ?? (numTokens === 1 ? 'decode' : 'prefill');
  const gateUpPathMode = resolveGateUpPathMode({ kernelPath, phase, layerIdx });

  if (!layerWeights.down || !layerWeights.postFeedforwardNorm) {
    throw new Error('Missing down or norm weights');
  }

  const downWeight = getWeightBuffer(layerWeights.down, 'ffn_down');
  const normWeightBuf = getNormWeightBuffer(layerWeights.postFeedforwardNorm, 'post_feedforward_norm', weightConfig, debugFlags);

  
  let activatedOutput;
  const useF16 = inputTensor.dtype === 'f16';
  const defaultMatmulOutputDtype = selectRuleValue('shared', 'dtype', 'f16OrFallbackByFlag', {
    useF16,
    fallback: inputTensor.dtype,
  });
  const matmulOutputDtype = resolveMatmulStepDtype(
    'ffn_gate_up',
    phase,
    layerIdx,
    kernelPath,
    defaultMatmulOutputDtype,
    'outputDtype',
    ffnStepPrecision
  );
  const {
    fusedGateUpInputDtype,
  } = resolveDenseFFNFusedPathDtypes({
    phase,
    layerIdx,
    kernelPath,
    ffnStepPrecision,
    fallbackInputDtype: inputTensor.dtype,
    fallbackOutputDtype: matmulOutputDtype,
  });

  if (layerWeights.gateUp) {
    const gateUpWeight = getWeightBuffer(layerWeights.gateUp, 'ffn_gate_up');
    let gateUpInput = inputTensor;
    let gateUpInputOwned = false;
    if (fusedGateUpInputDtype && fusedGateUpInputDtype !== inputTensor.dtype) {
      gateUpInput = await coerceTensorDtype(inputTensor, fusedGateUpInputDtype, recorder, {
        executionPolicies: context.executionPolicies ?? null,
        op: 'ffn_gate_up_input',
        transitionDeclaredBy: 'step_precision',
      });
      gateUpInputOwned = gateUpInput !== inputTensor;
    }
    let gateUpOutput = await doMatmul(
      gateUpInput, gateUpWeight,
      numTokens, intermediateSize * 2, hiddenSize,
        {
          transposeB: 'auto',
          outputDtype: matmulOutputDtype,
          role: 'ffn_gate_up',
          label: `L${layerIdx}.ffn_gate_up`,
          layerIdx,
          kernelPath,
          executionPolicies: context.executionPolicies ?? null,
        },
        recorder
      );

    const loraGateUp = getLoRAModule(lora, layerIdx, 'gate_up_proj');
    if (loraGateUp) {
      const combined = await applyLoRA(
        gateUpInput,
        gateUpOutput,
        loraGateUp,
        { M: numTokens, N: intermediateSize * 2, K: hiddenSize },
        getWeightBuffer,
        recorder,
        { kernelPath }
      );
      if (combined.buffer !== gateUpOutput.buffer) {
        if (recorder) {
          recorder.trackTemporaryBuffer(gateUpOutput.buffer);
        } else {
          releaseBuffer(gateUpOutput.buffer);
        }
        gateUpOutput = combined;
      }
    }

    if (!isGpuBufferInstance(layerWeights.gateUp) && !isWeightBuffer(layerWeights.gateUp)) {
      releaseOrTrack(recorder, isWeightBuffer(gateUpWeight) ? gateUpWeight.buffer : gateUpWeight);
    }

    activatedOutput = await doSiLURowSplit(gateUpOutput, {
      numTokens,
      dim: intermediateSize,
      activation: resolveActivationOp(hiddenActivation),
      swigluLimit,
    }, recorder);

    if (recorder) {
      if (gateUpInputOwned) {
        recorder.trackTemporaryBuffer(gateUpInput.buffer);
      }
      recorder.trackTemporaryBuffer(gateUpOutput.buffer);
    } else {
      if (gateUpInputOwned) {
        releaseBuffer(gateUpInput.buffer);
      }
      releaseBuffer(gateUpOutput.buffer);
    }
  } else {
    const hiddenSizeAligned32 = hiddenSize % 32 === 0;
    const activationDtype = selectRuleValue('shared', 'dtype', 'f16OrFallbackByFlag', {
      useF16,
      fallback: inputTensor.dtype,
    });
    const fusedGateUpWeights = resolveFusedGateUpWeights(layerWeights, {
      activationDtype,
      hiddenSize,
      kernelPath,
      phase,
      layerIdx,
    });
    const fusedGateWeight = getWeightBuffer(fusedGateUpWeights.gate ?? layerWeights.gate, 'ffn_gate');
    const fusedUpWeight = getWeightBuffer(fusedGateUpWeights.up ?? layerWeights.up, 'ffn_up');
    const gateDtype = requireFusedWeightDtype(fusedGateUpWeights.gateDtype, 'gate');
    const upDtype = requireFusedWeightDtype(fusedGateUpWeights.upDtype, 'up');
    const hasLoRAGate = Boolean(getLoRAModule(lora, layerIdx, 'gate_proj'));
    const hasLoRAUp = Boolean(getLoRAModule(lora, layerIdx, 'up_proj'));
    const dtypeMatches = gateDtype != null && upDtype != null && gateDtype === upDtype;
    const q4kFusedAllowed = gateDtype !== 'q4k' || !isFusedQ4KDisabled({ kernelPath });
    const dtypeSupported = gateDtype === 'f16' || gateDtype === 'f32' || (gateDtype === 'q4k' && q4kFusedAllowed);
    const canUseFusedGateUpByRule = selectRuleValue('inference', 'ffn', 'useFusedGateUp', {
      hasGate: true,
      hasUp: true,
      hasDown: true,
      hasFusedWeights: false,
      inputIsSupported: inputTensor.dtype === 'f32' || inputTensor.dtype === 'f16',
      hasLoRA: hasLoRAGate || hasLoRAUp,
      dtypeMatches,
      dtypeSupported,
      weightDtype: gateDtype,
      hasQ4KMaterialization: fusedGateUpWeights.hasQ4KMaterialization,
      activationDtype,
      f16BatchSupported: getKernelCapabilities().hasF16,
      useLargeBatchF16F32FusedGateUp: context.useLargeBatchF16F32FusedGateUp === true,
      batchSize: numTokens,
      hiddenSizeAligned32,
      useDoubleWideMlp: Boolean(useDoubleWideMlp),
    });
    const splitPrefillF16FusionAllowed = gateUpPathMode === 'split'
      && canFuseSplitPrefillF16GateUpPath({
        kernelPath,
        phase,
        layerIdx,
        gateDtype,
        upDtype,
      });
    const canUseFusedGateUp = gateUpPathMode === 'split' && !splitPrefillF16FusionAllowed
      ? false
      : canUseFusedGateUpByRule;
    trace.ffn(
      layerIdx,
      `useFusedGateUpWithPostNorm=${canUseFusedGateUp} gateUpPathMode=${gateUpPathMode} splitPrefillF16FusionAllowed=${splitPrefillF16FusionAllowed} ` +
      `inputDtype=${inputTensor.dtype} activationDtype=${activationDtype} ` +
      `gateDtype=${gateDtype} upDtype=${upDtype} hasQ4KMaterialization=${fusedGateUpWeights.hasQ4KMaterialization} ` +
      `dtypeMatches=${dtypeMatches} dtypeSupported=${dtypeSupported} hiddenSizeAligned32=${hiddenSizeAligned32} ` +
      `largeBatchF16F32FusedGateUp=${context.useLargeBatchF16F32FusedGateUp === true} batchSize=${numTokens}`
    );
    const gateWeight = canUseFusedGateUp
      ? fusedGateWeight
      : getWeightBuffer(layerWeights.gate, 'ffn_gate');
	    const upWeight = canUseFusedGateUp
	      ? fusedUpWeight
	      : getWeightBuffer(layerWeights.up, 'ffn_up');

	    if (canUseFusedGateUp) {
	      const fusedGateUpPipelineConstants = resolveFusedGateUpPipelineConstants({
	        kernelPath,
	        phase,
	        layerIdx,
	      });
	      const fusedGateUpVariant = resolveFusedGateUpVariant({ phase });
	      const {
	        fusedGateUpInputDtype,
	      } = resolveDenseFFNFusedPathDtypes({
        phase,
        layerIdx,
        kernelPath,
        ffnStepPrecision,
        fallbackInputDtype: inputTensor.dtype,
        fallbackOutputDtype: matmulOutputDtype,
      });
      let fusedInput = inputTensor;
      let fusedInputOwned = false;
      if (fusedGateUpInputDtype && fusedGateUpInputDtype !== inputTensor.dtype) {
        fusedInput = await coerceTensorDtype(inputTensor, fusedGateUpInputDtype, recorder, {
          executionPolicies: context.executionPolicies ?? null,
          op: 'ffn_gate_up_input',
          transitionDeclaredBy: 'step_precision',
        });
        fusedInputOwned = fusedInput !== inputTensor;
      }
      activatedOutput = await dispatchFusedGateUp({
	        inputTensor: fusedInput, gateWeight, upWeight, gateDtype,
	        hiddenSize, intermediateSize, numTokens,
	        hiddenActivation, swigluLimit, recorder,
	        executionPolicies: context.executionPolicies ?? null,
	        pipelineConstants: fusedGateUpPipelineConstants,
	        variant: fusedGateUpVariant,
	      });
      if (fusedInputOwned) {
        if (recorder) {
          recorder.trackTemporaryBuffer(fusedInput.buffer);
        } else {
          releaseBuffer(fusedInput.buffer);
        }
      }
    } else {
      const gateOutput = await doMatmul(
        inputTensor, gateWeight,
        numTokens, intermediateSize, hiddenSize,
        {
          transposeB: 'auto',
          outputDtype: matmulOutputDtype,
          role: 'ffn_gate',
          label: `L${layerIdx}.ffn_gate`,
          layerIdx,
          kernelPath,
          executionPolicies: context.executionPolicies ?? null,
        },
        recorder
      );

      const upOutput = await doMatmul(
        inputTensor, upWeight,
        numTokens, intermediateSize, hiddenSize,
        {
          transposeB: 'auto',
          outputDtype: matmulOutputDtype,
          role: 'ffn_up',
          label: `L${layerIdx}.ffn_up`,
          layerIdx,
          kernelPath,
          executionPolicies: context.executionPolicies ?? null,
        },
        recorder
      );

      activatedOutput = await dispatchActivation(hiddenActivation, upOutput, {
        size: numTokens * intermediateSize,
        gate: gateOutput,
        inputActivation: 'identity',
        swigluLimit,
      }, recorder);

      if (recorder) {
        recorder.trackTemporaryBuffer(gateOutput.buffer);
        recorder.trackTemporaryBuffer(upOutput.buffer);
      } else {
        releaseBuffer(gateOutput.buffer);
        releaseBuffer(upOutput.buffer);
      }
    }

    if (!isGpuBufferInstance(layerWeights.gate) && !isWeightBuffer(layerWeights.gate)) {
      releaseOrTrack(recorder, isWeightBuffer(gateWeight) ? gateWeight.buffer : gateWeight);
    }
    if (!isGpuBufferInstance(layerWeights.up) && !isWeightBuffer(layerWeights.up)) {
      releaseOrTrack(recorder, isWeightBuffer(upWeight) ? upWeight.buffer : upWeight);
    }
  }

  const outputTensor = await doMatmulRMSNormFused(
    activatedOutput,
    downWeight,
    normWeightBuf,
    {
      N: hiddenSize,
      K: intermediateSize,
      eps,
      residual: residualTensor,
      outputBuffer,
      transposeB,
      label: `L${layerIdx}.ffn_down`,
      rmsNormWeightOffset: config.postNormWeightOffset ?? config.rmsNormWeightOffset,
    },
    recorder
  );

  const loraDown = getLoRAModule(lora, layerIdx, 'down_proj');
  if (loraDown) {
    log.warn('Layer', `L${layerIdx} LoRA down_proj with fused kernel not yet optimized`);
  }

  if (!isGpuBufferInstance(layerWeights.down) && !isWeightBuffer(layerWeights.down)) {
    releaseOrTrack(recorder, isWeightBuffer(downWeight) ? downWeight.buffer : downWeight);
  }
  if (!isGpuBufferInstance(layerWeights.postFeedforwardNorm) && !isWeightBuffer(layerWeights.postFeedforwardNorm)) releaseOrTrack(recorder, normWeightBuf);
  if (recorder) {
    recorder.trackTemporaryBuffer(activatedOutput.buffer);
  } else {
    releaseBuffer(activatedOutput.buffer);
  }

  return outputTensor;
}
