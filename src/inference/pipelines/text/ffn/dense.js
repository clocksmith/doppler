

import {
  doMatmul, doSiLU, doGeLU, doSiLURowSplit, doMatmulRMSNormFused,
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
import { acquireBuffer, releaseBuffer } from '../../../../memory/buffer-pool.js';
import {
  runFusedFFN,
  recordFusedFFN,
  castF16ToF32,
  castF32ToF16,
  recordCastF16ToF32,
  recordCastF32ToF16,
  isFusedQ4KDisabled
} from '../../../../gpu/kernel-selector.js';
import { log, trace } from '../../../../debug/index.js';
import { isKernelDebugEnabled, dumpTokenVector } from '../debug-utils.js';
import { applyLoRA } from '../lora-apply.js';
import { getLoRAModule } from '../lora.js';
import { getWeightBuffer, getNormWeightBuffer } from '../weights.js';
import { runProbes } from '../probes.js';
import { selectRuleValue } from '../../../../rules/rule-registry.js';
import {
  getKernelPathMatmulPrecision,
  getKernelPathMatmulVariant,
} from '../../../../config/kernel-path-loader.js';
import { resolveLayerIntermediateSize } from '../config.js';
import { assertImplicitDtypeTransitionAllowed } from '../dtype-contract.js';

const ACTIVATION_FN_MAP = {
  gelu: doGeLU,
  silu: doSiLU,
};

function resolveActivationOp(hiddenActivation) {
  return selectRuleValue('inference', 'ffn', 'activationOp', { hiddenActivation });
}

function hasQ4KMaterialization(weight) {
  return isWeightBuffer(weight) && !!weight.materializations?.q4k?.buffer;
}

function isQ4KMatmulVariant(variant) {
  return typeof variant === 'string' && variant.startsWith('q4_');
}

export function resolveDenseFFNMatmulStepDtype(options = {}) {
  const precision = getKernelPathMatmulPrecision(
    options.role,
    options.phase,
    options.layerIdx,
    options.kernelPath
  );
  const requested = precision?.[options.field]
    ?? options.ffnStepPrecision?.[options.field]
    ?? options.fallback;
  if (requested == null) {
    return options.fallback;
  }
  return selectRuleValue('shared', 'dtype', 'f16OrF32FromDtype', { dtype: requested });
}

function resolveMatmulStepDtype(role, phase, layerIdx, kernelPath, fallback, field, ffnStepPrecision = null) {
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

export function resolveDenseFFNFusedPathDtypes(options = {}) {
  const phase = options.phase ?? null;
  const layerIdx = Number.isFinite(options.layerIdx) ? options.layerIdx : 0;
  const kernelPath = options.kernelPath ?? null;
  const ffnStepPrecision = options.ffnStepPrecision ?? null;
  const fallbackInputDtype = options.fallbackInputDtype ?? null;
  const fallbackOutputDtype = options.fallbackOutputDtype ?? fallbackInputDtype;

  const fusedGateUpInputDtype = resolveDenseFFNMatmulStepDtype({
    role: 'ffn_gate_up',
    phase,
    layerIdx,
    kernelPath,
    fallback: null,
    field: 'inputDtype',
    ffnStepPrecision,
  });
  const gateInputDtype = resolveDenseFFNMatmulStepDtype({
    role: 'ffn_gate',
    phase,
    layerIdx,
    kernelPath,
    fallback: null,
    field: 'inputDtype',
    ffnStepPrecision,
  });
  const upInputDtype = resolveDenseFFNMatmulStepDtype({
    role: 'ffn_up',
    phase,
    layerIdx,
    kernelPath,
    fallback: null,
    field: 'inputDtype',
    ffnStepPrecision,
  });
  const resolvedFusedGateUpInputDtype = fusedGateUpInputDtype
    ?? (gateInputDtype && gateInputDtype === upInputDtype ? gateInputDtype : fallbackInputDtype);

  const fusedGateUpOutputDtype = resolveDenseFFNMatmulStepDtype({
    role: 'ffn_gate_up',
    phase,
    layerIdx,
    kernelPath,
    fallback: fallbackOutputDtype,
    field: 'outputDtype',
    ffnStepPrecision,
  });
  const downInputDtype = resolveDenseFFNMatmulStepDtype({
    role: 'ffn_down',
    phase,
    layerIdx,
    kernelPath,
    fallback: fusedGateUpOutputDtype,
    field: 'inputDtype',
    ffnStepPrecision,
  });

  return {
    fusedGateUpInputDtype: resolvedFusedGateUpInputDtype,
    fusedGateUpOutputDtype,
    downInputDtype,
  };
}

function hasExplicitMatmulPrecision(role, phase, layerIdx, kernelPath) {
  // Only treat precision as "explicit split" when declared on the role's OWN
  // step. The FUSED_FFN_PRECISION_FALLBACK_ROLES fallback that resolves via
  // the aggregate `ffn` step's precision applies identically to gate/up/down
  // and must NOT force the split path — otherwise a manifest-declared `ffn`
  // precision permanently blocks the fused gate_up_activation kernel.
  const precision = getKernelPathMatmulPrecision(role, phase, layerIdx, kernelPath);
  if (!precision) return false;
  const fallbackPrecision = getKernelPathMatmulPrecision('ffn', phase, layerIdx, kernelPath);
  const inheritedFromFfn = fallbackPrecision
    && fallbackPrecision.inputDtype === precision.inputDtype
    && fallbackPrecision.outputDtype === precision.outputDtype
    && fallbackPrecision.activationDtype === precision.activationDtype;
  if (inheritedFromFfn) return false;
  return precision.inputDtype != null || precision.outputDtype != null;
}

export function canUseNativeF16FusedGateUp(options = {}) {
  if (options.inputDtype !== 'f16' || options.hasF16 !== true) {
    return false;
  }
  return options.gateDtype === 'f16' || options.gateDtype === 'q4k';
}

async function coerceTensorDtype(tensor, targetDtype, recorder, options = {}) {
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

export function resolveGateUpPathMode(options = {}) {
  const kernelPath = options.kernelPath ?? null;
  const phase = options.phase ?? null;
  const layerIdx = Number.isFinite(options.layerIdx) ? options.layerIdx : 0;
  if (!kernelPath || !phase) {
    return 'implicit';
  }

  const fusedVariant = getKernelPathMatmulVariant('ffn_gate_up', phase, layerIdx, kernelPath);
  if (fusedVariant != null) {
    return 'fused';
  }

  const gateVariant = getKernelPathMatmulVariant('ffn_gate', phase, layerIdx, kernelPath);
  const upVariant = getKernelPathMatmulVariant('ffn_up', phase, layerIdx, kernelPath);
  const hasExplicitGatePrecision = hasExplicitMatmulPrecision('ffn_gate', phase, layerIdx, kernelPath);
  const hasExplicitUpPrecision = hasExplicitMatmulPrecision('ffn_up', phase, layerIdx, kernelPath);
  const hasExplicitDownPrecision = hasExplicitMatmulPrecision('ffn_down', phase, layerIdx, kernelPath);
  const hasExplicitSplitPrecision = hasExplicitGatePrecision || hasExplicitUpPrecision || hasExplicitDownPrecision;
  if (hasExplicitSplitPrecision) {
    const decodeQ4PrecisionCanStayFused = phase === 'decode'
      && !hasExplicitDownPrecision
      && isQ4KMatmulVariant(gateVariant)
      && gateVariant === upVariant
      && (hasExplicitGatePrecision || hasExplicitUpPrecision);
    if (!decodeQ4PrecisionCanStayFused) {
      return 'split';
    }
  }
  if (
    gateVariant != null
    && upVariant != null
  ) {
    if (
      phase === 'prefill'
      && !isQ4KMatmulVariant(gateVariant)
      && !isQ4KMatmulVariant(upVariant)
    ) {
      return 'split';
    }
  }

  return 'implicit';
}

export function resolveFusedGateUpWeights(layerWeights, options = {}) {
  const gate = layerWeights?.gate ?? null;
  const up = layerWeights?.up ?? null;
  const hiddenSize = Number.isFinite(options.hiddenSize) ? options.hiddenSize : 0;
  const q4kAllowed = !isFusedQ4KDisabled({ kernelPath: options.kernelPath ?? null });
  const hasMixedQ4KMaterialization = hasQ4KMaterialization(gate) && hasQ4KMaterialization(up);
  const preferQ4KMaterialization = hiddenSize > 0
    && hiddenSize % 32 === 0
    && q4kAllowed
    && hasMixedQ4KMaterialization;
  const resolvedGate = preferQ4KMaterialization
    ? resolveWeightBufferMaterialization(gate, 'q4k')
    : gate;
  const resolvedUp = preferQ4KMaterialization
    ? resolveWeightBufferMaterialization(up, 'q4k')
    : up;

  return {
    gate: resolvedGate,
    up: resolvedUp,
    gateDtype: resolvedGate ? getWeightDtype(resolvedGate) : null,
    upDtype: resolvedUp ? getWeightDtype(resolvedUp) : null,
    hasQ4KMaterialization: hasMixedQ4KMaterialization,
  };
}

async function dispatchActivation(hiddenActivation, input, options, recorder) {
  const op = resolveActivationOp(hiddenActivation);
  const fn = ACTIVATION_FN_MAP[op];
  if (!fn) {
    throw new Error(`Unsupported FFN activation op "${op}".`);
  }
  return fn(input, options, recorder);
}

async function dispatchFusedGateUp({
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
  const fusedOutput = recorder
    ? await recordFusedFFN(
      recorder, fusedInput, gateWeight, upWeight,
      hiddenSize, intermediateSize,
      { batchSize: numTokens, activation, swigluLimit }
    )
    : await runFusedFFN(
      fusedInput, gateWeight, upWeight,
      hiddenSize, intermediateSize,
      { batchSize: numTokens, activation, swigluLimit }
    );

  if (!recorder && fusedInput !== inputTensor) {
    releaseBuffer(fusedInput.buffer);
  }

  return fusedOutput;
}


export async function runDenseFFNGPU(
  layerIdx,
  inputTensor,
  numTokens,
  context,
  layerWeights
) {
  const device = getDevice();
  if (!device) throw new Error('No GPU device');

  const { config, recorder } = context;
  const { hiddenSize, hiddenActivation, swigluLimit } = config;
  const intermediateSize = resolveLayerIntermediateSize(config, layerIdx);
  const lastTokenIdx = Math.max(0, numTokens - 1);
  const lora = context.lora || null;
  const ffnStepPrecision = context.ffnStepPrecision ?? null;
  const kernelPath = context.kernelPath ?? null;
  const phase = context.phase ?? (numTokens === 1 ? 'decode' : 'prefill');
  const gateUpPathMode = resolveGateUpPathMode({ kernelPath, phase, layerIdx });

  if (layerWeights?.gateUp && layerWeights?.down) {
    const gateUpWeight = getWeightBuffer(layerWeights.gateUp, 'ffn_gate_up');
    const downWeight = getWeightBuffer(layerWeights.down, 'ffn_down');

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
    const downOutputDtype = resolveMatmulStepDtype(
      'ffn_down',
      phase,
      layerIdx,
      kernelPath,
      'f32',
      'outputDtype',
      ffnStepPrecision
    );
    let gateUpOutput = await doMatmul(
      inputTensor, gateUpWeight,
      numTokens, intermediateSize * 2, hiddenSize,
      {
        transposeB: 'auto',
        label: `L${layerIdx}.ffn_gate_up`,
        layerIdx,
        kernelPath,
        outputDtype: matmulOutputDtype,
        role: 'ffn_gate_up',
        executionPolicies: context.executionPolicies ?? null,
      },
      recorder
    );

    const loraGateUp = getLoRAModule(lora, layerIdx, 'gate_up_proj');
    if (loraGateUp) {
      const combined = await applyLoRA(
        inputTensor,
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

    if (isKernelDebugEnabled(layerIdx) && !recorder) {
      await dumpTokenVector(gateUpOutput.buffer, 'ffn_gate_up', {
        layerIdx,
        tokenIdx: lastTokenIdx,
        rowSize: intermediateSize * 2,
        dtype: gateUpOutput.dtype,
      });
    }

    if (!isGpuBufferInstance(layerWeights.gateUp) && !isWeightBuffer(layerWeights.gateUp)) {
      releaseOrTrack(recorder, isWeightBuffer(gateUpWeight) ? gateUpWeight.buffer : gateUpWeight);
    }

    const activatedOutput = await doSiLURowSplit(gateUpOutput, {
      numTokens,
      dim: intermediateSize,
      activation: resolveActivationOp(hiddenActivation),
      swigluLimit,
      label: `L${layerIdx}.ffn_activation`,
      layerIdx,
    }, recorder);

    if (isKernelDebugEnabled(layerIdx) && !recorder) {
      await dumpTokenVector(activatedOutput.buffer, 'ffn_activated', {
        layerIdx,
        tokenIdx: lastTokenIdx,
        rowSize: intermediateSize,
        dtype: activatedOutput.dtype,
      });
    }

    if (recorder) {
      recorder.trackTemporaryBuffer(gateUpOutput.buffer);
    } else {
      releaseBuffer(gateUpOutput.buffer);
    }

    // Opt-in WideTile+residual fusion: if the caller (processFFNStandard)
    // staged a residual tensor on the context AND no LoRA is present on
    // down_proj (LoRA requires the pre-residual output for its add), route
    // this matmul to the q4_fused_widetile_residual variant which produces
    // (ffn_down_result + residual) in one dispatch. Tell the caller via a
    // context flag so processFFNStandard skips its downstream doResidualAdd.
    const pendingResidual = context.__pendingFfnResidualTensor;
    const downLoraProbe = getLoRAModule(lora, layerIdx, 'down_proj');
    const mergedSession = getRuntimeConfig().inference?.session;
    const tryFuseDownResidual = pendingResidual != null
      && !downLoraProbe
      && numTokens > 1
      && activatedOutput.dtype === 'f32'
      && downOutputDtype === 'f32'
      && pendingResidual.dtype === 'f32';
    let residualFusedHere = false;
    let output = await doMatmul(
      activatedOutput, downWeight,
      numTokens, hiddenSize, intermediateSize,
      {
        transposeB: 'auto',
        label: `L${layerIdx}.ffn_down`,
        layerIdx,
        kernelPath,
        outputDtype: downOutputDtype,
        role: 'ffn_down',
        executionPolicies: context.executionPolicies ?? null,
        residualTensor: tryFuseDownResidual ? pendingResidual : null,
      },
      recorder
    );
    // Detect whether the fusion fired: if the selected variant for ffn_down
    // matmul was q4_fused_widetile_residual, then output IS post-residual.
    // We infer this cheaply by re-checking the conditions the selector uses.
    // (A cleaner signal would require a return-shape change across all
    // dense.js paths; this local signal is enough for correctness.)
    {
      if (tryFuseDownResidual
          && getKernelCapabilities().hasF16 === true
          && mergedSession?.useWideTileResidualFusion === true
          && mergedSession?.useWideTileQ4KPrefill === true
          && mergedSession?.retainQ4KMaterialization === true
      ) {
        residualFusedHere = true;
        context.__ffnResidualFusedFired = true;
      }
    }

    const loraDown = getLoRAModule(lora, layerIdx, 'down_proj');
    if (loraDown) {
      const combined = await applyLoRA(
        activatedOutput,
        output,
        loraDown,
        { M: numTokens, N: hiddenSize, K: intermediateSize },
        getWeightBuffer,
        recorder,
        { kernelPath }
      );
      if (combined.buffer !== output.buffer) {
        if (recorder) {
          recorder.trackTemporaryBuffer(output.buffer);
        } else {
          releaseBuffer(output.buffer);
        }
        output = combined;
      }
    }

    if (isKernelDebugEnabled(layerIdx) && !recorder) {
      await dumpTokenVector(output.buffer, 'ffn_down_out', {
        layerIdx,
        tokenIdx: lastTokenIdx,
        rowSize: hiddenSize,
        dtype: output.dtype,
      });
    }

    if (!isGpuBufferInstance(layerWeights.down) && !isWeightBuffer(layerWeights.down)) {
      releaseOrTrack(recorder, isWeightBuffer(downWeight) ? downWeight.buffer : downWeight);
    }
    if (recorder) {
      recorder.trackTemporaryBuffer(activatedOutput.buffer);
    } else {
      releaseBuffer(activatedOutput.buffer);
    }

    return output;
  }

  const hasGate = Boolean(layerWeights?.gate);
  const hasUp = Boolean(layerWeights?.up);
  const hasDown = Boolean(layerWeights?.down);
  const hasFusedWeights = Boolean(layerWeights?.gateUp);
  const inputIsSupported = inputTensor.dtype === 'f32' || inputTensor.dtype === 'f16';
  const hasLoRA = Boolean(
    (hasGate ? getLoRAModule(lora, layerIdx, 'gate_proj') : null) ||
    (hasUp ? getLoRAModule(lora, layerIdx, 'up_proj') : null)
  );
  const hiddenSizeAligned32 = hiddenSize % 32 === 0;
  const activationDtype = selectRuleValue('shared', 'dtype', 'f16OrFallbackByFlag', {
    useF16: inputTensor.dtype === 'f16',
    fallback: inputTensor.dtype,
  });
  const defaultMatmulOutputDtype = selectRuleValue('shared', 'dtype', 'f16OrFallbackByFlag', {
    useF16: inputTensor.dtype === 'f16',
    fallback: inputTensor.dtype,
  });
  const fusedGateUpWeights = resolveFusedGateUpWeights(layerWeights, {
    activationDtype,
    hiddenSize,
    kernelPath,
    phase,
    layerIdx,
  });
  const gateDtype = fusedGateUpWeights.gateDtype ?? (hasGate ? 'f32' : null);
  const upDtype = fusedGateUpWeights.upDtype ?? (hasUp ? 'f32' : null);
  const dtypeMatches = gateDtype != null && upDtype != null && gateDtype === upDtype;
  const q4kFusedAllowed = gateDtype !== 'q4k' || !isFusedQ4KDisabled({ kernelPath });
  const dtypeSupported = gateDtype === 'f16' || gateDtype === 'f32' || (gateDtype === 'q4k' && q4kFusedAllowed);
  const f16BatchSupported = getKernelCapabilities().hasF16;
  const useFusedGateUpByRule = selectRuleValue('inference', 'ffn', 'useFusedGateUp', {
    hasGate,
    hasUp,
    hasDown,
    hasFusedWeights,
    inputIsSupported,
    hasLoRA,
    dtypeMatches,
    dtypeSupported,
    weightDtype: gateDtype,
    hasQ4KMaterialization: fusedGateUpWeights.hasQ4KMaterialization,
    activationDtype,
    f16BatchSupported,
    batchSize: numTokens,
    hiddenSizeAligned32,
  });
  const useFusedGateUp = gateUpPathMode === 'split'
    ? false
    : useFusedGateUpByRule;
  trace.ffn(
    layerIdx,
    `useFusedGateUp=${useFusedGateUp} gateUpPathMode=${gateUpPathMode} inputDtype=${inputTensor.dtype} activationDtype=${activationDtype} ` +
    `gateDtype=${gateDtype} upDtype=${upDtype} hasQ4KMaterialization=${fusedGateUpWeights.hasQ4KMaterialization} ` +
    `dtypeMatches=${dtypeMatches} dtypeSupported=${dtypeSupported} hiddenSizeAligned32=${hiddenSizeAligned32} batchSize=${numTokens}`
  );

  if (useFusedGateUp) {
    const {
      fusedGateUpInputDtype,
      downInputDtype,
    } = resolveDenseFFNFusedPathDtypes({
      phase,
      layerIdx,
      kernelPath,
      ffnStepPrecision,
      fallbackInputDtype: inputTensor.dtype,
      fallbackOutputDtype: defaultMatmulOutputDtype,
    });
    const gateWeight = getWeightBuffer(fusedGateUpWeights.gate ?? layerWeights.gate, 'ffn_gate');
    const upWeight = getWeightBuffer(fusedGateUpWeights.up ?? layerWeights.up, 'ffn_up');
    const downWeight = getWeightBuffer(layerWeights.down, 'ffn_down');
    const fusedDownOutputDtype = resolveMatmulStepDtype(
      'ffn_down',
      phase,
      layerIdx,
      kernelPath,
      'f32',
      'outputDtype',
      ffnStepPrecision
    );
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
    const fusedOutput = await dispatchFusedGateUp({
      inputTensor: fusedInput, gateWeight, upWeight, gateDtype,
      hiddenSize, intermediateSize, numTokens,
      hiddenActivation, swigluLimit, recorder,
      executionPolicies: context.executionPolicies ?? null,
    });

    let downInput = fusedOutput;
    if (downInputDtype && fusedOutput.dtype !== downInputDtype) {
      downInput = await coerceTensorDtype(fusedOutput, downInputDtype, recorder, {
        executionPolicies: context.executionPolicies ?? null,
        op: 'ffn_down_input',
        transitionDeclaredBy: 'step_precision',
      });
      if (recorder) {
        recorder.trackTemporaryBuffer(downInput.buffer);
      }
    }

    if (!isGpuBufferInstance(layerWeights.gate) && !isWeightBuffer(layerWeights.gate)) {
      releaseOrTrack(recorder, isWeightBuffer(gateWeight) ? gateWeight.buffer : gateWeight);
    }
    if (!isGpuBufferInstance(layerWeights.up) && !isWeightBuffer(layerWeights.up)) {
      releaseOrTrack(recorder, isWeightBuffer(upWeight) ? upWeight.buffer : upWeight);
    }

    // Opt-in WideTile+residual fusion (fused-gate-up path).
    const pendingResidualFused = context.__pendingFfnResidualTensor;
    const downLoraProbeFused = getLoRAModule(lora, layerIdx, 'down_proj');
    const mergedSessionFused = getRuntimeConfig().inference?.session;
    const tryFuseDownResidualFused = pendingResidualFused != null
      && !downLoraProbeFused
      && numTokens > 1
      && downInput.dtype === 'f32'
      && fusedDownOutputDtype === 'f32'
      && pendingResidualFused.dtype === 'f32';
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
        outputDtype: fusedDownOutputDtype,
        role: 'ffn_down',
        executionPolicies: context.executionPolicies ?? null,
        residualTensor: tryFuseDownResidualFused ? pendingResidualFused : null,
      },
      recorder
    );
    {
      if (tryFuseDownResidualFused
          && getKernelCapabilities().hasF16 === true
          && mergedSessionFused?.useWideTileResidualFusion === true
          && mergedSessionFused?.useWideTileQ4KPrefill === true
          && mergedSessionFused?.retainQ4KMaterialization === true
      ) {
        context.__ffnResidualFusedFired = true;
      }
    }

    const loraDown = getLoRAModule(lora, layerIdx, 'down_proj');
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
        if (recorder) {
          recorder.trackTemporaryBuffer(output.buffer);
        } else {
          releaseBuffer(output.buffer);
        }
        output = combined;
      }
    }

    if (!isGpuBufferInstance(layerWeights.down) && !isWeightBuffer(layerWeights.down)) {
      releaseOrTrack(recorder, isWeightBuffer(downWeight) ? downWeight.buffer : downWeight);
    }

    if (recorder) {
      if (downInput !== fusedOutput) {
        recorder.trackTemporaryBuffer(downInput.buffer);
      }
      if (fusedInputOwned) {
        recorder.trackTemporaryBuffer(fusedInput.buffer);
      }
      recorder.trackTemporaryBuffer(fusedOutput.buffer);
    } else {
      if (downInput !== fusedOutput) {
        releaseBuffer(downInput.buffer);
      }
      if (fusedInputOwned) {
        releaseBuffer(fusedInput.buffer);
      }
      releaseBuffer(fusedOutput.buffer);
    }

    return output;
  }

  if (!layerWeights?.gate || !layerWeights?.up || !layerWeights?.down) {
    log.warn('Layer', `L${layerIdx} FFN: no weights found`);
    const bytesPerElement = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype: inputTensor.dtype });
    const byteSize = numTokens * hiddenSize * bytesPerElement;
    const outputBuffer = acquireBuffer(byteSize, undefined, 'ffn_output');
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(inputTensor.buffer, 0, outputBuffer, 0, byteSize);
    device.queue.submit([encoder.finish()]);
    return createTensor(outputBuffer, inputTensor.dtype, [...inputTensor.shape], 'ffn_output_copy');
  }

  const gateInputDtype = resolveMatmulStepDtype(
    'ffn_gate',
    phase,
    layerIdx,
    kernelPath,
    inputTensor.dtype,
    'inputDtype',
    ffnStepPrecision
  );
  const gateOutputDtype = resolveMatmulStepDtype(
    'ffn_gate',
    phase,
    layerIdx,
    kernelPath,
    defaultMatmulOutputDtype,
    'outputDtype',
    ffnStepPrecision
  );
  const upInputDtype = resolveMatmulStepDtype(
    'ffn_up',
    phase,
    layerIdx,
    kernelPath,
    inputTensor.dtype,
    'inputDtype',
    ffnStepPrecision
  );
  const upOutputDtype = resolveMatmulStepDtype(
    'ffn_up',
    phase,
    layerIdx,
    kernelPath,
    defaultMatmulOutputDtype,
    'outputDtype',
    ffnStepPrecision
  );
  const downOutputDtype = resolveMatmulStepDtype(
    'ffn_down',
    phase,
    layerIdx,
    kernelPath,
    'f32',
    'outputDtype',
    ffnStepPrecision
  );
  const downInputDtype = resolveMatmulStepDtype(
    'ffn_down',
    phase,
    layerIdx,
    kernelPath,
    downOutputDtype,
    'inputDtype',
    ffnStepPrecision
  );
  const sharedInputDtype = gateInputDtype === upInputDtype ? gateInputDtype : null;
  let sharedInputTensor = inputTensor;
  let sharedInputOwned = false;
  if (sharedInputDtype && sharedInputDtype !== inputTensor.dtype) {
    sharedInputTensor = await coerceTensorDtype(inputTensor, sharedInputDtype, recorder, {
      executionPolicies: context.executionPolicies ?? null,
      op: 'ffn_shared_input',
      transitionDeclaredBy: 'step_precision',
    });
    sharedInputOwned = sharedInputTensor !== inputTensor;
  }
  // Opt-in fused gate + up + GeGLU path. Replaces 3 separate dispatches
  // (gate_proj + up_proj + gelu activation) with a single fused kernel when
  // preconditions match: prefill (M>1), f16-materialisable weights + f16
  // activations, gelu activation, no LoRA on gate/up. Gated by
  // runtime.inference.session.useFusedGateUpGelu (default false).
  const gateF16 = resolveWeightBufferMaterialization(layerWeights.gate, 'f16');
  const upF16 = resolveWeightBufferMaterialization(layerWeights.up, 'f16');
  const gateF16Dtype = getWeightDtype(gateF16);
  const upF16Dtype = getWeightDtype(upF16);
  const earlyLoraGate = getLoRAModule(lora, layerIdx, 'gate_proj');
  const earlyLoraUp = getLoRAModule(lora, layerIdx, 'up_proj');
  const fusedGateUpGeluCandidate = context.useFusedGateUpGelu === true
    && numTokens > 1
    && hiddenActivation === 'gelu'
    && !earlyLoraGate && !earlyLoraUp
    && sharedInputTensor.dtype === 'f16'
    && gateF16Dtype === 'f16'
    && upF16Dtype === 'f16';
  if (fusedGateUpGeluCandidate) {
    const { runFusedGateUpGelu, recordFusedGateUpGelu } =
      await import('../../../../gpu/kernels/fused-gate-up-gelu.js');
    const fused = recorder
      ? await recordFusedGateUpGelu(recorder, sharedInputTensor, gateF16, upF16, {
        M: numTokens,
        hiddenSize,
        intermediateSize,
        transposeB: true,
      })
      : await runFusedGateUpGelu(sharedInputTensor, gateF16, upF16, {
        M: numTokens,
        hiddenSize,
        intermediateSize,
        transposeB: true,
      });
    if (sharedInputOwned) {
      if (recorder) { recorder.trackTemporaryBuffer(sharedInputTensor.buffer); }
      else { releaseBuffer(sharedInputTensor.buffer); }
    }
    // Proceed directly to down_proj with the fused activation output.
    const downWeight = getWeightBuffer(layerWeights.down, 'ffn_down');
    let downInputTensor = fused;
    let downInputOwned = false;
    if (downInputDtype && fused.dtype !== downInputDtype) {
      downInputTensor = await coerceTensorDtype(fused, downInputDtype, recorder, {
        executionPolicies: context.executionPolicies ?? null,
        op: 'ffn_down_input',
        transitionDeclaredBy: 'step_precision',
      });
      downInputOwned = downInputTensor !== fused;
    }
    let outFused = await doMatmul(
      downInputTensor,
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
    if (!isGpuBufferInstance(layerWeights.down) && !isWeightBuffer(layerWeights.down)) {
      releaseOrTrack(recorder, isWeightBuffer(downWeight) ? downWeight.buffer : downWeight);
    }
    if (downInputOwned) {
      if (recorder) { recorder.trackTemporaryBuffer(downInputTensor.buffer); }
      else { releaseBuffer(downInputTensor.buffer); }
    }
    if (recorder) { recorder.trackTemporaryBuffer(fused.buffer); }
    else { releaseBuffer(fused.buffer); }
    return outFused;
  }
  const gateWeight = getWeightBuffer(layerWeights.gate, 'ffn_gate');
  let gateOutput = await doMatmul(
    gateInputDtype === sharedInputTensor.dtype ? sharedInputTensor : inputTensor,
    gateWeight,
    numTokens,
    intermediateSize,
    hiddenSize,
    {
      transposeB: 'auto',
      label: `L${layerIdx}.ffn_gate`,
      layerIdx,
      kernelPath,
      outputDtype: gateOutputDtype,
      role: 'ffn_gate',
      executionPolicies: context.executionPolicies ?? null,
    },
    recorder
  );
  if (!isGpuBufferInstance(layerWeights.gate) && !isWeightBuffer(layerWeights.gate)) {
    releaseOrTrack(recorder, isWeightBuffer(gateWeight) ? gateWeight.buffer : gateWeight);
  }

  const loraGate = getLoRAModule(lora, layerIdx, 'gate_proj');
  if (loraGate) {
    const combined = await applyLoRA(
      inputTensor,
      gateOutput,
      loraGate,
      { M: numTokens, N: intermediateSize, K: hiddenSize },
      getWeightBuffer,
      recorder,
      { kernelPath }
    );
    if (combined.buffer !== gateOutput.buffer) {
      if (recorder) {
        recorder.trackTemporaryBuffer(gateOutput.buffer);
      } else {
        releaseBuffer(gateOutput.buffer);
      }
      gateOutput = combined;
    }
  }

  const upWeight = getWeightBuffer(layerWeights.up, 'ffn_up');
  let upOutput = await doMatmul(
    upInputDtype === sharedInputTensor.dtype ? sharedInputTensor : inputTensor,
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
  if (!isGpuBufferInstance(layerWeights.up) && !isWeightBuffer(layerWeights.up)) {
    releaseOrTrack(recorder, isWeightBuffer(upWeight) ? upWeight.buffer : upWeight);
  }

  const loraUp = getLoRAModule(lora, layerIdx, 'up_proj');
  if (loraUp) {
    const combined = await applyLoRA(
      inputTensor,
      upOutput,
      loraUp,
      { M: numTokens, N: intermediateSize, K: hiddenSize },
      getWeightBuffer,
      recorder,
      { kernelPath }
    );
    if (combined.buffer !== upOutput.buffer) {
      if (recorder) {
        recorder.trackTemporaryBuffer(upOutput.buffer);
      } else {
        releaseBuffer(upOutput.buffer);
      }
      upOutput = combined;
    }
  }

  if (isKernelDebugEnabled(layerIdx) && !recorder) {
    await dumpTokenVector(gateOutput.buffer, 'ffn_gate', {
      layerIdx,
      tokenIdx: lastTokenIdx,
      rowSize: intermediateSize,
      dtype: gateOutput.dtype,
    });
    await dumpTokenVector(upOutput.buffer, 'ffn_up', {
      layerIdx,
      tokenIdx: lastTokenIdx,
      rowSize: intermediateSize,
      dtype: upOutput.dtype,
    });
  }

  await runProbes('ffn_gate', gateOutput.buffer, {
    layerIdx,
    numTokens,
    hiddenSize: intermediateSize,
    probes: context.debugProbes,
    recorder,
    operatorDiagnostics: context.operatorDiagnostics,
    dtype: gateOutput.dtype,
  });
  await runProbes('ffn_up', upOutput.buffer, {
    layerIdx,
    numTokens,
    hiddenSize: intermediateSize,
    probes: context.debugProbes,
    recorder,
    operatorDiagnostics: context.operatorDiagnostics,
    dtype: upOutput.dtype,
  });

  const activatedOutput = await dispatchActivation(hiddenActivation, upOutput, {
    size: numTokens * intermediateSize,
    gate: gateOutput,
    swigluLimit,
    label: `L${layerIdx}.ffn_activation`,
    layerIdx,
  }, recorder);

  if (isKernelDebugEnabled(layerIdx) && !recorder) {
    await dumpTokenVector(activatedOutput.buffer, 'ffn_activated', {
      layerIdx,
      tokenIdx: lastTokenIdx,
      rowSize: intermediateSize,
      dtype: activatedOutput.dtype,
    });
  }

  await runProbes('ffn_act', activatedOutput.buffer, {
    layerIdx,
    numTokens,
    hiddenSize: intermediateSize,
    probes: context.debugProbes,
    recorder,
    operatorDiagnostics: context.operatorDiagnostics,
    dtype: activatedOutput.dtype,
  });

  if (recorder) {
    recorder.trackTemporaryBuffer(gateOutput.buffer);
    recorder.trackTemporaryBuffer(upOutput.buffer);
  } else {
    releaseBuffer(gateOutput.buffer);
    releaseBuffer(upOutput.buffer);
  }
  if (sharedInputOwned) {
    if (recorder) {
      recorder.trackTemporaryBuffer(sharedInputTensor.buffer);
    } else {
      releaseBuffer(sharedInputTensor.buffer);
    }
  }

  const downWeight = getWeightBuffer(layerWeights.down, 'ffn_down');
  let downInputTensor = activatedOutput;
  let downInputOwned = false;
  if (downInputDtype && activatedOutput.dtype !== downInputDtype) {
    downInputTensor = await coerceTensorDtype(activatedOutput, downInputDtype, recorder, {
      executionPolicies: context.executionPolicies ?? null,
      op: 'ffn_down_input',
      transitionDeclaredBy: 'step_precision',
    });
    downInputOwned = downInputTensor !== activatedOutput;
  }
  let output = await doMatmul(
    downInputTensor,
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

  const loraDown = getLoRAModule(lora, layerIdx, 'down_proj');
  if (loraDown) {
    const combined = await applyLoRA(
      downInputTensor,
      output,
      loraDown,
      { M: numTokens, N: hiddenSize, K: intermediateSize },
      getWeightBuffer,
      recorder,
      { kernelPath }
    );
    if (combined.buffer !== output.buffer) {
      if (recorder) {
        recorder.trackTemporaryBuffer(output.buffer);
      } else {
        releaseBuffer(output.buffer);
      }
      output = combined;
    }
  }

  if (isKernelDebugEnabled(layerIdx) && !recorder) {
    await dumpTokenVector(output.buffer, 'ffn_down_out', {
      layerIdx,
      tokenIdx: lastTokenIdx,
      rowSize: hiddenSize,
      dtype: output.dtype,
    });
  }

  if (!isGpuBufferInstance(layerWeights.down) && !isWeightBuffer(layerWeights.down)) {
    releaseOrTrack(recorder, isWeightBuffer(downWeight) ? downWeight.buffer : downWeight);
  }
  if (downInputOwned) {
    if (recorder) {
      recorder.trackTemporaryBuffer(downInputTensor.buffer);
    } else {
      releaseBuffer(downInputTensor.buffer);
    }
  }
  if (recorder) {
    recorder.trackTemporaryBuffer(activatedOutput.buffer);
  } else {
    releaseBuffer(activatedOutput.buffer);
  }

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
  const { hiddenSize, hiddenActivation, swigluLimit } = config;
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
    const gateDtype = fusedGateUpWeights.gateDtype ?? 'f32';
    const upDtype = fusedGateUpWeights.upDtype ?? 'f32';
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
      batchSize: numTokens,
      hiddenSizeAligned32,
    });
    const canUseFusedGateUp = gateUpPathMode === 'split'
      ? false
      : canUseFusedGateUpByRule;
    trace.ffn(
      layerIdx,
      `useFusedGateUpWithPostNorm=${canUseFusedGateUp} gateUpPathMode=${gateUpPathMode} inputDtype=${inputTensor.dtype} activationDtype=${activationDtype} ` +
      `gateDtype=${gateDtype} upDtype=${upDtype} hasQ4KMaterialization=${fusedGateUpWeights.hasQ4KMaterialization} ` +
      `dtypeMatches=${dtypeMatches} dtypeSupported=${dtypeSupported} hiddenSizeAligned32=${hiddenSizeAligned32} batchSize=${numTokens}`
    );
    const gateWeight = canUseFusedGateUp
      ? fusedGateWeight
      : getWeightBuffer(layerWeights.gate, 'ffn_gate');
    const upWeight = canUseFusedGateUp
      ? fusedUpWeight
      : getWeightBuffer(layerWeights.up, 'ffn_up');

    if (canUseFusedGateUp) {
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
      rmsNormWeightOffset: weightConfig.rmsNormWeightOffset,
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
