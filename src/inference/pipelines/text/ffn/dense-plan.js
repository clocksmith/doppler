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

export function hasQ4KMaterialization(weight) {
  return isWeightBuffer(weight) && !!weight.materializations?.q4k?.buffer;
}

export function isQ4KMatmulVariant(variant) {
  return typeof variant === 'string' && variant.startsWith('q4_');
}

export function normalizeFusedGateUpPipelineConstants(constants) {
  if (!constants || typeof constants !== 'object') {
    return null;
  }
  const colsPerWorkgroup = constants.COLS_PER_WG;
  const threadsPerCol = constants.THREADS_PER_COL ?? constants.THREADS_PER_COL_GEMV;
  const workgroupSize = constants.WORKGROUP_SIZE;
  const useFullBlockFastPath = constants.USE_FULL_BLOCK_FAST_PATH;
  if (
    colsPerWorkgroup === undefined &&
    threadsPerCol === undefined &&
    workgroupSize === undefined &&
    useFullBlockFastPath === undefined
  ) {
    return null;
  }
  return {
    ...(workgroupSize !== undefined ? { WORKGROUP_SIZE: workgroupSize } : {}),
    ...(colsPerWorkgroup !== undefined ? { COLS_PER_WG: colsPerWorkgroup } : {}),
    ...(threadsPerCol !== undefined ? { THREADS_PER_COL: threadsPerCol } : {}),
    ...(useFullBlockFastPath !== undefined ? { USE_FULL_BLOCK_FAST_PATH: useFullBlockFastPath } : {}),
  };
}

export function constantsEqual(a, b) {
  const aKeys = Object.keys(a).sort();
  const bKeys = Object.keys(b).sort();
  if (aKeys.length !== bKeys.length) return false;
  for (let i = 0; i < aKeys.length; i++) {
    if (aKeys[i] !== bKeys[i] || a[aKeys[i]] !== b[bKeys[i]]) {
      return false;
    }
  }
  return true;
}

export function resolveFusedGateUpPipelineConstants(options = {}) {
  const phase = options.phase ?? null;
  const layerIdx = Number.isFinite(options.layerIdx) ? options.layerIdx : 0;
  const kernelPath = options.kernelPath ?? null;
  if (!phase) {
    return null;
  }
  if (kernelPath) {
    const fusedConstants = normalizeFusedGateUpPipelineConstants(
      getKernelPathMatmulConstants('ffn_gate_up', phase, layerIdx, kernelPath)
    );
    if (fusedConstants) {
      return fusedConstants;
    }
    const gateConstants = normalizeFusedGateUpPipelineConstants(
      getKernelPathMatmulConstants('ffn_gate', phase, layerIdx, kernelPath)
    );
    const upConstants = normalizeFusedGateUpPipelineConstants(
      getKernelPathMatmulConstants('ffn_up', phase, layerIdx, kernelPath)
    );
    if (gateConstants || upConstants) {
      if (!gateConstants || !upConstants || !constantsEqual(gateConstants, upConstants)) {
        throw new Error(
          `[FFN] Fused gate/up requires matching gate and up kernel constants; ` +
          `got gate=${JSON.stringify(gateConstants)} up=${JSON.stringify(upConstants)}.`
        );
      }
      return gateConstants;
    }
  }
  const sessionConstants = getRuntimeConfig()
    .inference?.session?.fusedFfnQ4K?.[phase]?.pipelineConstants;
  return normalizeFusedGateUpPipelineConstants(sessionConstants);
}

export function resolveFusedGateUpVariant(options = {}) {
  const phase = options.phase ?? null;
  if (!phase) {
    return null;
  }
  const variant = getRuntimeConfig()
    .inference?.session?.fusedFfnQ4K?.[phase]?.variant;
  if (variant == null) {
    return null;
  }
  if (typeof variant !== 'string' || variant.length === 0) {
    throw new Error(`[FFN] fusedFfnQ4K.${phase}.variant must be a non-empty string or null.`);
  }
  return variant;
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

export function hasExplicitMatmulPrecision(role, phase, layerIdx, kernelPath) {
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

export function canFuseSplitPrefillF16GateUpPath(options = {}) {
  const kernelPath = options.kernelPath ?? null;
  const phase = options.phase ?? null;
  const layerIdx = Number.isFinite(options.layerIdx) ? options.layerIdx : 0;
  if (!kernelPath || phase !== 'prefill') {
    return false;
  }
  if (options.gateDtype !== 'f16' || options.upDtype !== 'f16') {
    return false;
  }
  if (
    hasExplicitMatmulPrecision('ffn_gate', phase, layerIdx, kernelPath)
    || hasExplicitMatmulPrecision('ffn_up', phase, layerIdx, kernelPath)
    || hasExplicitMatmulPrecision('ffn_down', phase, layerIdx, kernelPath)
  ) {
    return false;
  }

  const gateVariant = getKernelPathMatmulVariant('ffn_gate', phase, layerIdx, kernelPath);
  const upVariant = getKernelPathMatmulVariant('ffn_up', phase, layerIdx, kernelPath);
  return gateVariant != null
    && gateVariant === upVariant
    && !isQ4KMatmulVariant(gateVariant);
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
