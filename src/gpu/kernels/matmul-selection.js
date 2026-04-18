
import { getKernelCapabilities } from '../device.js';
import { getBuffer, getLayout } from '../weight-buffer.js';
import { log, trace, isTraceEnabled } from '../../debug/index.js';
import { acquireBuffer } from '../../memory/buffer-pool.js';
import { ALIGNMENT, QUANTIZATION, TILE_SIZES } from './constants.js';
import { getKernelConfig, hasRequiredFeatures } from './utils.js';
import { getKernelThresholds } from '../../config/schema/index.js';
import {
  getKernelPathMatmulConstants,
  getKernelPathMatmulVariant,
  getKernelPathStrict,
} from '../../config/kernel-path-loader.js';
import { selectRuleValue as selectKernelRuleValue } from './rule-registry.js';
import { selectRuleValue as selectSharedRuleValue } from '../../rules/rule-registry.js';
import { logKernelSelectionOnce } from '../kernel-selection-log.js';

// =============================================================================
// Q4K Variant Lookup Tables
// =============================================================================


function selectQ4KFusedVariant(isM1, wantF16Output, aDtype, phase) {
  const useF16A = wantF16Output && aDtype === 'f16';
  const useF16Out = wantF16Output && aDtype !== 'f16';
  const isPrefill = phase === 'prefill' && !isM1;
  return selectKernelRuleValue('matmul', 'q4kFusedVariant', { useF16A, useF16Out, isM1, isPrefill });
}


export function resolveMatmulPhase(M, phaseOverride = null) {
  if (phaseOverride != null) {
    if (phaseOverride !== 'decode' && phaseOverride !== 'prefill') {
      throw new Error(`[Matmul] Invalid phase override "${phaseOverride}". Expected "decode" or "prefill".`);
    }
    return phaseOverride;
  }
  return selectKernelRuleValue('matmul', 'phase', { isDecode: M === 1 });
}


export function resolveMatmulConstants(options, phase) {
  if (options.constants && Object.keys(options.constants).length > 0) {
    return options.constants;
  }
  const pathConstants = getKernelPathMatmulConstants(
    options.role,
    phase,
    options.layerIdx,
    options.kernelPath
  );
  if (pathConstants && Object.keys(pathConstants).length > 0) {
    return pathConstants;
  }
  return null;
}


function applyMatmulConstants(config, constants) {
  if (!constants) return config;

  let workgroupSize = config.workgroupSize;
  let variantMetadata = config.variantMetadata;
  let updated = false;

  if (Number.isFinite(constants.WORKGROUP_SIZE)) {
    workgroupSize = [constants.WORKGROUP_SIZE, workgroupSize[1], workgroupSize[2]];
    updated = true;
  }
  if (Number.isFinite(constants.TILE_M)) {
    variantMetadata = { ...(variantMetadata ?? {}), tileM: constants.TILE_M };
    updated = true;
  }
  if (Number.isFinite(constants.COLS_PER_WG)) {
    variantMetadata = { ...(variantMetadata ?? {}), colsPerWg: constants.COLS_PER_WG };
    updated = true;
  }
  if (Number.isFinite(constants.MULTICOL_COLS_PER_WG)) {
    variantMetadata = { ...(variantMetadata ?? {}), colsPerWg: constants.MULTICOL_COLS_PER_WG };
    updated = true;
  }

  if (!updated) return config;
  return { ...config, workgroupSize, variantMetadata };
}


export function getMatmulConfig(variant, constants) {
  return applyMatmulConstants(getKernelConfig('matmul', variant), constants);
}


export function isFusedQ4KDisabled(options = {}) {
  if (options.disableFusedQ4K === true) return true;
  const capabilities = getKernelCapabilities();
  const hasSubgroups = capabilities?.hasSubgroups === true;
  // Subgroups are the only hardware gate for fused Q4K. When available,
  // fused Q4K is always selectable regardless of kernel path contents.
  // The preferredWeightDtype rule decides whether to route weights to Q4K;
  // this function only gates whether the fused variant is *allowed*.
  return !hasSubgroups;
}


export function toMatmulDtype(dtype) {
  return selectSharedRuleValue('shared', 'dtype', 'matmulDtype', { dtype });
}


export function selectMatmulKernel(options = {}) {
  const capabilities = getKernelCapabilities();
  const {
    preferF16 = true,
    useVec4 = false,
    outputDtype = 'f32',
    aDtype = null,
    bDtype = null,
    isPrefill = false,
    prefillRows = 0,
    transposeB = true,
  } = options;
  const { tiledPrefillMinRows } = getKernelThresholds().matmul;

  const inputsAreF16 = aDtype === 'f16' && bDtype === 'f16';
  // F16 weights needing F32a path: weights are F16 and either activation is already F32,
  // or both inputs are F16 but output is F32 (activation will be cast to F32 by executeMatmul)
  const weightsAreF16 = bDtype === 'f16' && (aDtype !== 'f16' || outputDtype !== 'f16');
  const useF16Matmul = outputDtype === 'f16' && preferF16 && inputsAreF16 && capabilities.hasF16;
  const useF16wF32a = preferF16 && weightsAreF16 && capabilities.hasF16;
  const useTiled = isPrefill
    && useF16Matmul
    && transposeB === true
    && prefillRows >= tiledPrefillMinRows;

  return selectKernelRuleValue(
    'matmul',
    'matmulKernel',
    { useF16Matmul, useF16wF32a, useVec4, useTiled }
  );
}

// Debug counter to limit logging
let _transposeDebugCount = 0;
const MATMUL_OVERRIDE_WARNINGS = new Set();


export function resolveTransposeB(B, transposeBOption) {
  if (transposeBOption === 'auto') {
    const weightLayout = getLayout(B);
    const buffer = getBuffer(B);
    const isColMajor = weightLayout === 'column';
    const result = !isColMajor;
    if (isTraceEnabled('kernels') && _transposeDebugCount < 50) {
      _transposeDebugCount++;
      trace.kernels(`resolveTransposeB: layout=${weightLayout}, isColumnMajor=${isColMajor}, transposeB=${result}, bufSize=${buffer.size}`);
    }
    return result;
  }
  return transposeBOption;
}


export function validateMatmulDimensions(label, M, N, K) {
  if (!Number.isFinite(M) || !Number.isFinite(N) || !Number.isFinite(K)) {
    throw new Error(`[${label}] Invalid dimensions: M=${M}, N=${N}, K=${K}`);
  }
  if (M <= 0 || N <= 0 || K <= 0) {
    throw new Error(`[${label}] Dimensions must be positive: M=${M}, N=${N}, K=${K}`);
  }
}


export function validateMatmulOffsets(label, aOffset, bOffset, cOffset) {
  if (!Number.isFinite(aOffset) || aOffset < 0 ||
      !Number.isFinite(bOffset) || bOffset < 0 ||
      !Number.isFinite(cOffset) || cOffset < 0) {
    throw new Error(`[${label}] Invalid buffer offsets: aOffset=${aOffset}, bOffset=${bOffset}, cOffset=${cOffset}`);
  }

  const storageAlignment = ALIGNMENT.STORAGE;
  if (aOffset % storageAlignment !== 0 ||
      bOffset % storageAlignment !== 0 ||
      cOffset % storageAlignment !== 0) {
    throw new Error(
      `[${label}] Buffer offsets must be ${storageAlignment}-byte aligned: ` +
      `aOffset=${aOffset}, bOffset=${bOffset}, cOffset=${cOffset}`
    );
  }
}


export function getMatmulBindingSizes(label, A, B, M, N, K, aDtype, bDtype, transposeB, aOffset, bOffset) {
  const aBytesPerElem = aDtype === 'f16' ? 2 : 4;
  const aBindingSize = Math.ceil((M * K * aBytesPerElem) / 4) * 4;
  const aRequired = aOffset + aBindingSize;
  if (A.size < aRequired) {
    throw new Error(`[${label}] A buffer too small: ${A.size} < ${aRequired} (M=${M}, K=${K}, aDtype=${aDtype})`);
  }

  const QK_K = TILE_SIZES.Q4K_SUPER_BLOCK_SIZE;
  const Q4K_BLOCK_BYTES = QUANTIZATION.Q4K_BLOCK_BYTES;

  let bBindingSize;
  let bRequired;

  if (bDtype === 'q4k') {
    const numBlocksPerRow = Math.ceil(K / QK_K);
    bBindingSize = Math.ceil((N * numBlocksPerRow * Q4K_BLOCK_BYTES) / 4) * 4;
    bRequired = bOffset + bBindingSize;
  } else {
    const bBytesPerElem = bDtype === 'f16' ? 2 : 4;
    const bElements = transposeB ? N * K : K * N;
    bBindingSize = Math.ceil((bElements * bBytesPerElem) / 4) * 4;
    bRequired = bOffset + bBindingSize;
  }

  if (B.size < bRequired) {
    throw new Error(
      `[${label}] B buffer too small: ${B.size} < ${bRequired} ` +
      `(N=${N}, K=${K}, bDtype=${bDtype}, transposeB=${transposeB})`
    );
  }

  return { aBindingSize, bBindingSize };
}


function isQ4KFusedVariant(variant) {
  return variant.startsWith('q4_fused');
}


function isGemvVariant(variant) {
  return variant.startsWith('gemv');
}


function supportsF16Input(variant) {
  return variant === 'f16' || variant === 'f16_vec4' || variant === 'f16_tiled' || variant.endsWith('_f16a');
}

export function requiresF32Input(variant) {
  return !supportsF16Input(variant);
}

function resolveRequiredWeightDtype(config) {
  return config?.weightDtype ?? null;
}


function resolveMatmulOverride(
  variantOverride,
  M,
  K,
  aDtype,
  bDtype,
  requestedOutputDtype,
  capabilities,
  strict,
  fusedQ4KDisabled
) {
  const override = variantOverride.trim();
  if (!override) return null;

  const failOrWarn = (message) => {
    if (strict) {
      throw new Error(message);
    }
    if (!MATMUL_OVERRIDE_WARNINGS.has(message)) {
      MATMUL_OVERRIDE_WARNINGS.add(message);
      log.warn('Matmul', message);
    }
    return null;
  };

  let config;
  try {
    config = getKernelConfig('matmul', override);
  } catch {
    return failOrWarn(`Unknown matmul kernel variant "${variantOverride}".`);
  }

  const outputDtype = config.outputDtype;
  if (!outputDtype) {
    return failOrWarn(`Matmul kernel "${variantOverride}" is missing outputDtype.`);
  }
  if (requestedOutputDtype && outputDtype !== requestedOutputDtype) {
    return failOrWarn(
      `Matmul kernel "${variantOverride}" outputs ${outputDtype} but ${requestedOutputDtype} was requested. [M=${M} K=${K} aDtype=${aDtype} bDtype=${bDtype}]`
    );
  }

  const requiredWeightDtype = resolveRequiredWeightDtype(config);
  const weightDtypeOk = !requiredWeightDtype
    || bDtype === requiredWeightDtype;
  if (!weightDtypeOk) {
    const overridePolicy = selectKernelRuleValue(
      'matmul', 'weightDtypeMismatchPolicy',
      { requiredWeightDtype, actualWeightDtype: bDtype }
    );
    if (overridePolicy === 'fallthrough') {
      return null;
    }
    return failOrWarn(
      `Matmul kernel "${variantOverride}" requires ${requiredWeightDtype} weights but B dtype is ${bDtype}.`
    );
  }

  if (supportsF16Input(override) && aDtype !== 'f16') {
    return failOrWarn(`Matmul kernel "${variantOverride}" requires f16 activations but A dtype is ${aDtype}.`);
  }

  if (override.includes('vec4') && (K % 4 !== 0)) {
    return failOrWarn(`Matmul kernel "${variantOverride}" requires K divisible by 4 but got K=${K}.`);
  }

  if (!hasRequiredFeatures(config.requires, capabilities)) {
    return failOrWarn(`Matmul kernel "${variantOverride}" requires unsupported GPU features.`);
  }

  const useQ4KFused = isQ4KFusedVariant(override);
  if (useQ4KFused) {
    if (bDtype !== 'q4k') {
      return failOrWarn(`Matmul kernel "${variantOverride}" requires Q4K weights but B dtype is ${bDtype}.`);
    }
    if (fusedQ4KDisabled) {
      return failOrWarn(`Matmul kernel "${variantOverride}" blocked by kernel path (fused Q4K disabled).`);
    }
  }

  const useGemv = isGemvVariant(override);
  if (useGemv && M !== 1) {
    return failOrWarn(`Matmul kernel "${variantOverride}" requires M=1 but got M=${M}.`);
  }

  return { variant: override, useQ4KFused, useGemv };
}

function resolveGemvPathVariant(pathVariant, aDtype, requestedOutputDtype, N, multicolThreshold) {
  const useF16GemvPath = pathVariant === 'gemv_f16a' && aDtype === 'f16' && requestedOutputDtype === 'f16';
  const useF32GemvPath = pathVariant === 'gemv' && aDtype === 'f32';
  const useMulticol = N > multicolThreshold;
  return selectKernelRuleValue(
    'matmul',
    'gemvPathVariant',
    { useF16GemvPath, useF32GemvPath, useMulticol, pathVariant }
  );
}

function selectGemvVariant(useF16Gemv, useF32Gemv, hasSubgroups, useVec4, N, multicolThreshold) {
  const useMulticol = N > multicolThreshold;
  return selectKernelRuleValue(
    'matmul',
    'gemvVariant',
    { hasSubgroups, useF16Gemv, useF32Gemv, useVec4, useMulticol }
  );
}


export function selectMatmulVariantAndFlags(mode, M, N, K, aDtype, bDtype, transposeB, requestedOutputDtype, options) {
  const capabilities = getKernelCapabilities();
  const strict = getKernelPathStrict();
  const phase = resolveMatmulPhase(M, options.phaseOverride ?? null);
  let pathVariant = getKernelPathMatmulVariant(options.role, phase, options.layerIdx, options.kernelPath);
  const hadPathVariant = Boolean(pathVariant);

  if (pathVariant && !strict && M === 1 && bDtype === 'f16' && capabilities.hasSubgroups) {
    const { multicolThreshold } = getKernelThresholds().matmul;
    pathVariant = resolveGemvPathVariant(pathVariant, aDtype, requestedOutputDtype, N, multicolThreshold);
  }

  const fusedQ4KDisabled = isFusedQ4KDisabled(options);

  if (pathVariant) {
    const override = resolveMatmulOverride(
      pathVariant,
      M,
      K,
      aDtype,
      bDtype,
      requestedOutputDtype,
      capabilities,
      strict,
      fusedQ4KDisabled
    );
    if (!override && strict) {
      // When weights resolved to Q4K (precision upgrade) but the path variant requires F16,
      // fall through to auto-selection rather than throwing. The auto path will pick
      // the correct fused Q4K variant for the actual weight dtype.
      if (bDtype === 'q4k') {
        logKernelSelectionOnce('matmul', {
          variant: pathVariant,
          reason: 'path_override_q4k_fallthrough',
        });
      } else {
        throw new Error(`[Matmul] Path variant "${pathVariant}" rejected for role=${options.role ?? '?'} layerIdx=${options.layerIdx ?? '?'} phase=${phase} M=${M} K=${K} aDtype=${aDtype} bDtype=${bDtype} outDtype=${requestedOutputDtype}`);
      }
    }
    if (override) {
      if (
        phase === 'prefill'
        && override.variant === 'f16_tiled'
        && aDtype === 'f16'
        && bDtype === 'f16'
        && requestedOutputDtype === 'f16'
        && transposeB === true
      ) {
        const { tiledPrefillMinRows } = getKernelThresholds().matmul;
        if (M <= tiledPrefillMinRows) {
          const adaptiveVariant = selectMatmulKernel({
            ...options,
            aDtype,
            bDtype,
            outputDtype: requestedOutputDtype,
            isPrefill: true,
            prefillRows: M,
            transposeB,
          });
          if (adaptiveVariant !== override.variant) {
            const adaptiveSelection = {
              variant: adaptiveVariant,
              useQ4KFused: false,
              useGemv: false,
            };
            logKernelSelectionOnce('matmul', {
              variant: adaptiveSelection.variant,
              reason: 'path_override_adaptive_fallback',
            });
            return adaptiveSelection;
          }
        }
      }
      logKernelSelectionOnce('matmul', {
        variant: override.variant,
        reason: 'path_override',
      });
      return override;
    }
  }

  const fusedAllowed = !fusedQ4KDisabled;
  const isQ4K = bDtype === 'q4k';
  const wantF16Output = requestedOutputDtype === 'f16' && capabilities.hasF16;
  let q4kVariant = isQ4K && capabilities.hasSubgroups && fusedAllowed
    ? selectQ4KFusedVariant(M === 1, wantF16Output, aDtype, phase)
    : null;
  // Opt-in override: when useTiledQ4KPrefill is set on options, replace the
  // default batched f16-output variant with the register-tiled variant
  // whenever the shape (Q4_K weights, f16 output, prefill M>=16) and
  // capability (hasF16) match. Matches both batched_f16 (f32 activations)
  // and batched_f16a (f16 activations); the tiled kernel takes f32 A and
  // f16 C per the q4_fused_batched_f16 binding contract. Decoupled from
  // subgroups because the tiled kernel does not use subgroup reduction.
  if (
    q4kVariant === 'q4_fused_batched_f16'
    && options.useTiledQ4KPrefill === true
    && phase === 'prefill'
    && M >= 16
    && aDtype === 'f32'
    && wantF16Output
    && capabilities.hasF16 === true
  ) {
    q4kVariant = 'q4_fused_prefill_tiled_f16';
  }
  // WideTile override (ported from ORT MatMulNBitsWideTile): register-tiled,
  // 1 thread per output column × TILE_M rows accumulated in registers.
  // Orthogonal to useTiledQ4KPrefill — if both flags are set, WideTile wins
  // because it has materially fewer workgroups at Gemma-4-scale prefill.
  // Routes to the f16-output or f32-output WideTile variant depending on
  // what the caller asked for (Gemma 4's FFN picks q4_fused_batched with
  // f32 output; attention projections may want f16).
  if (
    options.useWideTileQ4KPrefill === true
    && phase === 'prefill'
    && M >= 4
    && aDtype === 'f32'
    && capabilities.hasF16 === true
  ) {
    if (
      wantF16Output
      && (q4kVariant === 'q4_fused_batched_f16' || q4kVariant === 'q4_fused_prefill_tiled_f16')
    ) {
      q4kVariant = 'q4_fused_widetile_f16';
    } else if (
      !wantF16Output
      && q4kVariant === 'q4_fused_batched'
    ) {
      q4kVariant = 'q4_fused_widetile';
    }
  }

  const effectiveBDtype = bDtype === 'q4k' ? 'f32' : bDtype;
  const matmulVariant = selectMatmulKernel({
    ...options,
    aDtype: aDtype === 'q4k' ? 'f32' : aDtype,
    bDtype: effectiveBDtype,
    outputDtype: requestedOutputDtype,
    isPrefill: M > 1,
    prefillRows: M,
    transposeB,
  });

  const canGemv = M === 1 && effectiveBDtype === 'f16' && capabilities.hasF16;
  const useF16Gemv = canGemv && aDtype === 'f16' && wantF16Output;
  // F32 GEMV: activation is F32, or activation is F16 with F32 output (will be cast to F32)
  const useF32Gemv = canGemv && (aDtype === 'f32' || (aDtype === 'f16' && !wantF16Output));
  const useGemv = useF16Gemv || useF32Gemv;
  const useVec4 = (K % 4 === 0);
  const { multicolThreshold } = getKernelThresholds().matmul;
  const gemvVariant = useGemv
    ? selectGemvVariant(useF16Gemv, useF32Gemv, capabilities.hasSubgroups, useVec4, N, multicolThreshold)
    : null;

  const selection = selectKernelRuleValue(
    'matmul',
    'matmulSelection',
    { isQ4K, hasSubgroups: capabilities.hasSubgroups, fusedAllowed, useGemv, q4kVariant, gemvVariant, matmulVariant }
  );
  const reason = selection.useQ4KFused
    ? 'q4k_fused'
    : selection.useGemv
      ? 'gemv'
      : hadPathVariant
        ? 'path_override_fallback'
        : 'default';

  logKernelSelectionOnce('matmul', {
    variant: selection.variant,
    reason,
  });

  return selection;
}


export function resolveMatmulOutput(variant, M, N, outputBuffer) {
  const config = getKernelConfig('matmul', variant);
  if (!config.outputDtype) {
    throw new Error(`Matmul kernel "${variant}" is missing outputDtype.`);
  }
  const outputsF16 = config.outputDtype === 'f16';
  const elementSize = outputsF16 ? 2 : 4;

  const actualOutputDtype = selectSharedRuleValue('shared', 'dtype', 'f16OrF32FromDtype', {
    dtype: config.outputDtype,
  });
  const outputSize = M * N * elementSize;
  const cBindingSize = Math.ceil(outputSize / 4) * 4;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'matmul_output');
  return { output, outputSize, cBindingSize, actualOutputDtype };
}
