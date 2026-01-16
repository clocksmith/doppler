

import { getDevice, getKernelCapabilities } from '../device.js';
import { createTensor } from '../tensor.js';
import { getBuffer, getLayout, getWeightDtype } from '../weight-buffer.js';
import { log, trace, isTraceEnabled } from '../../debug/index.js';
import { acquireBuffer, releaseBuffer } from '../buffer-pool.js';
import { KernelBase } from './kernel-base.js';
import { ALIGNMENT, GPU_LIMITS, QUANTIZATION, TILE_SIZES } from './constants.js';
import { getKernelConfig, createUniformBufferWithView, getOrCreateBindGroupLayout, getCachedPipeline, createPipeline, getPipelineFast, hasRequiredFeatures } from './utils.js';
import { releaseUniformBuffer } from '../uniform-cache.js';
import { getKernelThresholds } from '../../config/schema/index.js';
import { getKernelPathMatmulConstants, getKernelPathMatmulVariant, getKernelPathStrict, isActiveKernelPathFusedQ4K } from '../../config/kernel-path-loader.js';
import { castF16ToF32, recordCastF16ToF32 } from './cast.js';
import { selectByRules } from './rule-matcher.js';
import { logKernelSelectionOnce } from '../kernel-selection-log.js';

// =============================================================================
// Q4K Variant Lookup Tables
// =============================================================================


function selectQ4KFusedVariant(isM1, wantF16Output, aDtype) {
  const useF16A = wantF16Output && aDtype === 'f16';
  const useF16Out = wantF16Output && aDtype !== 'f16';
  const rules = [
    { match: { useF16A: true, isM1: true }, value: 'q4_fused_multicol_f16a' },
    { match: { useF16A: true }, value: 'q4_fused_batched_f16a' },
    { match: { useF16Out: true, isM1: true }, value: 'q4_fused_multicol_f16' },
    { match: { useF16Out: true }, value: 'q4_fused_batched_f16' },
    { match: { isM1: true }, value: 'q4_fused_multicol' },
    { match: {}, value: 'q4_fused_batched' },
  ];
  return selectByRules(rules, { useF16A, useF16Out, isM1 });
}


function resolveMatmulConstants(options, phase) {
  if (options.constants && Object.keys(options.constants).length > 0) {
    return options.constants;
  }
  const pathConstants = getKernelPathMatmulConstants(options.role, phase, options.layerIdx);
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

  if (!updated) return config;
  return { ...config, workgroupSize, variantMetadata };
}


export function isFusedQ4KDisabled() {
  // Check window override first (debug flag)
  const debugFlags = typeof window !== 'undefined'
    ?  (window)
    : null;
  if (debugFlags?.DOPPLER_DISABLE_FUSED_Q4K) return true;

  // Check active kernel path - if explicitly set to dequant, disable fused
  if (!isActiveKernelPathFusedQ4K()) return true;

  return false;
}




function toMatmulDtype(dtype) {
  if (dtype === 'f16' || dtype === 'bf16') return 'f16';  // bf16 weights use f16 kernel
  if (dtype === 'q4k') return 'q4k';
  return 'f32';
}


export function selectMatmulKernel(options = {}) {
  const capabilities = getKernelCapabilities();
  const {
    preferF16 = true,
    useVec4 = false,
    outputDtype = 'f32',
    aDtype = null,
    bDtype = null,
  } = options;

  const inputsAreF16 = aDtype === 'f16' && bDtype === 'f16';
  const weightsAreF16 = bDtype === 'f16' && aDtype !== 'f16';
  const useF16Matmul = outputDtype === 'f16' && preferF16 && inputsAreF16 && capabilities.hasF16;
  const useF16wF32a = preferF16 && weightsAreF16 && capabilities.hasF16;

  const rules = [
    { match: { useF16Matmul: true, useVec4: true }, value: 'f16_vec4' },
    { match: { useF16Matmul: true }, value: 'f16' },
    { match: { useF16wF32a: true }, value: 'f16w_f32a' },
    { match: {}, value: 'f32' },
  ];

  return selectByRules(
    rules,
    { useF16Matmul, useF16wF32a, useVec4 }
  );
}

class MatmulKernel extends KernelBase {
  
  async getPipeline(variant) {
    return this.getPipelineFor('matmul', variant);
  }

  
  dispatch(pipeline, bindGroup, workgroups) {
    this.dispatchKernel(pipeline, bindGroup, workgroups, 'matmul');
  }

  
  record(recorder, pipeline, bindGroup, workgroups) {
    this.recordKernel(recorder, pipeline, bindGroup, workgroups, 'matmul');
  }
}



// Debug counter to limit logging
let _transposeDebugCount = 0;
const MATMUL_OVERRIDE_WARNINGS = new Set();


function resolveTransposeB(B, transposeBOption) {
  if (transposeBOption === 'auto') {
    // Get layout from WeightBuffer (buffer-dtypes WeakMap removed)
    const weightLayout = getLayout(B);
    const buffer = getBuffer(B);
    // WeightBuffer has explicit layout; raw GPUBuffer defaults to row-major
    const isColMajor = weightLayout === 'column';
    const result = !isColMajor;
    // Log first 50 calls to avoid flooding
    if (isTraceEnabled('kernels') && _transposeDebugCount < 50) {
      _transposeDebugCount++;
      trace.kernels(`resolveTransposeB: layout=${weightLayout}, isColumnMajor=${isColMajor}, transposeB=${result}, bufSize=${buffer.size}`);
    }
    return result;
  }
  return transposeBOption;
}


function validateMatmulDimensions(label, M, N, K) {
  if (!Number.isFinite(M) || !Number.isFinite(N) || !Number.isFinite(K)) {
    throw new Error(`[${label}] Invalid dimensions: M=${M}, N=${N}, K=${K}`);
  }
  if (M <= 0 || N <= 0 || K <= 0) {
    throw new Error(`[${label}] Dimensions must be positive: M=${M}, N=${N}, K=${K}`);
  }
}


function validateMatmulOffsets(label, aOffset, bOffset, cOffset) {
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


function getMatmulBindingSizes(label, A, B, M, N, K, aDtype, bDtype, transposeB, aOffset, bOffset) {
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
  return variant === 'f16' || variant === 'f16_vec4' || variant.endsWith('_f16a');
}

function requiresF32Input(variant) {
  return !supportsF16Input(variant);
}


function resolveMatmulOverride(variantOverride, M, K, aDtype, bDtype, requestedOutputDtype, capabilities, strict) {
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

  const outputDtype = config.outputDtype ?? 'f32';
  if (requestedOutputDtype && outputDtype !== requestedOutputDtype) {
    return failOrWarn(
      `Matmul kernel "${variantOverride}" outputs ${outputDtype} but ${requestedOutputDtype} was requested.`
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
    if (isFusedQ4KDisabled()) {
      return failOrWarn(`Matmul kernel "${variantOverride}" blocked by DOPPLER_DISABLE_FUSED_Q4K.`);
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
  const rules = [
    { match: { useF16GemvPath: true, useMulticol: true }, value: 'gemv_subgroup_multicol_f16a' },
    { match: { useF16GemvPath: true }, value: 'gemv_subgroup_f16a' },
    { match: { useF32GemvPath: true, useMulticol: true }, value: 'gemv_subgroup_multicol' },
    { match: { useF32GemvPath: true }, value: 'gemv_subgroup' },
    { match: {}, value: pathVariant },
  ];

  return selectByRules(
    rules,
    { useF16GemvPath, useF32GemvPath, useMulticol }
  );
}

function selectGemvVariant(useF16Gemv, useF32Gemv, hasSubgroups, useVec4, N, multicolThreshold) {
  const useMulticol = N > multicolThreshold;
  const rules = [
    { match: { hasSubgroups: true, useF16Gemv: true, useVec4: true, useMulticol: false }, value: 'gemv_subgroup_vec4_f16a' },
    { match: { hasSubgroups: true, useF32Gemv: true, useVec4: true, useMulticol: false }, value: 'gemv_subgroup_vec4' },
    { match: { hasSubgroups: true, useF16Gemv: true, useMulticol: true }, value: 'gemv_subgroup_multicol_f16a' },
    { match: { hasSubgroups: true, useF16Gemv: true }, value: 'gemv_subgroup_f16a' },
    { match: { hasSubgroups: true, useF32Gemv: true, useMulticol: true }, value: 'gemv_subgroup_multicol' },
    { match: { hasSubgroups: true, useF32Gemv: true }, value: 'gemv_subgroup' },
    { match: { useF16Gemv: true }, value: 'gemv_f16a' },
    { match: { useF32Gemv: true }, value: 'gemv' },
    { match: {}, value: 'f32' },
  ];

  return selectByRules(
    rules,
    { hasSubgroups, useF16Gemv, useF32Gemv, useVec4, useMulticol }
  );
}


function selectMatmulVariantAndFlags(mode, M, N, K, aDtype, bDtype, transposeB, requestedOutputDtype, options) {
  const capabilities = getKernelCapabilities();
  const strict = getKernelPathStrict();
  const phase = M === 1 ? 'decode' : 'prefill';
  let pathVariant = getKernelPathMatmulVariant(options.role, phase, options.layerIdx);
  const hadPathVariant = Boolean(pathVariant);

  if (pathVariant && !strict && M === 1 && bDtype === 'f16' && capabilities.hasSubgroups) {
    const { multicolThreshold } = getKernelThresholds().matmul;
    pathVariant = resolveGemvPathVariant(pathVariant, aDtype, requestedOutputDtype, N, multicolThreshold);
  }

  if (pathVariant) {
    const override = resolveMatmulOverride(pathVariant, M, K, aDtype, bDtype, requestedOutputDtype, capabilities, strict);
    if (override) {
      logKernelSelectionOnce('matmul', {
        variant: override.variant,
        reason: 'path_override',
      });
      return override;
    }
  }

  const allowFused = !isFusedQ4KDisabled();
  const canFused = bDtype === 'q4k' && capabilities.hasSubgroups && allowFused;
  const wantF16Output = requestedOutputDtype === 'f16' && capabilities.hasF16;
  const q4kVariant = canFused ? selectQ4KFusedVariant(M === 1, wantF16Output, aDtype) : null;

  const effectiveBDtype = bDtype === 'q4k' ? 'f32' : bDtype;
  const matmulVariant = selectMatmulKernel({
    ...options,
    aDtype: aDtype === 'q4k' ? 'f32' : aDtype,
    bDtype: effectiveBDtype,
    outputDtype: requestedOutputDtype,
  });

  const canGemv = M === 1 && effectiveBDtype === 'f16' && capabilities.hasF16;
  const useF16Gemv = canGemv && aDtype === 'f16' && wantF16Output;
  const useF32Gemv = canGemv && aDtype === 'f32';
  const useGemv = useF16Gemv || useF32Gemv;
  const useVec4 = (K % 4 === 0);
  const { multicolThreshold } = getKernelThresholds().matmul;
  const gemvVariant = useGemv
    ? selectGemvVariant(useF16Gemv, useF32Gemv, capabilities.hasSubgroups, useVec4, N, multicolThreshold)
    : null;

  const rules = [
    { match: { canFused: true }, value: { variant: q4kVariant, useQ4KFused: true, useGemv: false } },
    { match: { useGemv: true }, value: { variant: gemvVariant, useQ4KFused: false, useGemv: true } },
    { match: {}, value: { variant: matmulVariant, useQ4KFused: false, useGemv: false } },
  ];

  const selection = selectByRules(
    rules,
    { canFused, useGemv }
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


function resolveMatmulOutput(variant, M, N, outputBuffer) {
  // Use kernel config's outputDtype instead of string matching
  const config = getKernelConfig('matmul', variant);
  const outputsF16 = config.outputDtype === 'f16';
  const elementSize = outputsF16 ? 2 : 4;
  
  const actualOutputDtype = outputsF16 ? 'f16' : 'f32';
  const outputSize = M * N * elementSize;
  const cBindingSize = Math.ceil(outputSize / 4) * 4;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'matmul_output');
  return { output, outputSize, cBindingSize, actualOutputDtype };
}


function calculateMatmulDispatch(variant, useQ4KFused, useGemv, M, N, config) {
  const maxWorkgroups = GPU_LIMITS.MAX_WORKGROUPS;
  const [wgX, wgY] = config.workgroupSize;
  let workgroupsX = 1;
  let workgroupsY = 1;
  
  let uniformWorkgroupsX;

  // Get colsPerWg from variantMetadata (default 4 for non-multicol GEMV)
  const colsPerWg = config.variantMetadata?.colsPerWg ?? 4;
  // Get tileM from variantMetadata (default 4 for batched variants)
  const tileM = config.variantMetadata?.tileM ?? 4;

  if (useGemv && variant.startsWith('gemv_subgroup')) {
    const gemvWorkgroupsX = Math.ceil(N / colsPerWg);
    if (gemvWorkgroupsX > maxWorkgroups) {
      workgroupsX = maxWorkgroups;
      workgroupsY = Math.ceil(gemvWorkgroupsX / maxWorkgroups);
    } else {
      workgroupsX = gemvWorkgroupsX;
      workgroupsY = 1;
    }
    uniformWorkgroupsX = workgroupsX;
    return { workgroups: [workgroupsX, workgroupsY, 1], uniformWorkgroupsX };
  }

  if (useQ4KFused) {
    if (variant === 'q4_fused') {
      workgroupsX = N;
      workgroupsY = 1;
    } else if (config.variantMetadata?.colsPerWg) {
      // Multicol variants: q4_fused_multicol, q4_fused_multicol_f16
      workgroupsX = Math.ceil(N / colsPerWg);
      workgroupsY = 1;
    } else if (config.variantMetadata?.tileM) {
      // Batched variants: q4_fused_batched, q4_fused_batched_f16
      workgroupsX = N;
      workgroupsY = Math.ceil(M / tileM);
    } else {
      // Fallback for q4_fused (1 col per workgroup)
      workgroupsX = N;
      workgroupsY = 1;
    }
  } else if (useGemv) {
    workgroupsX = N;
    workgroupsY = 1;
  } else {
    const colsPerThread = variant === 'f16_vec4' ? 4 : 1;
    workgroupsX = Math.ceil(M / wgX);
    workgroupsY = Math.ceil(N / (wgY * colsPerThread));
  }

  return { workgroups: [workgroupsX, workgroupsY, 1], uniformWorkgroupsX };
}


function createMatmulUniformBuffer(label, M, N, K, alpha, useQ4KFused, transposeB, uniformWorkgroupsX, recorder, device) {
  // Shader struct is 32 bytes: M, N, K, alpha, transpose_b/num_blocks, workgroups_x/_pad0, _pad1, _pad2
  const uniformSize = 32;

  return createUniformBufferWithView(
    label,
    uniformSize,
    (view) => {
      view.setUint32(0, M, true);
      view.setUint32(4, N, true);
      view.setUint32(8, K, true);
      view.setFloat32(12, alpha, true);
      if (useQ4KFused) {
        const numBlocksPerRow = Math.ceil(K / TILE_SIZES.Q4K_SUPER_BLOCK_SIZE);
        view.setUint32(16, numBlocksPerRow, true);
      } else {
        view.setUint32(16, transposeB ? 1 : 0, true);
      }
      // workgroups_x (or _pad0 if not needed)
      view.setUint32(20, uniformWorkgroupsX ?? 0, true);
      // _pad1, _pad2 - leave as zeros (already zero-initialized)
    },
    recorder,
    device
  );
}


export function createMatmulBindGroupLayout() {
  return getOrCreateBindGroupLayout('matmul_bind_group_layout', [
    {
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: 'uniform' },
    },
    {
      binding: 1,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: 'read-only-storage' },
    },
    {
      binding: 2,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: 'read-only-storage' },
    },
    {
      binding: 3,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: 'storage' },
    },
  ]);
}

// Debug counter for runMatmul
let _runMatmulDebugCount = 0;


export async function runMatmul(A, B, M, N, K, options = {}) {
  const device = getDevice();
  const {
    alpha = 1.0,
    outputBuffer = null,
    transposeB: transposeBOption = true,  // Default: assume row-major (SafeTensors)
    aOffset = 0,
    bOffset = 0,
    cOffset = 0,
  } = options;

  // Extract underlying GPUBuffer from WeightBuffer if needed
  const bBuffer = getBuffer(B);
  const weightDtype = getWeightDtype(B);

  // Debug: log what options are being passed
  if (isTraceEnabled('kernels') && _runMatmulDebugCount < 20) {
    _runMatmulDebugCount++;
    const weightLayout = getLayout(B);
    trace.kernels(`runMatmul: M=${M}, N=${N}, K=${K}, transposeBOption=${transposeBOption}, weightLayout=${weightLayout}, weightDtype=${weightDtype}`);
  }

  const transposeB = resolveTransposeB(B, transposeBOption);

  validateMatmulDimensions('runMatmul', M, N, K);

  // Get activation dtype from Tensor, weight dtype from WeightBuffer or options
  const aDtype = toMatmulDtype(A.dtype);
  // Prefer WeightBuffer dtype, fall back to options.bDtype
  const bDtype = toMatmulDtype(weightDtype ?? options.bDtype);
  const requestedOutputDtype = options.outputDtype || A.dtype;

  // Warn if B buffer dtype is unknown - this can cause wrong kernel selection
  if (isTraceEnabled('kernels') && !weightDtype && !options.bDtype && M <= 2) {
    log.warn('Matmul', `runMatmul: B buffer dtype unknown! size=${bBuffer.size}, M=${M}, N=${N}, K=${K}. Assuming f32.`);
  }

  validateMatmulOffsets('runMatmul', aOffset, bOffset, cOffset);

  const { variant, useQ4KFused, useGemv } = selectMatmulVariantAndFlags(
    'run',
    M,
    N,
    K,
    aDtype,
    bDtype,
    transposeB,
    requestedOutputDtype,
    options
  );

  const phase = M === 1 ? 'decode' : 'prefill';
  const constants = resolveMatmulConstants(options, phase);

  let matmulInput = A;
  let matmulADtype = aDtype;
  let castedInput = null;
  if (matmulADtype === 'f16' && requiresF32Input(variant)) {
    if (isTraceEnabled('kernels')) {
      trace.kernels(`Matmul: casting f16 activations to f32 for variant=${variant}`);
    }
    castedInput = await castF16ToF32(A);
    matmulInput = castedInput;
    matmulADtype = 'f32';
  }

  const { aBindingSize, bBindingSize } = getMatmulBindingSizes(
    'runMatmul',
    matmulInput.buffer,
    bBuffer,
    M,
    N,
    K,
    matmulADtype,
    bDtype,
    transposeB,
    aOffset,
    bOffset
  );

  if (isTraceEnabled('kernels') && bDtype === 'q4k') {
    if (useQ4KFused) {
      trace.kernels(`Q4K FUSED: M=${M}, N=${N}, K=${K}, variant=${variant} (WARNING: 2.3x slower than dequant)`);
    } else {
      trace.kernels(`Q4K DEQUANT: M=${M}, N=${N}, K=${K}, will dequant first then matmul with variant=${variant}`);
    }
  }

  // Debug: Log kernel selection for large matmuls (lm_head projection)
  if (isTraceEnabled('kernels') && N > 100000) {
    trace.kernels(`MATMUL_LARGE: N=${N}, variant=${variant}, aDtype=${aDtype}, bDtype=${bDtype}, transposeB=${transposeB}`);
  }

  const config = applyMatmulConstants(getKernelConfig('matmul', variant), constants);
  const kernel = new MatmulKernel(device);

  // Fast path: use synchronously cached pipeline if available
  let pipeline = getCachedPipeline('matmul', variant, constants);
  if (!pipeline) {
    pipeline = await createPipeline('matmul', variant, null, constants);
  }

  const { output: C, outputSize, cBindingSize, actualOutputDtype } = resolveMatmulOutput(
    variant,
    M,
    N,
    outputBuffer
  );

  if (!Number.isFinite(outputSize) || outputSize <= 0) {
    throw new Error(`[runMatmul] Invalid output size: ${outputSize} (M=${M}, N=${N})`);
  }

  const cRequired = cOffset + cBindingSize;
  if (C.size < cRequired) {
    throw new Error(`[runMatmul] Output buffer too small: ${C.size} < ${cRequired} (M=${M}, N=${N})`);
  }

  const dispatchPlan = calculateMatmulDispatch(variant, useQ4KFused, useGemv, M, N, config);
  const uniformBuffer = createMatmulUniformBuffer(
    'matmul_uniforms',
    M,
    N,
    K,
    alpha,
    useQ4KFused,
    transposeB,
    dispatchPlan.uniformWorkgroupsX,
    null,
    device
  );

  // Q4K F16 variants use binding 4 for output (F16), all other variants use binding 3
  const isQ4KF16 = variant === 'q4_fused_multicol_f16' ||
    variant === 'q4_fused_f16a' ||
    variant === 'q4_fused_batched_f16' ||
    variant === 'q4_fused_multicol_f16a' ||
    variant === 'q4_fused_batched_f16a';
  
  const entries = [
    { binding: 0, resource: { buffer: uniformBuffer } },
    { binding: 1, resource: { buffer: matmulInput.buffer, offset: aOffset, size: aBindingSize } },
    { binding: 2, resource: { buffer: bBuffer, offset: bOffset, size: bBindingSize } },
  ];

  if (isQ4KF16) {
    entries.push({ binding: 4, resource: { buffer: C, offset: cOffset, size: cBindingSize } });
  } else {
    entries.push({ binding: 3, resource: { buffer: C, offset: cOffset, size: cBindingSize } });
  }

  const bindGroup = device.createBindGroup({
    label: 'matmul_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });

  kernel.dispatch(pipeline, bindGroup, dispatchPlan.workgroups);
  releaseUniformBuffer(uniformBuffer);
  if (castedInput) {
    releaseBuffer(castedInput.buffer);
  }

  return createTensor(C, actualOutputDtype, [M, N], 'matmul_output');
}

// Debug counter for recordMatmul
let _recordMatmulDebugCount = 0;


export async function recordMatmul(recorder, A, B, M, N, K, options = {}) {
  const device = recorder.device;
  const {
    alpha = 1.0,
    outputBuffer = null,
    transposeB: transposeBOption = true,  // Default: assume row-major (SafeTensors)
    aOffset = 0,
    bOffset = 0,
    cOffset = 0,
  } = options;

  // Extract underlying GPUBuffer from WeightBuffer if needed
  const bBuffer = getBuffer(B);
  const weightDtype = getWeightDtype(B);

  // Debug: log what options are being passed
  if (isTraceEnabled('kernels') && _recordMatmulDebugCount < 20) {
    _recordMatmulDebugCount++;
    const weightLayout = getLayout(B);
    trace.kernels(`recordMatmul: M=${M}, N=${N}, K=${K}, transposeBOption=${transposeBOption}, weightLayout=${weightLayout}, weightDtype=${weightDtype}`);
  }

  const transposeB = resolveTransposeB(B, transposeBOption);
  validateMatmulDimensions('recordMatmul', M, N, K);

  // Get activation dtype from Tensor, weight dtype from WeightBuffer or options
  const aDtype = toMatmulDtype(A.dtype);
  // Prefer WeightBuffer dtype, fall back to options.bDtype
  const bDtype = toMatmulDtype(weightDtype ?? options.bDtype);
  const requestedOutputDtype = options.outputDtype || A.dtype;

  validateMatmulOffsets('recordMatmul', aOffset, bOffset, cOffset);

  const { variant, useQ4KFused, useGemv } = selectMatmulVariantAndFlags(
    'record',
    M,
    N,
    K,
    aDtype,
    bDtype,
    transposeB,
    requestedOutputDtype,
    options
  );

  const phase = M === 1 ? 'decode' : 'prefill';
  const constants = resolveMatmulConstants(options, phase);

  let matmulInput = A;
  let matmulADtype = aDtype;
  let castedInput = null;
  if (matmulADtype === 'f16' && requiresF32Input(variant)) {
    if (isTraceEnabled('kernels')) {
      trace.kernels(`Matmul: casting f16 activations to f32 for variant=${variant}`);
    }
    castedInput = await recordCastF16ToF32(recorder, A);
    recorder.trackTemporaryBuffer(castedInput.buffer);
    matmulInput = castedInput;
    matmulADtype = 'f32';
  }

  const { aBindingSize, bBindingSize } = getMatmulBindingSizes(
    'recordMatmul',
    matmulInput.buffer,
    bBuffer,
    M,
    N,
    K,
    matmulADtype,
    bDtype,
    transposeB,
    aOffset,
    bOffset
  );

  const config = applyMatmulConstants(getKernelConfig('matmul', variant), constants);
  const kernel = new MatmulKernel(device);

  // Fast path: use synchronously cached pipeline if available
  let pipeline = getCachedPipeline('matmul', variant, constants);
  if (!pipeline) {
    pipeline = await createPipeline('matmul', variant, null, constants);
  }

  const { output: C, outputSize, cBindingSize, actualOutputDtype } = resolveMatmulOutput(
    variant,
    M,
    N,
    outputBuffer
  );

  if (!Number.isFinite(outputSize) || outputSize <= 0) {
    throw new Error(`[recordMatmul] Invalid output size: ${outputSize} (M=${M}, N=${N})`);
  }

  const cRequired = cOffset + cBindingSize;
  if (C.size < cRequired) {
    throw new Error(`[recordMatmul] Output buffer too small: ${C.size} < ${cRequired} (M=${M}, N=${N})`);
  }

  const dispatchPlan = calculateMatmulDispatch(variant, useQ4KFused, useGemv, M, N, config);
  const uniformBuffer = createMatmulUniformBuffer(
    'matmul_uniforms',
    M,
    N,
    K,
    alpha,
    useQ4KFused,
    transposeB,
    dispatchPlan.uniformWorkgroupsX,
    recorder,
    device
  );

  // Q4K F16 variants use binding 4 for output (F16), all other variants use binding 3
  const isQ4KF16 = variant === 'q4_fused_multicol_f16' ||
    variant === 'q4_fused_f16a' ||
    variant === 'q4_fused_batched_f16' ||
    variant === 'q4_fused_multicol_f16a' ||
    variant === 'q4_fused_batched_f16a';
  
  const entries = [
    { binding: 0, resource: { buffer: uniformBuffer } },
    { binding: 1, resource: { buffer: matmulInput.buffer, offset: aOffset, size: aBindingSize } },
    { binding: 2, resource: { buffer: bBuffer, offset: bOffset, size: bBindingSize } },
  ];

  if (isQ4KF16) {
    entries.push({ binding: 4, resource: { buffer: C, offset: cOffset, size: cBindingSize } });
  } else {
    entries.push({ binding: 3, resource: { buffer: C, offset: cOffset, size: cBindingSize } });
  }

  const bindGroup = device.createBindGroup({
    label: 'matmul_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });

  kernel.record(recorder, pipeline, bindGroup, dispatchPlan.workgroups);
  return createTensor(C, actualOutputDtype, [M, N], 'matmul_output');
}
