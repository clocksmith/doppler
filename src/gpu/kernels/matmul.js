

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
import { getKernelPathMatmulVariant, getKernelPathStrict, isActiveKernelPathFusedQ4K } from '../../config/kernel-path-loader.js';
import { castF16ToF32, recordCastF16ToF32 } from './cast.js';

// =============================================================================
// Q4K Variant Lookup Tables
// =============================================================================


function selectQ4KFusedVariant(isM1, wantF16Output, aDtype) {
  if (!wantF16Output) {
    return isM1 ? 'q4_fused_multicol' : 'q4_fused_batched';
  }
  if (aDtype === 'f16') {
    return isM1 ? 'q4_fused_multicol_f16a' : 'q4_fused_batched_f16a';
  }
  return isM1 ? 'q4_fused_multicol_f16' : 'q4_fused_batched_f16';
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

  // Full f16 matmul only when both inputs are f16 and caller wants f16 output.
  if (outputDtype === 'f16' && preferF16 && inputsAreF16 && capabilities.hasF16) {
    return useVec4 ? 'f16_vec4' : 'f16';
  }

  // Mixed precision: f32 activations, f16 weights.
  // Use f16w_f32a kernel regardless of output dtype - it will produce f32 output,
  // and cast to f16 if needed afterwards. This is better than using f32 kernel
  // which can't read f16 weights at all!
  if (preferF16 && weightsAreF16 && capabilities.hasF16) {
    return 'f16w_f32a';
  }

  return 'f32';
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


function resolveMatmulOverride(variantOverride, M, aDtype, bDtype, requestedOutputDtype, capabilities, strict) {
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


function selectMatmulVariantAndFlags(mode, M, N, K, aDtype, bDtype, transposeB, requestedOutputDtype, options) {
  const capabilities = getKernelCapabilities();
  const strict = getKernelPathStrict();
  const phase = M === 1 ? 'decode' : 'prefill';
  let pathVariant = getKernelPathMatmulVariant(options.role, phase, options.layerIdx);

  if (pathVariant && !strict && M === 1 && bDtype === 'f16' && capabilities.hasSubgroups) {
    const { multicolThreshold } = getKernelThresholds().matmul;
    if (pathVariant === 'gemv_f16a' && aDtype === 'f16' && requestedOutputDtype === 'f16') {
      pathVariant = N > multicolThreshold ? 'gemv_subgroup_multicol_f16a' : 'gemv_subgroup_f16a';
    } else if (pathVariant === 'gemv' && aDtype === 'f32') {
      pathVariant = N > multicolThreshold ? 'gemv_subgroup_multicol' : 'gemv_subgroup';
    }
  }

  if (pathVariant) {
    const override = resolveMatmulOverride(pathVariant, M, aDtype, bDtype, requestedOutputDtype, capabilities, strict);
    if (override) {
      return override;
    }
  }

  let variant = 'f32';
  let useQ4KFused = false;
  let useGemv = false;

  // For Q4K weights, prefer fused path if available
  if (bDtype === 'q4k') {
    const allowFused = !isFusedQ4KDisabled();
    const canFused = capabilities.hasSubgroups && allowFused;

    if (canFused) {
      useQ4KFused = true;
      const wantF16Output = requestedOutputDtype === 'f16' && capabilities.hasF16;
      variant = selectQ4KFusedVariant(M === 1, wantF16Output, aDtype);
    }
  }

  if (!useQ4KFused) {
    const effectiveBDtype = bDtype === 'q4k' ? 'f32' : bDtype;
    variant = selectMatmulKernel({
      ...options,
      aDtype: aDtype === 'q4k' ? 'f32' : aDtype,
      bDtype: effectiveBDtype,
      outputDtype: requestedOutputDtype,
    });

    const canGemv = M === 1 && effectiveBDtype === 'f16' && capabilities.hasF16;
    const wantsF16Output = requestedOutputDtype === 'f16' && capabilities.hasF16;
    const useF16Gemv = canGemv && aDtype === 'f16' && wantsF16Output;
    const useF32Gemv = canGemv && aDtype === 'f32';

    useGemv = useF16Gemv || useF32Gemv;
    if (useGemv) {
      if (capabilities.hasSubgroups) {
        // Use configurable threshold from schema
        const { multicolThreshold } = getKernelThresholds().matmul;
        if (N > multicolThreshold) {
          variant = useF16Gemv ? 'gemv_subgroup_multicol_f16a' : 'gemv_subgroup_multicol';
        } else {
          variant = useF16Gemv ? 'gemv_subgroup_f16a' : 'gemv_subgroup';
        }
      } else {
        variant = useF16Gemv ? 'gemv_f16a' : 'gemv';
      }
    }
  }

  return { variant, useQ4KFused, useGemv };
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

  const config = getKernelConfig('matmul', variant);
  const kernel = new MatmulKernel(device);

  // Fast path: use synchronously cached pipeline if available
  let pipeline = getCachedPipeline('matmul', variant);
  if (!pipeline) {
    pipeline = await createPipeline('matmul', variant);
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

  const config = getKernelConfig('matmul', variant);
  const kernel = new MatmulKernel(device);

  // Fast path: use synchronously cached pipeline if available
  let pipeline = getCachedPipeline('matmul', variant);
  if (!pipeline) {
    pipeline = await createPipeline('matmul', variant);
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
