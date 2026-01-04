/**
 * Matrix Multiplication Kernels
 *
 * Provides optimized matmul operations with support for:
 * - F16/F32 inputs and outputs
 * - Mixed precision (F16 weights, F32 activations)
 * - Tiled and naive variants
 * - Command recording for batched execution
 */

import { getDevice, getKernelCapabilities } from '../device.js';
import { Tensor, createTensor, type TensorDtype } from '../tensor.js';
import { type WeightBuffer, getBuffer, getLayout, getWeightDtype } from '../weight-buffer.js';
import { log, trace, isTraceEnabled } from '../../debug/index.js';
import { acquireBuffer } from '../buffer-pool.js';
import type { CommandRecorder } from '../command-recorder.js';
import { KernelBase } from './kernel-base.js';
import { ALIGNMENT, GPU_LIMITS, QUANTIZATION, TILE_SIZES } from './constants.js';
import { getKernelConfig, createUniformBufferWithView, getOrCreateBindGroupLayout, getCachedPipeline, createPipeline, getPipelineFast } from './utils.js';
import { shouldUseFusedQ4K } from '../kernel-hints.js';
import { releaseUniformBuffer } from '../uniform-cache.js';
import type { OutputBufferOptions, OutputDtypeOptions, Vec4Options } from './types.js';
import { getKernelThresholds } from '../../config/schema/index.js';

// =============================================================================
// Q4K Variant Lookup Tables
// =============================================================================

/**
 * Q4K fused variant lookup table keyed by "${m1}/${f16out}".
 * Replaces duplicated if-else chains in selectMatmulVariantAndFlags.
 */
const Q4K_FUSED_VARIANTS: Record<string, string> = {
  'true/true': 'q4_fused_multicol_f16',
  'true/false': 'q4_fused_multicol',
  'false/true': 'q4_fused_batched_f16',
  'false/false': 'q4_fused_batched',
};

/**
 * Select Q4K fused variant based on M dimension and output dtype.
 */
function selectQ4KFusedVariant(isM1: boolean, wantF16Output: boolean): string {
  const key = `${isM1}/${wantF16Output}`;
  return Q4K_FUSED_VARIANTS[key] ?? 'q4_fused_batched';
}

/**
 * Debug flag to disable fused Q4K kernels.
 * When true, Q4K weights will be dequantized first, then use standard matmul.
 *
 * Check order:
 * 1. window.DOPPLER_DISABLE_FUSED_Q4K (debug override)
 * 2. Kernel hints from manifest (optimizations.kernelHints.q4kMatmul)
 * 3. Default: false (use dequant path - 2x faster based on benchmarks)
 */
export function isFusedQ4KDisabled(): boolean {
  // Check window override first (debug flag)
  const debugFlags = typeof window !== 'undefined'
    ? (window as unknown as { DOPPLER_DISABLE_FUSED_Q4K?: boolean })
    : null;
  if (debugFlags?.DOPPLER_DISABLE_FUSED_Q4K) return true;

  // Check kernel hints - shouldUseFusedQ4K returns false if hints say to use dequant
  return !shouldUseFusedQ4K();
}

/** Matmul-supported buffer types (includes q4k for fused W4A16) */
type MatmulDtype = 'f16' | 'f32' | 'q4k';

/** Helper to narrow TensorDtype to matmul-supported types */
function toMatmulDtype(dtype: TensorDtype | 'q4k' | 'bf16' | 'q8' | null | undefined): MatmulDtype {
  if (dtype === 'f16' || dtype === 'bf16') return 'f16';  // bf16 weights use f16 kernel
  if (dtype === 'q4k') return 'q4k';
  return 'f32';
}

/** Matmul kernel options */
export interface MatmulOptions extends OutputBufferOptions, OutputDtypeOptions, Vec4Options {
  alpha?: number;
  /**
   * Whether B matrix is stored transposed.
   * - true: B is [N,K] (SafeTensors/row-major), needs transpose
   * - false: B is [K,N] (column-major/pre-transposed), direct access
   * - 'auto': Auto-detect from buffer layout metadata (default)
   */
  transposeB?: boolean | 'auto';
  aOffset?: number;
  bOffset?: number;
  cOffset?: number;
  aDtype?: 'f16' | 'f32' | null;
  bDtype?: 'f16' | 'f32' | 'q4k' | null;
  preferF16?: boolean;
}

/**
 * Select the best matmul kernel variant
 */
export function selectMatmulKernel(options: MatmulOptions = {}): string {
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
  async getPipeline(variant: string): Promise<GPUComputePipeline> {
    return this.getPipelineFor('matmul', variant);
  }

  dispatch(
    pipeline: GPUComputePipeline,
    bindGroup: GPUBindGroup,
    workgroups: [number, number, number]
  ): void {
    this.dispatchKernel(pipeline, bindGroup, workgroups, 'matmul');
  }

  record(
    recorder: CommandRecorder,
    pipeline: GPUComputePipeline,
    bindGroup: GPUBindGroup,
    workgroups: [number, number, number]
  ): void {
    this.recordKernel(recorder, pipeline, bindGroup, workgroups, 'matmul');
  }
}

type MatmulSelectionMode = 'run' | 'record';

// Debug counter to limit logging
let _transposeDebugCount = 0;

function resolveTransposeB(B: GPUBuffer | WeightBuffer, transposeBOption: boolean | 'auto'): boolean {
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

function validateMatmulDimensions(label: string, M: number, N: number, K: number): void {
  if (!Number.isFinite(M) || !Number.isFinite(N) || !Number.isFinite(K)) {
    throw new Error(`[${label}] Invalid dimensions: M=${M}, N=${N}, K=${K}`);
  }
  if (M <= 0 || N <= 0 || K <= 0) {
    throw new Error(`[${label}] Dimensions must be positive: M=${M}, N=${N}, K=${K}`);
  }
}

function validateMatmulOffsets(
  label: string,
  aOffset: number,
  bOffset: number,
  cOffset: number
): void {
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

function getMatmulBindingSizes(
  label: string,
  A: GPUBuffer,
  B: GPUBuffer,
  M: number,
  N: number,
  K: number,
  aDtype: MatmulDtype,
  bDtype: MatmulDtype,
  transposeB: boolean,
  aOffset: number,
  bOffset: number
): { aBindingSize: number; bBindingSize: number } {
  const aBytesPerElem = aDtype === 'f16' ? 2 : 4;
  const aBindingSize = Math.ceil((M * K * aBytesPerElem) / 4) * 4;
  const aRequired = aOffset + aBindingSize;
  if (A.size < aRequired) {
    throw new Error(`[${label}] A buffer too small: ${A.size} < ${aRequired} (M=${M}, K=${K}, aDtype=${aDtype})`);
  }

  const QK_K = TILE_SIZES.Q4K_SUPER_BLOCK_SIZE;
  const Q4K_BLOCK_BYTES = QUANTIZATION.Q4K_BLOCK_BYTES;
  let bBindingSize: number;
  let bRequired: number;

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

function selectMatmulVariantAndFlags(
  mode: MatmulSelectionMode,
  M: number,
  N: number,
  K: number,
  aDtype: MatmulDtype,
  bDtype: MatmulDtype,
  transposeB: boolean,
  requestedOutputDtype: 'f16' | 'f32',
  options: MatmulOptions
): { variant: string; useQ4KFused: boolean; useGemv: boolean } {
  const capabilities = getKernelCapabilities();
  let variant = 'f32';
  let useQ4KFused = false;
  let useGemv = false;

  if (bDtype === 'q4k') {
    const useFused = mode === 'record' ? true : !isFusedQ4KDisabled();
    if (useFused) {
      if (!capabilities.hasSubgroups) {
        if (mode === 'record') {
          throw new Error(
            'Q4_K fused matmul requires subgroup support. ' +
            'Your GPU/browser may not support WebGPU subgroups. ' +
            'Consider using a dequantized model (F16) as fallback.'
          );
        }
        log.warn('Matmul', 'Q4K fused requested but no subgroup support. Falling back to dequant path. ' +
          'Your GPU/browser may not support WebGPU subgroups.');
      } else {
        useQ4KFused = true;
        const wantF16Output = requestedOutputDtype === 'f16' && capabilities.hasF16;
        // Use lookup table for Q4K variant selection
        variant = selectQ4KFusedVariant(M === 1, wantF16Output);
      }
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

    useGemv = M === 1 && effectiveBDtype === 'f16' && aDtype === 'f32';
    if (useGemv) {
      if (capabilities.hasSubgroups) {
        // Use configurable threshold from schema
        const { multicolThreshold } = getKernelThresholds().matmul;
        if (N > multicolThreshold) {
          variant = 'gemv_subgroup_multicol';
        } else {
          variant = 'gemv_subgroup';
        }
      } else {
        variant = 'gemv';
      }
    } else if (M === 1 && effectiveBDtype === 'f16' && aDtype === 'f32') {
      variant = 'f16w_f32a_naive';
    }
  }

  return { variant, useQ4KFused, useGemv };
}

function resolveMatmulOutput(
  variant: string,
  M: number,
  N: number,
  outputBuffer: GPUBuffer | null
): { output: GPUBuffer; outputSize: number; cBindingSize: number; actualOutputDtype: 'f16' | 'f32' } {
  // Use kernel config's outputDtype instead of string matching
  const config = getKernelConfig('matmul', variant);
  const outputsF16 = config.outputDtype === 'f16';
  const elementSize = outputsF16 ? 2 : 4;
  const actualOutputDtype: 'f16' | 'f32' = outputsF16 ? 'f16' : 'f32';
  const outputSize = M * N * elementSize;
  const cBindingSize = Math.ceil(outputSize / 4) * 4;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'matmul_output');
  return { output, outputSize, cBindingSize, actualOutputDtype };
}

function calculateMatmulDispatch(
  variant: string,
  useQ4KFused: boolean,
  useGemv: boolean,
  M: number,
  N: number,
  config: { workgroupSize: [number, number, number]; variantMetadata?: { colsPerWg?: number; tileM?: number } }
): { workgroups: [number, number, number]; uniformWorkgroupsX?: number } {
  const maxWorkgroups = GPU_LIMITS.MAX_WORKGROUPS;
  const [wgX, wgY] = config.workgroupSize;
  let workgroupsX = 1;
  let workgroupsY = 1;
  let uniformWorkgroupsX: number | undefined;

  // Get colsPerWg from variantMetadata (default 4 for non-multicol GEMV)
  const colsPerWg = config.variantMetadata?.colsPerWg ?? 4;
  // Get tileM from variantMetadata (default 4 for batched variants)
  const tileM = config.variantMetadata?.tileM ?? 4;

  if (useGemv && (variant === 'gemv_subgroup' || variant === 'gemv_subgroup_multicol')) {
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
  } else if (variant === 'f16w_f32a_naive') {
    workgroupsX = Math.ceil(N / wgX);
    workgroupsY = 1;
  } else {
    workgroupsX = Math.ceil(M / wgX);
    workgroupsY = Math.ceil(N / wgY);
  }

  return { workgroups: [workgroupsX, workgroupsY, 1], uniformWorkgroupsX };
}

function createMatmulUniformBuffer(
  label: string,
  M: number,
  N: number,
  K: number,
  alpha: number,
  useQ4KFused: boolean,
  transposeB: boolean,
  uniformWorkgroupsX: number | undefined,
  recorder: CommandRecorder | null,
  device: GPUDevice
): GPUBuffer {
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

/**
 * Create bind group layout for matmul operation
 */
export function createMatmulBindGroupLayout(): GPUBindGroupLayout {
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

/**
 * Run matrix multiplication
 *
 * @param A - Activation tensor (Tensor with explicit dtype)
 * @param B - Weight buffer (GPUBuffer or WeightBuffer)
 * @returns Output tensor with computed dtype
 */
export async function runMatmul(
  A: Tensor,
  B: GPUBuffer | WeightBuffer,
  M: number,
  N: number,
  K: number,
  options: MatmulOptions = {}
): Promise<Tensor> {
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
  const { aBindingSize, bBindingSize } = getMatmulBindingSizes(
    'runMatmul',
    A.buffer,
    bBuffer,
    M,
    N,
    K,
    aDtype,
    bDtype,
    transposeB,
    aOffset,
    bOffset
  );

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

  // Q4K F16 variants use binding 4 for output (F16), F32 variants use binding 3
  const isQ4KF16 = variant === 'q4_fused_multicol_f16' || variant === 'q4_fused_batched_f16';
  const entries: GPUBindGroupEntry[] = [
    { binding: 0, resource: { buffer: uniformBuffer } },
    { binding: 1, resource: { buffer: A.buffer, offset: aOffset, size: aBindingSize } },
    { binding: 2, resource: { buffer: bBuffer, offset: bOffset, size: bBindingSize } },
  ];

  if (isQ4KF16) {
    // F16 output at binding 4, dummy at binding 3 (shader declares both)
    const dummyBuffer = acquireBuffer(4, undefined, 'q4k_dummy');
    entries.push({ binding: 3, resource: { buffer: dummyBuffer, size: 4 } });
    entries.push({ binding: 4, resource: { buffer: C, offset: cOffset, size: cBindingSize } });
  } else {
    entries.push({ binding: 3, resource: { buffer: C, offset: cOffset, size: cBindingSize } });
    // Only add binding 4 for Q4K F32 variants (shader declares it)
    if (useQ4KFused) {
      const dummyBuffer = acquireBuffer(4, undefined, 'q4k_dummy');
      entries.push({ binding: 4, resource: { buffer: dummyBuffer, size: 4 } });
    }
  }

  const bindGroup = device.createBindGroup({
    label: 'matmul_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });

  kernel.dispatch(pipeline, bindGroup, dispatchPlan.workgroups);
  releaseUniformBuffer(uniformBuffer);

  return createTensor(C, actualOutputDtype, [M, N], 'matmul_output');
}

// Debug counter for recordMatmul
let _recordMatmulDebugCount = 0;

/**
 * Record matrix multiplication (batched, no submit)
 *
 * @param recorder - Command recorder for batched execution
 * @param A - Activation tensor (Tensor with explicit dtype)
 * @param B - Weight buffer (GPUBuffer or WeightBuffer)
 * @returns Output tensor with computed dtype
 */
export async function recordMatmul(
  recorder: CommandRecorder,
  A: Tensor,
  B: GPUBuffer | WeightBuffer,
  M: number,
  N: number,
  K: number,
  options: MatmulOptions = {}
): Promise<Tensor> {
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
  const { aBindingSize, bBindingSize } = getMatmulBindingSizes(
    'recordMatmul',
    A.buffer,
    bBuffer,
    M,
    N,
    K,
    aDtype,
    bDtype,
    transposeB,
    aOffset,
    bOffset
  );

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

  // Q4K F16 variants use binding 4 for output (F16), F32 variants use binding 3
  const isQ4KF16 = variant === 'q4_fused_multicol_f16' || variant === 'q4_fused_batched_f16';
  const entries: GPUBindGroupEntry[] = [
    { binding: 0, resource: { buffer: uniformBuffer } },
    { binding: 1, resource: { buffer: A.buffer, offset: aOffset, size: aBindingSize } },
    { binding: 2, resource: { buffer: bBuffer, offset: bOffset, size: bBindingSize } },
  ];

  if (isQ4KF16) {
    // F16 output at binding 4, dummy at binding 3 (shader declares both)
    const dummyBuffer = acquireBuffer(4, undefined, 'q4k_dummy');
    entries.push({ binding: 3, resource: { buffer: dummyBuffer, size: 4 } });
    entries.push({ binding: 4, resource: { buffer: C, offset: cOffset, size: cBindingSize } });
  } else {
    entries.push({ binding: 3, resource: { buffer: C, offset: cOffset, size: cBindingSize } });
    // Only add binding 4 for Q4K F32 variants (shader declares it)
    if (useQ4KFused) {
      const dummyBuffer = acquireBuffer(4, undefined, 'q4k_dummy');
      entries.push({ binding: 4, resource: { buffer: dummyBuffer, size: 4 } });
    }
  }

  const bindGroup = device.createBindGroup({
    label: 'matmul_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });

  kernel.record(recorder, pipeline, bindGroup, dispatchPlan.workgroups);
  return createTensor(C, actualOutputDtype, [M, N], 'matmul_output');
}
