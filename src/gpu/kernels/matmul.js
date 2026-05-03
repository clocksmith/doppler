import { getDevice, getKernelCapabilities } from '../device.js';
import { createTensor } from '../tensor.js';
import {
  getBuffer,
  getLayout,
  getWeightDtype,
  isWeightBuffer,
  resolveWeightBufferMaterialization,
} from '../weight-buffer.js';
import { log, trace, isTraceEnabled } from '../../debug/index.js';
import { releaseBuffer } from '../../memory/buffer-pool.js';
import { releaseUniformBuffer } from '../uniform-cache.js';
import { castF16ToF32, recordCastF16ToF32 } from './cast.js';
import { getKernelPathMatmulVariant } from '../../config/kernel-path-loader.js';
import { assertImplicitDtypeTransitionAllowed } from '../../inference/pipelines/text/dtype-contract.js';
import {
  resolveMatmulPhase,
  resolveMatmulConstants,
  getMatmulConfig,
  isFusedQ4KDisabled,
  toMatmulDtype,
  resolveTransposeB,
  validateMatmulDimensions,
  validateMatmulOffsets,
  getMatmulBindingSizes,
  requiresF32Input,
  selectMatmulVariantAndFlags,
  resolveMatmulOutput,
  selectMatmulKernel,
} from './matmul-selection.js';
import { selectRuleValue as selectKernelRuleValue } from './rule-registry.js';
import { getRuntimeConfig } from '../../config/runtime.js';
import {
  MatmulKernel,
  calculateMatmulDispatch,
  createMatmulUniformBuffer,
  createMatmulBindGroupLayout,
  getMatmulPipeline,
} from './matmul-dispatch.js';
import { __dbgRecord } from './utils.js';

export { isFusedQ4KDisabled, selectMatmulKernel };
export { createMatmulBindGroupLayout };

let _runMatmulDebugCount = 0;
let _recordMatmulDebugCount = 0;

function normalizeMatmulDebugConfig(config) {
  if (!config || typeof config !== 'object') {
    return null;
  }
  return {
    enabled: config.enabled === true,
    forceSplitQKV: config.forceSplitQKV === true,
    validateAttentionWeightBuffer: config.validateAttentionWeightBuffer === true,
    failOnSmallAttentionWeightBuffer: config.failOnSmallAttentionWeightBuffer === true,
    logAttentionWeightBuffer: config.logAttentionWeightBuffer === true,
    logProjectionValues: config.logProjectionValues === true,
  };
}

function isAttentionProjectionRole(role = '') {
  return role === 'qkv_proj' || role === 'q_proj' || role === 'k_proj' || role === 'v_proj';
}

function getDebugCounter(isRecord) {
  return isRecord ? _recordMatmulDebugCount : _runMatmulDebugCount;
}

function incrementDebugCounter(isRecord) {
  if (isRecord) {
    _recordMatmulDebugCount += 1;
    return;
  }
  _runMatmulDebugCount += 1;
}

function buildProfileLabel(options = {}) {
  const layerLabel = Number.isFinite(options.layerIdx) ? `:L${options.layerIdx}` : '';
  const roleLabel = options.role ? `:${options.role}` : '';
  return `matmul${roleLabel}${layerLabel}`;
}

function assertBindGroupBuffer(kernelName, variant, bindingIndex, bindingLabel, buffer, details = []) {
  const isGpuBuffer = buffer && (
    typeof GPUBuffer === 'undefined'
      ? true
      : buffer instanceof GPUBuffer
  );
  if (isGpuBuffer) {
    return;
  }
  const detailText = details.filter(Boolean).join(', ');
  throw new Error(
    `[${kernelName}] variant="${variant}" binding ${bindingIndex} "${bindingLabel}" requires a GPUBuffer` +
    (detailText ? ` (${detailText})` : '') +
    '.'
  );
}

function createMatmulBindGroupEntries(variant, uniformBuffer, matmulInput, bBuffer, outputBuffer, offsets, bindingSizes, residualBuffer = null, normWeightBuffer = null) {
  const isQ4KF16 = variant === 'q4_fused_multicol_f16'
    || variant === 'q4_fused_f16a'
    || variant === 'q4_fused_batched_f16'
    || variant === 'q4_fused_multicol_f16a'
    || variant === 'q4_fused_multicol_f16a_f32acc'
    || variant === 'q4_fused_batched_f16a'
    || variant === 'q4_fused_batched_f16acc_f16a'
    || variant === 'q4_fused_prefill_tiled_f16'
    || variant === 'q4_fused_widetile_f16'
    || variant === 'q4_fused_widetile_f16a';
  // 5-entry WideTile epilogue/prologue variants: output at binding 3 + one
  // extra read-only buffer at binding 4 (residual for _residual, norm weight
  // for _rmsnorm). Distinct from isQ4KF16 (which puts output at binding 4).
  const isWideTileResidual = variant === 'q4_fused_widetile_residual';
  const isWideTileRmsnorm = variant === 'q4_fused_rmsnorm_widetile';

  assertBindGroupBuffer('matmul', variant, 0, 'uniforms', uniformBuffer);
  assertBindGroupBuffer('matmul', variant, 1, 'input', matmulInput?.buffer, [
    `inputLabel=${matmulInput?.label ?? 'unknown'}`,
    `inputDtype=${matmulInput?.dtype ?? 'unknown'}`,
  ]);
  assertBindGroupBuffer('matmul', variant, 2, 'weights', bBuffer);
  assertBindGroupBuffer('matmul', variant, isQ4KF16 ? 4 : 3, 'output', outputBuffer);
  if (isWideTileResidual) {
    if (!residualBuffer) {
      throw new Error(`[Matmul] variant "${variant}" requires a residual buffer but none was provided.`);
    }
    assertBindGroupBuffer('matmul', variant, 4, 'residual', residualBuffer);
  }
  if (isWideTileRmsnorm) {
    if (!normWeightBuffer) {
      throw new Error(`[Matmul] variant "${variant}" requires a norm weight buffer but none was provided.`);
    }
    assertBindGroupBuffer('matmul', variant, 4, 'norm_weight', normWeightBuffer);
  }

  const entries = [
    { binding: 0, resource: { buffer: uniformBuffer } },
    { binding: 1, resource: { buffer: matmulInput.buffer, offset: offsets.aOffset, size: bindingSizes.aBindingSize } },
    { binding: 2, resource: { buffer: bBuffer, offset: offsets.bOffset, size: bindingSizes.bBindingSize } },
  ];

  if (isQ4KF16) {
    entries.push({
      binding: 4,
      resource: { buffer: outputBuffer, offset: offsets.cOffset, size: bindingSizes.cBindingSize },
    });
  } else {
    entries.push({
      binding: 3,
      resource: { buffer: outputBuffer, offset: offsets.cOffset, size: bindingSizes.cBindingSize },
    });
    if (isWideTileResidual) {
      entries.push({
        binding: 4,
        resource: { buffer: residualBuffer },
      });
    }
    if (isWideTileRmsnorm) {
      entries.push({
        binding: 4,
        resource: { buffer: normWeightBuffer },
      });
    }
  }

  return entries;
}

function resolvePreferredWeightDtype(variant, hasQ4KMaterialization) {
  if (typeof variant !== 'string' || variant.length === 0) {
    return null;
  }

  let config;
  try {
    config = getMatmulConfig(variant, null);
  } catch {
    return null;
  }

  const variantWeightDtype = config?.weightDtype ?? null;
  if (!variantWeightDtype) {
    return null;
  }

  return selectKernelRuleValue('matmul', 'preferredWeightDtype', {
    variantWeightDtype,
    hasQ4KMaterialization,
  });
}

async function executeMatmul(recorder, A, B, M, N, K, options = {}) {
  const isRecord = Boolean(recorder);
  const mode = isRecord ? 'record' : 'run';
  const opLabel = isRecord ? 'recordMatmul' : 'runMatmul';
  const device = recorder?.device || getDevice();
  const capabilities = getKernelCapabilities();

  const {
    alpha = 1.0,
    outputBuffer = null,
    transposeB: transposeBOption = true,
    aOffset = 0,
    bOffset = 0,
    cOffset = 0,
  } = options;

  const phase = resolveMatmulPhase(M, options.phaseOverride ?? null);
  const pathVariant = getKernelPathMatmulVariant(options.role, phase, options.layerIdx, options.kernelPath);
  const hasQ4KMat = isWeightBuffer(B) && B.materializations?.q4k?.buffer != null;
  const preferredWeightDtype = resolvePreferredWeightDtype(pathVariant, hasQ4KMat);
  const resolvedWeight = resolveWeightBufferMaterialization(B, preferredWeightDtype);
  const bBuffer = getBuffer(resolvedWeight);
  const weightDtype = getWeightDtype(resolvedWeight);
  const weightLabel = (resolvedWeight && typeof resolvedWeight === 'object' ? resolvedWeight.label : null) ?? bBuffer?.label ?? null;
  const weightLayout = getLayout(resolvedWeight);
  const weightShape = resolvedWeight?.shape ? `[${resolvedWeight.shape.join(', ')}]` : null;
  const matmulDebug = normalizeMatmulDebugConfig(options.matmulDebug);
  const debugAttention = matmulDebug?.enabled === true;
  const isAttnProj = isAttentionProjectionRole(options.role ?? '');
  const shouldValidateAttentionWeightBuffer = debugAttention && matmulDebug.validateAttentionWeightBuffer;
  const shouldFailOnSmallAttentionWeightBuffer = debugAttention && matmulDebug.failOnSmallAttentionWeightBuffer;
  const shouldLogAttentionWeightBuffer = debugAttention && matmulDebug.logAttentionWeightBuffer;

  if (isTraceEnabled('kernels') && getDebugCounter(isRecord) < 20) {
    incrementDebugCounter(isRecord);
    const modeLabel = isRecord ? 'recordMatmul' : 'runMatmul';
    trace.kernels(`${modeLabel}: M=${M}, N=${N}, K=${K}, transposeBOption=${transposeBOption}, weightLayout=${weightLayout}, weightDtype=${weightDtype}`);
  }

  const transposeB = resolveTransposeB(resolvedWeight, transposeBOption);
  validateMatmulDimensions(opLabel, M, N, K);

  const aDtype = toMatmulDtype(A.dtype);
  const bDtype = toMatmulDtype(weightDtype ?? options.bDtype);
  const requestedOutputDtype = options.outputDtype || A.dtype;

  if (bDtype === 'f16' && capabilities?.hasF16 !== true) {
    throw new Error(`[${opLabel}] f16 weights require shader-f16 support.`);
  }
  if (requestedOutputDtype === 'f16' && capabilities?.hasF16 !== true) {
    throw new Error(`[${opLabel}] f16 output requires shader-f16 support.`);
  }

  if (!isRecord && isTraceEnabled('kernels') && !weightDtype && !options.bDtype && M <= 2) {
    log.warn('Matmul', `runMatmul: B buffer dtype unknown! size=${bBuffer.size}, M=${M}, N=${N}, K=${K}. Assuming f32.`);
  }

  validateMatmulOffsets(opLabel, aOffset, bOffset, cOffset);

  const runtimeSession = getRuntimeConfig().inference?.session;
  const effectiveOptions = (
    options.useTiledQ4KPrefill == null
    || options.useWideTileQ4KPrefill == null
    || options.useWideTileResidualFusion == null
    || options.useFusedRmsnormWideTile == null
  )
    ? {
        ...options,
        useTiledQ4KPrefill: options.useTiledQ4KPrefill ?? (runtimeSession?.useTiledQ4KPrefill === true),
        useWideTileQ4KPrefill: options.useWideTileQ4KPrefill ?? (runtimeSession?.useWideTileQ4KPrefill === true),
        useWideTileResidualFusion: options.useWideTileResidualFusion ?? (runtimeSession?.useWideTileResidualFusion === true),
        useFusedRmsnormWideTile: options.useFusedRmsnormWideTile ?? (runtimeSession?.useFusedRmsnormWideTile === true),
      }
    : options;

  let { variant, useQ4KFused, useGemv } = selectMatmulVariantAndFlags(
    mode,
    M,
    N,
    K,
    aDtype,
    bDtype,
    transposeB,
    requestedOutputDtype,
    effectiveOptions
  );

  if (
    runtimeSession?.useF32AccumF16ioMatmul === true
    && useQ4KFused
    && variant === 'q4_fused_multicol_f16a'
  ) {
    variant = 'q4_fused_multicol_f16a_f32acc';
  }

  let constants = resolveMatmulConstants(options, phase);
  if (variant === 'f32' && constants && options.constants == null) {
    constants = null;
  }
  // For the rmsnorm-fused WideTile variant, forward the caller's
  // rmsNormOffset flag as a pipeline override constant. Gemma-family norm
  // weights encode `(w - 1.0)`; other models encode `w`.
  // Also forward WEIGHT_IS_F16 based on the norm weight buffer dtype so the
  // kernel correctly unpacks f16-packed weights (Gemma hidden weights) vs
  // f32 weights.
  if (variant === 'q4_fused_rmsnorm_widetile' && options.rmsNormOffset != null) {
    const normWeightDtype = getWeightDtype(options.normWeight) ?? 'f32';
    constants = {
      ...(constants ?? {}),
      RMS_NORM_OFFSET: options.rmsNormOffset === true,
      WEIGHT_IS_F16: normWeightDtype === 'f16',
    };
  }

  let matmulInput = A;
  let matmulADtype = aDtype;
  let castedInput = null;
  if (matmulADtype === 'f16' && requiresF32Input(variant)) {
    assertImplicitDtypeTransitionAllowed({
      executionPolicies: options.executionPolicies ?? null,
      fromDtype: 'f16',
      toDtype: 'f32',
      op: options.role ? `matmul(${options.role})` : 'matmul',
      detail: `Variant "${variant}" would widen activations internally.`,
    });
    if (isTraceEnabled('kernels')) {
      trace.kernels(`Matmul: casting f16 activations to f32 for variant=${variant}`);
    }
    if (isRecord) {
      castedInput = await recordCastF16ToF32(recorder, A);
      recorder.trackTemporaryBuffer(castedInput.buffer);
    } else {
      castedInput = await castF16ToF32(A);
    }
    matmulInput = castedInput;
    matmulADtype = 'f32';
  }

  let bindingSizes;
  try {
    bindingSizes = getMatmulBindingSizes(
      opLabel,
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
  } catch (err) {
    const detailParts = [];
    if (options.role) detailParts.push(`role=${options.role}`);
    if (Number.isFinite(options.layerIdx)) detailParts.push(`layer=${options.layerIdx}`);
    if (weightLabel) detailParts.push(`label=${weightLabel}`);
    if (weightDtype) detailParts.push(`weightDtype=${weightDtype}`);
    if (weightLayout) detailParts.push(`layout=${weightLayout}`);
    if (weightShape) detailParts.push(`shape=${weightShape}`);
    if (Number.isFinite(bBuffer?.size)) detailParts.push(`bSize=${bBuffer.size}`);
    if (Number.isFinite(bOffset) && bOffset > 0) detailParts.push(`bOffset=${bOffset}`);
    const detail = detailParts.length ? ` (${detailParts.join(', ')})` : '';
    if (shouldValidateAttentionWeightBuffer && isAttnProj && err instanceof Error && err.message.includes('B buffer too small')) {
      const probeDetail = [
        `role=${options.role ?? ''}`,
        `layer=${Number.isFinite(options.layerIdx) ? options.layerIdx : '?'}`,
        `M=${M}`,
        `N=${N}`,
        `K=${K}`,
        ...(weightDtype ? [`weightDtype=${weightDtype}`] : []),
        ...(weightLayout ? [`weightLayout=${weightLayout}`] : []),
        ...(weightShape ? [`shape=${weightShape}`] : []),
        ...(weightLabel ? [`label=${weightLabel}`] : []),
        ...(Number.isFinite(bBuffer?.size) ? [`bSize=${bBuffer.size}`] : []),
      ].join(' ');
      if (shouldLogAttentionWeightBuffer) {
        log.warn('MatmulQKVProbe', `${err.message} | ${probeDetail}`);
      }
      if (shouldFailOnSmallAttentionWeightBuffer) {
        throw new Error(`${err.message}${detail}`);
      }
    }
    if (err instanceof Error && err.message.includes('B buffer too small')) {
      throw new Error(`${err.message}${detail}`);
    }
    throw err;
  }

  if (!isRecord && isTraceEnabled('kernels') && bDtype === 'q4k') {
    if (useQ4KFused) {
      trace.kernels(`Q4K FUSED: M=${M}, N=${N}, K=${K}, variant=${variant} (WARNING: 2.3x slower than dequant)`);
    } else {
      trace.kernels(`Q4K DEQUANT: M=${M}, N=${N}, K=${K}, will dequant first then matmul with variant=${variant}`);
    }
  }

  if (!isRecord && isTraceEnabled('kernels') && N > 100000) {
    trace.kernels(`MATMUL_LARGE: N=${N}, variant=${variant}, aDtype=${aDtype}, bDtype=${bDtype}, transposeB=${transposeB}`);
  }

  if (isAttnProj && shouldLogAttentionWeightBuffer) {
    log.warn('MatmulQKVProbe',
      `role=${options.role ?? ''} layer=${Number.isFinite(options.layerIdx) ? options.layerIdx : '?'} ` +
      `M=${M} N=${N} K=${K} transposeB=${transposeB} bSize=${bBuffer?.size ?? 0} ` +
      `requiredB=${bindingSizes?.bBindingSize ?? 'n/a'} weightShape=${weightShape ?? 'n/a'} ` +
      `weightDtype=${weightDtype ?? 'unknown'} weightLayout=${weightLayout ?? 'unknown'}`
    );
  }

  const __dbg = (typeof process !== "undefined" ? process : { env: {} }).env.DOPPLER_DBG_RECORD === '1';
  const __t0 = __dbg ? performance.now() : 0;
  const config = getMatmulConfig(variant, constants);
  const kernel = new MatmulKernel(device);
  const pipeline = await getMatmulPipeline(variant, constants);
  const __tPipeline = __dbg ? performance.now() : 0;

  const { output: C, outputSize, cBindingSize, actualOutputDtype } = resolveMatmulOutput(
    variant,
    M,
    N,
    outputBuffer
  );
  const ownsOutput = outputBuffer == null;

  if (isAttnProj && shouldLogAttentionWeightBuffer) {
    log.warn('MatmulVariantDiag',
      `role=${options.role ?? ''} layer=${Number.isFinite(options.layerIdx) ? options.layerIdx : '?'} mode=${mode} ` +
      `variant=${variant} useQ4KFused=${useQ4KFused} useGemv=${useGemv} ` +
      `aDtype=${aDtype} bDtype=${bDtype} output=${actualOutputDtype}`
    );
  }

  if (!Number.isFinite(outputSize) || outputSize <= 0) {
    throw new Error(`[${opLabel}] Invalid output size: ${outputSize} (M=${M}, N=${N})`);
  }

  const cRequired = cOffset + cBindingSize;
  if (C.size < cRequired) {
    throw new Error(`[${opLabel}] Output buffer too small: ${C.size} < ${cRequired} (M=${M}, N=${N})`);
  }

  const dispatchPlan = calculateMatmulDispatch(variant, useQ4KFused, useGemv, M, N, config);
  let uniformBuffer = null;
  let completed = false;
  try {
    const uniformExtras = variant === 'q4_fused_rmsnorm_widetile' && Number.isFinite(options.rmsNormEps)
      ? { eps: options.rmsNormEps }
      : null;
    uniformBuffer = createMatmulUniformBuffer(
      'matmul_uniforms',
      M,
      N,
      K,
      alpha,
      useQ4KFused,
      transposeB,
      dispatchPlan.uniformWorkgroupsX,
      recorder || null,
      device,
      uniformExtras
    );

    const residualBuffer = options.residualTensor?.buffer ?? null;
    const normWeightBuffer = options.normWeight?.buffer ?? options.normWeight ?? null;
    const entries = createMatmulBindGroupEntries(
      variant,
      uniformBuffer,
      matmulInput,
      bBuffer,
      C,
      { aOffset, bOffset, cOffset },
      {
        aBindingSize: bindingSizes.aBindingSize,
        bBindingSize: bindingSizes.bBindingSize,
        cBindingSize,
      },
      residualBuffer,
      normWeightBuffer
    );

    const __tBgStart = __dbg ? performance.now() : 0;
    const bindGroup = device.createBindGroup({
      label: 'matmul_bind_group',
      layout: pipeline.getBindGroupLayout(0),
      entries,
    });
    const __tBg = __dbg ? performance.now() : 0;

    if (isRecord) {
      kernel.record(recorder, pipeline, bindGroup, dispatchPlan.workgroups, buildProfileLabel(options));
    } else {
      kernel.dispatch(pipeline, bindGroup, dispatchPlan.workgroups);
    }
    if (__dbg) {
      const __tEnd = performance.now();
      __dbgRecord('matmul', variant, __tPipeline - __t0, __tBgStart - __tPipeline, __tBg - __tBgStart, __tEnd - __tBg);
    }
    completed = true;
    return createTensor(C, actualOutputDtype, [M, N], 'matmul_output');
  } finally {
    if (!isRecord && uniformBuffer) {
      releaseUniformBuffer(uniformBuffer);
    }
    if (!isRecord && castedInput) {
      releaseBuffer(castedInput.buffer);
    }
    if (!completed && ownsOutput) {
      releaseBuffer(C);
    }
  }
}


export async function runMatmul(A, B, M, N, K, options = {}) {
  return executeMatmul(null, A, B, M, N, K, options);
}


export async function recordMatmul(recorder, A, B, M, N, K, options = {}) {
  return executeMatmul(recorder, A, B, M, N, K, options);
}
