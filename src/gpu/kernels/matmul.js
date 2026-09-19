import { createKernelBindingEntries } from './kernel-bindings.js';
import { getDevice, getKernelCapabilities } from '../device.js';
import { getKernelConfig } from './kernel-configs.js';
import { createTensor } from '../tensor.js';
import {
  getBuffer,
  getLayout,
  getWeightMetadata,
  getWeightDtype,
  isWeightBuffer,
  resolveWeightBufferMaterialization,
} from '../weight-buffer.js';
import { log, trace, isTraceEnabled } from '../../debug/index.js';
import { releaseBuffer } from '../../memory/buffer-pool.js';
import { releaseUniformBuffer } from '../uniform-cache.js';
import { castF16ToF32, recordCastF16ToF32 } from './cast.js';
import { getKernelPathMatmulVariant } from '../../config/kernel-path-loader.js';
import { assertImplicitDtypeTransitionAllowed } from '../../config/dtype-transition-contract.js';
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
import { getPipelineBindGroupLayout } from './pipeline-cache.js';

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
  return role === 'qkv_proj' || role === 'q_proj' || role === 'q_gate_proj'
    || role === 'k_proj' || role === 'v_proj';
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

function resolveLiteRTInt4StorageEncoding(weight, label) {
  const storageEncoding = String(getWeightMetadata(weight)?.storageEncoding ?? '').toLowerCase();
  if (storageEncoding !== 'signed' && storageEncoding !== 'offset_binary') {
    throw new Error(
      `[Matmul] LiteRT INT4 weight "${label ?? 'unknown'}" requires metadata.storageEncoding ` +
      `"signed" or "offset_binary", got "${storageEncoding || 'missing'}".`
    );
  }
  return storageEncoding;
}

function resolveW4A16ScaleDtype(weight, label) {
  const metadata = getWeightMetadata(weight);
  const scaleDtype = String(metadata?.scaleDtype ?? '').toLowerCase();
  if (scaleDtype !== 'f16' && scaleDtype !== 'bf16' && scaleDtype !== 'f32') {
    throw new Error(
      `[Matmul] W4A16 weight "${label ?? 'unknown'}" requires metadata.scaleDtype ` +
      `"f16", "bf16", or "f32", got "${scaleDtype || 'missing'}".`
    );
  }
  return scaleDtype;
}

function resolveW4A16ScaleDtypeConstant(scaleDtype) {
  if (scaleDtype === 'f16') return 0;
  if (scaleDtype === 'bf16') return 1;
  if (scaleDtype === 'f32') return 2;
  throw new Error(`[Matmul] unsupported W4A16 scale dtype "${scaleDtype}".`);
}

function createMatmulBindGroupEntries(variant, uniformBuffer, matmulInput, bBuffer, outputBuffer, offsets, bindingSizes, residualBuffer = null, normWeightBuffer = null, scaleBuffer = null) {
  const resources = {
    uniforms: { buffer: uniformBuffer },
    input: { buffer: matmulInput?.buffer, offset: offsets.aOffset, size: bindingSizes.aBindingSize },
    weights: { buffer: bBuffer, offset: offsets.bOffset, size: bindingSizes.bBindingSize },
    scales: { buffer: scaleBuffer },
    output: { buffer: outputBuffer, offset: offsets.cOffset, size: bindingSizes.cBindingSize },
    residual: { buffer: residualBuffer },
    norm_weight: { buffer: normWeightBuffer },
  };
  const entries = createKernelBindingEntries(getKernelConfig('matmul', variant), resources);

  return entries;
}

function resolvePreferredWeightDtype(variant, hasQ4KMaterialization, capabilities) {
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
    hasSubgroups: capabilities?.hasSubgroups === true,
  });
}

function buildF16CapabilityErrorDetail({
  role,
  layerIdx,
  pathVariant,
  preferredWeightDtype,
  weightDtype,
  weightLabel,
  weightLayout,
  weightShape,
  hasQ4KMaterialization,
}) {
  const parts = [];
  if (role) parts.push(`role=${role}`);
  if (Number.isFinite(layerIdx)) parts.push(`layer=${layerIdx}`);
  if (pathVariant) parts.push(`variant=${pathVariant}`);
  if (preferredWeightDtype) parts.push(`preferredWeightDtype=${preferredWeightDtype}`);
  if (weightDtype) parts.push(`weightDtype=${weightDtype}`);
  if (weightLabel) parts.push(`label=${weightLabel}`);
  if (weightLayout) parts.push(`layout=${weightLayout}`);
  if (weightShape) parts.push(`shape=${weightShape}`);
  if (hasQ4KMaterialization) parts.push('q4kMaterialization=true');
  return parts.length > 0 ? ` (${parts.join(', ')})` : '';
}

function requireMatmulOutputDtype(dtype, opLabel) {
  if (dtype === 'f16' || dtype === 'f32') {
    return dtype;
  }
  throw new Error(`[${opLabel}] options.outputDtype is required and must be "f16" or "f32", got ${String(dtype)}.`);
}

function requireMatmulWeightDtype(dtype, opLabel, details = []) {
  if (dtype != null && dtype !== '') {
    return dtype;
  }
  const suffix = details.filter(Boolean).length > 0
    ? ` (${details.filter(Boolean).join(', ')})`
    : '';
  throw new Error(`[${opLabel}] B dtype is required for matmul dispatch${suffix}.`);
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
  const preferredWeightDtype = resolvePreferredWeightDtype(pathVariant, hasQ4KMat, capabilities);
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
  const rawBDtype = requireMatmulWeightDtype(weightDtype ?? options.bDtype, opLabel, [
    options.role ? `role=${options.role}` : null,
    Number.isFinite(options.layerIdx) ? `layer=${options.layerIdx}` : null,
    weightLabel ? `label=${weightLabel}` : null,
  ]);
  const bDtype = toMatmulDtype(rawBDtype);
  const requestedOutputDtype = requireMatmulOutputDtype(options.outputDtype, opLabel);

  if (bDtype === 'f16' && capabilities?.hasF16 !== true) {
    const detail = buildF16CapabilityErrorDetail({
      role: options.role,
      layerIdx: options.layerIdx,
      pathVariant,
      preferredWeightDtype,
      weightDtype,
      weightLabel,
      weightLayout,
      weightShape,
      hasQ4KMaterialization: hasQ4KMat,
    });
    throw new Error(`[${opLabel}] f16 weights require shader-f16 support.${detail}`);
  }
  if (requestedOutputDtype === 'f16' && capabilities?.hasF16 !== true) {
    throw new Error(`[${opLabel}] f16 output requires shader-f16 support.`);
  }

  validateMatmulOffsets(opLabel, aOffset, bOffset, cOffset);

  const runtimeSession = getRuntimeConfig().inference?.session;
  const effectiveOptions = (
    options.useTiledQ4KPrefill == null
    || options.useWideTileQ4KPrefill == null
    || options.useWideTileQ4KDecode == null
    || options.useWideTileResidualFusion == null
    || options.useFusedRmsnormWideTile == null
  )
    ? {
        ...options,
        useTiledQ4KPrefill: options.useTiledQ4KPrefill ?? (runtimeSession?.useTiledQ4KPrefill === true),
        useWideTileQ4KPrefill: options.useWideTileQ4KPrefill ?? (runtimeSession?.useWideTileQ4KPrefill === true),
        useWideTileQ4KDecode: options.useWideTileQ4KDecode ?? (runtimeSession?.useWideTileQ4KDecode === true),
        useWideTileResidualFusion: options.useWideTileResidualFusion ?? (runtimeSession?.useWideTileResidualFusion === true),
        useFusedRmsnormWideTile: options.useFusedRmsnormWideTile ?? (runtimeSession?.useFusedRmsnormWideTile === true),
      }
    : options;

  let { variant, useQ4KFused, useGemv, useLiteRTInt4Fused = false, useW4A16Fused = false } = selectMatmulVariantAndFlags(
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
  if (bDtype === 'litert_int4') {
    const storageEncoding = resolveLiteRTInt4StorageEncoding(resolvedWeight, weightLabel);
    constants = {
      ...(constants ?? {}),
      STORAGE_OFFSET_BINARY: storageEncoding === 'offset_binary' ? 1 : 0,
    };
  }
  let w4a16ScaleBuffer = null;
  if (bDtype === 'w4a16') {
    const metadata = getWeightMetadata(resolvedWeight);
    w4a16ScaleBuffer = metadata?.scaleBuffer ?? null;
    const scaleDtype = resolveW4A16ScaleDtype(resolvedWeight, weightLabel);
    constants = {
      ...(constants ?? {}),
      SCALE_DTYPE: resolveW4A16ScaleDtypeConstant(scaleDtype),
    };
  }
  // For the rmsnorm-fused WideTile variant, forward the caller's
  // rmsNormOffset flag as a pipeline override constant. Gemma-family norm
  // weights encode `(w - 1.0)`; other models encode `w`.
  // Also forward WEIGHT_IS_F16 based on the norm weight buffer dtype so the
  // kernel correctly unpacks f16-packed weights (Gemma hidden weights) vs
  // f32 weights.
  if (variant === 'q4_fused_rmsnorm_widetile' && options.rmsNormOffset != null) {
    const normWeightDtype = getWeightDtype(options.normWeight);
    if (!normWeightDtype) {
      throw new Error('[Matmul] q4_fused_rmsnorm_widetile requires normWeight dtype metadata.');
    }
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
        if (!isRecord && castedInput) {
          releaseBuffer(castedInput.buffer);
          castedInput = null;
        }
        throw new Error(`${err.message}${detail}`);
      }
    }
    if (err instanceof Error && err.message.includes('B buffer too small')) {
      if (!isRecord && castedInput) {
        releaseBuffer(castedInput.buffer);
        castedInput = null;
      }
      throw new Error(`${err.message}${detail}`);
    }
    if (!isRecord && castedInput) {
      releaseBuffer(castedInput.buffer);
      castedInput = null;
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

  let config;
  let kernel;
  let pipeline;
  let C = null;
  let outputSize;
  let cBindingSize;
  let actualOutputDtype;
  let ownsOutput = false;
  let dispatchPlan;
  try {
    config = getMatmulConfig(variant, constants);
    kernel = new MatmulKernel(device);
    pipeline = await getMatmulPipeline(variant, constants);

    const outputInfo = resolveMatmulOutput(
      variant,
      M,
      N,
      outputBuffer
    );
    C = outputInfo.output;
    outputSize = outputInfo.outputSize;
    cBindingSize = outputInfo.cBindingSize;
    actualOutputDtype = outputInfo.actualOutputDtype;
    ownsOutput = outputBuffer == null;

    if (isAttnProj && shouldLogAttentionWeightBuffer) {
      log.warn('MatmulVariantDiag',
        `role=${options.role ?? ''} layer=${Number.isFinite(options.layerIdx) ? options.layerIdx : '?'} mode=${mode} ` +
        `variant=${variant} useQ4KFused=${useQ4KFused} useGemv=${useGemv} useLiteRTInt4Fused=${useLiteRTInt4Fused} ` +
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

    dispatchPlan = calculateMatmulDispatch(variant, useQ4KFused, useGemv, useLiteRTInt4Fused, M, N, config, useW4A16Fused);
  } catch (error) {
    if (!isRecord && castedInput) {
      releaseBuffer(castedInput.buffer);
      castedInput = null;
    }
    if (ownsOutput && C) {
      releaseBuffer(C);
    }
    throw error;
  }
  let uniformBuffer = null;
  let completed = false;
  try {
    const uniformExtras = { eps: options.rmsNormEps };
    uniformBuffer = createMatmulUniformBuffer(
      'matmul_uniforms',
      config,
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
      normWeightBuffer,
      w4a16ScaleBuffer
    );

    const bindGroup = device.createBindGroup({
      label: 'matmul_bind_group',
      layout: getPipelineBindGroupLayout(pipeline, 0),
      entries,
    });

    if (isRecord) {
      kernel.record(recorder, pipeline, bindGroup, dispatchPlan.workgroups, buildProfileLabel(options));
    } else {
      kernel.dispatch(pipeline, bindGroup, dispatchPlan.workgroups);
    }
    const tensor = createTensor(C, actualOutputDtype, [M, N], 'matmul_output');
    completed = true;
    return tensor;
  } finally {
    if (!isRecord && uniformBuffer) {
      releaseUniformBuffer(uniformBuffer);
    }
    if (!isRecord && castedInput) {
      releaseBuffer(castedInput.buffer);
      castedInput = null;
    }
    if (!completed && ownsOutput && C) {
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
