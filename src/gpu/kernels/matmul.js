import { getDevice } from '../device.js';
import { createTensor } from '../tensor.js';
import { getBuffer, getLayout, getWeightDtype } from '../weight-buffer.js';
import { log, trace, isTraceEnabled } from '../../debug/index.js';
import { releaseBuffer } from '../../memory/buffer-pool.js';
import { releaseUniformBuffer } from '../uniform-cache.js';
import { castF16ToF32, recordCastF16ToF32 } from './cast.js';
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
import {
  MatmulKernel,
  calculateMatmulDispatch,
  createMatmulUniformBuffer,
  createMatmulBindGroupLayout,
  getMatmulPipeline,
} from './matmul-dispatch.js';

export { isFusedQ4KDisabled, selectMatmulKernel };
export { createMatmulBindGroupLayout };

let _runMatmulDebugCount = 0;
let _recordMatmulDebugCount = 0;

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

function createMatmulBindGroupEntries(variant, uniformBuffer, matmulInput, bBuffer, outputBuffer, offsets, bindingSizes) {
  const isQ4KF16 = variant === 'q4_fused_multicol_f16'
    || variant === 'q4_fused_f16a'
    || variant === 'q4_fused_batched_f16'
    || variant === 'q4_fused_multicol_f16a'
    || variant === 'q4_fused_batched_f16a';

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
  }

  return entries;
}

async function executeMatmul(recorder, A, B, M, N, K, options = {}) {
  const isRecord = Boolean(recorder);
  const mode = isRecord ? 'record' : 'run';
  const opLabel = isRecord ? 'recordMatmul' : 'runMatmul';
  const device = recorder?.device || getDevice();

  const {
    alpha = 1.0,
    outputBuffer = null,
    transposeB: transposeBOption = true,
    aOffset = 0,
    bOffset = 0,
    cOffset = 0,
  } = options;

  const bBuffer = getBuffer(B);
  const weightDtype = getWeightDtype(B);
  const weightLabel = (B && typeof B === 'object' ? B.label : null) ?? bBuffer?.label ?? null;
  const weightLayout = getLayout(B);
  const weightShape = B?.shape ? `[${B.shape.join(', ')}]` : null;

  if (isTraceEnabled('kernels') && getDebugCounter(isRecord) < 20) {
    incrementDebugCounter(isRecord);
    const modeLabel = isRecord ? 'recordMatmul' : 'runMatmul';
    trace.kernels(`${modeLabel}: M=${M}, N=${N}, K=${K}, transposeBOption=${transposeBOption}, weightLayout=${weightLayout}, weightDtype=${weightDtype}`);
  }

  const transposeB = resolveTransposeB(B, transposeBOption);
  validateMatmulDimensions(opLabel, M, N, K);

  const aDtype = toMatmulDtype(A.dtype);
  const bDtype = toMatmulDtype(weightDtype ?? options.bDtype);
  const requestedOutputDtype = options.outputDtype || A.dtype;

  if (!isRecord && isTraceEnabled('kernels') && !weightDtype && !options.bDtype && M <= 2) {
    log.warn('Matmul', `runMatmul: B buffer dtype unknown! size=${bBuffer.size}, M=${M}, N=${N}, K=${K}. Assuming f32.`);
  }

  validateMatmulOffsets(opLabel, aOffset, bOffset, cOffset);

  const { variant, useQ4KFused, useGemv } = selectMatmulVariantAndFlags(
    mode,
    M,
    N,
    K,
    aDtype,
    bDtype,
    transposeB,
    requestedOutputDtype,
    options
  );

  const phase = resolveMatmulPhase(M);
  const constants = resolveMatmulConstants(options, phase);

  let matmulInput = A;
  let matmulADtype = aDtype;
  let castedInput = null;
  if (matmulADtype === 'f16' && requiresF32Input(variant)) {
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
    if (!isRecord && err instanceof Error && err.message.includes('B buffer too small')) {
      const detailParts = [];
      if (weightLabel) detailParts.push(`label=${weightLabel}`);
      if (weightDtype) detailParts.push(`weightDtype=${weightDtype}`);
      if (weightLayout) detailParts.push(`layout=${weightLayout}`);
      if (weightShape) detailParts.push(`shape=${weightShape}`);
      if (Number.isFinite(bBuffer?.size)) detailParts.push(`bSize=${bBuffer.size}`);
      const detail = detailParts.length ? ` (${detailParts.join(', ')})` : '';
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

  const config = getMatmulConfig(variant, constants);
  const kernel = new MatmulKernel(device);
  const pipeline = await getMatmulPipeline(variant, constants);

  const { output: C, outputSize, cBindingSize, actualOutputDtype } = resolveMatmulOutput(
    variant,
    M,
    N,
    outputBuffer
  );

  if (!Number.isFinite(outputSize) || outputSize <= 0) {
    throw new Error(`[${opLabel}] Invalid output size: ${outputSize} (M=${M}, N=${N})`);
  }

  const cRequired = cOffset + cBindingSize;
  if (C.size < cRequired) {
    throw new Error(`[${opLabel}] Output buffer too small: ${C.size} < ${cRequired} (M=${M}, N=${N})`);
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
    recorder || null,
    device
  );

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
    }
  );

  const bindGroup = device.createBindGroup({
    label: 'matmul_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });

  if (isRecord) {
    kernel.record(recorder, pipeline, bindGroup, dispatchPlan.workgroups, buildProfileLabel(options));
  } else {
    kernel.dispatch(pipeline, bindGroup, dispatchPlan.workgroups);
    releaseUniformBuffer(uniformBuffer);
    if (castedInput) {
      releaseBuffer(castedInput.buffer);
    }
  }

  return createTensor(C, actualOutputDtype, [M, N], 'matmul_output');
}


export async function runMatmul(A, B, M, N, K, options = {}) {
  return executeMatmul(null, A, B, M, N, K, options);
}


export async function recordMatmul(recorder, A, B, M, N, K, options = {}) {
  return executeMatmul(recorder, A, B, M, N, K, options);
}
