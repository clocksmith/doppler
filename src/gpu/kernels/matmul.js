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

  const phase = resolveMatmulPhase(M);
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

  const phase = resolveMatmulPhase(M);
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

  const layerLabel = Number.isFinite(options.layerIdx) ? `:L${options.layerIdx}` : '';
  const roleLabel = options.role ? `:${options.role}` : '';
  const profileLabel = `matmul${roleLabel}${layerLabel}`;
  kernel.record(recorder, pipeline, bindGroup, dispatchPlan.workgroups, profileLabel);
  return createTensor(C, actualOutputDtype, [M, N], 'matmul_output');
}
