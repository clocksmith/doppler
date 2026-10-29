import { executePairedLinearAttention } from './paired-linear-attention.js';
import { getDevice } from '../device.js';
import { getShaderScopeCacheKey } from './shader-source-scope.js';
import { WORKGROUP_SIZES } from './constants.js';
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor } from '../tensor.js';
import { getRuntimeConfig } from '../../config/runtime.js';
import { castF32ToF16, recordCastF32ToF16 } from './cast.js';
import { createUniformBufferFromData } from './uniform-utils.js';
import { getShaderModule } from './shader-cache.js';
import {
  getOrCreateBindGroupLayout,
  getOrCreatePipelineLayout,
} from './pipeline-cache.js';
import { recordDispatch } from './dispatch.js';
import { selectRuleValue } from '../../rules/rule-registry.js';
import { assertImplicitDtypeTransitionAllowed } from '../../config/dtype-transition-contract.js';
import { getKernelConfig } from './kernel-configs.js';
import { requireNormGateActivation, resolveGatedDeltaNormGateKernel } from './gated-delta-norm-gate.js';

const CONV_WORKGROUP_SIZE = WORKGROUP_SIZES.DEFAULT;
const HEAD_WORKGROUP_SIZE = 128;

let cachedEpoch = -1;
const pipelineCache = new Map();
let convBindGroupLayout = null;
let recurrentBindGroupLayout = null;
let fusedDecodeBindGroupLayout = null;

function createBindGroupLayouts(device) {
  convBindGroupLayout = getOrCreateBindGroupLayout(
    'linear_attention_conv_layout',
    [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
    device
  );
  recurrentBindGroupLayout = getOrCreateBindGroupLayout(
    'linear_attention_recurrent_layout',
    [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
    device
  );
  fusedDecodeBindGroupLayout = getOrCreateBindGroupLayout(
    'linear_attention_fused_decode_layout',
    [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
    device
  );
}

async function createPipelines(device, inputDtype, stateComponents, normGateActivation, useRegisterPrefill, useRegisterDecode) {
  createBindGroupLayouts(device);
  const variant = inputDtype === 'f16' ? 'f16' : 'f32';
  const suffix = variant === 'f16' ? '_f16' : '';
  // The gate-applying kernels come from their rule maps and must declare the
  // model's norm-gate activation; silu resolves to the original shader files.
  const gateContext = { inputDtype: variant, normGateActivation, useRegisterPrefill, useRegisterDecode };
  const recurrentConfig = resolveGatedDeltaNormGateKernel(
    'gated_delta_recurrent', 'gatedDeltaRecurrent', 'scalarVariant', gateContext
  );
  const fusedDecodeConfig = resolveGatedDeltaNormGateKernel(
    'gated_delta_fused_decode', 'gatedDeltaFusedDecode', 'scalarVariant', { ...gateContext, stateComponents }
  );
  const convConfig = recurrentConfig.variantMetadata?.convVariant
    ? getKernelConfig('gated_delta_conv', recurrentConfig.variantMetadata.convVariant) : null;
  const [convModule, recurrentModule, fusedDecodeModule] = await Promise.all([
    getShaderModule(device, convConfig?.shaderFile ?? `gated_delta_conv${suffix}.wgsl`, `gated_delta_conv_${variant}`),
    getShaderModule(device, recurrentConfig.shaderFile, `gated_delta_recurrent_${variant}`),
    getShaderModule(device, fusedDecodeConfig.shaderFile, `gated_delta_fused_decode_${variant}`),
  ]);

  const convPipeline = device.createComputePipeline({
    label: `linear_attention_conv_pipeline_${variant}`,
    layout: getOrCreatePipelineLayout('linear_attention_conv_pipeline_layout', [convBindGroupLayout], device),
    compute: {
      module: convModule,
      entryPoint: convConfig?.entryPoint ?? 'main',
      constants: {
        WORKGROUP_SIZE: CONV_WORKGROUP_SIZE,
      },
    },
  });
  let parallelConvPipeline = null;
  if (recurrentConfig.variantMetadata?.parallelConvVariant) {
    const parallel = getKernelConfig('gated_delta_conv', recurrentConfig.variantMetadata.parallelConvVariant);
    parallelConvPipeline = device.createComputePipeline({
      label: 'linear_attention_parallel_conv',
      layout: getOrCreatePipelineLayout('linear_attention_conv_pipeline_layout', [convBindGroupLayout], device),
      compute: { module: convModule, entryPoint: parallel.entryPoint, constants: { WORKGROUP_SIZE: CONV_WORKGROUP_SIZE } },
    });
  }
  const recurrentPipeline = device.createComputePipeline({
    label: `linear_attention_recurrent_pipeline_${variant}`,
    layout: getOrCreatePipelineLayout('linear_attention_recurrent_pipeline_layout', [recurrentBindGroupLayout], device),
    compute: {
      module: recurrentModule,
      entryPoint: 'main',
      constants: {
        WORKGROUP_SIZE: recurrentConfig.workgroupSize[0],
        RECURRENT_STATE_COMPONENTS: stateComponents,
      },
    },
  });
  const normGateVariant = recurrentConfig.variantMetadata?.normGateVariant;
  let normGatePipeline = null;
  if (normGateVariant) {
    const normConfig = getKernelConfig('gated_delta_recurrent', normGateVariant);
    const normModule = await getShaderModule(device, normConfig.shaderFile, 'linear_prefill_norm_gate');
    normGatePipeline = device.createComputePipeline({
      label: 'linear_prefill_norm_gate',
      layout: getOrCreatePipelineLayout('linear_attention_recurrent_pipeline_layout', [recurrentBindGroupLayout], device),
      compute: { module: normModule, entryPoint: normConfig.entryPoint,
        constants: { WORKGROUP_SIZE: normConfig.workgroupSize[0], RECURRENT_STATE_COMPONENTS: stateComponents } },
    });
  }
  const fusedDecodePipeline = device.createComputePipeline({
    label: `linear_attention_fused_decode_pipeline_${variant}`,
    layout: getOrCreatePipelineLayout(
      'linear_attention_fused_decode_pipeline_layout',
      [fusedDecodeBindGroupLayout],
      device
    ),
    compute: {
      module: fusedDecodeModule,
      entryPoint: 'main',
      constants: {
        WORKGROUP_SIZE: HEAD_WORKGROUP_SIZE,
        RECURRENT_STATE_COMPONENTS: stateComponents,
      },
    },
  });

  return { convPipeline, parallelConvPipeline, recurrentPipeline, fusedDecodePipeline, normGatePipeline,
    valueSliceWidth: recurrentConfig.variantMetadata?.valueSliceWidth ?? null };
}

function normalizeInputDtype(dtype) {
  return selectRuleValue('shared', 'dtype', 'f16OrF32FromDtype', { dtype });
}

function ensurePipelines(device, inputDtype, stateComponents, normGateActivation, useRegisterPrefill, useRegisterDecode) {
  const epoch = getShaderScopeCacheKey();
  if (epoch !== cachedEpoch) {
      pipelineCache.clear();
      convBindGroupLayout = null;
      recurrentBindGroupLayout = null;
      fusedDecodeBindGroupLayout = null;
      cachedEpoch = epoch;
  }
  const dtype = normalizeInputDtype(inputDtype);
  const variant = `${dtype}:${stateComponents}:${normGateActivation}:${useRegisterPrefill}:${useRegisterDecode}`;
  const cached = pipelineCache.get(variant);
  if (cached) return cached;
  const task = createPipelines(device, dtype, stateComponents, normGateActivation, useRegisterPrefill, useRegisterDecode).then(value => {
    if (cachedEpoch === epoch && pipelineCache.get(variant) === task) pipelineCache.set(variant, value);
    return value;
  }, error => {
    if (pipelineCache.get(variant) === task) pipelineCache.delete(variant);
    throw error;
  });
  pipelineCache.set(variant, task);
  return task;
}

function buildParamsData(params) {
  const data = new ArrayBuffer(64);
  const view = new DataView(data);
  view.setUint32(0, params.numTokens, true);
  view.setUint32(4, params.convDim, true);
  view.setUint32(8, params.convKernelSize, true);
  view.setUint32(12, params.numVHeads, true);
  view.setUint32(16, params.numKHeads, true);
  view.setUint32(20, params.headKDim, true);
  view.setUint32(24, params.headVDim, true);
  view.setUint32(28, params.qSize, true);
  view.setUint32(32, params.kSize, true);
  view.setUint32(36, params.valueDim, true);
  view.setUint32(40, params.qRep, true);
  view.setUint32(44, params.normMode, true);
  view.setFloat32(48, params.rmsNormEps, true);
  view.setFloat32(52, params.qkL2NormEps, true);
  const packedFlags = (params.abPacked ? 1 : 0) | (params.qkvzPacked ? 2 : 0);
  view.setUint32(56, packedFlags, true);
  view.setUint32(60, params.bProjOffsetElements ?? 0, true);
  return data;
}

// Epsilons come from the manifest (rmsNormEps) and the caller (qkL2NormEps);
// a missing or non-positive value is a contract error, not a default.
function requirePositiveEpsilon(value, label) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    throw new Error(`[LinearAttention] ${label} must be a positive finite number; got ${JSON.stringify(value)}.`);
  }
  return value;
}

function requireGpuBuffer(buffer, label) {
  if (!(buffer instanceof GPUBuffer)) {
    throw new Error(`linear_attention kernel requires GPUBuffer for ${label}.`);
  }
}

function resolveOutputDtype(outputDtype) {
  const normalized = selectRuleValue('shared', 'dtype', 'f16OrF32FromDtype', { dtype: outputDtype });
  if (normalized === 'f16' || normalized === 'f32') {
    return normalized;
  }
  throw new Error(`linear_attention core output dtype "${outputDtype}" is invalid.`);
}

function canUseFusedDecodeCore(layerState, numTokens, options, aTensor, bTensor) {
  const session = options.runtimeSession ?? getRuntimeConfig().inference.session;
  return session?.useLinearAttentionFusedDecodeCore === true
    && numTokens === 1
    && options?.abPacked === true
    && Number.isFinite(options?.bProjOffsetElements)
    && options.bProjOffsetElements > 0
    && aTensor?.buffer === bTensor?.buffer
    && layerState.qRep === 1
    && layerState.numKHeads === layerState.numVHeads
    && layerState.headKDim <= HEAD_WORKGROUP_SIZE
    && layerState.headVDim <= HEAD_WORKGROUP_SIZE
    && layerState.qSize === layerState.numKHeads * layerState.headKDim
    && layerState.kSize === layerState.numKHeads * layerState.headKDim
    && layerState.vSize === layerState.numVHeads * layerState.headVDim
    && layerState.convDim === layerState.qSize + layerState.kSize + layerState.vSize;
}

export async function runLinearAttentionCoreGPU(qkvTensor, zTensor, aTensor, bTensor, layerState, options = {}) {
  if (qkvTensor?.precision === 'two-f32') {
    const session = options.runtimeSession ?? getRuntimeConfig().inference.session;
    if ((options.numTokens === 1 && session.useLinearAttentionRegisterDecode === true)
        || (options.numTokens > 1 && session.useLinearAttentionRegisterPrefill === true)) {
      throw new Error('Register linear attention requires scalar F32 inputs and state.');
    }
    return executePairedLinearAttention(qkvTensor, zTensor, aTensor, bTensor, layerState, options);
  }
  const device = getDevice();
  if (!device) {
    throw new Error('No GPU device available for linear_attention core.');
  }
  const recorder = options.recorder ?? null;
  const useRecorder = recorder
    && typeof recorder.getEncoder === 'function'
    && typeof recorder.trackTemporaryBuffer === 'function';

  requireGpuBuffer(qkvTensor?.buffer, 'qkvTensor');
  requireGpuBuffer(zTensor?.buffer, 'zTensor');
  requireGpuBuffer(aTensor?.buffer, 'aTensor');
  requireGpuBuffer(bTensor?.buffer, 'bTensor');
  requireGpuBuffer(layerState?.convWeightGPU, 'convWeightGPU');
  requireGpuBuffer(layerState?.dtBiasGPU, 'dtBiasGPU');
  requireGpuBuffer(layerState?.aLogGPU, 'aLogGPU');
  requireGpuBuffer(layerState?.normWeightGPU, 'normWeightGPU');
  requireGpuBuffer(layerState?.convStateGPU, 'convStateGPU');
  requireGpuBuffer(layerState?.recurrentStateGPU, 'recurrentStateGPU');

  const numTokens = Number(options.numTokens ?? 0);
  if (!Number.isFinite(numTokens) || numTokens <= 0) {
    throw new Error('runLinearAttentionCoreGPU requires numTokens > 0.');
  }
  if (!Number.isFinite(layerState.headVDim) || layerState.headVDim <= 0) {
    throw new Error(`linear_attention requires positive headVDim, got ${layerState.headVDim}.`);
  }
  if (layerState.normMode !== 'shared' && layerState.normMode !== 'per_head') {
    throw new Error(`linear_attention requires supported normMode, got ${layerState.normMode}.`);
  }
  const normGateActivation = requireNormGateActivation(options.normGateActivation, 'linear_attention core');

  const inputDtype = normalizeInputDtype(qkvTensor?.dtype);
  if (normalizeInputDtype(zTensor?.dtype) !== inputDtype) {
    throw new Error(`linear_attention core requires matching qkv/z dtypes; got ${qkvTensor?.dtype} and ${zTensor?.dtype}.`);
  }
  if (normalizeInputDtype(aTensor?.dtype) !== inputDtype) {
    throw new Error(`linear_attention core requires matching qkv/a dtypes; got ${qkvTensor?.dtype} and ${aTensor?.dtype}.`);
  }
  if (normalizeInputDtype(bTensor?.dtype) !== inputDtype) {
    throw new Error(`linear_attention core requires matching qkv/b dtypes; got ${qkvTensor?.dtype} and ${bTensor?.dtype}.`);
  }

  const precision = layerState.recurrentStatePrecision;
  if (precision !== 'f32' && precision !== 'two-f32') {
    throw new Error('linear_attention core requires an explicit recurrent state precision.');
  }
  const stateComponents = selectRuleValue('kernels', 'linearAttention', 'stateComponents', { precision });
  const stateBytes = layerState.numVHeads * layerState.headKDim * layerState.headVDim * stateComponents * 4;
  if (![layerState.numVHeads, layerState.headKDim, layerState.headVDim].every(value => Number.isSafeInteger(value) && value > 0)
      || !Number.isSafeInteger(stateBytes) || stateBytes <= 0
      || layerState.recurrentStateComponents !== stateComponents
      || !Number.isSafeInteger(layerState.recurrentStateGPU.size)
      || layerState.recurrentStateGPU.size < stateBytes) {
    throw new Error('linear_attention recurrent state binding does not match declared storage layout.');
  }
  const prefillSession = options.runtimeSession ?? getRuntimeConfig().inference.session;
  if (numTokens > 1 && typeof prefillSession.useLinearAttentionRegisterPrefill !== 'boolean') {
    throw new Error('Linear attention prefill requires boolean useLinearAttentionRegisterPrefill.');
  }
  const useRegisterPrefill = numTokens > 1 && prefillSession.useLinearAttentionRegisterPrefill;
  if (useRegisterPrefill) {
    const selected = resolveGatedDeltaNormGateKernel('gated_delta_recurrent', 'gatedDeltaRecurrent', 'scalarVariant',
      { inputDtype, normGateActivation, useRegisterPrefill });
    const shape = selected.variantMetadata;
    if (!Number.isSafeInteger(shape?.headKDim) || layerState.headKDim !== shape.headKDim
        || stateComponents !== shape.stateComponents || layerState.headVDim > shape.maxHeadVDim) {
      throw new Error('Register prefill requires its registered head width, F32 state/inputs and silu norm gating.');
    }
  }
  if (numTokens === 1 && typeof prefillSession.useLinearAttentionRegisterDecode !== 'boolean') {
    throw new Error('Linear attention decode requires boolean useLinearAttentionRegisterDecode.');
  }
  const useRegisterDecode = numTokens === 1 && prefillSession.useLinearAttentionRegisterDecode;
  const useFusedDecodeCore = canUseFusedDecodeCore(layerState, numTokens, options, aTensor, bTensor);
  if (useRegisterDecode) {
    const selected = resolveGatedDeltaNormGateKernel('gated_delta_fused_decode', 'gatedDeltaFusedDecode', 'scalarVariant',
      { inputDtype, normGateActivation, stateComponents, useRegisterDecode });
    const shape = selected.variantMetadata;
    if (!useFusedDecodeCore || layerState.headKDim !== shape?.headKDim || stateComponents !== shape.stateComponents
        || layerState.headVDim > shape.maxHeadVDim) {
      throw new Error('Register decode requires its declared F32 fused-core geometry and packed AB input.');
    }
  }
  const prepared = ensurePipelines(device, inputDtype, stateComponents, normGateActivation, useRegisterPrefill, useRegisterDecode);
  const pipelines = prepared instanceof Promise ? await prepared : prepared;
  if (!pipelines) {
    throw new Error(`linear_attention core failed to resolve pipelines for dtype "${inputDtype}".`);
  }

  const convOutSize = numTokens * layerState.convDim * Float32Array.BYTES_PER_ELEMENT;
  const outputSize = numTokens * layerState.valueDim * Float32Array.BYTES_PER_ELEMENT;
  let convOutBuffer = null;
  const outputDtype = resolveOutputDtype(options.outputDtype);
  const outputShape = [numTokens, layerState.valueDim];
  const paramsPayload = {
    numTokens,
    convDim: layerState.convDim,
    convKernelSize: layerState.convKernelSize,
    numVHeads: layerState.numVHeads,
    numKHeads: layerState.numKHeads,
    headKDim: layerState.headKDim,
    headVDim: layerState.headVDim,
    qSize: layerState.qSize,
    kSize: layerState.kSize,
    valueDim: layerState.valueDim,
    qRep: layerState.qRep,
    normMode: layerState.normMode === 'per_head' ? 1 : 0,
    rmsNormEps: requirePositiveEpsilon(layerState.rmsNormEps, 'layerState.rmsNormEps'),
    qkL2NormEps: requirePositiveEpsilon(options.qkL2NormEps, 'options.qkL2NormEps'),
    abPacked: options.abPacked === true,
    qkvzPacked: options.qkvzPacked === true,
    bProjOffsetElements: options.bProjOffsetElements,
  };
  const workspace = useFusedDecodeCore && numTokens === 1 && outputDtype === 'f32'
    && Number.isInteger(options.layerIdx) && options.layerIdx >= 0 ? recorder?.decodeWorkspace : null;
  const outputBuffer = workspace
    ? workspace.getBuffer(`linear_core:${options.layerIdx}`, outputSize)
    : acquireBuffer(outputSize, undefined, `L${options.layerIdx ?? 0}.linear_attention_core_out`);
  if (useFusedDecodeCore) {
    if (useRecorder) {
      try {
        const paramsBuffer = createUniformBufferFromData(
          'linear_attention_params',
          buildParamsData(paramsPayload),
          recorder
        );
        const descriptor = {
          label: 'linear_attention_fused_decode_bind_group',
          layout: fusedDecodeBindGroupLayout,
          entries: [
            { binding: 0, resource: { buffer: paramsBuffer } },
            { binding: 1, resource: { buffer: qkvTensor.buffer } },
            { binding: 2, resource: { buffer: zTensor.buffer } },
            { binding: 3, resource: { buffer: aTensor.buffer } },
            { binding: 4, resource: { buffer: layerState.convWeightGPU } },
            { binding: 5, resource: { buffer: layerState.convStateGPU } },
            { binding: 6, resource: { buffer: layerState.dtBiasGPU } },
            { binding: 7, resource: { buffer: layerState.aLogGPU } },
            { binding: 8, resource: { buffer: layerState.normWeightGPU } },
            { binding: 9, resource: { buffer: layerState.recurrentStateGPU } },
            { binding: 10, resource: { buffer: outputBuffer } },
          ],
        };
        const fusedBindGroup = workspace
          ? workspace.getBindGroup(`linear_core:${options.layerIdx}`, descriptor)
          : device.createBindGroup(descriptor);
        recordDispatch(
          recorder,
          pipelines.fusedDecodePipeline,
          fusedBindGroup,
          [layerState.numVHeads, 1, 1],
          'linear_attention_fused_decode_core'
        );
        const output = createTensor(
          outputBuffer,
          'f32',
          outputShape,
          `L${options.layerIdx ?? 0}.linear_attention_core`
        );
        if (outputDtype === 'f16') {
          assertImplicitDtypeTransitionAllowed({
            executionPolicies: options.executionPolicies ?? null,
            fromDtype: output.dtype,
            toDtype: 'f16',
            op: 'linear_attention_core',
            detail: 'Linear attention core would narrow activations implicitly.',
          });
          const casted = await recordCastF32ToF16(recorder, output);
          recorder.trackTemporaryBuffer(outputBuffer);
          return casted;
        }
        return output;
      } catch (error) {
        if (!workspace) releaseBuffer(outputBuffer);
        throw error;
      }
    }

    let paramsBuffer = null;
    let submitted = false;
    try {
      paramsBuffer = createUniformBufferFromData(
        'linear_attention_params',
        buildParamsData(paramsPayload),
        null,
        device,
        { useCache: false }
      );
      const fusedBindGroup = device.createBindGroup({
        label: 'linear_attention_fused_decode_bind_group',
        layout: fusedDecodeBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: qkvTensor.buffer } },
          { binding: 2, resource: { buffer: zTensor.buffer } },
          { binding: 3, resource: { buffer: aTensor.buffer } },
          { binding: 4, resource: { buffer: layerState.convWeightGPU } },
          { binding: 5, resource: { buffer: layerState.convStateGPU } },
          { binding: 6, resource: { buffer: layerState.dtBiasGPU } },
          { binding: 7, resource: { buffer: layerState.aLogGPU } },
          { binding: 8, resource: { buffer: layerState.normWeightGPU } },
          { binding: 9, resource: { buffer: layerState.recurrentStateGPU } },
          { binding: 10, resource: { buffer: outputBuffer } },
        ],
      });
      const encoder = device.createCommandEncoder({ label: 'linear_attention_fused_decode_core' });
      const pass = encoder.beginComputePass({ label: 'linear_attention_fused_decode_core_pass' });
      pass.setPipeline(pipelines.fusedDecodePipeline);
      pass.setBindGroup(0, fusedBindGroup);
      pass.dispatchWorkgroups(layerState.numVHeads, 1, 1);
      pass.end();
      device.queue.submit([encoder.finish()]);
      submitted = true;

      const output = createTensor(
        outputBuffer,
        'f32',
        outputShape,
        `L${options.layerIdx ?? 0}.linear_attention_core`
      );
      if (outputDtype === 'f16') {
        assertImplicitDtypeTransitionAllowed({
          executionPolicies: options.executionPolicies ?? null,
          fromDtype: output.dtype,
          toDtype: 'f16',
          op: 'linear_attention_core',
          detail: 'Linear attention core would narrow activations implicitly.',
        });
        const casted = await castF32ToF16(output);
        if (!workspace) releaseBuffer(outputBuffer);
        return casted;
      }
      return output;
    } catch (error) {
      if (!workspace) releaseBuffer(outputBuffer);
      throw error;
    } finally {
      if (submitted) {
        device.queue.onSubmittedWorkDone()
          .then(() => {
            paramsBuffer?.destroy();
          })
          .catch(() => {
            paramsBuffer?.destroy();
          });
      } else {
        paramsBuffer?.destroy();
      }
    }
  }

  try {
    convOutBuffer = acquireBuffer(convOutSize, undefined, `L${options.layerIdx ?? 0}.linear_conv_out`);
  } catch (error) {
    if (!workspace) releaseBuffer(outputBuffer);
    throw error;
  }
  if (useRecorder) {
    try {
      const paramsBuffer = createUniformBufferFromData(
        'linear_attention_params',
        buildParamsData({
          numTokens,
          convDim: layerState.convDim,
          convKernelSize: layerState.convKernelSize,
          numVHeads: layerState.numVHeads,
          numKHeads: layerState.numKHeads,
          headKDim: layerState.headKDim,
          headVDim: layerState.headVDim,
          qSize: layerState.qSize,
          kSize: layerState.kSize,
          valueDim: layerState.valueDim,
          qRep: layerState.qRep,
          normMode: layerState.normMode === 'per_head' ? 1 : 0,
          rmsNormEps: requirePositiveEpsilon(layerState.rmsNormEps, 'layerState.rmsNormEps'),
          qkL2NormEps: requirePositiveEpsilon(options.qkL2NormEps, 'options.qkL2NormEps'),
          abPacked: options.abPacked === true,
          qkvzPacked: options.qkvzPacked === true,
          bProjOffsetElements: options.bProjOffsetElements,
        }),
        recorder
      );
      const convBindGroup = device.createBindGroup({
        label: 'linear_attention_conv_bind_group',
        layout: convBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: qkvTensor.buffer } },
          { binding: 2, resource: { buffer: layerState.convWeightGPU } },
          { binding: 3, resource: { buffer: layerState.convStateGPU } },
          { binding: 4, resource: { buffer: convOutBuffer } },
        ],
      });

      const recurrentBindGroup = device.createBindGroup({
        label: 'linear_attention_recurrent_bind_group',
        layout: recurrentBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: convOutBuffer } },
          { binding: 2, resource: { buffer: zTensor.buffer } },
          { binding: 3, resource: { buffer: aTensor.buffer } },
          { binding: 4, resource: { buffer: bTensor.buffer } },
          { binding: 5, resource: { buffer: layerState.dtBiasGPU } },
          { binding: 6, resource: { buffer: layerState.aLogGPU } },
          { binding: 7, resource: { buffer: layerState.normWeightGPU } },
          { binding: 8, resource: { buffer: layerState.recurrentStateGPU } },
          { binding: 9, resource: { buffer: outputBuffer } },
        ],
      });

      recordDispatch(
        recorder,
        pipelines.convPipeline,
        convBindGroup,
        [Math.ceil(layerState.convDim / CONV_WORKGROUP_SIZE), 1, 1],
        'linear_attention_conv'
      );
      if (pipelines.parallelConvPipeline) recordDispatch(recorder, pipelines.parallelConvPipeline,
        convBindGroup, [Math.ceil(numTokens * layerState.convDim / CONV_WORKGROUP_SIZE), 1, 1], 'linear_attention_parallel_conv');
      recordDispatch(
        recorder,
        pipelines.recurrentPipeline,
        recurrentBindGroup,
        [layerState.numVHeads, pipelines.valueSliceWidth ? Math.ceil(layerState.headVDim / pipelines.valueSliceWidth) : 1, 1],
        'linear_attention_recurrent'
      );
      if (pipelines.normGatePipeline) recordDispatch(recorder, pipelines.normGatePipeline,
        recurrentBindGroup, [layerState.numVHeads, numTokens, 1], 'linear_prefill_norm_gate');

      recorder.trackTemporaryBuffer(convOutBuffer);

      const output = createTensor(
        outputBuffer,
        'f32',
        outputShape,
        `L${options.layerIdx ?? 0}.linear_attention_core`
      );
      if (outputDtype === 'f16') {
        assertImplicitDtypeTransitionAllowed({
          executionPolicies: options.executionPolicies ?? null,
          fromDtype: output.dtype,
          toDtype: 'f16',
          op: 'linear_attention_core',
          detail: 'Linear attention core would narrow activations implicitly.',
        });
        const casted = await recordCastF32ToF16(recorder, output);
        recorder.trackTemporaryBuffer(outputBuffer);
        return casted;
      }
      return output;
    } catch (error) {
      releaseBuffer(convOutBuffer);
      if (!workspace) releaseBuffer(outputBuffer);
      throw error;
    }
  }

  let paramsBuffer = null;
  let submitted = false;
  try {
    paramsBuffer = createUniformBufferFromData(
      'linear_attention_params',
      buildParamsData({
        numTokens,
        convDim: layerState.convDim,
        convKernelSize: layerState.convKernelSize,
        numVHeads: layerState.numVHeads,
        numKHeads: layerState.numKHeads,
        headKDim: layerState.headKDim,
        headVDim: layerState.headVDim,
        qSize: layerState.qSize,
        kSize: layerState.kSize,
        valueDim: layerState.valueDim,
        qRep: layerState.qRep,
        normMode: layerState.normMode === 'per_head' ? 1 : 0,
        rmsNormEps: requirePositiveEpsilon(layerState.rmsNormEps, 'layerState.rmsNormEps'),
        qkL2NormEps: requirePositiveEpsilon(options.qkL2NormEps, 'options.qkL2NormEps'),
        abPacked: options.abPacked === true,
        qkvzPacked: options.qkvzPacked === true,
        bProjOffsetElements: options.bProjOffsetElements,
      }),
      null,
      device,
      { useCache: false }
    );
    const convBindGroup = device.createBindGroup({
      label: 'linear_attention_conv_bind_group',
      layout: convBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: paramsBuffer } },
        { binding: 1, resource: { buffer: qkvTensor.buffer } },
        { binding: 2, resource: { buffer: layerState.convWeightGPU } },
        { binding: 3, resource: { buffer: layerState.convStateGPU } },
        { binding: 4, resource: { buffer: convOutBuffer } },
      ],
    });

    const recurrentBindGroup = device.createBindGroup({
      label: 'linear_attention_recurrent_bind_group',
      layout: recurrentBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: paramsBuffer } },
        { binding: 1, resource: { buffer: convOutBuffer } },
        { binding: 2, resource: { buffer: zTensor.buffer } },
        { binding: 3, resource: { buffer: aTensor.buffer } },
        { binding: 4, resource: { buffer: bTensor.buffer } },
        { binding: 5, resource: { buffer: layerState.dtBiasGPU } },
        { binding: 6, resource: { buffer: layerState.aLogGPU } },
        { binding: 7, resource: { buffer: layerState.normWeightGPU } },
        { binding: 8, resource: { buffer: layerState.recurrentStateGPU } },
        { binding: 9, resource: { buffer: outputBuffer } },
      ],
    });

    const encoder = device.createCommandEncoder({ label: 'linear_attention_core' });

    {
      const pass = encoder.beginComputePass({ label: 'linear_attention_conv_pass' });
      pass.setPipeline(pipelines.convPipeline);
      pass.setBindGroup(0, convBindGroup);
      pass.dispatchWorkgroups(Math.ceil(layerState.convDim / CONV_WORKGROUP_SIZE), 1, 1);
      if (pipelines.parallelConvPipeline) {
        pass.setPipeline(pipelines.parallelConvPipeline);
        pass.setBindGroup(0, convBindGroup);
        pass.dispatchWorkgroups(Math.ceil(numTokens * layerState.convDim / CONV_WORKGROUP_SIZE), 1, 1);
      }
      pass.end();
    }

    {
      const pass = encoder.beginComputePass({ label: 'linear_attention_recurrent_pass' });
      pass.setPipeline(pipelines.recurrentPipeline);
      pass.setBindGroup(0, recurrentBindGroup);
      pass.dispatchWorkgroups(layerState.numVHeads,
        pipelines.valueSliceWidth ? Math.ceil(layerState.headVDim / pipelines.valueSliceWidth) : 1, 1);
      if (pipelines.normGatePipeline) {
        pass.setPipeline(pipelines.normGatePipeline);
        pass.setBindGroup(0, recurrentBindGroup);
        pass.dispatchWorkgroups(layerState.numVHeads, numTokens, 1);
      }
      pass.end();
    }

    device.queue.submit([encoder.finish()]);
    submitted = true;

    const output = createTensor(
      outputBuffer,
      'f32',
      outputShape,
      `L${options.layerIdx ?? 0}.linear_attention_core`
    );
    if (outputDtype === 'f16') {
      assertImplicitDtypeTransitionAllowed({
        executionPolicies: options.executionPolicies ?? null,
        fromDtype: output.dtype,
        toDtype: 'f16',
        op: 'linear_attention_core',
        detail: 'Linear attention core would narrow activations implicitly.',
      });
      const casted = await castF32ToF16(output);
      if (!workspace) releaseBuffer(outputBuffer);
      return casted;
    }
    return output;
  } catch (error) {
    if (!workspace) releaseBuffer(outputBuffer);
    throw error;
  } finally {
    if (submitted) {
      device.queue.onSubmittedWorkDone()
        .then(() => {
          paramsBuffer?.destroy();
        })
        .catch(() => {
          paramsBuffer?.destroy();
        });
    } else {
      paramsBuffer?.destroy();
    }
    releaseBuffer(convOutBuffer);
  }
}
