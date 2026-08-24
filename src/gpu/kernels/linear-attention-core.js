import { getDevice, getDeviceEpoch } from '../device.js';
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

async function createPipelines(device, inputDtype) {
  createBindGroupLayouts(device);
  const variant = inputDtype === 'f16' ? 'f16' : 'f32';
  const suffix = variant === 'f16' ? '_f16' : '';
  const [convModule, recurrentModule, fusedDecodeModule] = await Promise.all([
    getShaderModule(device, `gated_delta_conv${suffix}.wgsl`, `gated_delta_conv_${variant}`),
    getShaderModule(device, `gated_delta_recurrent${suffix}.wgsl`, `gated_delta_recurrent_${variant}`),
    getShaderModule(device, `gated_delta_fused_decode${suffix}.wgsl`, `gated_delta_fused_decode_${variant}`),
  ]);

  const convPipeline = device.createComputePipeline({
    label: `linear_attention_conv_pipeline_${variant}`,
    layout: getOrCreatePipelineLayout('linear_attention_conv_pipeline_layout', [convBindGroupLayout], device),
    compute: {
      module: convModule,
      entryPoint: 'main',
      constants: {
        WORKGROUP_SIZE: CONV_WORKGROUP_SIZE,
      },
    },
  });
  const recurrentPipeline = device.createComputePipeline({
    label: `linear_attention_recurrent_pipeline_${variant}`,
    layout: getOrCreatePipelineLayout('linear_attention_recurrent_pipeline_layout', [recurrentBindGroupLayout], device),
    compute: {
      module: recurrentModule,
      entryPoint: 'main',
      constants: {
        WORKGROUP_SIZE: HEAD_WORKGROUP_SIZE,
      },
    },
  });
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
      },
    },
  });

  return { convPipeline, recurrentPipeline, fusedDecodePipeline };
}

function normalizeInputDtype(dtype) {
  return selectRuleValue('shared', 'dtype', 'f16OrF32FromDtype', { dtype });
}

async function ensurePipelines(device, inputDtype) {
  const epoch = getDeviceEpoch();
  if (epoch !== cachedEpoch) {
      pipelineCache.clear();
      convBindGroupLayout = null;
      recurrentBindGroupLayout = null;
      fusedDecodeBindGroupLayout = null;
      cachedEpoch = epoch;
  }
  const variant = normalizeInputDtype(inputDtype);
  if (!pipelineCache.has(variant)) {
    pipelineCache.set(variant, createPipelines(device, variant));
    cachedEpoch = epoch;
  }
  try {
    return await pipelineCache.get(variant);
  } catch (error) {
    pipelineCache.delete(variant);
    throw error;
  }
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
  const session = getRuntimeConfig()?.inference?.session;
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

  const pipelines = await ensurePipelines(device, inputDtype);
  if (!pipelines) {
    throw new Error(`linear_attention core failed to resolve pipelines for dtype "${inputDtype}".`);
  }

  const convOutSize = numTokens * layerState.convDim * Float32Array.BYTES_PER_ELEMENT;
  const outputSize = numTokens * layerState.valueDim * Float32Array.BYTES_PER_ELEMENT;
  let convOutBuffer = null;
  const outputBuffer = acquireBuffer(outputSize, undefined, `L${options.layerIdx ?? 0}.linear_attention_core_out`);
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
    rmsNormEps: Number(layerState.rmsNormEps) || 1e-6,
    qkL2NormEps: Number(options.qkL2NormEps) || 1e-6,
    abPacked: options.abPacked === true,
    qkvzPacked: options.qkvzPacked === true,
    bProjOffsetElements: options.bProjOffsetElements,
  };
  const useFusedDecodeCore = canUseFusedDecodeCore(layerState, numTokens, options, aTensor, bTensor);
  if (useFusedDecodeCore) {
    if (useRecorder) {
      const paramsBuffer = createUniformBufferFromData(
        'linear_attention_params',
        buildParamsData(paramsPayload),
        recorder
      );
      try {
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
        releaseBuffer(outputBuffer);
        throw error;
      }
    }

    const paramsBuffer = createUniformBufferFromData(
      'linear_attention_params',
      buildParamsData(paramsPayload),
      null,
      device,
      { useCache: false }
    );
    let submitted = false;
    try {
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
        releaseBuffer(outputBuffer);
        return casted;
      }
      return output;
    } catch (error) {
      releaseBuffer(outputBuffer);
      throw error;
    } finally {
      if (submitted) {
        device.queue.onSubmittedWorkDone()
          .then(() => {
            paramsBuffer.destroy();
          })
          .catch(() => {
            paramsBuffer.destroy();
          });
      } else {
        paramsBuffer.destroy();
      }
    }
  }

  convOutBuffer = acquireBuffer(convOutSize, undefined, `L${options.layerIdx ?? 0}.linear_conv_out`);
  if (useRecorder) {
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
        rmsNormEps: Number(layerState.rmsNormEps) || 1e-6,
        qkL2NormEps: Number(options.qkL2NormEps) || 1e-6,
        abPacked: options.abPacked === true,
        qkvzPacked: options.qkvzPacked === true,
        bProjOffsetElements: options.bProjOffsetElements,
      }),
      recorder
    );
    try {
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
      recordDispatch(
        recorder,
        pipelines.recurrentPipeline,
        recurrentBindGroup,
        [layerState.numVHeads, 1, 1],
        'linear_attention_recurrent'
      );

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
      releaseBuffer(outputBuffer);
      throw error;
    }
  }

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
      rmsNormEps: Number(layerState.rmsNormEps) || 1e-6,
      qkL2NormEps: Number(options.qkL2NormEps) || 1e-6,
      abPacked: options.abPacked === true,
      qkvzPacked: options.qkvzPacked === true,
      bProjOffsetElements: options.bProjOffsetElements,
    }),
    null,
    device,
    { useCache: false }
  );
  let submitted = false;

  try {
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
      pass.end();
    }

    {
      const pass = encoder.beginComputePass({ label: 'linear_attention_recurrent_pass' });
      pass.setPipeline(pipelines.recurrentPipeline);
      pass.setBindGroup(0, recurrentBindGroup);
      pass.dispatchWorkgroups(layerState.numVHeads, 1, 1);
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
      releaseBuffer(outputBuffer);
      return casted;
    }
    return output;
  } catch (error) {
    releaseBuffer(outputBuffer);
    throw error;
  } finally {
    if (submitted) {
      device.queue.onSubmittedWorkDone()
        .then(() => {
          paramsBuffer.destroy();
        })
        .catch(() => {
          paramsBuffer.destroy();
        });
    } else {
      paramsBuffer.destroy();
    }
    releaseBuffer(convOutBuffer);
  }
}
