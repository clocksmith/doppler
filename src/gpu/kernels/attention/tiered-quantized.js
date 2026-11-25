import { getKernelCapabilities } from "../../device.js";
import { acquireBuffer } from "../../../memory/buffer-pool.js";
import { createTensor } from "../../tensor.js";
import { KernelBase } from "../kernel-base.js";
import { padToQ4KBlock } from "../../../config/schema/index.js";
import { createUniformBufferWithView } from "../uniform-utils.js";
import { getKernelConfig } from "../kernel-configs.js";
import { getPipelineBindGroupLayout } from "../pipeline-cache.js";
import { hasRequiredFeatures } from "../feature-check.js";
import { selectRuleValue as selectKernelRuleValue } from "../rule-registry.js";
import { getRequiredVariantMaxKVLen } from "./plan.js";
import { assertAttentionBindGroupBuffer, dispatchAttentionKernel, releaseAttentionUniform, resolveAttentionExecution } from "./executors.js";

function getTieredQuantMaxKVLen() {
  return getRequiredVariantMaxKVLen(
    'attention_tiered_quant',
    'decode_tiered_int8_f16kv',
    'attention_tiered_quant.decode_tiered_int8_f16kv'
  );
}

class AttentionTieredQuantKernel extends KernelBase {

  async getPipeline(variant) {
    return this.getPipelineFor('attention_tiered_quant', variant);
  }

  dispatch(
    pipeline,
    bindGroup,
    workgroups
  ) {
    this.dispatchKernel(pipeline, bindGroup, workgroups, 'attention_tiered_quant');
  }

  record(
    recorder,
    pipeline,
    bindGroup,
    workgroups
  ) {
    this.recordKernel(recorder, pipeline, bindGroup, workgroups, 'attention_tiered_quant');
  }
}

function createTieredQuantAttentionUniformBuffer(
  device,
  recorder,
  params
) {
  return createUniformBufferWithView(
    'attention_tiered_quant_uniforms',
    64,
    (view) => {
      view.setUint32(0, params.numHeads, true);
      view.setUint32(4, params.numKVHeads, true);
      view.setUint32(8, params.headDim, true);
      view.setUint32(12, params.coldLen, true);
      view.setUint32(16, params.hotLen, true);
      view.setUint32(20, params.seqLen, true);
      view.setFloat32(24, params.scale, true);
      view.setUint32(28, params.causal ? 1 : 0, true);
      view.setUint32(32, params.startPos, true);
      view.setFloat32(36, params.attnSoftcap, true);
      view.setUint32(40, params.slidingWindow, true);
      view.setUint32(44, params.hotWindow, true);
      view.setUint32(48, params.hotStart, true);
      view.setUint32(52, params.packedStride, true);
      view.setUint32(56, 0, true);
    },
    recorder,
    device
  );
}

async function executeAttentionTieredQuant(
  recorder,
  Q,
  hotK,
  hotV,
  coldPackedK,
  coldPackedV,
  coldScalesK,
  coldScalesV,
  numHeads,
  headDim,
  options = {}
) {
  const execution = resolveAttentionExecution(recorder);
  const {
    seqLen = 1,
    coldLen = 0,
    hotLen = 0,
    numKVHeads = numHeads,
    scale = 1.0 / Math.sqrt(headDim),
    causal = true,
    startPos = 0,
    outputBuffer = null,
    attnSoftcap = 0,
    slidingWindow = 0,
    hotWindow = hotLen,
    hotStart = 0,
    packedStride = 0,
    mode = 'int8',
    // TurboQuant additional buffers
    rotationMatrixBuffer = null,
    codebookCentroidsBuffer = null,
    residualKBuffer = null,
    residualVBuffer = null,
    residualNormsKBuffer = null,
    residualNormsVBuffer = null,
    qjlMatrixBuffer = null,
  } = options;

  if (mode === 'turboquant-outlier') {
    throw new Error(
      'TurboQuant outlier attention is not supported yet. ' +
      'Outlier-mode decode kernels are not wired end to end.'
    );
  }

  const isTurboQuant = mode === 'turboquant' || mode === 'turboquant-prod';
  const isProd = mode === 'turboquant-prod';

  const totalLen = coldLen + hotLen;
  const maxKVLen = getTieredQuantMaxKVLen();
  if (totalLen > maxKVLen) {
    throw new Error(`Tiered quant attention requires total KV len <= ${maxKVLen} but got ${totalLen}.`);
  }
  if (!Number.isFinite(packedStride) || packedStride <= 0) {
    throw new Error('Tiered quant attention requires packedStride > 0.');
  }

  if (Q.dtype !== 'f32') {
    throw new Error('Tiered quant attention requires f32 Q.');
  }

  if (isTurboQuant && !rotationMatrixBuffer) {
    throw new Error('TurboQuant tiered quant attention requires rotationMatrixBuffer.');
  }
  if (isTurboQuant && !codebookCentroidsBuffer) {
    throw new Error('TurboQuant tiered quant attention requires codebookCentroidsBuffer.');
  }
  if (isProd && !qjlMatrixBuffer) {
    throw new Error('TurboQuant prod tiered quant attention requires qjlMatrixBuffer.');
  }

  const variant = selectKernelRuleValue('attention', 'tieredQuantVariant', { mode });
  const caps = getKernelCapabilities();
  const config = getKernelConfig('attention_tiered_quant', variant);
  if (!hasRequiredFeatures(config.requires, caps)) {
    throw new Error(`Tiered quant attention kernel "${variant}" requires unsupported GPU features.`);
  }

  const kernel = new AttentionTieredQuantKernel(execution.device);
  const pipeline = await kernel.getPipeline(variant);

  const outputDtype = config.outputDtype;
  if (!outputDtype) {
    throw new Error(`Kernel config missing outputDtype for attention_tiered_quant variant "${variant}".`);
  }
  const bytesPerElement = outputDtype === 'f16' ? 2 : 4;
  const paddedHiddenSize = padToQ4KBlock(numHeads * headDim);
  const outputSize = seqLen * paddedHiddenSize * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'attention_tiered_quant_output');

  const uniformBuffer = createTieredQuantAttentionUniformBuffer(execution.device, execution.recorder, {
    numHeads,
    numKVHeads,
    headDim,
    coldLen,
    hotLen,
    seqLen,
    scale,
    causal,
    startPos,
    attnSoftcap,
    slidingWindow,
    hotWindow,
    hotStart,
    packedStride,
  });

  assertAttentionBindGroupBuffer('attention_tiered_quant', variant, 0, 'uniforms', uniformBuffer);
  assertAttentionBindGroupBuffer('attention_tiered_quant', variant, 1, 'Q', Q?.buffer, [
    `QLabel=${Q?.label ?? 'unknown'}`,
    `QDtype=${Q?.dtype ?? 'unknown'}`,
  ]);
  assertAttentionBindGroupBuffer('attention_tiered_quant', variant, 2, 'hotK', hotK?.buffer, [
    `hotKLabel=${hotK?.label ?? 'unknown'}`,
    `hotKDtype=${hotK?.dtype ?? 'unknown'}`,
  ]);
  assertAttentionBindGroupBuffer('attention_tiered_quant', variant, 3, 'hotV', hotV?.buffer, [
    `hotVLabel=${hotV?.label ?? 'unknown'}`,
    `hotVDtype=${hotV?.dtype ?? 'unknown'}`,
  ]);
  assertAttentionBindGroupBuffer('attention_tiered_quant', variant, 4, 'coldPackedK', coldPackedK);
  assertAttentionBindGroupBuffer('attention_tiered_quant', variant, 5, 'coldPackedV', coldPackedV);
  assertAttentionBindGroupBuffer('attention_tiered_quant', variant, 6, 'coldScalesK', coldScalesK);
  assertAttentionBindGroupBuffer('attention_tiered_quant', variant, 7, 'coldScalesV', coldScalesV);

  let entries;
  if (isProd) {
    // TurboQuant prod tiered: 16 bindings
    assertAttentionBindGroupBuffer('attention_tiered_quant', variant, 8, 'residual_k', residualKBuffer);
    assertAttentionBindGroupBuffer('attention_tiered_quant', variant, 15, 'qjl_matrix', qjlMatrixBuffer);
    entries = [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: Q.buffer } },
      { binding: 2, resource: { buffer: hotK.buffer } },
      { binding: 3, resource: { buffer: hotV.buffer } },
      { binding: 4, resource: { buffer: coldPackedK } },
      { binding: 5, resource: { buffer: coldPackedV } },
      { binding: 6, resource: { buffer: coldScalesK } },
      { binding: 7, resource: { buffer: coldScalesV } },
      { binding: 8, resource: { buffer: residualKBuffer } },
      { binding: 9, resource: { buffer: residualVBuffer } },
      { binding: 10, resource: { buffer: residualNormsKBuffer } },
      { binding: 11, resource: { buffer: residualNormsVBuffer } },
      { binding: 12, resource: { buffer: outputBuf } },
      { binding: 13, resource: { buffer: rotationMatrixBuffer } },
      { binding: 14, resource: { buffer: codebookCentroidsBuffer } },
      { binding: 15, resource: { buffer: qjlMatrixBuffer } },
    ];
  } else if (isTurboQuant) {
    // TurboQuant MSE tiered: 11 bindings
    entries = [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: Q.buffer } },
      { binding: 2, resource: { buffer: hotK.buffer } },
      { binding: 3, resource: { buffer: hotV.buffer } },
      { binding: 4, resource: { buffer: coldPackedK } },
      { binding: 5, resource: { buffer: coldPackedV } },
      { binding: 6, resource: { buffer: coldScalesK } },
      { binding: 7, resource: { buffer: coldScalesV } },
      { binding: 8, resource: { buffer: outputBuf } },
      { binding: 9, resource: { buffer: rotationMatrixBuffer } },
      { binding: 10, resource: { buffer: codebookCentroidsBuffer } },
    ];
  } else {
    // Standard int4/int8: 9 bindings
    assertAttentionBindGroupBuffer('attention_tiered_quant', variant, 8, 'output', outputBuf);
    entries = [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: Q.buffer } },
      { binding: 2, resource: { buffer: hotK.buffer } },
      { binding: 3, resource: { buffer: hotV.buffer } },
      { binding: 4, resource: { buffer: coldPackedK } },
      { binding: 5, resource: { buffer: coldPackedV } },
      { binding: 6, resource: { buffer: coldScalesK } },
      { binding: 7, resource: { buffer: coldScalesV } },
      { binding: 8, resource: { buffer: outputBuf } },
    ];
  }

  const bindGroup = execution.device.createBindGroup({
    label: 'attention_tiered_quant_bind_group',
    layout: getPipelineBindGroupLayout(pipeline, 0),
    entries,
  });

  dispatchAttentionKernel(execution, kernel, pipeline, bindGroup, numHeads);
  releaseAttentionUniform(execution, uniformBuffer);

  return createTensor(outputBuf, outputDtype, [seqLen, numHeads, headDim], 'attention_tiered_quant_output');
}

export async function runAttentionTieredQuant(
  Q,
  hotK,
  hotV,
  coldPackedK,
  coldPackedV,
  coldScalesK,
  coldScalesV,
  numHeads,
  headDim,
  options = {}
) {
  return executeAttentionTieredQuant(
    null,
    Q,
    hotK,
    hotV,
    coldPackedK,
    coldPackedV,
    coldScalesK,
    coldScalesV,
    numHeads,
    headDim,
    options
  );
}

export async function recordAttentionTieredQuant(
  recorder,
  Q,
  hotK,
  hotV,
  coldPackedK,
  coldPackedV,
  coldScalesK,
  coldScalesV,
  numHeads,
  headDim,
  options = {}
) {
  return executeAttentionTieredQuant(
    recorder,
    Q,
    hotK,
    hotV,
    coldPackedK,
    coldPackedV,
    coldScalesK,
    coldScalesV,
    numHeads,
    headDim,
    options
  );
}
