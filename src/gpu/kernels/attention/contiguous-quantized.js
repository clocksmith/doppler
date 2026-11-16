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

function getContiguousQuantMaxKVLen() {
  return getRequiredVariantMaxKVLen(
    'attention_contiguous_quant',
    'decode_contiguous_turboquant_f16kv',
    'attention_contiguous_quant.decode_contiguous_turboquant_f16kv'
  );
}

class AttentionContiguousQuantKernel extends KernelBase {

  async getPipeline(variant) {
    return this.getPipelineFor('attention_contiguous_quant', variant);
  }

  dispatch(
    pipeline,
    bindGroup,
    workgroups
  ) {
    this.dispatchKernel(pipeline, bindGroup, workgroups, 'attention_contiguous_quant');
  }

  record(
    recorder,
    pipeline,
    bindGroup,
    workgroups
  ) {
    this.recordKernel(recorder, pipeline, bindGroup, workgroups, 'attention_contiguous_quant');
  }
}

function createContiguousQuantAttentionUniformBuffer(
  device,
  recorder,
  params
) {
  const hasProdFields = params.packedStrideMSE != null;
  const size = hasProdFields ? 64 : 48;
  return createUniformBufferWithView(
    'attention_contiguous_quant_uniforms',
    size,
    (view) => {
      view.setUint32(0, params.numHeads, true);
      view.setUint32(4, params.numKVHeads, true);
      view.setUint32(8, params.headDim, true);
      view.setUint32(12, params.kvLen, true);
      view.setUint32(16, params.seqLen, true);
      view.setFloat32(20, params.scale, true);
      view.setUint32(24, params.causal ? 1 : 0, true);
      view.setUint32(28, params.startPos, true);
      view.setFloat32(32, params.attnSoftcap, true);
      view.setUint32(36, params.slidingWindow, true);
      if (hasProdFields) {
        view.setUint32(40, params.packedStrideMSE, true);
        view.setUint32(44, params.packedStrideResidual, true);
        view.setUint32(48, 0, true);
        view.setUint32(52, 0, true);
        view.setUint32(56, 0, true);
        view.setUint32(60, 0, true);
      } else {
        view.setUint32(40, params.packedStride, true);
        view.setUint32(44, 0, true);
      }
    },
    recorder,
    device
  );
}

// =============================================================================
// Contiguous Quantized Attention (TurboQuant for full-attention models)
// =============================================================================

async function executeAttentionContiguousQuant(
  recorder,
  Q,
  packedK,
  packedV,
  scalesK,
  scalesV,
  numHeads,
  headDim,
  options = {}
) {
  const execution = resolveAttentionExecution(recorder);
  const {
    seqLen = 1,
    kvLen = 0,
    numKVHeads = numHeads,
    scale = 1.0 / Math.sqrt(headDim),
    causal = true,
    startPos = 0,
    outputBuffer = null,
    attnSoftcap = 0,
    slidingWindow = 0,
    packedStride = 0,
    mode = 'turboquant',
    rotationMatrixBuffer = null,
    codebookCentroidsBuffer = null,
    // Prod-mode additional buffers
    residualKBuffer = null,
    residualVBuffer = null,
    residualNormsKBuffer = null,
    residualNormsVBuffer = null,
    qjlMatrixBuffer = null,
    packedStrideMSE = 0,
    packedStrideResidual = 0,
  } = options;

  const maxKVLen = getContiguousQuantMaxKVLen();
  if (kvLen > maxKVLen) {
    throw new Error(`Contiguous quant attention requires kvLen <= ${maxKVLen} but got ${kvLen}.`);
  }
  if (!Number.isFinite(packedStride) || packedStride <= 0) {
    throw new Error('Contiguous quant attention requires packedStride > 0.');
  }
  if (Q.dtype !== 'f32') {
    throw new Error('Contiguous quant attention requires f32 Q.');
  }
  if (!rotationMatrixBuffer) {
    throw new Error('Contiguous quant attention requires rotationMatrixBuffer.');
  }
  if (!codebookCentroidsBuffer) {
    throw new Error('Contiguous quant attention requires codebookCentroidsBuffer.');
  }

  const isProd = mode === 'turboquant_prod';
  const variant = selectKernelRuleValue('attention', 'contiguousQuantVariant', { mode });
  const caps = getKernelCapabilities();
  const config = getKernelConfig('attention_contiguous_quant', variant);
  if (!hasRequiredFeatures(config.requires, caps)) {
    throw new Error(`Contiguous quant attention kernel "${variant}" requires unsupported GPU features.`);
  }

  const kernel = new AttentionContiguousQuantKernel(execution.device);
  const pipeline = await kernel.getPipeline(variant);

  const outputDtype = config.outputDtype;
  if (!outputDtype) {
    throw new Error(`Kernel config missing outputDtype for attention_contiguous_quant variant "${variant}".`);
  }
  const bytesPerElement = outputDtype === 'f16' ? 2 : 4;
  const paddedHiddenSize = padToQ4KBlock(numHeads * headDim);
  const outputSize = seqLen * paddedHiddenSize * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'attention_contiguous_quant_output');

  const uniformParams = isProd
    ? {
      numHeads, numKVHeads, headDim, kvLen, seqLen, scale, causal, startPos,
      attnSoftcap, slidingWindow, packedStrideMSE, packedStrideResidual,
    }
    : {
      numHeads, numKVHeads, headDim, kvLen, seqLen, scale, causal, startPos,
      attnSoftcap, slidingWindow, packedStride,
    };
  const uniformBuffer = createContiguousQuantAttentionUniformBuffer(
    execution.device, execution.recorder, uniformParams
  );

  let entries;
  if (isProd) {
    // Contiguous prod: 14 bindings
    assertAttentionBindGroupBuffer('attention_contiguous_quant', variant, 6, 'residual_k', residualKBuffer);
    assertAttentionBindGroupBuffer('attention_contiguous_quant', variant, 7, 'residual_v', residualVBuffer);
    assertAttentionBindGroupBuffer('attention_contiguous_quant', variant, 13, 'qjl_matrix', qjlMatrixBuffer);
    entries = [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: Q.buffer } },
      { binding: 2, resource: { buffer: packedK } },
      { binding: 3, resource: { buffer: packedV } },
      { binding: 4, resource: { buffer: scalesK } },
      { binding: 5, resource: { buffer: scalesV } },
      { binding: 6, resource: { buffer: residualKBuffer } },
      { binding: 7, resource: { buffer: residualVBuffer } },
      { binding: 8, resource: { buffer: residualNormsKBuffer } },
      { binding: 9, resource: { buffer: residualNormsVBuffer } },
      { binding: 10, resource: { buffer: outputBuf } },
      { binding: 11, resource: { buffer: rotationMatrixBuffer } },
      { binding: 12, resource: { buffer: codebookCentroidsBuffer } },
      { binding: 13, resource: { buffer: qjlMatrixBuffer } },
    ];
  } else {
    // Contiguous MSE: 9 bindings
    entries = [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: Q.buffer } },
      { binding: 2, resource: { buffer: packedK } },
      { binding: 3, resource: { buffer: packedV } },
      { binding: 4, resource: { buffer: scalesK } },
      { binding: 5, resource: { buffer: scalesV } },
      { binding: 6, resource: { buffer: outputBuf } },
      { binding: 7, resource: { buffer: rotationMatrixBuffer } },
      { binding: 8, resource: { buffer: codebookCentroidsBuffer } },
    ];
  }

  const bindGroup = execution.device.createBindGroup({
    label: 'attention_contiguous_quant_bind_group',
    layout: getPipelineBindGroupLayout(pipeline, 0),
    entries,
  });

  dispatchAttentionKernel(execution, kernel, pipeline, bindGroup, numHeads);
  releaseAttentionUniform(execution, uniformBuffer);

  return createTensor(outputBuf, outputDtype, [seqLen, numHeads, headDim], 'attention_contiguous_quant_output');
}

export async function runAttentionContiguousQuant(
  Q,
  packedK,
  packedV,
  scalesK,
  scalesV,
  numHeads,
  headDim,
  options = {}
) {
  return executeAttentionContiguousQuant(
    null, Q, packedK, packedV, scalesK, scalesV, numHeads, headDim, options
  );
}

export async function recordAttentionContiguousQuant(
  recorder,
  Q,
  packedK,
  packedV,
  scalesK,
  scalesV,
  numHeads,
  headDim,
  options = {}
) {
  return executeAttentionContiguousQuant(
    recorder, Q, packedK, packedV, scalesK, scalesV, numHeads, headDim, options
  );
}
