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
import { assertAttentionBindGroupBuffer, dispatchAttentionKernel, getPageTableFallbackBuffer, releaseAttentionUniform, resolveAttentionExecution } from "./executors.js";

// Track if we've logged the attention tier selection (avoid spam)

function getTieredMaxKVLen() {
  return getRequiredVariantMaxKVLen('attention_tiered', 'decode_tiered_f16', 'attention_tiered.decode_tiered_f16');
}

class AttentionTieredKernel extends KernelBase {

  async getPipeline(variant) {
    return this.getPipelineFor('attention_tiered', variant);
  }

  dispatch(
    pipeline,
    bindGroup,
    workgroups
  ) {
    this.dispatchKernel(pipeline, bindGroup, workgroups, 'attention_tiered');
  }

  record(
    recorder,
    pipeline,
    bindGroup,
    workgroups
  ) {
    this.recordKernel(recorder, pipeline, bindGroup, workgroups, 'attention_tiered');
  }
}

function createTieredAttentionUniformBuffer(
  device,
  recorder,
  params
) {
  return createUniformBufferWithView(
    'attention_tiered_uniforms',
    80,
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
      view.setUint32(52, params.coldPageSize, true);
      view.setUint32(56, params.coldLayout ?? 0, true);
      view.setUint32(60, params.hotLayout ?? 1, true);
      view.setUint32(64, 0, true);
    },
    recorder,
    device
  );
}

// -----------------------------------------------------------------------------
// Flash-attention prefill path (head_dim = 256, f16 KV)
// -----------------------------------------------------------------------------
// Two-pass kernel to raise RDNA3 occupancy: pass 1 processes one KV slice per
// workgroup and writes per-split (acc, m, l) partials; pass 2 merges across
// splits with online softmax. Single recorder, two dispatches — queue order
// handles the read-after-write between passes.

// Single-pass flash-attention dispatcher (ORT-style). 7-binding contract —
// same as attention_head256_f16kv. One kernel launch, no reduce pass.
// Workgroups: (num_heads, ceil(seqLen / ORT_FLASH_WG), 1) with ORT_FLASH_WG=64.

async function executeAttentionTiered(
  recorder,
  Q,
  hotK,
  hotV,
  coldK,
  coldV,
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
    coldPageTable = null,
    coldPageSize = 0,
    coldLayout = 2,
    hotLayout = 1,
  } = options;

  const totalLen = coldLen + hotLen;
  const maxKVLen = getTieredMaxKVLen();
  if (totalLen > maxKVLen) {
    throw new Error(`Tiered attention requires total KV len <= ${maxKVLen} but got ${totalLen}.`);
  }

  const useF16 = Q.dtype === 'f16' && hotK.dtype === 'f16' && coldK.dtype === 'f16';
  const useF16KV = hotK.dtype === 'f16' && coldK.dtype === 'f16';
  const variant = selectKernelRuleValue('attention', 'tieredVariant', { useF16 });
  const caps = getKernelCapabilities();
  const config = getKernelConfig('attention_tiered', variant);
  if (!hasRequiredFeatures(config.requires, caps)) {
    throw new Error(`Tiered attention kernel "${variant}" requires unsupported GPU features.`);
  }
  if (!useF16KV) {
    throw new Error('Tiered attention requires f16 KV buffers.');
  }

  const kernel = new AttentionTieredKernel(execution.device);
  const pipeline = await kernel.getPipeline(variant);

  const outputDtype = config.outputDtype;
  if (!outputDtype) {
    throw new Error(`Kernel config missing outputDtype for attention_tiered variant "${variant}".`);
  }
  const bytesPerElement = outputDtype === 'f16' ? 2 : 4;
  const paddedHiddenSize = padToQ4KBlock(numHeads * headDim);
  const outputSize = seqLen * paddedHiddenSize * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'attention_tiered_output');

  const uniformBuffer = createTieredAttentionUniformBuffer(execution.device, execution.recorder, {
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
    coldPageSize,
    coldLayout,
    hotLayout,
  });

  const pageTableBinding = coldPageTable || getPageTableFallbackBuffer(execution.device);
  assertAttentionBindGroupBuffer('attention_tiered', variant, 0, 'uniforms', uniformBuffer);
  assertAttentionBindGroupBuffer('attention_tiered', variant, 1, 'Q', Q?.buffer, [
    `QLabel=${Q?.label ?? 'unknown'}`,
    `QDtype=${Q?.dtype ?? 'unknown'}`,
  ]);
  assertAttentionBindGroupBuffer('attention_tiered', variant, 2, 'hotK', hotK?.buffer, [
    `hotKLabel=${hotK?.label ?? 'unknown'}`,
    `hotKDtype=${hotK?.dtype ?? 'unknown'}`,
  ]);
  assertAttentionBindGroupBuffer('attention_tiered', variant, 3, 'hotV', hotV?.buffer, [
    `hotVLabel=${hotV?.label ?? 'unknown'}`,
    `hotVDtype=${hotV?.dtype ?? 'unknown'}`,
  ]);
  assertAttentionBindGroupBuffer('attention_tiered', variant, 4, 'coldK', coldK?.buffer, [
    `coldKLabel=${coldK?.label ?? 'unknown'}`,
    `coldKDtype=${coldK?.dtype ?? 'unknown'}`,
  ]);
  assertAttentionBindGroupBuffer('attention_tiered', variant, 5, 'coldV', coldV?.buffer, [
    `coldVLabel=${coldV?.label ?? 'unknown'}`,
    `coldVDtype=${coldV?.dtype ?? 'unknown'}`,
  ]);
  assertAttentionBindGroupBuffer('attention_tiered', variant, 6, 'output', outputBuf);
  assertAttentionBindGroupBuffer('attention_tiered', variant, 7, 'pageTable', pageTableBinding, [
    `coldLayout=${coldLayout}`,
  ]);
  const bindGroup = execution.device.createBindGroup({
    label: 'attention_tiered_bind_group',
    layout: getPipelineBindGroupLayout(pipeline, 0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: Q.buffer } },
      { binding: 2, resource: { buffer: hotK.buffer } },
      { binding: 3, resource: { buffer: hotV.buffer } },
      { binding: 4, resource: { buffer: coldK.buffer } },
      { binding: 5, resource: { buffer: coldV.buffer } },
      { binding: 6, resource: { buffer: outputBuf } },
      { binding: 7, resource: { buffer: pageTableBinding } },
    ],
  });

  dispatchAttentionKernel(execution, kernel, pipeline, bindGroup, numHeads);
  releaseAttentionUniform(execution, uniformBuffer);

  return createTensor(outputBuf, outputDtype, [seqLen, numHeads, headDim], 'attention_tiered_output');
}

export async function runAttentionTiered(
  Q,
  hotK,
  hotV,
  coldK,
  coldV,
  numHeads,
  headDim,
  options = {}
) {
  return executeAttentionTiered(null, Q, hotK, hotV, coldK, coldV, numHeads, headDim, options);
}

export async function recordAttentionTiered(
  recorder,
  Q,
  hotK,
  hotV,
  coldK,
  coldV,
  numHeads,
  headDim,
  options = {}
) {
  return executeAttentionTiered(recorder, Q, hotK, hotV, coldK, coldV, numHeads, headDim, options);
}
