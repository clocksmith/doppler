import { getDevice } from '../../../gpu/device.js';
import { getKernelConfig } from '../../../gpu/kernels/kernel-configs.js';
import { getKernelPathMatmulVariant } from '../../../config/kernel-path-loader.js';
import { createWeightBuffer, getWeightDtype, isGpuBufferInstance, isWeightBuffer } from '../../../gpu/weight-buffer.js';
import { selectRuleValue } from '../../../rules/rule-registry.js';
import { log } from '../../../debug/index.js';

// ============================================================================
// QKV Fusion
// ============================================================================


function kernelPathVariantRequiresQ4KWeights(variant) {
  if (typeof variant !== 'string' || variant.length === 0) {
    return false;
  }
  let config;
  try {
    config = getKernelConfig('matmul', variant);
  } catch {
    return false;
  }
  const shaderFile = String(config?.shaderFile ?? config?.wgsl ?? '');
  return shaderFile.startsWith('fused_matmul_q4');
}

function shouldSkipQKVFusionForLayer(layerIdx, kernelPath) {
  if (!kernelPath) {
    return false;
  }
  const prefillVariant = getKernelPathMatmulVariant('qkv_proj', 'prefill', layerIdx, kernelPath);
  if (kernelPathVariantRequiresQ4KWeights(prefillVariant)) {
    return true;
  }
  const decodeVariant = getKernelPathMatmulVariant('qkv_proj', 'decode', layerIdx, kernelPath);
  return kernelPathVariantRequiresQ4KWeights(decodeVariant);
}

export function fuseQKVWeights(layerWeights, modelConfig, kernelPath = null) {
  const device = getDevice();
  if (!device) {
    log.debug('QKV Fusion', 'No GPU device, skipping fusion');
    return;
  }
  if (
    Number.isFinite(modelConfig?.globalHeadDim)
    && modelConfig.globalHeadDim !== modelConfig.headDim
  ) {
    log.debug(
      'QKV Fusion',
      'Skipping QKV fusion for mixed-head-dim model; per-layer attention geometry must stay explicit.'
    );
    return;
  }

  const { numLayers, numHeads, numKVHeads, headDim, hiddenSize } = modelConfig;
  const qSize = numHeads * headDim;
  const kSize = numKVHeads * headDim;
  const vSize = numKVHeads * headDim;
  const qkvSize = qSize + kSize + vSize;

  const resolveWeight = (value) => {
    if (isWeightBuffer(value)) {
      return {
        buffer: value.buffer,
        dtype: value.dtype ?? null,
        layout: value.layout ?? null,
        shape: Array.isArray(value.shape) ? value.shape : null,
      };
    }
    if (isGpuBufferInstance(value)) {
      return {
        buffer: value,
        dtype: getWeightDtype(value),
        layout: null,
        shape: null,
      };
    }
    return null;
  };

  log.debug('QKV Fusion', `Fusing Q/K/V weights for ${numLayers} layers (${qSize}+${kSize}+${vSize}=${qkvSize})`);

  const resolveBytesPerElement = (weight, expectedElements) => {
    const dtype = typeof weight?.dtype === 'string'
      ? weight.dtype.toLowerCase()
      : null;
    if (dtype === 'f16' || dtype === 'bf16') return 2;
    if (dtype === 'f32') return 4;
    const minF16Bytes = expectedElements * 2;
    const minF32Bytes = expectedElements * 4;
    if (weight?.buffer?.size >= minF32Bytes) return 4;
    if (weight?.buffer?.size >= minF16Bytes) return 2;
    return 0;
  };

  let fusedCount = 0;
  for (let l = 0; l < numLayers; l++) {
    const weights = layerWeights.get(`layer_${l}`);
    if (!weights) continue;

    // Skip if already fused or if weights are not GPUBuffers
    if (weights.qkvProj) continue;
    if (shouldSkipQKVFusionForLayer(l, kernelPath)) {
      log.debug(
        'QKV Fusion',
        `Layer ${l}: skipped because active kernel path requires Q4K weights for qkv_proj`
      );
      continue;
    }
    const qProj = resolveWeight(weights.qProj);
    const kProj = resolveWeight(weights.kProj);
    const vProj = resolveWeight(weights.vProj);
    if (!qProj || !kProj || !vProj) {
      continue;
    }

    const qExpectedElements = qSize * hiddenSize;
    const kExpectedElements = kSize * hiddenSize;
    const vExpectedElements = vSize * hiddenSize;
    const bytesPerElement = resolveBytesPerElement(qProj, qExpectedElements);
    const kBytesPerElement = resolveBytesPerElement(kProj, kExpectedElements);
    const vBytesPerElement = resolveBytesPerElement(vProj, vExpectedElements);

    // Pool allocation can round GPUBuffer.size up, so infer logical dtype first and
    // only use buffer size as a minimum-size inference.
    if ((bytesPerElement !== 2 && bytesPerElement !== 4)
      || kBytesPerElement !== bytesPerElement
      || vBytesPerElement !== bytesPerElement) {
      log.debug(
        'QKV Fusion',
        `Layer ${l}: inconsistent projection dtypes (q=${bytesPerElement}, k=${kBytesPerElement}, v=${vBytesPerElement}), skipping`
      );
      continue;
    }

    const normalizedDtype = typeof qProj.dtype === 'string'
      ? qProj.dtype.toLowerCase()
      : null;
    const dtype = normalizedDtype === 'bf16'
      ? 'f16'
      : (
        normalizedDtype === 'f16' || normalizedDtype === 'f32'
          ? normalizedDtype
          : selectRuleValue('inference', 'dtype', 'f16OrF32FromBytes', { bytesPerElement })
      );
    const layout = qProj.layout ?? kProj.layout ?? vProj.layout ?? 'row';
    let fusedShape = [qkvSize, hiddenSize];
    if (Array.isArray(qProj.shape) && qProj.shape.length === 2) {
      if (qProj.shape[0] === qSize && qProj.shape[1] === hiddenSize) {
        fusedShape = [qkvSize, hiddenSize];
      } else if (qProj.shape[1] === qSize && qProj.shape[0] === hiddenSize) {
        fusedShape = [hiddenSize, qkvSize];
      }
    }

    // Create fused QKV buffer: [qkvSize, hiddenSize] row-major
    // Each row is concatenated: [q_row, k_row, v_row]
    const qkvBuffer = device.createBuffer({
      label: `layer_${l}_qkv_proj`,
      size: qkvSize * hiddenSize * bytesPerElement,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Copy Q, K, V weights into fused buffer
    // Q: [qSize, hiddenSize] -> offset 0
    // K: [kSize, hiddenSize] -> offset qSize * hiddenSize * bytesPerElement
    // V: [vSize, hiddenSize] -> offset (qSize + kSize) * hiddenSize * bytesPerElement
    const encoder = device.createCommandEncoder({ label: 'qkv_fusion' });
    encoder.copyBufferToBuffer(
      qProj.buffer, 0,
      qkvBuffer, 0,
      qSize * hiddenSize * bytesPerElement
    );
    encoder.copyBufferToBuffer(
      kProj.buffer, 0,
      qkvBuffer, qSize * hiddenSize * bytesPerElement,
      kSize * hiddenSize * bytesPerElement
    );
    encoder.copyBufferToBuffer(
      vProj.buffer, 0,
      qkvBuffer, (qSize + kSize) * hiddenSize * bytesPerElement,
      vSize * hiddenSize * bytesPerElement
    );
    device.queue.submit([encoder.finish()]);

    // Store fused buffer, sizes, and dtype
    weights.qkvProj = createWeightBuffer(
      qkvBuffer,
      dtype,
      layout,
      fusedShape,
      `layer_${l}_qkv_proj`
    );
    weights.qkvSizes = [qSize, kSize, vSize];
    weights.qkvDtype = dtype;
    fusedCount++;
  }

  log.debug('QKV Fusion', `Fused ${fusedCount}/${numLayers} layers`);
}
