import { getDevice } from '../../../gpu/device.js';
import { Q4K_BLOCK_BYTES, q4kBlockCount } from '../../../config/schema/index.js';
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

function qkvProjectionRequiresQ4KWeights(layerIdx, kernelPath) {
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

function normalizeProjectionDtype(weight) {
  const dtype = typeof weight?.dtype === 'string'
    ? weight.dtype.toLowerCase()
    : null;
  if (dtype === 'bf16') return 'f16';
  if (dtype === 'f16' || dtype === 'f32' || dtype === 'q4k') return dtype;
  return null;
}

function resolveProjectionStorageFormat(weight, expectedRows, hiddenSize) {
  const dtype = normalizeProjectionDtype(weight);
  if (dtype === 'q4k') {
    const layout = String(weight?.layout ?? 'row').toLowerCase();
    if (layout !== 'row') {
      return null;
    }
    const bytesPerRow = q4kBlockCount(hiddenSize) * Q4K_BLOCK_BYTES;
    const byteLength = expectedRows * bytesPerRow;
    if (weight?.buffer?.size < byteLength) {
      return null;
    }
    return {
      dtype,
      layout,
      byteLength,
      bytesPerElement: null,
      bytesPerRow,
    };
  }

  if (dtype === 'f16' || dtype === 'f32') {
    const bytesPerElement = dtype === 'f16' ? 2 : 4;
    const byteLength = expectedRows * hiddenSize * bytesPerElement;
    if (weight?.buffer?.size < byteLength) {
      return null;
    }
    return {
      dtype,
      layout: weight?.layout ?? null,
      byteLength,
      bytesPerElement,
      bytesPerRow: hiddenSize * bytesPerElement,
    };
  }

  const minF16Bytes = expectedRows * hiddenSize * 2;
  const minF32Bytes = expectedRows * hiddenSize * 4;
  if (weight?.buffer?.size >= minF32Bytes) {
    return {
      dtype: 'f32',
      layout: weight?.layout ?? null,
      byteLength: minF32Bytes,
      bytesPerElement: 4,
      bytesPerRow: hiddenSize * 4,
    };
  }
  if (weight?.buffer?.size >= minF16Bytes) {
    return {
      dtype: 'f16',
      layout: weight?.layout ?? null,
      byteLength: minF16Bytes,
      bytesPerElement: 2,
      bytesPerRow: hiddenSize * 2,
    };
  }
  return null;
}

function formatsMatch(a, b) {
  return a?.dtype === b?.dtype
    && a?.layout === b?.layout
    && a?.bytesPerElement === b?.bytesPerElement
    && a?.bytesPerRow === b?.bytesPerRow;
}

export function fuseQKVWeights(layerWeights, modelConfig, kernelPath = null, options = {}) {
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
  const hasAttentionOutputGate = modelConfig?.attentionOutputGate === true;
  const qProjSize = hasAttentionOutputGate ? qSize * 2 : qSize;
  const kSize = numKVHeads * headDim;
  const vSize = numKVHeads * headDim;
  const qkvSize = qSize + kSize + vSize;
  const allowQ4K = options.allowQ4K === true;

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

  let fusedCount = 0;
  for (let l = 0; l < numLayers; l++) {
    const weights = layerWeights.get(`layer_${l}`);
    if (!weights) continue;

    // Skip if already fused or if weights are not GPUBuffers
    if (weights.qkvProj) continue;
    const requiresQ4K = qkvProjectionRequiresQ4KWeights(l, kernelPath);
    const qProj = resolveWeight(weights.qProj);
    const kProj = resolveWeight(weights.kProj);
    const vProj = resolveWeight(weights.vProj);
    if (!qProj || !kProj || !vProj) {
      continue;
    }

    const qFormat = resolveProjectionStorageFormat(qProj, qProjSize, hiddenSize);
    const kFormat = resolveProjectionStorageFormat(kProj, kSize, hiddenSize);
    const vFormat = resolveProjectionStorageFormat(vProj, vSize, hiddenSize);

    // Pool allocation can round GPUBuffer.size up, so infer logical dtype first and
    // only use buffer size as a minimum-size inference.
    if (!qFormat || !formatsMatch(qFormat, kFormat) || !formatsMatch(qFormat, vFormat)) {
      log.debug(
        'QKV Fusion',
        `Layer ${l}: inconsistent projection storage formats, skipping`
      );
      continue;
    }

    if (requiresQ4K && qFormat.dtype !== 'q4k') {
      log.debug('QKV Fusion', `Layer ${l}: qkv_proj requires Q4K weights, skipping ${qFormat.dtype} pack`);
      continue;
    }
    if (qFormat.dtype === 'q4k' && !allowQ4K) {
      log.debug('QKV Fusion', `Layer ${l}: Q4K QKV fusion requires explicit runtime opt-in`);
      continue;
    }

    const dtype = qFormat.dtype ?? selectRuleValue('inference', 'dtype', 'f16OrF32FromBytes', {
      bytesPerElement: qFormat.bytesPerElement,
    });
    const layout = qProj.layout ?? kProj.layout ?? vProj.layout ?? 'row';
    let fusedShape = [qkvSize, hiddenSize];
    if (qFormat.dtype !== 'q4k' && Array.isArray(qProj.shape) && qProj.shape.length === 2) {
      if (qProj.shape[0] === qProjSize && qProj.shape[1] === hiddenSize) {
        fusedShape = [qkvSize, hiddenSize];
      } else if (qProj.shape[1] === qProjSize && qProj.shape[0] === hiddenSize) {
        fusedShape = [hiddenSize, qkvSize];
      }
    }

    const qBytes = qSize * qFormat.bytesPerRow;
    const qGateBytes = hasAttentionOutputGate ? qBytes : 0;

    // Create fused QKV buffer: [qkvSize, hiddenSize] row-major.
    // attentionOutputGate models keep Q rows in qkvProj and gate rows in qGateProj.
    const qkvBuffer = device.createBuffer({
      label: `layer_${l}_qkv_proj`,
      size: qBytes + kFormat.byteLength + vFormat.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const qGateBuffer = hasAttentionOutputGate
      ? device.createBuffer({
        label: `layer_${l}_q_gate_proj`,
        size: qGateBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      })
      : null;

    const encoder = device.createCommandEncoder({ label: 'qkv_fusion' });
    if (hasAttentionOutputGate) {
      for (let head = 0; head < numHeads; head++) {
        const srcHeadOffset = head * headDim * 2 * qFormat.bytesPerRow;
        const dstHeadOffset = head * headDim * qFormat.bytesPerRow;
        const headBytes = headDim * qFormat.bytesPerRow;
        encoder.copyBufferToBuffer(
          qProj.buffer,
          srcHeadOffset,
          qkvBuffer,
          dstHeadOffset,
          headBytes
        );
        encoder.copyBufferToBuffer(
          qProj.buffer,
          srcHeadOffset + headBytes,
          qGateBuffer,
          dstHeadOffset,
          headBytes
        );
      }
    } else {
      encoder.copyBufferToBuffer(
        qProj.buffer, 0,
        qkvBuffer, 0,
        qBytes
      );
    }
    encoder.copyBufferToBuffer(
      kProj.buffer, 0,
      qkvBuffer, qBytes,
      kFormat.byteLength
    );
    encoder.copyBufferToBuffer(
      vProj.buffer, 0,
      qkvBuffer, qBytes + kFormat.byteLength,
      vFormat.byteLength
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
    if (qGateBuffer) {
      weights.qGateProj = createWeightBuffer(
        qGateBuffer,
        dtype,
        layout,
        [qSize, hiddenSize],
        `layer_${l}_q_gate_proj`
      );
    }
    fusedCount++;
  }

  log.debug('QKV Fusion', `Fused ${fusedCount}/${numLayers} layers`);
}
