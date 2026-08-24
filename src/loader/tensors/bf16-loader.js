import { getDevice, getKernelCapabilities } from '../../gpu/device.js';
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { runBF16ToF16 } from '../../gpu/kernel-selector.js';
import { createWeightBuffer } from '../../gpu/weight-buffer.js';
import {
  applyBufferLayout,
  convertBF16ToF32GPU,
  shouldDequantizeToF16,
} from '../dtype-utils.js';
import { trace as debugTrace } from '../../debug/index.js';
import { selectRuleValue } from '../../rules/rule-registry.js';

function isGpuBufferInstance(value) {
  return typeof GPUBuffer !== 'undefined' && value instanceof GPUBuffer;
}

function isReleasableBuffer(value) {
  return typeof value === 'object' && value !== null && 'size' in value;
}

function releaseOwnedGpuBuffer(buffer, owned) {
  if (owned && isReleasableBuffer(buffer)) releaseBuffer(buffer);
}

function alignTo4(size) {
  return Math.ceil(size / 4) * 4;
}

function toUint8View(data) {
  if (data instanceof Uint8Array) return data;
  if (ArrayBuffer.isView(data)) {
    return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  }
  return new Uint8Array(data);
}

function writeBufferAligned(device, buffer, data) {
  const bytes = toUint8View(data);
  const alignedSize = alignTo4(bytes.byteLength);
  if (alignedSize === bytes.byteLength) {
    device.queue.writeBuffer(buffer, 0, bytes);
    return;
  }
  const padded = new Uint8Array(alignedSize);
  padded.set(bytes);
  device.queue.writeBuffer(buffer, 0, padded);
}

export async function loadBF16(shardData, location, name, config) {
  const device = getDevice();
  let ownsSrcBuffer = !isGpuBufferInstance(shardData);
  const srcBuffer = isGpuBufferInstance(shardData)
    ? shardData
    : acquireBuffer(alignTo4(location.size), undefined, `${name}_bf16`);
  let resultBuffer = null;
  try {
    if (ownsSrcBuffer) writeBufferAligned(device, srcBuffer, shardData);

    const numElements = location.size / 2;
    const caps = config.gpuCapabilities || getKernelCapabilities();
    const isMatmulWeight = shouldDequantizeToF16(location);
    const keepF32Weights = config.keepF32Weights === true;
    const keepPackedBF16 = config.keepBF16Weights === true
      && (isMatmulWeight || location.role === 'embedding');
    if (keepPackedBF16) {
      const layout = selectRuleValue('loader', 'weights', 'weightLayout', {
        layout: location.layout ?? null,
        useColumnWise: false,
      });
      const data = isMatmulWeight
        ? createWeightBuffer(srcBuffer, 'bf16', layout, location.shape, name)
        : applyBufferLayout(srcBuffer, location, 'bf16');
      ownsSrcBuffer = false;
      debugTrace.loader(`Retaining packed BF16 storage for selected kernel path: ${name}`);
      return { data, allocatedBuffers: [srcBuffer] };
    }

    if (caps?.hasF16 && isMatmulWeight && !keepF32Weights) {
      const f16Tensor = await runBF16ToF16(srcBuffer, [numElements], name);
      resultBuffer = f16Tensor.buffer;
      releaseOwnedGpuBuffer(srcBuffer, ownsSrcBuffer);
      ownsSrcBuffer = false;
      debugTrace.loader(`BF16->F16 for matmul weight: ${name} (${numElements} elements)`);
      const layout = selectRuleValue('loader', 'weights', 'weightLayout', {
        layout: location.layout ?? null,
        useColumnWise: false,
      });
      return {
        data: createWeightBuffer(f16Tensor.buffer, 'f16', layout, location.shape, name),
        allocatedBuffers: [f16Tensor.buffer],
      };
    }

    if (isMatmulWeight && keepF32Weights) {
      debugTrace.loader(`Keeping BF16 matmul weight in f32: ${name} (keepF32Weights=true)`);
    }
    const dstBuffer = await convertBF16ToF32GPU(srcBuffer, numElements, name);
    resultBuffer = dstBuffer;
    releaseOwnedGpuBuffer(srcBuffer, ownsSrcBuffer);
    ownsSrcBuffer = false;

    if (isGpuBufferInstance(dstBuffer)) {
      if (isMatmulWeight) {
        const layout = selectRuleValue('loader', 'weights', 'weightLayout', {
          layout: location.layout ?? null,
          useColumnWise: false,
        });
        return {
          data: createWeightBuffer(dstBuffer, 'f32', layout, location.shape, name),
          allocatedBuffers: [dstBuffer],
        };
      }
      return {
        data: applyBufferLayout(dstBuffer, location, 'f32'),
        allocatedBuffers: [dstBuffer],
      };
    }
    return { data: dstBuffer, allocatedBuffers: [] };
  } catch (error) {
    if (isReleasableBuffer(resultBuffer)) releaseBuffer(resultBuffer);
    throw error;
  } finally {
    releaseOwnedGpuBuffer(srcBuffer, ownsSrcBuffer);
  }
}
