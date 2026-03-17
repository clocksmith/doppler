

import { getDevice, getKernelCapabilities } from '../../gpu/device.js';
import { acquireBuffer, releaseBuffer, readBuffer } from '../../memory/buffer-pool.js';
import { dequantize, dequantizeRowwise, dequantizeQ6K, castF16ToF32, runBF16ToF16 } from '../../gpu/kernel-selector.js';
import { createTensor } from '../../gpu/tensor.js';
import { createWeightBuffer } from '../../gpu/weight-buffer.js';
import { f16ToF32, convertBF16ToF32GPU, shouldDequantizeToF16, applyBufferLayout } from '../dtype-utils.js';
import { QK_K, Q4K_BLOCK_BYTES, Q6K_BLOCK_BYTES } from '../quantization-constants.js';
import { log, trace as debugTrace } from '../../debug/index.js';
import { selectRuleValue } from '../../rules/rule-registry.js';
import { dequantizeQ4KM, dequantizeQ4KMRowWise } from '../../converter/quantizer.js';

// ============================================================================
// Q4K Detection
// ============================================================================

let loggedF32UpcastNonMatmul = false;

function isGpuBufferInstance(value) {
  return typeof GPUBuffer !== 'undefined' && value instanceof GPUBuffer;
}

function isReleasableBuffer(value) {
  return typeof value === 'object' && value !== null && 'size' in value;
}

function releaseOwnedGpuBuffer(buffer, owned) {
  if (!owned || !isReleasableBuffer(buffer)) {
    return;
  }
  releaseBuffer(buffer);
}

function logF32UpcastNonMatmul(name, numElements, bufferSize) {
  if (loggedF32UpcastNonMatmul) {
    return;
  }
  loggedF32UpcastNonMatmul = true;
  log.warn(
    'Loader',
    `F16->F32 upcast for non-matmul weights enabled ` +
    `(runtime.loading.allowF32UpcastNonMatmul=true). ` +
    `Example: ${name} (${numElements} elements, bufSize=${bufferSize}).`
  );
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

function acquireAlignedBuffer(size, label) {
  return acquireBuffer(alignTo4(size), undefined, label);
}


export function isPackedQ4K(location) {
  if (!Array.isArray(location.shape) || location.shape.length !== 2) {
    return false;
  }
  const [rows, cols] = location.shape;
  const expectedRowwise = rows * Math.ceil(cols / QK_K) * Q4K_BLOCK_BYTES;
  return location.size < expectedRowwise;
}


function isEmbeddingRole(location) {
  if (!location?.role) {
    throw new Error('Tensor role is required to determine embedding layout.');
  }
  return location.role === 'embedding';
}


export function shouldUseFusedQ4K(location, config) {
  if (!config.useFusedQ4K) return false;

  const caps = config.gpuCapabilities || getKernelCapabilities();
  if (!caps?.hasSubgroups) return false;

  const isMatmulWeight = shouldDequantizeToF16(location);
  if (!isMatmulWeight) return false;

  if (isEmbeddingRole(location)) return false;
  if (isPackedQ4K(location)) return false;

  return true;
}

// ============================================================================
// Dtype Output Selection
// ============================================================================


export function getQ4KOutputDtype(location, config) {
  const isMatmulWeight = shouldDequantizeToF16(location);
  const caps = config.gpuCapabilities || getKernelCapabilities();
  return selectRuleValue('loader', 'weights', 'q4kOutputDtype', {
    isMatmulWeight,
    keepF32Weights: Boolean(config.keepF32Weights),
    hasF16: Boolean(caps?.hasF16),
  });
}


export function getWeightLayout(location, config) {
  const isMatmulWeight = shouldDequantizeToF16(location);
  // Layout: 'col' = column-wise, 'row' = row-wise (default)
  const useColumnWise = config.q4kLayout === 'col' && isMatmulWeight;
  return selectRuleValue('loader', 'weights', 'weightLayout', {
    layout: location.layout ?? null,
    useColumnWise,
  });
}

// ============================================================================
// CPU Path Helpers
// ============================================================================


export function convertBF16ToF32CPU(bf16Data) {
  const f32 = new Float32Array(bf16Data.length);
  const tmp = new ArrayBuffer(4);
  const u32View = new Uint32Array(tmp);
  const f32View = new Float32Array(tmp);

  for (let i = 0; i < bf16Data.length; i++) {
    u32View[0] = bf16Data[i] << 16;
    f32[i] = f32View[0];
  }

  return f32;
}


export function convertF16ToF32CPU(f16Data) {
  const f32 = new Float32Array(f16Data.length);
  for (let i = 0; i < f16Data.length; i++) {
    f32[i] = f16ToF32(f16Data[i]);
  }
  return f32;
}

// ============================================================================
// GPU Tensor Loading
// ============================================================================


export async function loadQ4KFused(shardData, location, name) {
  const device = getDevice();
  const ownsBuffer = !isGpuBufferInstance(shardData);
  const buffer = isGpuBufferInstance(shardData)
    ? shardData
    : acquireAlignedBuffer(location.size, `q4k_${name}`);
  try {
    if (ownsBuffer) {
      writeBufferAligned(device, buffer, shardData);
    }
    return {
      data: createWeightBuffer(buffer, 'q4k', 'row', location.shape, name),
      allocatedBuffers: [buffer],
    };
  } catch (error) {
    releaseOwnedGpuBuffer(buffer, ownsBuffer);
    throw error;
  }
}


export async function loadQ4KDequant(shardData, location, name, config) {
  const device = getDevice();
  let ownsQuantBuffer = !isGpuBufferInstance(shardData);
  const quantBuffer = isGpuBufferInstance(shardData)
    ? shardData
    : acquireAlignedBuffer(location.size, `quant_${name}`);
  let dequantized = null;
  try {
    if (ownsQuantBuffer) {
      writeBufferAligned(device, quantBuffer, shardData);
    }

    const outputDtype = getQ4KOutputDtype(location, config);

    const is2DMatrix = Array.isArray(location.shape) && location.shape.length === 2;
    const K = is2DMatrix ? location.shape[1] : 0;
    const needsRowwise = is2DMatrix && K > 0 && K % QK_K !== 0;
    const layout = getWeightLayout(location, config);
    // TEMP: Force GPU dequant to test parity
    const FORCE_GPU_DEQUANT = true;
    const canUseCpuReference = !FORCE_GPU_DEQUANT && (
      outputDtype === 'f32'
      && !isGpuBufferInstance(shardData)
      && (!needsRowwise || layout === 'row')
    );

    if (canUseCpuReference) {
      const quantizedBytes = toUint8View(shardData);
      const numBlocks = Math.ceil(location.size / Q4K_BLOCK_BYTES);
      debugTrace.loader(
        `Dequantizing ${name} with CPU reference path: ` +
        `shape=[${location.shape.join(',')}], layout=${layout}, needsRowwise=${needsRowwise}`
      );
      const f32Weights = needsRowwise
        ? dequantizeQ4KMRowWise(quantizedBytes, location.shape)
        : dequantizeQ4KM(quantizedBytes, numBlocks, location.shape);
      const outputBuffer = acquireAlignedBuffer(f32Weights.byteLength, `dequant_cpu_${name}`);
      writeBufferAligned(device, outputBuffer, new Uint8Array(f32Weights.buffer));
      releaseOwnedGpuBuffer(quantBuffer, ownsQuantBuffer);
      ownsQuantBuffer = false;
      return {
        data: createWeightBuffer(outputBuffer, 'f32', layout, location.shape, name),
        allocatedBuffers: [outputBuffer],
      };
    }

    let dequantizedTensor;
    if (needsRowwise) {
      const rows = location.shape[0];
      debugTrace.loader(
        `Dequantizing ${name} (row-wise): [${rows},${K}], K not 256-aligned, ` +
        `outputDtype=${outputDtype}`
      );
      dequantizedTensor = await dequantizeRowwise(quantBuffer, rows, K, { outputDtype });
    } else {
      const numBlocks = Math.ceil(location.size / Q4K_BLOCK_BYTES);
      debugTrace.loader(
        `Dequantizing ${name}: size=${location.size}, numBlocks=${numBlocks}, ` +
        `outputDtype=${outputDtype}, expectedOutput=${numBlocks * QK_K * (outputDtype === 'f16' ? 2 : 4)}`
      );
      dequantizedTensor = await dequantize(quantBuffer, numBlocks, { outputDtype });
    }
    dequantized = dequantizedTensor.buffer;

    debugTrace.loader(`Dequantized ${name}: resultSize=${dequantized.size}`);

    // DEBUG: GPU-vs-CPU dequant parity check on first Q4K tensor
    if (!loadQ4KDequant._parityChecked && !isGpuBufferInstance(shardData)) {
      loadQ4KDequant._parityChecked = true;
      try {
        const quantizedBytes = toUint8View(shardData);
        const cpuNumBlocks = Math.ceil(location.size / Q4K_BLOCK_BYTES);
        const cpuRef = needsRowwise
          ? dequantizeQ4KMRowWise(quantizedBytes, location.shape)
          : dequantizeQ4KM(quantizedBytes, cpuNumBlocks, location.shape);
        // Flush GPU work before readback
        await device.queue.onSubmittedWorkDone();
        // Read back GPU result
        const readSize = Math.min(dequantized.size, 256 * 4); // first 256 f32 values
        const gpuData = await readBuffer(dequantized, readSize);
        log.warn('DequantParity', `readSize=${readSize}, gpuData.length=${gpuData?.byteLength ?? 'null'}, bufSize=${dequantized.size}`);
        const gpuF32 = new Float32Array(gpuData.buffer, gpuData.byteOffset, readSize / 4);
        const cpuSlice = cpuRef.slice(0, readSize / 4);
        let maxDiff = 0;
        let diffIdx = -1;
        for (let i = 0; i < cpuSlice.length; i++) {
          const d = Math.abs(gpuF32[i] - cpuSlice[i]);
          if (d > maxDiff) { maxDiff = d; diffIdx = i; }
        }
        log.warn('DequantParity',
          `GPU-vs-CPU check on "${name}" [${location.shape}]: ` +
          `maxDiff=${maxDiff.toFixed(8)} at idx=${diffIdx}, ` +
          `gpu[0..3]=[${Array.from(gpuF32.slice(0, 4)).map(v => v.toFixed(6))}], ` +
          `cpu[0..3]=[${Array.from(cpuSlice.slice(0, 4)).map(v => v.toFixed(6))}]`
        );
      } catch (e) {
        log.warn('DequantParity', `Parity check failed: ${e.message}`);
      }
    }

    releaseOwnedGpuBuffer(quantBuffer, ownsQuantBuffer);
    ownsQuantBuffer = false;

    const dtype = outputDtype;

    return {
      data: createWeightBuffer(dequantized, dtype, layout, location.shape, name),
      allocatedBuffers: [dequantized],
    };
  } catch (error) {
    if (isReleasableBuffer(dequantized)) {
      releaseBuffer(dequantized);
    }
    throw error;
  } finally {
    releaseOwnedGpuBuffer(quantBuffer, ownsQuantBuffer);
  }
}


export async function loadQ6K(shardData, location, name) {
  const device = getDevice();

  debugTrace.loader(`Loading Q6_K tensor "${name}", size=${location.size}`);
  let ownsQuantBuffer = !isGpuBufferInstance(shardData);
  const quantBuffer = isGpuBufferInstance(shardData)
    ? shardData
    : acquireAlignedBuffer(location.size, `quant_${name}`);
  let dequantized = null;
  try {
    if (ownsQuantBuffer) {
      writeBufferAligned(device, quantBuffer, shardData);
    }

    const numBlocks = Math.floor(location.size / Q6K_BLOCK_BYTES);
    debugTrace.loader(
      `Dequantizing Q6_K ${name}: size=${location.size}, numBlocks=${numBlocks}, ` +
      `expectedOutput=${numBlocks * 256 * 2} (f16)`
    );

    const dequantizedTensor = await dequantizeQ6K(quantBuffer, numBlocks, { outputDtype: 'f16' });
    dequantized = dequantizedTensor.buffer;

    debugTrace.loader(`Dequantized Q6_K ${name}: resultSize=${dequantized.size}`);
    releaseOwnedGpuBuffer(quantBuffer, ownsQuantBuffer);
    ownsQuantBuffer = false;

    const isMatmulWeight = shouldDequantizeToF16(location);
    if (isMatmulWeight) {
      return {
        data: createWeightBuffer(dequantized, 'f16', 'row', location.shape, name),
        allocatedBuffers: [dequantized],
      };
    }

    return {
      data: applyBufferLayout(dequantized, location, 'f16'),
      allocatedBuffers: [dequantized],
    };
  } catch (error) {
    if (isReleasableBuffer(dequantized)) {
      releaseBuffer(dequantized);
    }
    throw error;
  } finally {
    releaseOwnedGpuBuffer(quantBuffer, ownsQuantBuffer);
  }
}


export async function loadBF16(shardData, location, name, config) {
  const device = getDevice();
  let ownsSrcBuffer = !isGpuBufferInstance(shardData);
  const srcBuffer = isGpuBufferInstance(shardData)
    ? shardData
    : acquireAlignedBuffer(location.size, `${name}_bf16`);
  let resultBuffer = null;
  try {
    if (ownsSrcBuffer) {
      writeBufferAligned(device, srcBuffer, shardData);
    }

    const numElements = location.size / 2;
    const caps = config.gpuCapabilities || getKernelCapabilities();
    const isMatmulWeight = shouldDequantizeToF16(location);
    const keepF32Weights = config.keepF32Weights === true;

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

    return {
      data: dstBuffer,
      allocatedBuffers: [],
    };
  } catch (error) {
    if (isReleasableBuffer(resultBuffer)) {
      releaseBuffer(resultBuffer);
    }
    throw error;
  } finally {
    releaseOwnedGpuBuffer(srcBuffer, ownsSrcBuffer);
  }
}


export async function loadFloat(shardData, location, name, config) {
  if (!config) {
    throw new Error('Tensor load config is required.');
  }
  const device = getDevice();
  let ownsBuffer = !isGpuBufferInstance(shardData);
  const buffer = isGpuBufferInstance(shardData)
    ? shardData
    : acquireAlignedBuffer(location.size, name);
  let resultBuffer = null;
  try {
    if (ownsBuffer) {
      writeBufferAligned(device, buffer, shardData);
    }

    const dtype = selectRuleValue('loader', 'weights', 'floatLocationDtype', {
      locationDtype: location.dtype,
    });
    const layout = selectRuleValue('loader', 'weights', 'weightLayout', {
      layout: location.layout ?? null,
      useColumnWise: false,
    });
    const isMatmulWeight = shouldDequantizeToF16(location);

    if (isMatmulWeight) {
      ownsBuffer = false;
      return {
        data: createWeightBuffer(buffer, dtype, layout, location.shape, name),
        allocatedBuffers: [buffer],
      };
    }

    if (dtype === 'f16') {
      if (config.allowF32UpcastNonMatmul === false) {
        ownsBuffer = false;
        return {
          data: applyBufferLayout(buffer, location, 'f16'),
          allocatedBuffers: [buffer],
        };
      }
      const numElements = location.shape.reduce((a, b) => a * b, 1);
      logF32UpcastNonMatmul(name, numElements, buffer.size);
      debugTrace.loader(`F16->F32 upcast for non-matmul: ${name} (${numElements} elements, bufSize=${buffer.size})`);
      const inputTensor = createTensor(buffer, 'f16', [numElements], `${name}_f16`);
      const f32Tensor = await castF16ToF32(inputTensor);
      resultBuffer = f32Tensor.buffer;
      debugTrace.loader(`F16->F32 complete: ${name} resultSize=${f32Tensor.buffer.size}`);
      releaseOwnedGpuBuffer(buffer, ownsBuffer);
      ownsBuffer = false;
      return {
        data: applyBufferLayout(f32Tensor.buffer, location, 'f32'),
        allocatedBuffers: [f32Tensor.buffer],
      };
    }

    ownsBuffer = false;
    return {
      data: applyBufferLayout(buffer, location, dtype),
      allocatedBuffers: [buffer],
    };
  } catch (error) {
    if (isReleasableBuffer(resultBuffer)) {
      releaseBuffer(resultBuffer);
    }
    throw error;
  } finally {
    releaseOwnedGpuBuffer(buffer, ownsBuffer);
  }
}

// ============================================================================
// Main GPU Loading Entry Point
// ============================================================================


const GPU_LOADER_DISPATCH = {
  q4k_fused: (shardData, location, name, _config) => {
    debugTrace.loader(`Loading Q4K weight (fused): ${name} (size=${location.size})`);
    return loadQ4KFused(shardData, location, name);
  },
  q4k_dequant: (shardData, location, name, config) => {
    if (config.useFusedQ4K && isPackedQ4K(location)) {
      const [rows, cols] = location.shape;
      debugTrace.loader(`Packed Q4K weight ${name} [${rows},${cols}] incompatible with fused matmul, using dequant`);
    }
    return loadQ4KDequant(shardData, location, name, config);
  },
  q6k: (shardData, location, name, _config) => loadQ6K(shardData, location, name),
  bf16: (shardData, location, name, config) => loadBF16(shardData, location, name, config),
  float: (shardData, location, name, config) => loadFloat(shardData, location, name, config),
};

export async function loadTensorToGPU(shardData, location, name, config) {
  const dtype = location.dtype;
  const useFusedQ4K = shouldUseFusedQ4K(location, config);
  const loaderPath = selectRuleValue('loader', 'tensorLoader', 'gpuLoaderPath', { dtype, useFusedQ4K });
  const loader = GPU_LOADER_DISPATCH[loaderPath];
  if (!loader) {
    throw new Error(`Unknown GPU loader path: "${loaderPath}" for dtype "${dtype}"`);
  }
  return loader(shardData, location, name, config);
}


const CPU_LOADER_DISPATCH = {
  raw: (shardData, _location) => shardData,
  bf16_to_f32: (shardData, _location) => {
    const bf16 = new Uint16Array(shardData.slice().buffer);
    return convertBF16ToF32CPU(bf16);
  },
  f16_to_f32: (shardData, _location) => {
    const f16 = new Uint16Array(shardData.slice().buffer);
    return convertF16ToF32CPU(f16);
  },
  f32: (shardData, _location) => new Float32Array(shardData.slice().buffer),
};

export function loadTensorToCPU(shardData, location) {
  const dtype = location.dtype;
  const loaderPath = selectRuleValue('loader', 'tensorLoader', 'cpuLoaderPath', { dtype });
  const loader = CPU_LOADER_DISPATCH[loaderPath];
  if (!loader) {
    throw new Error(`Unknown CPU loader path: "${loaderPath}" for dtype "${dtype}"`);
  }
  return loader(shardData, location);
}
