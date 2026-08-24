import { getDevice, getKernelCapabilities, getPlatformConfig } from '../../gpu/device.js';
import { acquireBuffer, releaseBuffer, readBuffer } from '../../memory/buffer-pool.js';
import { dequantize, dequantizeRowwise, dequantizeQ6K, castF16ToF32 } from '../../gpu/kernel-selector.js';
import { createTensor } from '../../gpu/tensor.js';
import { createWeightBuffer } from '../../gpu/weight-buffer.js';
import { f16ToF32, shouldDequantizeToF16, applyBufferLayout } from '../dtype-utils.js';
import { QK_K, Q4K_BLOCK_BYTES, Q6K_BLOCK_BYTES } from '../quantization-constants.js';
import { log, trace as debugTrace } from '../../debug/index.js';
import { selectRuleValue } from '../../rules/rule-registry.js';
import { dequantizeQ4KM, dequantizeQ4KMRowWise, float32ToFloat16 } from '../../converter/quantizer.js';
import { hasSourceTransform } from './source-transform.js';
import { loadFunctionalDescriptor } from './functional-descriptor-loader.js';
import { loadBF16 } from './bf16-loader.js';

export let loggedF32UpcastNonMatmul = false;

export function isGpuBufferInstance(value) {
  return typeof GPUBuffer !== 'undefined' && value instanceof GPUBuffer;
}

export function isReleasableBuffer(value) {
  return typeof value === 'object' && value !== null && 'size' in value;
}

export function releaseOwnedGpuBuffer(buffer, owned) {
  if (!owned || !isReleasableBuffer(buffer)) {
    return;
  }
  releaseBuffer(buffer);
}

export function normalizeLoaderDebugConfig(config) {
  const debug = config?.loaderDebug;
  if (!debug || typeof debug !== 'object') {
    return null;
  }

  return {
    enabled: debug.enabled === true,
    forceGpuDequant: debug.forceGpuDequant === true,
    preferCpuDequant: debug.preferCpuDequant === true,
    failOnCpuDequantPath: debug.failOnCpuDequantPath === true,
    runQ4KDequantParity: debug.runQ4KDequantParity === true,
    q4kDequantParitySamples: Number.isFinite(debug.q4kDequantParitySamples)
      ? Math.min(4096, Math.max(1, Math.trunc(debug.q4kDequantParitySamples)))
      : 256,
  };
}

export function logF32UpcastNonMatmul(name, numElements, bufferSize) {
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

export function alignTo4(size) {
  return Math.ceil(size / 4) * 4;
}

export function toUint8View(data) {
  if (data instanceof Uint8Array) return data;
  if (ArrayBuffer.isView(data)) {
    return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  }
  return new Uint8Array(data);
}

export function toUint16View(data, label) {
  const bytes = toUint8View(data);
  if (bytes.byteLength % 2 !== 0) {
    throw new Error(`${label}: byte length must be divisible by 2.`);
  }
  if (bytes.byteOffset % 2 === 0) {
    return new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
  }
  return new Uint16Array(bytes.slice().buffer);
}

export function resolveInputByteLength(data, fallbackSize) {
  if (data instanceof Uint8Array) return data.byteLength;
  if (ArrayBuffer.isView(data)) return data.byteLength;
  if (data instanceof ArrayBuffer) return data.byteLength;
  if (isGpuBufferInstance(data)) return data.size;
  return fallbackSize;
}

export function writeBufferAligned(device, buffer, data) {
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

export function acquireAlignedBuffer(size, label) {
  return acquireBuffer(alignTo4(size), undefined, label);
}

export function getShapeElementCount(shape) {
  if (!Array.isArray(shape)) {
    throw new Error('Tensor shape must be an array.');
  }
  return shape.reduce((product, value) => product * value, 1);
}

export function isPackedQ4K(location) {
  if (!Array.isArray(location.shape) || location.shape.length < 2) {
    return false;
  }
  const cols = location.shape[location.shape.length - 1];
  const rows = location.shape.slice(0, -1).reduce((a, b) => a * b, 1);
  const expectedRowwise = rows * Math.ceil(cols / QK_K) * Q4K_BLOCK_BYTES;
  return location.size < expectedRowwise;
}

export function isEmbeddingRole(location) {
  if (!location?.role) {
    throw new Error('Tensor role is required to determine embedding layout.');
  }
  return location.role === 'embedding';
}

export function shouldUseFusedQ4K(location, config) {
  if (!config.useFusedQ4K) return false;
  return canUseFusedQ4KStorage(location, config);
}

export function canUseFusedQ4KStorage(location, config) {
  const caps = config.gpuCapabilities || getKernelCapabilities();
  if (!caps?.hasSubgroups) return false;

  const isMatmulWeight = shouldDequantizeToF16(location);
  if (!isMatmulWeight) return false;

  if (isEmbeddingRole(location)) return false;
  if (isPackedQ4K(location)) return false;

  return true;
}

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

export function convertF16ToF32CPU(f16Data) {
  const f32 = new Float32Array(f16Data.length);
  for (let i = 0; i < f16Data.length; i++) {
    f32[i] = f16ToF32(f16Data[i]);
  }
  return f32;
}

export async function materializeQ4KDenseBuffer(quantBuffer, shardData, location, name, config) {
  let dequantized = null;
  const outputDtype = getQ4KOutputDtype(location, config);
  const loaderDebug = normalizeLoaderDebugConfig(config);
  const debugEnabled = loaderDebug?.enabled === true;
  const forceGpuDequant = loaderDebug?.forceGpuDequant === true;
  const failOnCpuDequantPath = loaderDebug?.failOnCpuDequantPath === true;
  const runQ4KDequantParity = loaderDebug?.runQ4KDequantParity === true;
  const paritySamples = loaderDebug?.q4kDequantParitySamples ?? 256;

  const q4kCpuReferenceContext = getQ4KCpuReferenceContext(shardData, location, config);
  const { needsRowwise, layout, K } = q4kCpuReferenceContext;
  const preferCpuDequant = loaderDebug?.preferCpuDequant === true;
  const canUseCpuReference = !forceGpuDequant && preferCpuDequant && q4kCpuReferenceContext.eligible;

  if (canUseCpuReference && failOnCpuDequantPath) {
    throw new Error(
      `[LoaderDebug] CPU dequant path taken for ${name}; this run is configured fail-closed. ` +
      'Set runtime.shared.debug.loader.forceGpuDequant=true to isolate GPU dequant.'
    );
  }

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
    try {
      writeBufferAligned(getDevice(), outputBuffer, new Uint8Array(f32Weights.buffer));
      return {
        buffer: outputBuffer,
        outputDtype: 'f32',
        layout,
        allocatedBuffers: [outputBuffer],
      };
    } catch (error) {
      releaseBuffer(outputBuffer);
      throw error;
    }
  }

  let numBlocks = null;
  let rowCount = null;
  let blocksPerRow = null;
  let denseElementCount = null;
  let dequantizedTensor;
  if (needsRowwise) {
    rowCount = location.shape.slice(0, -1).reduce((a, b) => a * b, 1);
    blocksPerRow = Math.ceil(K / QK_K);
    numBlocks = rowCount * blocksPerRow;
    denseElementCount = rowCount * K;
    debugTrace.loader(
      `Dequantizing ${name} (row-wise): [${rowCount},${K}], K not 256-aligned, ` +
      `outputDtype=${outputDtype}`
    );
    dequantizedTensor = await dequantizeRowwise(quantBuffer, rowCount, K, { outputDtype });
  } else {
    numBlocks = Math.ceil(location.size / Q4K_BLOCK_BYTES);
    denseElementCount = numBlocks * QK_K;
    debugTrace.loader(
      `Dequantizing ${name}: size=${location.size}, numBlocks=${numBlocks}, ` +
      `outputDtype=${outputDtype}, expectedOutput=${numBlocks * QK_K * (outputDtype === 'f16' ? 2 : 4)}`
    );
    dequantizedTensor = await dequantize(quantBuffer, numBlocks, { outputDtype });
  }
  dequantized = dequantizedTensor.buffer;

  debugTrace.loader(`Dequantized ${name}: resultSize=${dequantized.size}`);

  if (runQ4KDequantParity && !isGpuBufferInstance(shardData) && dequantized && numBlocks !== null) {
    const isProbeTarget = debugEnabled &&
      (name.includes('.self_attn.q_proj.weight') || name.includes('.self_attn.k_proj.weight') ||
        name.includes('.self_attn.v_proj.weight') || name.includes('.self_attn.qkv_proj.weight'));

    if (isProbeTarget) {
      try {
        const bytesPerElem = outputDtype === 'f16' ? 2 : 4;
        const requestedOutputBytes = denseElementCount * bytesPerElem;
        const sampleCount = paritySamples;
        const readSize = Math.min(sampleCount * bytesPerElem, requestedOutputBytes, dequantized.size);
        const gpuRaw = await readBuffer(dequantized, readSize);
        const gpuBytes = gpuRaw instanceof ArrayBuffer
          ? new Uint8Array(gpuRaw)
          : new Uint8Array(gpuRaw.buffer, gpuRaw.byteOffset, gpuRaw.byteLength);

        let gpuVals;
        if (outputDtype === 'f16') {
          const u16 = new Uint16Array(gpuBytes.buffer, gpuBytes.byteOffset,
            Math.min(sampleCount, Math.floor(gpuBytes.byteLength / 2)));
          gpuVals = Array.from(u16, (half) => f16ToF32(half));
        } else {
          const f32 = new Float32Array(gpuBytes.buffer, gpuBytes.byteOffset,
            Math.min(sampleCount, Math.floor(gpuBytes.byteLength / 4)));
          gpuVals = Array.from(f32);
        }

        const quantizedBytes = toUint8View(shardData);
        const cpuRef = Array.from(
          needsRowwise
            ? dequantizeQ4KMRowWise(quantizedBytes, location.shape)
            : dequantizeQ4KM(quantizedBytes, numBlocks, location.shape)
        ).slice(0, gpuVals.length);

        let maxDiff = 0;
        let diffIdx = -1;
        for (let i = 0; i < gpuVals.length && i < cpuRef.length; i++) {
          const d = Math.abs(gpuVals[i] - cpuRef[i]);
          if (d > maxDiff) {
            maxDiff = d;
            diffIdx = i;
          }
        }

        log.warn('DequantProbe',
          `tensor="${name}" shape=[${location.shape}] ` +
          `location.size=${location.size} numBlocks=${numBlocks} blocksPerRow=${blocksPerRow} ` +
          `rows=${rowCount} K=${K} denseElements=${denseElementCount} outputDtype=${outputDtype} ` +
          `bytesPerElem=${bytesPerElem} requestedOutputBytes=${requestedOutputBytes} bufSize=${dequantized.size} ` +
          `runParity=true sampleCount=${sampleCount}`
        );
        log.warn('DequantProbe',
          `parity: maxDiff=${maxDiff.toFixed(8)} at idx=${diffIdx} ` +
          `gpu[0..3]=[${gpuVals.slice(0, 4).map((v) => v.toFixed(6))}] ` +
          `cpu[0..3]=[${cpuRef.slice(0, 4).map((v) => v.toFixed(6))}]`
        );
      } catch (e) {
        log.warn('DequantProbe', `Readback failed: ${e.message}`);
      }
    }
  }

  return {
    buffer: dequantized,
    outputDtype,
    layout,
    allocatedBuffers: [dequantized],
  };
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

    const dense = await materializeQ4KDenseBuffer(quantBuffer, shardData, location, name, config);
    dequantized = dense.buffer;
    releaseOwnedGpuBuffer(quantBuffer, ownsQuantBuffer);
    ownsQuantBuffer = false;

    return {
      data: createWeightBuffer(dequantized, dense.outputDtype, dense.layout, location.shape, name),
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

export async function loadQ4KMixed(shardData, location, name, config) {
  const canMaterializeMixed = shouldUseFusedQ4K(location, config)
    && config.q4kLayout === 'row';
  if (!canMaterializeMixed) {
    return loadQ4KDequant(shardData, location, name, config);
  }

  const device = getDevice();
  let ownsQuantBuffer = !isGpuBufferInstance(shardData);
  const quantBuffer = isGpuBufferInstance(shardData)
    ? shardData
    : acquireAlignedBuffer(location.size, `q4k_mixed_${name}`);
  let dequantized = null;
  try {
    if (ownsQuantBuffer) {
      writeBufferAligned(device, quantBuffer, shardData);
    }

    const dense = await materializeQ4KDenseBuffer(quantBuffer, shardData, location, name, config);
    dequantized = dense.buffer;
    ownsQuantBuffer = false;

    return {
      data: createWeightBuffer(
        dense.buffer,
        dense.outputDtype,
        dense.layout,
        location.shape,
        name,
        {
          q4k: { buffer: quantBuffer, layout: 'row' },
        }
      ),
      allocatedBuffers: [quantBuffer, dense.buffer],
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

export function getQ4KCpuReferenceContext(shardData, location, config) {
  const outputDtype = getQ4KOutputDtype(location, config);
  const isMatrixLike = Array.isArray(location.shape) && location.shape.length >= 2;
  const K = isMatrixLike ? location.shape[location.shape.length - 1] : 0;
  const layout = getWeightLayout(location, config);
  const needsRowwise = isMatrixLike && layout === 'row' && K > 0 && K % QK_K !== 0;
  const eligible = outputDtype === 'f32'
    && !isGpuBufferInstance(shardData)
    && (!needsRowwise || layout === 'row');
  return {
    eligible,
    outputDtype,
    needsRowwise,
    layout,
    K,
  };
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

export async function loadFloat(shardData, location, name, config) {
  if (!config) {
    throw new Error('Tensor load config is required.');
  }
  if (hasSourceTransform(location) && isGpuBufferInstance(shardData)) {
    throw new Error(
      `Tensor "${name}" requires CPU-side sourceTransform materialization before GPU upload. ` +
      'Disable streaming for this tensor and load from assembled bytes.'
    );
  }
  const device = getDevice();
  const inputByteLength = resolveInputByteLength(shardData, location.size);
  let ownsBuffer = !isGpuBufferInstance(shardData);
  const buffer = isGpuBufferInstance(shardData)
    ? shardData
    : acquireAlignedBuffer(inputByteLength, name);
  let resultBuffer = null;
  try {
    if (ownsBuffer) {
      writeBufferAligned(device, buffer, shardData);
    }

    const layout = selectRuleValue('loader', 'weights', 'weightLayout', {
      layout: location.layout ?? null,
      useColumnWise: false,
    });
    const isMatmulWeight = shouldDequantizeToF16(location);
    const caps = config.gpuCapabilities || getKernelCapabilities();
    const dtype = isMatmulWeight
      ? selectRuleValue('loader', 'weights', 'matmulWeightDtype', {
        locationDtype: location.dtype,
        hasF16: Boolean(caps?.hasF16),
        isMatmulWeight,
        keepF32Weights: Boolean(config.keepF32Weights),
      })
      : selectRuleValue('loader', 'weights', 'floatLocationDtype', {
        locationDtype: location.dtype,
      });

    if (location.dtype === 'F16' && dtype === 'f32') {
      if (isGpuBufferInstance(shardData)) {
        throw new Error(
          `Tensor "${name}" requires CPU F16->F32 materialization on this runtime, ` +
          'but the source was already uploaded to a GPU buffer.'
        );
      }
      const f32Values = convertF16ToF32CPU(toUint16View(shardData, `F16 tensor load for ${name}`));
      const f32Buffer = acquireAlignedBuffer(f32Values.byteLength, `${name}_f32`);
      resultBuffer = f32Buffer;
      writeBufferAligned(device, f32Buffer, new Uint8Array(f32Values.buffer, f32Values.byteOffset, f32Values.byteLength));
      releaseOwnedGpuBuffer(buffer, ownsBuffer);
      ownsBuffer = false;
      resultBuffer = null;
      if (isMatmulWeight) {
        return {
          data: createWeightBuffer(f32Buffer, 'f32', layout, location.shape, name),
          allocatedBuffers: [f32Buffer],
        };
      }
      return {
        data: applyBufferLayout(f32Buffer, location, 'f32'),
        allocatedBuffers: [f32Buffer],
      };
    }

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
      const numElements = getShapeElementCount(location.shape);
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
