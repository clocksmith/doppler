

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
import { acquireAlignedBuffer, canUseFusedQ4KStorage, convertF16ToF32CPU, getQ4KCpuReferenceContext, getQ4KOutputDtype, getShapeElementCount, isEmbeddingRole, isGpuBufferInstance, isPackedQ4K, loadFloat, loadQ4KDequant, loadQ4KMixed, loadQ6K, releaseOwnedGpuBuffer, resolveInputByteLength, shouldUseFusedQ4K, toUint16View, toUint8View, writeBufferAligned } from './load-executor.js';
export { convertF16ToF32CPU, getQ4KOutputDtype, getWeightLayout, isPackedQ4K, loadFloat, loadQ4KDequant, loadQ6K, shouldUseFusedQ4K } from './load-executor.js';

export { loadBF16 } from './bf16-loader.js';

// ============================================================================
// Q4K Detection
// ============================================================================

let loggedQ4KLimitFallback = false;

function toFloat32View(data, label) {
  const bytes = toUint8View(data);
  if (bytes.byteLength % 4 !== 0) {
    throw new Error(`${label}: byte length must be divisible by 4.`);
  }
  if (bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength) {
    return new Float32Array(bytes.buffer);
  }
  return new Float32Array(bytes.slice().buffer);
}

function getStorageCompanion(shardData, location, name, role) {
  const companion = shardData?.storageCompanions?.[role];
  if (!companion || !(companion.bytes instanceof Uint8Array)) {
    throw new Error(
      `W4A16 tensor "${name}" is missing required storage companion "${role}".`
    );
  }
  const declared = Array.isArray(location?.storage?.companions)
    ? location.storage.companions.find((entry) => entry.role === role)
    : null;
  if (declared && companion.tensorId !== declared.tensorId) {
    throw new Error(
      `W4A16 tensor "${name}" companion "${role}" resolved to "${companion.tensorId}", expected "${declared.tensorId}".`
    );
  }
  return companion;
}

function readW4A16LogicalShape(companion, fallbackShape, name) {
  const location = companion.location ?? null;
  const bytes = companion.bytes;
  const dtype = String(location?.dtype || '').toUpperCase();
  if (!Array.isArray(location?.shape) || location.shape.length !== 1 || location.shape[0] !== 2) {
    throw new Error(`W4A16 tensor "${name}" shape companion must have shape [2].`);
  }
  if (dtype === 'I64') {
    if (bytes.byteLength !== 16) {
      throw new Error(`W4A16 tensor "${name}" I64 shape companion must be 16 bytes.`);
    }
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    return [Number(view.getBigInt64(0, true)), Number(view.getBigInt64(8, true))];
  }
  if (dtype === 'I32' || dtype === 'U32') {
    if (bytes.byteLength !== 8) {
      throw new Error(`W4A16 tensor "${name}" ${dtype} shape companion must be 8 bytes.`);
    }
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const read = dtype === 'I32'
      ? (offset) => view.getInt32(offset, true)
      : (offset) => view.getUint32(offset, true);
    return [read(0), read(4)];
  }
  if (Array.isArray(fallbackShape) && fallbackShape.length === 2) {
    return fallbackShape;
  }
  throw new Error(`W4A16 tensor "${name}" has unsupported shape companion dtype "${location?.dtype}".`);
}

function assertW4A16Shape(shape, fallbackShape, name) {
  if (!Array.isArray(shape) || shape.length !== 2) {
    throw new Error(`W4A16 tensor "${name}" logical shape must be 2D.`);
  }
  const rows = Number(shape[0]);
  const cols = Number(shape[1]);
  if (!Number.isInteger(rows) || rows <= 0 || !Number.isInteger(cols) || cols <= 0) {
    throw new Error(`W4A16 tensor "${name}" has invalid logical shape ${JSON.stringify(shape)}.`);
  }
  if (Array.isArray(fallbackShape) && fallbackShape.length === 2) {
    if (rows !== fallbackShape[0] || cols !== fallbackShape[1]) {
      throw new Error(
        `W4A16 tensor "${name}" shape companion [${rows},${cols}] does not match manifest shape [${fallbackShape.join(',')}].`
      );
    }
  }
  return [rows, cols];
}

function bf16ToF32(bits) {
  const floats = new Float32Array(1);
  const uints = new Uint32Array(floats.buffer);
  uints[0] = (bits & 0xffff) << 16;
  return floats[0];
}

function readOffsetBinaryInt4(byte, highNibble) {
  const value = highNibble ? ((byte >> 4) & 0x0f) : (byte & 0x0f);
  return value - 8;
}

function readW4A16Scales(companion, expectedScales, name) {
  const dtype = String(companion.location?.dtype || '').toUpperCase();
  if (dtype === 'F16') {
    const packed = toUint16View(companion.bytes, `W4A16 scales for ${name}`);
    if (packed.length !== expectedScales) {
      throw new Error(
        `W4A16 tensor "${name}" scale count ${packed.length} does not match expected ${expectedScales}.`
      );
    }
    const scales = new Float32Array(packed.length);
    for (let i = 0; i < packed.length; i += 1) {
      scales[i] = f16ToF32(packed[i]);
    }
    return scales;
  }
  if (dtype === 'BF16') {
    const packed = toUint16View(companion.bytes, `W4A16 scales for ${name}`);
    if (packed.length !== expectedScales) {
      throw new Error(
        `W4A16 tensor "${name}" scale count ${packed.length} does not match expected ${expectedScales}.`
      );
    }
    const scales = new Float32Array(packed.length);
    for (let i = 0; i < packed.length; i += 1) {
      scales[i] = bf16ToF32(packed[i]);
    }
    return scales;
  }
  if (dtype === 'F32') {
    const scales = toFloat32View(companion.bytes, `W4A16 scales for ${name}`);
    if (scales.length !== expectedScales) {
      throw new Error(
        `W4A16 tensor "${name}" scale count ${scales.length} does not match expected ${expectedScales}.`
      );
    }
    return scales;
  }
  throw new Error(`W4A16 tensor "${name}" has unsupported scale companion dtype "${companion.location?.dtype}".`);
}

function validateW4A16PackedStorage(shardData, location, name) {
  const scaleCompanion = getStorageCompanion(shardData, location, name, 'scales');
  const shapeCompanion = getStorageCompanion(shardData, location, name, 'shape');
  const [rows, cols] = assertW4A16Shape(
    readW4A16LogicalShape(shapeCompanion, location.shape, name),
    location.shape,
    name
  );
  const groupsPerRow = Math.ceil(cols / 32);
  const expectedPackedBytes = rows * groupsPerRow * 16;
  if (shardData.byteLength !== expectedPackedBytes) {
    throw new Error(
      `W4A16 tensor "${name}" packed byte length ${shardData.byteLength} does not match expected ${expectedPackedBytes}.`
    );
  }
  const scaleDtype = String(scaleCompanion.location?.dtype || '').toUpperCase();
  const scaleBytesPerElement = scaleDtype === 'F32'
    ? 4
    : (scaleDtype === 'F16' || scaleDtype === 'BF16' ? 2 : null);
  if (scaleBytesPerElement == null) {
    throw new Error(`W4A16 tensor "${name}" has unsupported scale companion dtype "${scaleCompanion.location?.dtype}".`);
  }
  const expectedScaleBytes = rows * groupsPerRow * scaleBytesPerElement;
  if (scaleCompanion.bytes.byteLength !== expectedScaleBytes) {
    throw new Error(
      `W4A16 tensor "${name}" scale byte length ${scaleCompanion.bytes.byteLength} does not match expected ${expectedScaleBytes}.`
    );
  }
  return {
    rows,
    cols,
    groupsPerRow,
    scaleDtype: scaleDtype.toLowerCase(),
    scaleBytes: scaleCompanion.bytes,
  };
}

function dequantizeW4A16ToF16(shardData, location, name) {
  const scaleCompanion = getStorageCompanion(shardData, location, name, 'scales');
  const shapeCompanion = getStorageCompanion(shardData, location, name, 'shape');
  const [rows, cols] = assertW4A16Shape(
    readW4A16LogicalShape(shapeCompanion, location.shape, name),
    location.shape,
    name
  );
  const groupsPerRow = Math.ceil(cols / 32);
  const expectedPackedBytes = rows * groupsPerRow * 16;
  if (shardData.byteLength !== expectedPackedBytes) {
    throw new Error(
      `W4A16 tensor "${name}" packed byte length ${shardData.byteLength} does not match expected ${expectedPackedBytes}.`
    );
  }
  const expectedScales = rows * groupsPerRow;
  const scales = readW4A16Scales(scaleCompanion, expectedScales, name);
  const out = new Uint16Array(rows * cols);
  for (let row = 0; row < rows; row += 1) {
    for (let group = 0; group < groupsPerRow; group += 1) {
      const scale = scales[(row * groupsPerRow) + group];
      const packedOffset = ((row * groupsPerRow) + group) * 16;
      for (let lane = 0; lane < 32; lane += 1) {
        const col = (group * 32) + lane;
        if (col >= cols) break;
        const byte = shardData[packedOffset + Math.floor(lane / 2)];
        const quant = readOffsetBinaryInt4(byte, (lane % 2) === 1);
        out[(row * cols) + col] = float32ToFloat16(quant * scale);
      }
    }
  }
  return new Uint8Array(out.buffer, out.byteOffset, out.byteLength);
}

export function isLiteRTAffineInt4FusedEligible(location, config) {
  const caps = config?.gpuCapabilities || getKernelCapabilities();
  if (caps?.hasF16 !== true) return false;

  if (!Array.isArray(location?.shape) || location.shape.length !== 2) return false;
  if (!location?.role) return false;
  if (isEmbeddingRole(location)) return false;
  if (location.role !== 'matmul' && location.role !== 'lm_head') return false;
  if (!shouldDequantizeToF16(location)) return false;

  const transform = location.sourceTransform;
  if (!transform || typeof transform !== 'object') return false;
  const sourceDtype = String(transform.sourceDtype || '').toUpperCase();
  const targetDtype = String(transform.targetDtype || '').toUpperCase();
  const locationDtype = String(location.dtype || '').toUpperCase();
  const storageEncoding = String(transform.storageEncoding || '').toLowerCase();
  const scale = Number(transform.scale);
  const zeroPoint = Number(transform.zeroPoint);
  const storageEncodingSupported = storageEncoding === 'signed' || storageEncoding === 'offset_binary';

  return transform.kind === 'affine_dequant'
    && transform.scheme === 'per_tensor_affine'
    && sourceDtype === 'INT4'
    && targetDtype === 'F16'
    && locationDtype === 'F16'
    && storageEncodingSupported
    && Number.isFinite(scale)
    && Math.abs(scale - 0.0625) <= Number.EPSILON
    && Number.isSafeInteger(zeroPoint)
    && zeroPoint === 0;
}

export function isW4A16FusedEligible(location, config) {
  const caps = config?.gpuCapabilities || getKernelCapabilities();
  if (caps?.hasF16 !== true) return false;

  if (!Array.isArray(location?.shape) || location.shape.length !== 2) return false;
  if (String(location?.dtype || '').toUpperCase() !== 'W4A16') return false;
  if (!location?.role) return false;
  if (isEmbeddingRole(location)) return false;
  if (location.role !== 'matmul' && location.role !== 'lm_head') return false;
  if (!shouldDequantizeToF16(location)) return false;
  if (location?.storage?.packing !== 'w4a16') return false;
  if (!Array.isArray(location?.storage?.companions)) return false;

  const hasScales = location.storage.companions.some((entry) => entry.role === 'scales');
  const hasShape = location.storage.companions.some((entry) => entry.role === 'shape');
  return hasScales && hasShape;
}

function getQ4KDenseMaterializedSizeBytes(location, config) {
  if (!Array.isArray(location.shape) || location.shape.length === 0) {
    return null;
  }
  const elementCount = getShapeElementCount(location.shape);
  if (!Number.isFinite(elementCount) || elementCount <= 0) {
    return null;
  }
  const outputDtype = getQ4KOutputDtype(location, config);
  const bytesPerElement = outputDtype === 'f16' ? 2 : 4;
  return elementCount * bytesPerElement;
}

function getMaxStorageBufferBindingSize() {
  const device = getDevice();
  const maxStorage = device?.limits?.maxStorageBufferBindingSize;
  return Number.isFinite(maxStorage) && maxStorage > 0 ? maxStorage : null;
}

function resolveQ4KLimitFallback(location, config) {
  if (location?.dtype !== 'Q4_K_M' && location?.dtype !== 'Q4_K') {
    return {
      denseExceedsBindingLimit: false,
      limitFallbackEligible: false,
      denseSizeBytes: null,
      maxBindingSizeBytes: null,
    };
  }

  const denseSizeBytes = getQ4KDenseMaterializedSizeBytes(location, config);
  const maxBindingSizeBytes = getMaxStorageBufferBindingSize();
  const denseExceedsBindingLimit = (
    denseSizeBytes != null
    && maxBindingSizeBytes != null
    && denseSizeBytes > maxBindingSizeBytes
  );
  const packedSizeBytes = Number.isFinite(location.size) ? location.size : null;
  const packedFitsBindingLimit = (
    packedSizeBytes != null
    && maxBindingSizeBytes != null
    && packedSizeBytes <= maxBindingSizeBytes
  );
  const limitFallbackEligible = (
    denseExceedsBindingLimit
    && config.keepF32Weights !== true
    && packedFitsBindingLimit
    && canUseFusedQ4KStorage(location, config)
  );

  return {
    denseExceedsBindingLimit,
    limitFallbackEligible,
    denseSizeBytes,
    maxBindingSizeBytes,
  };
}

function logQ4KLimitFallbackOnce(name, fallback) {
  if (loggedQ4KLimitFallback) {
    return;
  }
  loggedQ4KLimitFallback = true;
  log.warn(
    'Loader',
    `Q4K dense materialization for "${name}" would require ${fallback.denseSizeBytes} bytes, ` +
    `exceeding maxStorageBufferBindingSize=${fallback.maxBindingSizeBytes}; retaining packed Q4K for fused matmul.`
  );
}

// ============================================================================
// Dtype Output Selection
// ============================================================================

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

export async function loadLiteRTInt4Fused(shardData, location, name, config = null) {
  if (isGpuBufferInstance(shardData)) {
    throw new Error(
      `LiteRT INT4 tensor "${name}" requires raw packed source bytes before GPU upload.`
    );
  }
  if (!isLiteRTAffineInt4FusedEligible(location, config ?? { gpuCapabilities: getKernelCapabilities() })) {
    throw new Error(
      `LiteRT INT4 tensor "${name}" does not match the fused fixed-affine contract ` +
      '(INT4 -> F16, storageEncoding=signed|offset_binary, per_tensor_affine, scale=0.0625, zeroPoint=0, 2D matmul/lm_head role).'
    );
  }

  const [rows, cols] = location.shape;
  const expectedBytes = rows * Math.ceil(cols / 2);
  const actualBytes = resolveInputByteLength(shardData, location.size);
  if (actualBytes !== expectedBytes) {
    throw new Error(
      `LiteRT INT4 tensor "${name}" packed byte size mismatch. ` +
      `Expected ${expectedBytes} bytes for shape [${rows},${cols}], got ${actualBytes}.`
    );
  }

  const device = getDevice();
  const buffer = acquireAlignedBuffer(actualBytes, `litert_int4_${name}`);
  try {
    writeBufferAligned(device, buffer, shardData);
    return {
      data: createWeightBuffer(buffer, 'litert_int4', 'row', location.shape, name, null, {
        storageEncoding: String(location.sourceTransform.storageEncoding).toLowerCase(),
      }),
      allocatedBuffers: [buffer],
    };
  } catch (error) {
    releaseBuffer(buffer);
    throw error;
  }
}

export async function loadW4A16Fused(shardData, location, name, config = null) {
  if (isGpuBufferInstance(shardData)) {
    throw new Error(
      `W4A16 tensor "${name}" requires raw packed source bytes before GPU upload.`
    );
  }
  if (!isW4A16FusedEligible(location, config ?? { gpuCapabilities: getKernelCapabilities() })) {
    throw new Error(
      `W4A16 tensor "${name}" does not match the fused packed contract ` +
      '(dtype=W4A16, storage.packing=w4a16, scales+shape companions, 2D matmul/lm_head role).'
    );
  }

  const storage = validateW4A16PackedStorage(shardData, location, name);
  const device = getDevice();
  const weightBuffer = acquireAlignedBuffer(shardData.byteLength, `w4a16_${name}`);
  const scaleBuffer = acquireAlignedBuffer(storage.scaleBytes.byteLength, `w4a16_scales_${name}`);
  try {
    writeBufferAligned(device, weightBuffer, shardData);
    writeBufferAligned(device, scaleBuffer, storage.scaleBytes);
    return {
      data: createWeightBuffer(weightBuffer, 'w4a16', 'row', location.shape, name, null, {
        scaleBuffer,
        scaleDtype: storage.scaleDtype,
        groupsPerRow: storage.groupsPerRow,
      }),
      allocatedBuffers: [weightBuffer, scaleBuffer],
    };
  } catch (error) {
    releaseBuffer(weightBuffer);
    releaseBuffer(scaleBuffer);
    throw error;
  }
}

export async function loadW4A16Dequant(shardData, location, name, config) {
  if (!config) {
    throw new Error('Tensor load config is required.');
  }
  if (isGpuBufferInstance(shardData)) {
    throw new Error(
      `W4A16 tensor "${name}" requires CPU-side storage companion materialization before GPU upload.`
    );
  }
  const f16Bytes = dequantizeW4A16ToF16(shardData, location, name);
  const device = getDevice();
  const buffer = acquireAlignedBuffer(f16Bytes.byteLength, `w4a16_dequant_${name}`);
  try {
    writeBufferAligned(device, buffer, f16Bytes);
    const layout = selectRuleValue('loader', 'weights', 'weightLayout', {
      layout: location.layout ?? null,
      useColumnWise: false,
    });
    if (shouldDequantizeToF16(location)) {
      return {
        data: createWeightBuffer(buffer, 'f16', layout, location.shape, name),
        allocatedBuffers: [buffer],
      };
    }
    return {
      data: applyBufferLayout(buffer, location, 'f16'),
      allocatedBuffers: [buffer],
    };
  } catch (error) {
    releaseBuffer(buffer);
    throw error;
  }
}

// ============================================================================
// Main GPU Loading Entry Point
// ============================================================================

const GPU_LOADER_DISPATCH = {
  functional_descriptor: (shardData, location, name, config) => {
    debugTrace.loader(`Loading functional descriptor: ${name}`);
    return loadFunctionalDescriptor(shardData, location, name, config);
  },
  litert_int4_fused: (shardData, location, name, config) => {
    debugTrace.loader(`Loading LiteRT INT4 weight (fused): ${name} (size=${location.size})`);
    return loadLiteRTInt4Fused(shardData, location, name, config);
  },
  w4a16_fused: (shardData, location, name, config) => {
    debugTrace.loader(`Loading W4A16 weight (fused): ${name} (size=${location.size})`);
    return loadW4A16Fused(shardData, location, name, config);
  },
  q4k_mixed: (shardData, location, name, config) => loadQ4KMixed(shardData, location, name, config),
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
  q4k_dequant_reference: (shardData, location, name, config) => loadQ4KDequant(
    shardData,
    location,
    name,
    {
      ...config,
      loaderDebug: {
        ...(config?.loaderDebug ?? {}),
        preferCpuDequant: true,
      },
    }
  ),
  q6k: (shardData, location, name, _config) => loadQ6K(shardData, location, name),
  w4a16_dequant_reference: (shardData, location, name, config) => loadW4A16Dequant(shardData, location, name, config),
  bf16: (shardData, location, name, config) => loadBF16(shardData, location, name, config),
  float: (shardData, location, name, config) => loadFloat(shardData, location, name, config),
  unsupported_packed_quantization: (_shardData, location, name, _config) => {
    throw new Error(
      `Unsupported packed quantization dtype "${location.dtype}" for tensor "${name}". ` +
      'Add a native loader and kernel path before enabling runtime execution.'
    );
  },
};

export async function loadTensorToGPU(shardData, location, name, config) {
  const dtype = location.dtype;
  const useFusedQ4K = shouldUseFusedQ4K(location, config);
  const requiresFusedQ4KRole = Array.isArray(config?.q4kFusedRoles)
    && config.q4kFusedRoles.includes(location.role);
  const caps = config?.gpuCapabilities || getKernelCapabilities();
  const platformId = getPlatformConfig()?.platform?.id ?? null;
  const q4kReferenceContext = getQ4KCpuReferenceContext(shardData, location, config);
  const q4kBasicBackendClass = platformId === 'basic'
    || (caps?.hasSubgroups !== true && caps?.hasF16 !== true);
  const q4kLimitFallback = resolveQ4KLimitFallback(location, config);
  const litertAffineInt4FusedEligible = isLiteRTAffineInt4FusedEligible(location, { ...config, gpuCapabilities: caps });
  const w4a16FusedEligible = isW4A16FusedEligible(location, { ...config, gpuCapabilities: caps });
  const loaderPath = selectRuleValue('loader', 'tensorLoader', 'gpuLoaderPath', {
    dtype,
    role: location.role ?? null,
    litertAffineInt4FusedEligible,
    w4a16FusedEligible,
    useFusedQ4K,
    requiresFusedQ4KRole,
    q4kMaterializationMode: config.q4kMaterializationMode ?? 'dense',
    q4kCpuReferenceEligible: q4kReferenceContext.eligible,
    q4kBasicBackendClass,
    q4kDenseExceedsBindingLimit: q4kLimitFallback.denseExceedsBindingLimit,
    q4kLimitFallbackEligible: q4kLimitFallback.limitFallbackEligible,
  });
  const loader = GPU_LOADER_DISPATCH[loaderPath];
  if (!loader) {
    throw new Error(`Unknown GPU loader path: "${loaderPath}" for dtype "${dtype}"`);
  }
  if (loaderPath === 'q4k_fused' && q4kLimitFallback.limitFallbackEligible) {
    logQ4KLimitFallbackOnce(name, q4kLimitFallback);
  }
  return loader(shardData, location, name, config);
}

const CPU_LOADER_DISPATCH = {
  unsupported_functional_descriptor: (_shardData, _location, name) => {
    throw new Error(
      `FUNCTIONAL_DESCRIPTOR tensor "${name ?? 'unknown'}" requires GPU materialization through loadTensorToGPU.`
    );
  },
  raw: (shardData, _location) => shardData,
  w4a16_dequant_reference: (shardData, location) => {
    const f16Bytes = dequantizeW4A16ToF16(shardData, location, 'cpu');
    return convertF16ToF32CPU(toUint16View(f16Bytes, 'W4A16 CPU dequantized tensor load'));
  },
  unsupported_packed_quantization: (_shardData, location) => {
    throw new Error(
      `Unsupported packed quantization dtype "${location.dtype}" for CPU tensor load. ` +
      'Add a native loader before enabling runtime execution.'
    );
  },
  bf16_to_f32: (shardData, _location) => convertBF16ToF32CPU(
    toUint16View(shardData, 'BF16 CPU tensor load')
  ),
  f16_to_f32: (shardData, _location) => convertF16ToF32CPU(
    toUint16View(shardData, 'F16 CPU tensor load')
  ),
  f32: (shardData, _location) => toFloat32View(shardData, 'F32 CPU tensor load'),
};

export function loadTensorToCPU(shardData, location, name = null) {
  const dtype = location.dtype;
  const loaderPath = selectRuleValue('loader', 'tensorLoader', 'cpuLoaderPath', { dtype });
  const loader = CPU_LOADER_DISPATCH[loaderPath];
  if (!loader) {
    throw new Error(`Unknown CPU loader path: "${loaderPath}" for dtype "${dtype}"`);
  }
  return loader(shardData, location, name);
}
