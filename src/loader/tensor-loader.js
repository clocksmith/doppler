

import { getDevice, getKernelCapabilities } from '../gpu/device.js';
import { acquireBuffer, releaseBuffer } from '../memory/buffer-pool.js';
import { dequantize, dequantizeRowwise, dequantizeQ6K, castF16ToF32, runBF16ToF16 } from '../gpu/kernel-selector.js';
import { createTensor } from '../gpu/tensor.js';
import { createWeightBuffer } from '../gpu/weight-buffer.js';
import { f16ToF32, convertBF16ToF32GPU, shouldDequantizeToF16, applyBufferLayout } from './dtype-utils.js';
import { QK_K, Q4K_BLOCK_BYTES, Q6K_BLOCK_BYTES } from './quantization-constants.js';
import { log, trace as debugTrace } from '../debug/index.js';
import { selectRuleValue } from '../rules/rule-registry.js';

// ============================================================================
// Q4K Detection
// ============================================================================

let loggedF32UpcastNonMatmul = false;

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
  const buffer = acquireBuffer(location.size, undefined, `q4k_${name}`);
  device.queue.writeBuffer(buffer, 0,  ( (shardData)));

  return {
    data: createWeightBuffer(buffer, 'q4k', 'row', location.shape, name),
    allocatedBuffers: [buffer],
  };
}


export async function loadQ4KDequant(shardData, location, name, config) {
  const device = getDevice();
  const quantBuffer = acquireBuffer(location.size, undefined, `quant_${name}`);
  device.queue.writeBuffer(quantBuffer, 0,  ( (shardData)));

  const outputDtype = getQ4KOutputDtype(location, config);

  // Check if this is a 2D matrix with K (columns) not aligned to QK_K (256).
  // If so, we need row-wise dequant to produce proper row-major output.
  const is2DMatrix = Array.isArray(location.shape) && location.shape.length === 2;
  const K = is2DMatrix ? location.shape[1] : 0;
  const needsRowwise = is2DMatrix && K > 0 && K % QK_K !== 0;

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
  const dequantized = dequantizedTensor.buffer;

  debugTrace.loader(`Dequantized ${name}: resultSize=${dequantized.size}`);
  releaseBuffer(quantBuffer);

  const layout = getWeightLayout(location, config);
  
  const dtype = outputDtype;

  return {
    data: createWeightBuffer(dequantized, dtype, layout, location.shape, name),
    allocatedBuffers: [dequantized],
  };
}


export async function loadQ6K(shardData, location, name) {
  const device = getDevice();

  debugTrace.loader(`Loading Q6_K tensor "${name}", size=${location.size}`);
  const quantBuffer = acquireBuffer(location.size, undefined, `quant_${name}`);
  device.queue.writeBuffer(quantBuffer, 0,  ( (shardData)));

  const numBlocks = Math.floor(location.size / Q6K_BLOCK_BYTES);
  debugTrace.loader(
    `Dequantizing Q6_K ${name}: size=${location.size}, numBlocks=${numBlocks}, ` +
    `expectedOutput=${numBlocks * 256 * 2} (f16)`
  );

  const dequantizedTensor = await dequantizeQ6K(quantBuffer, numBlocks, { outputDtype: 'f16' });
  const dequantized = dequantizedTensor.buffer;

  debugTrace.loader(`Dequantized Q6_K ${name}: resultSize=${dequantized.size}`);
  releaseBuffer(quantBuffer);

  const isMatmulWeight = shouldDequantizeToF16(location);
  if (isMatmulWeight) {
    return {
      data: createWeightBuffer(dequantized, 'f16', 'row', location.shape, name),
      allocatedBuffers: [dequantized],
    };
  }

  return {
    data: dequantized,
    allocatedBuffers: [dequantized],
  };
}


export async function loadBF16(shardData, location, name, config) {
  const device = getDevice();
  const srcBuffer = acquireBuffer(shardData.byteLength, undefined, `${name}_bf16`);
  device.queue.writeBuffer(srcBuffer, 0,  ( (shardData)));

  const numElements = shardData.byteLength / 2;
  const caps = config.gpuCapabilities || getKernelCapabilities();
  const isMatmulWeight = shouldDequantizeToF16(location);

  // For matmul weights with F16 support: BF16 → F16 directly
  if (caps?.hasF16 && isMatmulWeight) {
    const f16Tensor = await runBF16ToF16(srcBuffer, [numElements], name);
    releaseBuffer(srcBuffer);
    debugTrace.loader(`BF16→F16 for matmul weight: ${name} (${numElements} elements)`);

    
    const layout = selectRuleValue('loader', 'weights', 'weightLayout', {
      layout: location.layout ?? null,
      useColumnWise: false,
    });
    return {
      data: createWeightBuffer(f16Tensor.buffer, 'f16', layout, location.shape, name),
      allocatedBuffers: [f16Tensor.buffer],
    };
  }

  // Standard path: BF16 → F32
  const dstBuffer = await convertBF16ToF32GPU(srcBuffer, numElements, name);
  releaseBuffer(srcBuffer);

  if (dstBuffer instanceof GPUBuffer) {
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
      data: applyBufferLayout(dstBuffer, location),
      allocatedBuffers: [dstBuffer],
    };
  }

  // Float32Array returned (shouldn't happen in GPU path)
  return {
    data: dstBuffer,
    allocatedBuffers: [],
  };
}


export async function loadFloat(shardData, location, name, config) {
  if (!config) {
    throw new Error('Tensor load config is required.');
  }
  const device = getDevice();
  const buffer = acquireBuffer(location.size, undefined, name);
  device.queue.writeBuffer(buffer, 0,  ( (shardData)));

  const dtype = selectRuleValue('loader', 'weights', 'floatLocationDtype', {
    locationDtype: location.dtype,
  });
  const layout = selectRuleValue('loader', 'weights', 'weightLayout', {
    layout: location.layout ?? null,
    useColumnWise: false,
  });
  const isMatmulWeight = shouldDequantizeToF16(location);

  // Return WeightBuffer for matmul weights
  if (isMatmulWeight) {
    return {
      data: createWeightBuffer(buffer, dtype, layout, location.shape, name),
      allocatedBuffers: [buffer],
    };
  }

  // Non-matmul F16 weights need upcast to F32
  if (dtype === 'f16') {
    if (config.allowF32UpcastNonMatmul === false) {
      return {
        data: applyBufferLayout(buffer, location),
        allocatedBuffers: [buffer],
      };
    }
    const numElements = location.shape.reduce((a, b) => a * b, 1);
    logF32UpcastNonMatmul(name, numElements, buffer.size);
    debugTrace.loader(`F16→F32 upcast for non-matmul: ${name} (${numElements} elements, bufSize=${buffer.size})`);
    const inputTensor = createTensor(buffer, 'f16', [numElements], `${name}_f16`);
    const f32Tensor = await castF16ToF32(inputTensor);
    debugTrace.loader(`F16→F32 complete: ${name} resultSize=${f32Tensor.buffer.size}`);
    releaseBuffer(buffer);
    return {
      data: applyBufferLayout(f32Tensor.buffer, location),
      allocatedBuffers: [f32Tensor.buffer],
    };
  }

  return {
    data: applyBufferLayout(buffer, location),
    allocatedBuffers: [buffer],
  };
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
