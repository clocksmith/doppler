import { getKernelConfig } from './kernel-configs.js';
import { getKernelBindGroupLayout } from './kernel-bindings.js';
import { KernelBase } from './kernel-base.js';
import { GPU_LIMITS, TILE_SIZES } from './constants.js';
import { createKernelUniformBuffer } from './uniform-utils.js';
import {
  getCachedPipeline,
  createPipeline,
} from './pipeline-cache.js';


export class MatmulKernel extends KernelBase {
  
  async getPipeline(variant) {
    return this.getPipelineFor('matmul', variant);
  }

  
  dispatch(pipeline, bindGroup, workgroups) {
    this.dispatchKernel(pipeline, bindGroup, workgroups, 'matmul');
  }

  
  record(recorder, pipeline, bindGroup, workgroups, label = 'matmul') {
    this.recordKernel(recorder, pipeline, bindGroup, workgroups, label);
  }
}


export function calculateMatmulDispatch(variant, useQ4KFused, useGemv, useLiteRTInt4Fused, M, N, config, useW4A16Fused = false) {
  if (typeof useLiteRTInt4Fused === 'number') {
    config = N;
    N = M;
    M = useLiteRTInt4Fused;
    useLiteRTInt4Fused = false;
  }
  const maxWorkgroups = GPU_LIMITS.MAX_WORKGROUPS;
  const geometry = config.variantMetadata?.dispatchGeometry;
  const metadata = config.variantMetadata;
  const required = (field) => {
    const value = metadata?.[field];
    if (!Number.isInteger(value) || value <= 0) {
      throw new Error(`Matmul kernel "${variant}" is missing positive variantMetadata.${field}.`);
    }
    return value;
  };
  let x, y, uniformWorkgroupsX;
  switch (geometry) {
    case 'column':
      x = N; y = 1;
      break;
    case 'column-block':
      x = Math.ceil(N / required('colsPerWg')); y = 1;
      break;
    case 'row-column-block':
      x = Math.ceil(N / required('colsPerWg')); y = M;
      uniformWorkgroupsX = 0; // Declared but unused by these shader entry points.
      break;
    case 'row-column-tile':
      x = Math.ceil(N / required('colsPerWg')); y = Math.ceil(M / required('tileM'));
      uniformWorkgroupsX = 0;
      break;
    case 'column-row-tile':
      x = N; y = Math.ceil(M / required('tileM'));
      break;
    case 'linear-column-block': {
      const count = Math.ceil(N / required('colsPerWg'));
      x = Math.min(count, maxWorkgroups); y = Math.ceil(count / maxWorkgroups);
      uniformWorkgroupsX = x;
      break;
    }
    case 'matrix-tile':
      x = Math.ceil(M / required('tileM')); y = Math.ceil(N / required('tileN'));
      break;
    case 'matrix-workgroup':
      x = Math.ceil(M / config.workgroupSize[0]);
      y = Math.ceil(N / (config.workgroupSize[1] * required('colsPerThread')));
      break;
    default:
      throw new Error(`Matmul kernel "${variant}" has unsupported dispatchGeometry "${String(geometry)}".`);
  }
  if (![x, y].every((value) => Number.isInteger(value) && value > 0 && value <= maxWorkgroups)) {
    throw new Error(`Matmul kernel "${variant}" dispatch exceeds WebGPU workgroup limits: ${x} x ${y}, max=${maxWorkgroups}.`);
  }
  return { workgroups: [x, y, 1], uniformWorkgroupsX };
}


export function createMatmulUniformBuffer(label, config, M, N, K, alpha, useQ4KFused, transposeB, uniformWorkgroupsX, recorder, device, extras = null) {
  return createKernelUniformBuffer(
    label,
    config,
    {
      M, N, K, alpha,
      transpose_b: transposeB ? 1 : 0,
      num_blocks_per_row: Math.ceil(K / TILE_SIZES.Q4K_SUPER_BLOCK_SIZE),
      workgroups_x: uniformWorkgroupsX,
      eps: extras?.eps,
    },
    recorder,
    device
  );
}


export function createMatmulBindGroupLayout() {
  // Compatibility helper for the registry's dense matrix binding layout.
  return getKernelBindGroupLayout(getKernelConfig('matmul', 'f32'));
}


export async function getMatmulPipeline(variant, constants) {
  let pipeline = getCachedPipeline('matmul', variant, constants);
  if (!pipeline) {
    pipeline = await createPipeline('matmul', variant, null, constants);
  }
  return pipeline;
}
