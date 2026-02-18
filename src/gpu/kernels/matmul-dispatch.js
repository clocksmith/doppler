import { KernelBase } from './kernel-base.js';
import { GPU_LIMITS, TILE_SIZES } from './constants.js';
import { createUniformBufferWithView, getOrCreateBindGroupLayout, getCachedPipeline, createPipeline } from './utils.js';


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


export function calculateMatmulDispatch(variant, useQ4KFused, useGemv, M, N, config) {
  const maxWorkgroups = GPU_LIMITS.MAX_WORKGROUPS;
  const [wgX, wgY] = config.workgroupSize;
  let workgroupsX = 1;
  let workgroupsY = 1;
  
  let uniformWorkgroupsX;

  // Get colsPerWg from variantMetadata (required for multicol GEMV)
  const colsPerWg = config.variantMetadata?.colsPerWg;
  // Get tileM from variantMetadata (required for batched variants)
  const tileM = config.variantMetadata?.tileM;

  if (useQ4KFused && variant.includes('multicol') && colsPerWg == null) {
    throw new Error(`Matmul kernel "${variant}" is missing variantMetadata.colsPerWg.`);
  }
  if (useQ4KFused && variant.includes('batched') && tileM == null) {
    throw new Error(`Matmul kernel "${variant}" is missing variantMetadata.tileM.`);
  }

  if (useGemv && variant.startsWith('gemv_subgroup')) {
    if (colsPerWg == null) {
      throw new Error(`Matmul kernel "${variant}" is missing variantMetadata.colsPerWg.`);
    }
    const gemvWorkgroupsX = Math.ceil(N / colsPerWg);
    if (gemvWorkgroupsX > maxWorkgroups) {
      workgroupsX = maxWorkgroups;
      workgroupsY = Math.ceil(gemvWorkgroupsX / maxWorkgroups);
    } else {
      workgroupsX = gemvWorkgroupsX;
      workgroupsY = 1;
    }
    uniformWorkgroupsX = workgroupsX;
    return { workgroups: [workgroupsX, workgroupsY, 1], uniformWorkgroupsX };
  }

  if (useQ4KFused) {
    if (variant === 'q4_fused') {
      workgroupsX = N;
      workgroupsY = 1;
    } else if (config.variantMetadata?.colsPerWg) {
      // Multicol variants: q4_fused_multicol, q4_fused_multicol_f16
      workgroupsX = Math.ceil(N / colsPerWg);
      workgroupsY = 1;
    } else if (config.variantMetadata?.tileM) {
      // Batched variants: q4_fused_batched, q4_fused_batched_f16
      workgroupsX = N;
      workgroupsY = Math.ceil(M / tileM);
    } else {
      // Fallback for q4_fused (1 col per workgroup)
      workgroupsX = N;
      workgroupsY = 1;
    }
  } else if (useGemv) {
    workgroupsX = N;
    workgroupsY = 1;
  } else if (variant === 'f16_tiled') {
    const tileM = config.variantMetadata?.tileM ?? 64;
    const tileN = config.variantMetadata?.tileN ?? 64;
    workgroupsX = Math.ceil(M / tileM);
    workgroupsY = Math.ceil(N / tileN);
  } else {
    const colsPerThread = variant === 'f16_vec4' ? 4 : 1;
    workgroupsX = Math.ceil(M / wgX);
    workgroupsY = Math.ceil(N / (wgY * colsPerThread));
  }

  return { workgroups: [workgroupsX, workgroupsY, 1], uniformWorkgroupsX };
}


export function createMatmulUniformBuffer(label, M, N, K, alpha, useQ4KFused, transposeB, uniformWorkgroupsX, recorder, device) {
  // Shader struct is 32 bytes: M, N, K, alpha, transpose_b/num_blocks, workgroups_x/_pad0, _pad1, _pad2
  const uniformSize = 32;

  return createUniformBufferWithView(
    label,
    uniformSize,
    (view) => {
      view.setUint32(0, M, true);
      view.setUint32(4, N, true);
      view.setUint32(8, K, true);
      view.setFloat32(12, alpha, true);
      if (useQ4KFused) {
        const numBlocksPerRow = Math.ceil(K / TILE_SIZES.Q4K_SUPER_BLOCK_SIZE);
        view.setUint32(16, numBlocksPerRow, true);
      } else {
        view.setUint32(16, transposeB ? 1 : 0, true);
      }
      // workgroups_x (or _pad0 if not needed)
      view.setUint32(20, uniformWorkgroupsX ?? 0, true);
      // _pad1, _pad2 - leave as zeros (already zero-initialized)
    },
    recorder,
    device
  );
}


export function createMatmulBindGroupLayout() {
  return getOrCreateBindGroupLayout('matmul_bind_group_layout', [
    {
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: 'uniform' },
    },
    {
      binding: 1,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: 'read-only-storage' },
    },
    {
      binding: 2,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: 'read-only-storage' },
    },
    {
      binding: 3,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: 'storage' },
    },
  ]);
}


export async function getMatmulPipeline(variant, constants) {
  let pipeline = getCachedPipeline('matmul', variant, constants);
  if (!pipeline) {
    pipeline = await createPipeline('matmul', variant, null, constants);
  }
  return pipeline;
}
