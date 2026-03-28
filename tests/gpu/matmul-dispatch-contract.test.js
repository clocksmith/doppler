import assert from 'node:assert/strict';

import {
  calculateMatmulDispatch,
} from '../../src/gpu/kernels/matmul-dispatch.js';

import {
  validateMatmulDimensions,
  requiresF32Input,
} from '../../src/gpu/kernels/matmul-selection.js';

import { TILE_SIZES } from '../../src/gpu/kernels/constants.js';

// === validateMatmulDimensions: positive dimensions ===
{
  assert.doesNotThrow(() => validateMatmulDimensions('test', 1, 2048, 1024));
  assert.doesNotThrow(() => validateMatmulDimensions('test', 128, 4096, 512));
}

// === validateMatmulDimensions: rejects non-positive ===
{
  assert.throws(
    () => validateMatmulDimensions('test', 0, 2048, 1024),
    /Dimensions must be positive/
  );
  assert.throws(
    () => validateMatmulDimensions('test', 1, -1, 1024),
    /Dimensions must be positive/
  );
  assert.throws(
    () => validateMatmulDimensions('test', 1, 2048, 0),
    /Dimensions must be positive/
  );
}

// === validateMatmulDimensions: rejects non-finite ===
{
  assert.throws(
    () => validateMatmulDimensions('test', NaN, 2048, 1024),
    /Invalid dimensions/
  );
  assert.throws(
    () => validateMatmulDimensions('test', 1, Infinity, 1024),
    /Invalid dimensions/
  );
}

// === requiresF32Input: f16 variants accept f16 ===
{
  assert.equal(requiresF32Input('f16'), false);
  assert.equal(requiresF32Input('f16_vec4'), false);
  assert.equal(requiresF32Input('f16_tiled'), false);
  assert.equal(requiresF32Input('gemv_f16a'), false);
  assert.equal(requiresF32Input('q4_fused_f16a'), false);
  assert.equal(requiresF32Input('q4_fused_multicol_f16a'), false);
}

// === requiresF32Input: f32 variants require f32 ===
{
  assert.equal(requiresF32Input('f32'), true);
  assert.equal(requiresF32Input('gemv'), true);
  assert.equal(requiresF32Input('q4_fused'), true);
  assert.equal(requiresF32Input('gemv_subgroup'), true);
}

// === calculateMatmulDispatch: GEMV variant (M=1 decode) ===
// GEMV dispatches one workgroup per output column.
{
  const M = 1;
  const N = 2048;
  const config = {
    workgroupSize: [256, 1, 1],
    variantMetadata: null,
  };

  const result = calculateMatmulDispatch('gemv', false, true, M, N, config);
  assert.deepEqual(result.workgroups, [N, 1, 1]);
}

// === calculateMatmulDispatch: GEMV subgroup variant requires colsPerWg ===
{
  const config = {
    workgroupSize: [256, 1, 1],
    variantMetadata: null,
  };

  assert.throws(
    () => calculateMatmulDispatch('gemv_subgroup', false, true, 1, 2048, config),
    /missing variantMetadata.colsPerWg/
  );
}

// === calculateMatmulDispatch: GEMV subgroup with colsPerWg ===
{
  const M = 1;
  const N = 4096;
  const colsPerWg = 4;
  const config = {
    workgroupSize: [256, 1, 1],
    variantMetadata: { colsPerWg },
  };

  const result = calculateMatmulDispatch('gemv_subgroup', false, true, M, N, config);
  const expectedX = Math.ceil(N / colsPerWg);
  assert.deepEqual(result.workgroups, [expectedX, 1, 1]);
  assert.equal(result.uniformWorkgroupsX, expectedX);
}

// === calculateMatmulDispatch: GEMV subgroup f16a variant ===
{
  const M = 1;
  const N = 2048;
  const colsPerWg = 4;
  const config = {
    workgroupSize: [256, 1, 1],
    variantMetadata: { colsPerWg },
  };

  const result = calculateMatmulDispatch('gemv_subgroup_f16a', false, true, M, N, config);
  const expectedX = Math.ceil(N / colsPerWg);
  assert.deepEqual(result.workgroups, [expectedX, 1, 1]);
}

// === calculateMatmulDispatch: q4_fused variant (1 col per workgroup) ===
{
  const M = 1;
  const N = 1024;
  const config = {
    workgroupSize: [256, 1, 1],
    variantMetadata: null,
  };

  const result = calculateMatmulDispatch('q4_fused', true, false, M, N, config);
  assert.deepEqual(result.workgroups, [N, 1, 1]);
}

// === calculateMatmulDispatch: q4_fused_multicol requires colsPerWg ===
{
  const config = {
    workgroupSize: [256, 1, 1],
    variantMetadata: null,
  };

  assert.throws(
    () => calculateMatmulDispatch('q4_fused_multicol', true, false, 1, 2048, config),
    /missing variantMetadata.colsPerWg/
  );
}

// === calculateMatmulDispatch: q4_fused_multicol with colsPerWg ===
{
  const M = 1;
  const N = 4096;
  const colsPerWg = 8;
  const config = {
    workgroupSize: [256, 1, 1],
    variantMetadata: { colsPerWg },
  };

  const result = calculateMatmulDispatch('q4_fused_multicol', true, false, M, N, config);
  assert.deepEqual(result.workgroups, [Math.ceil(N / colsPerWg), 1, 1]);
}

// === calculateMatmulDispatch: q4_fused_multicol_f16 with colsPerWg ===
{
  const M = 1;
  const N = 2048;
  const colsPerWg = 4;
  const config = {
    workgroupSize: [256, 1, 1],
    variantMetadata: { colsPerWg },
  };

  const result = calculateMatmulDispatch('q4_fused_multicol_f16', true, false, M, N, config);
  assert.deepEqual(result.workgroups, [Math.ceil(N / colsPerWg), 1, 1]);
}

// === calculateMatmulDispatch: q4_fused_multicol_f16a with colsPerWg ===
{
  const M = 1;
  const N = 2048;
  const colsPerWg = 4;
  const config = {
    workgroupSize: [256, 1, 1],
    variantMetadata: { colsPerWg },
  };

  const result = calculateMatmulDispatch('q4_fused_multicol_f16a', true, false, M, N, config);
  assert.deepEqual(result.workgroups, [Math.ceil(N / colsPerWg), 1, 1]);
}

// === calculateMatmulDispatch: q4_fused_batched requires tileM ===
{
  const config = {
    workgroupSize: [256, 1, 1],
    variantMetadata: null,
  };

  assert.throws(
    () => calculateMatmulDispatch('q4_fused_batched', true, false, 8, 2048, config),
    /missing variantMetadata.tileM/
  );
}

// === calculateMatmulDispatch: q4_fused_batched with tileM ===
{
  const M = 16;
  const N = 2048;
  const tileM = 4;
  const config = {
    workgroupSize: [256, 1, 1],
    variantMetadata: { tileM },
  };

  const result = calculateMatmulDispatch('q4_fused_batched', true, false, M, N, config);
  assert.deepEqual(result.workgroups, [N, Math.ceil(M / tileM), 1]);
}

// === calculateMatmulDispatch: q4_fused_batched_f16a with tileM ===
{
  const M = 8;
  const N = 1024;
  const tileM = 4;
  const config = {
    workgroupSize: [256, 1, 1],
    variantMetadata: { tileM },
  };

  const result = calculateMatmulDispatch('q4_fused_batched_f16a', true, false, M, N, config);
  assert.deepEqual(result.workgroups, [N, Math.ceil(M / tileM), 1]);
}

// === calculateMatmulDispatch: q4_fused_batched_multicol_shared uses both tiles ===
{
  const M = 182;
  const N = 6912;
  const tileM = 4;
  const colsPerWg = 8;
  const config = {
    workgroupSize: [32, 4, 1],
    variantMetadata: { tileM, colsPerWg },
  };

  const result = calculateMatmulDispatch('q4_fused_batched_multicol_shared', true, false, M, N, config);
  assert.deepEqual(result.workgroups, [Math.ceil(N / colsPerWg), Math.ceil(M / tileM), 1]);
}

// === calculateMatmulDispatch: f16_tiled prefill variant ===
{
  const M = 128;
  const N = 4096;
  const tileM = 64;
  const tileN = 64;
  const config = {
    workgroupSize: [16, 16, 1],
    variantMetadata: { tileM, tileN },
  };

  const result = calculateMatmulDispatch('f16_tiled', false, false, M, N, config);
  assert.deepEqual(result.workgroups, [Math.ceil(M / tileM), Math.ceil(N / tileN), 1]);
}

// === calculateMatmulDispatch: f16w_f32a_tiled prefill variant ===
{
  const M = 64;
  const N = 2048;
  const tileM = 64;
  const tileN = 64;
  const config = {
    workgroupSize: [16, 16, 1],
    variantMetadata: { tileM, tileN },
  };

  const result = calculateMatmulDispatch('f16w_f32a_tiled', false, false, M, N, config);
  assert.deepEqual(result.workgroups, [Math.ceil(M / tileM), Math.ceil(N / tileN), 1]);
}

// === calculateMatmulDispatch: standard f32 matmul ===
{
  const M = 4;
  const N = 2048;
  const wgX = 16;
  const wgY = 16;
  const config = {
    workgroupSize: [wgX, wgY, 1],
    variantMetadata: null,
  };

  const result = calculateMatmulDispatch('f32', false, false, M, N, config);
  assert.deepEqual(result.workgroups, [Math.ceil(M / wgX), Math.ceil(N / wgY), 1]);
}

// === calculateMatmulDispatch: f16 matmul ===
{
  const M = 1;
  const N = 4096;
  const wgX = 16;
  const wgY = 16;
  const config = {
    workgroupSize: [wgX, wgY, 1],
    variantMetadata: null,
  };

  const result = calculateMatmulDispatch('f16', false, false, M, N, config);
  assert.deepEqual(result.workgroups, [Math.ceil(M / wgX), Math.ceil(N / wgY), 1]);
}

// === Output shape contracts: M=1 decode produces [1, N] tensor ===
// This verifies the semantic that GEMV (M=1) always produces a [1, N] shape output.
// The dispatch workgroups for GEMV must cover all N columns.
{
  const M = 1;
  const N = 2048;

  // GEMV variant
  const gemvConfig = {
    workgroupSize: [256, 1, 1],
    variantMetadata: null,
  };
  const gemvResult = calculateMatmulDispatch('gemv', false, true, M, N, gemvConfig);
  assert.equal(gemvResult.workgroups[0], N, 'GEMV dispatch X must cover all N columns');
  assert.equal(gemvResult.workgroups[1], 1);
  assert.equal(gemvResult.workgroups[2], 1);
}

// === Output shape contracts: prefill M>1 produces [M, N] tensor ===
// Tiled matmul must dispatch enough workgroups to cover the M x N output.
{
  const M = 64;
  const N = 4096;
  const tileM = 64;
  const tileN = 64;
  const config = {
    workgroupSize: [16, 16, 1],
    variantMetadata: { tileM, tileN },
  };

  const result = calculateMatmulDispatch('f16_tiled', false, false, M, N, config);
  // Every tile must be covered
  assert.ok(result.workgroups[0] * tileM >= M, 'tiled dispatch must cover all M rows');
  assert.ok(result.workgroups[1] * tileN >= N, 'tiled dispatch must cover all N columns');
}

// === GEMV subgroup dispatch: large N wraps to 2D ===
{
  const M = 1;
  const N = 65535 * 4 + 100;
  const colsPerWg = 4;
  const config = {
    workgroupSize: [256, 1, 1],
    variantMetadata: { colsPerWg },
  };

  const result = calculateMatmulDispatch('gemv_subgroup', false, true, M, N, config);
  const totalWg = Math.ceil(N / colsPerWg);
  assert.ok(result.workgroups[0] <= 65535, 'X must not exceed MAX_WORKGROUPS');
  assert.ok(
    result.workgroups[0] * result.workgroups[1] >= totalWg,
    '2D dispatch must cover all workgroups'
  );
}

// === Activation dtype routing: f32a vs f16a variant naming ===
// Variants ending in _f16a support f16 activations; all others require f32.
{
  // f16a variants
  const f16aVariants = [
    'gemv_f16a',
    'gemv_subgroup_f16a',
    'q4_fused_f16a',
    'q4_fused_multicol_f16a',
    'q4_fused_batched_f16a',
  ];
  for (const v of f16aVariants) {
    assert.equal(requiresF32Input(v), false, `${v} should accept f16 input`);
  }

  // f32a variants (no _f16a suffix)
  const f32aVariants = [
    'gemv',
    'gemv_subgroup',
    'q4_fused',
    'q4_fused_multicol',
    'q4_fused_batched',
    'q4_fused_batched_multicol_shared',
    'f32',
  ];
  for (const v of f32aVariants) {
    assert.equal(requiresF32Input(v), true, `${v} should require f32 input`);
  }
}

// === Q4K block size constant consistency ===
{
  assert.equal(TILE_SIZES.Q4K_SUPER_BLOCK_SIZE, 256);
}

console.log('matmul-dispatch-contract.test: ok');
