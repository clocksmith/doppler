import test from 'node:test';
import assert from 'node:assert/strict';
import { runRouteQ4Matmul, recordRouteQ4Matmul, runRouteScatter } from '../../src/gpu/kernels/moe-route.js';
import { runQwenRouteExperts } from '../../src/inference/pipelines/text/moe/qwen-route-executor.js';
import { createKernelRegistry, enterKernelRegistry, getKernelConfig } from '../../src/gpu/kernels/kernel-configs.js';

const buffer = size => ({ size, __dopplerFakeGPUBuffer: true });
const input = { buffer: buffer(2 * 257 * 4), dtype: 'f32', shape: [2, 257] };
const indices = buffer(4 * 4);
const weight = { buffer: buffer(3 * 7 * 2 * 144), dtype: 'q4k', layout: 'row', shape: [3, 7, 257] };
const options = { numRoutes: 4, numExperts: 3, topK: 2, N: 7, K: 257, inputMode: 'token', alpha: 1 };

test('route projection rejects invalid semantics before GPU preparation in both adapters', async () => {
  const changes = [
    [{ numRoutes: 3 }, /route\/topK/], [{ topK: 4 }, /route\/topK/],
    [{ N: 0 }, /positive u32/], [{ numExperts: 2 ** 32 }, /positive u32/],
    [{ inputMode: undefined }, /inputMode/], [{ alpha: NaN }, /alpha/],
    [{ alpha: Number.MAX_VALUE }, /alpha/], [{ K: 256 }, /geometry/],
  ];
  for (const [change, message] of changes) {
    await assert.rejects(runRouteQ4Matmul(input, indices, weight, { ...options, ...change }), message);
    await assert.rejects(recordRouteQ4Matmul({}, input, indices, weight, { ...options, ...change }), message);
  }
  await assert.rejects(runRouteQ4Matmul({ ...input, dtype: 'f16' }, indices, weight, options), /scalar F32/);
  await assert.rejects(runRouteQ4Matmul({ ...input, precision: 'two-f32' }, indices, weight, options), /scalar F32/);
});

test('packed bank metadata, extents, route count and borrowed output aliasing fail before allocation', async () => {
  for (const changed of [{ dtype: 'f32' }, { layout: 'column' }, { shape: [7, 257] }, { shape: [4, 7, 257] }]) {
    await assert.rejects(runRouteQ4Matmul(input, indices, { ...weight, ...changed }, options), /row-packed/);
  }
  await assert.rejects(runRouteQ4Matmul(input, indices, { ...weight, buffer: buffer(weight.buffer.size - 4) }, options), /weight bank/);
  await assert.rejects(runRouteQ4Matmul(input, buffer(12), weight, options), /routing indices/);
  await assert.rejects(runRouteQ4Matmul(input, indices, weight, { ...options, outputBuffer: input.buffer }), /alias/);
  await assert.rejects(runRouteQ4Matmul(input, indices, weight, { ...options, outputBuffer: buffer(4) }), /output/);
});

test('route combine requires matching scalar F32 geometry and route-weight extent', async () => {
  const routes = { buffer: buffer(4 * 7 * 4), dtype: 'f32', shape: [4, 7] };
  const config = { numTokens: 2, hiddenSize: 7, topK: 2 };
  await assert.rejects(runRouteScatter(routes, buffer(12), config), /routing weights/);
  await assert.rejects(runRouteScatter(routes, buffer(16), { ...config, numTokens: 3 }), /geometry/);
  await assert.rejects(runRouteScatter(routes, buffer(16), { ...config, outputBuffer: routes.buffer }), /alias/);
});

test('explicit full-block specialization rejects tails instead of dropping the final block', async () => {
  const original = getKernelConfig('moe_route_matmul', 'q4k_f32');
  const leave = enterKernelRegistry(createKernelRegistry({ extensions: { moe_route_matmul: {
    variants: { q4k_f32: { weightDtype: original.weightDtype, wgsl: original.shaderFile, entryPoint: original.entryPoint,
      wgslOverrides: { ...original.wgslOverrides, USE_FULL_BLOCK_FAST_PATH: true } } },
  } } }));
  try { await assert.rejects(runRouteQ4Matmul(input, indices, weight, options), /Full-block specialization/); }
  finally { leave(); }
});

test('Qwen route executor refuses unsupported arithmetic and foreign expert semantics', async () => {
  const config = { activationDtype: 'f32', hiddenActivation: 'silu', expectedExpertFormat: 'qwen3_5',
    weights: { expertFormat: 'qwen3_5' }, swigluLimit: null };
  for (const change of [{ activationDtype: 'f16' }, { hiddenActivation: 'gelu' },
    { expectedExpertFormat: 'gemma4' }, { swigluLimit: undefined }]) {
    await assert.rejects(runQwenRouteExperts({ ...config, ...change }), /Qwen topk-route/);
  }
});
