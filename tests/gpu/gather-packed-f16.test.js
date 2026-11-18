import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { registerHooks } from 'node:module';

const target = new URL('../../src/gpu/kernels/gather.js', import.meta.url).href;
globalThis.gatherDispatchProbe = { calls: [], released: [], allocated: [], limit: 2 };
const hooks = registerHooks({ resolve(specifier, context, next) {
  if (context.parentURL !== target) return next(specifier, context);
  const sources = {
    '../device.js': `export const getDevice = () => ({ limits: { maxComputeWorkgroupsPerDimension: globalThis.gatherDispatchProbe.limit, maxStorageBuffersPerShaderStage: 6, maxStorageBufferBindingSize: 128, maxBufferSize: 128 } });
      export const getKernelCapabilities = () => ({ hasF16: false });`,
    '../../memory/buffer-pool.js': `export const acquireBuffer = size => { const b = { size }; globalThis.gatherDispatchProbe.allocated.push(b); return b; };
      export const releaseBuffer = b => globalThis.gatherDispatchProbe.released.push(b);`,
    './kernel-execution.js': `export const unifiedKernelWrapper = async (...args) => globalThis.gatherDispatchProbe.calls.push(args);`,
    '../tensor.js': 'export const createTensor = (buffer, dtype, shape) => ({ buffer, dtype, shape });',
  };
  if (sources[specifier]) return { url: 'data:text/javascript,' + encodeURIComponent(sources[specifier]), shortCircuit: true };
  return next(specifier, context);
} });
try {
  const shader = await readFile(new URL('../../src/gpu/kernels/gather_split4_packed_f16_vec4_f32_out.wgsl', import.meta.url), 'utf8');
  assert.doesNotMatch(shader, /enable f16|array<f16>/);
  assert.match(shader, /unpack2x16float/);
  const { runGatherSplit, recordGatherSplit, resolveSplitGatherPipeline } = await import(target);
  const embedding = { dtype: 'f16', layout: 'row', metadata: { packing: 'packed-f16-u32', splitGatherSectionCount: 4 },
    sections: [{ buffer: { size: 64 }, rowStart: 0, rowCount: 2 }, { buffer: { size: 32 }, rowStart: 2, rowCount: 1 }] };
  const options = { embeddingDtype: 'f16', outputDtype: 'f32' };
  const pipeline = resolveSplitGatherPipeline(embedding, 'f32', null);
  assert.equal(pipeline.variant, 'packed_f16_vec4_f32_out');
  await runGatherSplit({}, embedding, 1, 16, 3, options);
  await recordGatherSplit({}, {}, embedding, 1, 16, 3, options);
  assert.ok(globalThis.gatherDispatchProbe.calls.every(args => args[2] === 'packed_f16_vec4_f32_out'));
  await assert.rejects(runGatherSplit({}, embedding, 1, 16, 3, { ...options, outputDtype: 'f16' }), /F32 output/);
  await assert.rejects(runGatherSplit({}, { ...embedding, metadata: {} }, 1, 16, 3, options), /shader-f16/);
  await assert.rejects(runGatherSplit({}, { ...embedding, sections: [{ buffer: { size: 2 }, rowStart: 0, rowCount: 3 }] }, 1, 16, 3, options), /declared rows/);
  await assert.rejects(runGatherSplit({}, embedding, 1, 16, 4, options), /logical|expose/);
} finally { hooks.deregister(); delete globalThis.gatherDispatchProbe; }
console.log('gather-packed-f16: passed');
