import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';

const target = new URL('../../src/gpu/kernels/gather.js', import.meta.url).href;
globalThis.gatherDispatchProbe = { calls: [], released: [], allocated: [], limit: 2 };
const hooks = registerHooks({ resolve(specifier, context, next) {
  if (context.parentURL !== target) return next(specifier, context);
  const sources = {
    '../device.js': `export const getDevice = () => ({ limits: { maxComputeWorkgroupsPerDimension: globalThis.gatherDispatchProbe.limit } });
      export const getKernelCapabilities = () => ({ hasF16: true });`,
    '../../memory/buffer-pool.js': `export const acquireBuffer = size => { const b = { size }; globalThis.gatherDispatchProbe.allocated.push(b); return b; };
      export const releaseBuffer = b => globalThis.gatherDispatchProbe.released.push(b);`,
    './kernel-execution.js': `export const unifiedKernelWrapper = async (...args) => globalThis.gatherDispatchProbe.calls.push(args);`,
    '../tensor.js': 'export const createTensor = (buffer, dtype, shape) => ({ buffer, dtype, shape });',
  };
  if (sources[specifier]) return { url: 'data:text/javascript,' + encodeURIComponent(sources[specifier]), shortCircuit: true };
  return next(specifier, context);
} });
try {
  const { runGather, recordGather } = await import(target);
  const p = globalThis.gatherDispatchProbe, options = { embeddingDtype: 'f32', outputDtype: 'f32' };
  for (const count of [256, 512, 768, 1024]) {
    await runGather({}, {}, 1, count, 8, options);
    await recordGather({}, {}, {}, 1, count, 8, options);
    const [a, b] = p.calls.slice(-2);
    assert.deepEqual(a[5], b[5], 'Immediate and recorded geometry agree');
    assert.deepEqual(a[5], [Math.min(count / 256, 2), Math.ceil(count / 512), 1]);
  }
  await assert.rejects(runGather({}, {}, 1, 1280, 8, options), /two-dimensional/);
  assert.equal(p.released.at(-1), p.allocated.at(-1), 'Rejected geometry releases its owned output');
  p.limit = undefined;
  await assert.rejects(runGather({}, {}, 1, 256, 8, options), /explicit device/);
} finally { hooks.deregister(); delete globalThis.gatherDispatchProbe; }
console.log('gather-dispatch: passed');
