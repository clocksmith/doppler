import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { setDevice } from '../../src/gpu/device.js';
import { PreparedDecodeWorkspace } from '../../src/gpu/prepared-decode-workspace.js';

// Replace allocation/submission ports; run the actual immediate and recorded wrappers.
const target = new URL('../../src/gpu/kernels/rmsnorm.js', import.meta.url).href;
const live = new Set(), dispatches = [];
globalThis.rmsnormContract = {
  acquire(size, label) {
    if (this.fail === 'placeholder' && label === 'rmsnorm_prenorm_placeholder') throw Error('placeholder allocation rejected');
    const buffer = { size }; live.add(buffer); return buffer;
  },
  release(buffer) { assert(live.delete(buffer)); },
  dispatch(...args) { if (this.fail === 'dispatch') throw Error('dispatch rejected'); dispatches.push(args); },
};
const modules = {
  '../../memory/buffer-pool.js': `export const acquireBuffer = (size, usage, label) => rmsnormContract.acquire(size, label);
    export const releaseBuffer = buffer => rmsnormContract.release(buffer);
    export const getBufferRequestedSize = buffer => buffer.size;`,
  './kernel-execution.js': 'export const unifiedKernelWrapper = async (...args) => rmsnormContract.dispatch(...args);',
};
const hooks = registerHooks({ resolve(specifier, context, next) {
  if (context.parentURL === target && modules[specifier]) return {
    url: `data:text/javascript,${encodeURIComponent(modules[specifier])}`, shortCircuit: true,
  };
  return next(specifier, context);
} });
const { selectRMSNormKernel, runRMSNorm, recordRMSNorm } = await import(target);

const step = { op: 'input_norm', kernel: 'rmsnorm.wgsl', entry: 'main' };
const path = steps => ({ activationDtype: 'f32', decode: { steps }, prefill: { steps } });
const options = { hiddenSize: 1024, phase: 'decode', layerIdx: 0,
  role: 'input_norm', section: 'layer', kernelPath: path([step]) };
setDevice({ features: new Set(['subgroups', 'shader-f16']), limits: {},
  createBindGroup() {}, createBuffer() {} }, { platformConfig: null });
try {
  assert.equal(selectRMSNormKernel(options, false), 'default',
    'An explicit main entry must not become a subgroup optimization');
  assert.equal(selectRMSNormKernel({ ...options, residual: {} }, false), 'default',
    'A residual binding must not replace an explicit entry with main_cached');
  assert.equal(selectRMSNormKernel({ ...options, kernelPath: path([{ ...step, entry: 'main_subgroup' }]) }, false), 'subgroup');
  assert.throws(() => selectRMSNormKernel({ ...options, kernelPath: path([]) }, false), /exactly one/);
  assert.throws(() => selectRMSNormKernel({ ...options, kernelPath: path([step, step]) }, false), /exactly one/);
  assert.throws(() => selectRMSNormKernel({ ...options, kernelPath: path([{ ...step, entry: 'missing' }]) }, false), /exact registered/);
  assert.throws(() => selectRMSNormKernel({ ...options, phase: undefined }, false), /explicit/);
  assert.throws(() => selectRMSNormKernel(options, true), /dtype/);
  for (const recorded of [false, true]) {
    const temporaries = new Set();
    const recorder = { device: { limits: { maxComputeWorkgroupsPerDimension: 65535 } },
      trackTemporaryBuffer: buffer => temporaries.add(buffer),
      abort() { for (const buffer of temporaries) rmsnormContract.release(buffer); temporaries.clear(); } };
    const input = { dtype: 'f32', shape: [1, 1024], buffer: { size: 4096 } };
    const weight = { size: 4096 };
    const run = config => recorded ? recordRMSNorm(recorder, input, weight, 1e-6, config)
      : runRMSNorm(input, weight, 1e-6, config);
    const result = await run({ ...options, kernelPath: path([{ ...step, constants: { WORKGROUP_SIZE: 128 } }]) });
    const dispatch = dispatches.at(-1);
    assert.equal(dispatch[1], recorded ? recorder : null);
    assert.equal(dispatch[2], 'default');
    assert.equal(dispatch[6].WORKGROUP_SIZE, 128);
    assert.equal(live.size, recorded ? 2 : 1, 'Recorded placeholder remains owned until recorder settlement');
    if (recorded) {
      assert(temporaries.has(dispatch[7][0].buffer), 'The recorder owns the extra binding');
      recorder.abort();
    }
    assert.equal(live.size, 1, 'Only the returned output remains after recorder settlement');
    globalThis.rmsnormContract.release(result.buffer);
    const count = dispatches.length;
    await assert.rejects(run({ ...options, kernelPath: path([{ ...step, constants: { RMS_NORM_OFFSET: true } }]) }), /RMS_NORM_OFFSET/);
    assert.equal(dispatches.length, count, 'Conflicting constants fail before dispatch');
    assert.equal(live.size, 0, 'Contract rejection allocates nothing');
    for (const failure of ['placeholder', 'dispatch']) {
      globalThis.rmsnormContract.fail = failure;
      await assert.rejects(run(options), /rejected/);
      recorder.abort();
      assert.equal(live.size, 0, `${failure} rejection releases every acquired buffer`);
      globalThis.rmsnormContract.fail = null;
    }
  }
  globalThis.GPUBufferUsage = { STORAGE: 128, COPY_SRC: 4, COPY_DST: 8 };
  const preparedBuffers = [];
  const workspace = new PreparedDecodeWorkspace({ createBuffer({ size }) {
    const buffer = { size, destroyed: false, destroy() { this.destroyed = true; } };
    preparedBuffers.push(buffer);
    return buffer;
  } });
  const recorder = { device: { limits: { maxComputeWorkgroupsPerDimension: 65535 } },
    decodeWorkspace: workspace,
    trackTemporaryBuffer() { throw Error('Prepared storage cannot become recorder-owned'); } };
  const input = { dtype: 'f32', shape: [1, 1024], buffer: { size: 4096 } };
  const weight = { size: 4096 };
  try {
    const first = await recordRMSNorm(recorder, input, weight, 1e-6, options);
    const placeholder = dispatches.at(-1)[7][0].buffer;
    assert.equal(dispatches.at(-1)[6].OUTPUT_PRENORM, false);
    const repeat = await recordRMSNorm(recorder, input, weight, 1e-6, options);
    assert.equal(first.buffer, repeat.buffer, 'Decode steps reuse the same named output');
    assert.equal(dispatches.at(-1)[7][0].buffer, placeholder, 'Disabled prenorm binding is session-owned');
    const other = await recordRMSNorm(recorder, input, weight, 1e-6, { ...options, layerIdx: 1 });
    assert.notEqual(first.buffer, other.buffer, 'Live layers cannot alias normalization outputs');
    const borrowed = { size: 4096 };
    const supplied = await recordRMSNorm(recorder, input, weight, 1e-6, { ...options, outputBuffer: borrowed });
    assert.equal(supplied.buffer, borrowed, 'Caller output retains precedence');
    globalThis.rmsnormContract.fail = 'dispatch';
    await assert.rejects(recordRMSNorm(recorder, input, weight, 1e-6, options), /rejected/);
    globalThis.rmsnormContract.fail = null;
    assert(preparedBuffers.every(buffer => !buffer.destroyed), 'Failed recording preserves session ownership');
    assert.equal(live.size, 0, 'Prepared normalization does not acquire transient pool storage');
    await recordRMSNorm(recorder, input, weight, 1e-6, options);
  } finally {
    globalThis.rmsnormContract.fail = null;
    workspace.release();
    assert(preparedBuffers.every(buffer => buffer.destroyed), 'Session release owns final destruction');
  }
} finally {
  setDevice(null, { platformConfig: null }); hooks.deregister(); delete globalThis.rmsnormContract;
}
console.log('rmsnorm-execution-contract: declared entries and invalid contracts passed');
