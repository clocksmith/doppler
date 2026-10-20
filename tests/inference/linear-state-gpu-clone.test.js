import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';

const target = new URL('../../src/inference/pipelines/text/linear-attention/state-clone.js', import.meta.url).href;
let live = new Set(), discarded = [], copies = [], submissions = 0, allocations = 0, failAt = -1, failure = null;
const device = {
  queue: {
    submit() { submissions++; if (failure === 'submit') throw Error('submit failed'); },
    async onSubmittedWorkDone() { if (failure === 'completion') throw Error('completion failed'); },
  },
  createCommandEncoder() {
    return { copyBufferToBuffer(src, from, dst, to, bytes) {
      if (failure === 'copy') throw Error('copy failed');
      dst.bytes.set(src.bytes.subarray(from, from + bytes), to); copies.push(bytes);
    }, finish() { return {}; } };
  },
};
globalThis.linearCloneControl = {
  device,
  acquire(size) {
    if (allocations++ === failAt) throw Error('allocation failed');
    const b = { size, bytes: new Uint8Array(size), __dopplerFakeGPUBuffer: true }; live.add(b); return b;
  },
  upload(b, values) {
    if (failure === 'upload') throw Error('upload failed');
    b.bytes.set(new Uint8Array(values.buffer, values.byteOffset, values.byteLength));
  },
  release(b) { assert(live.delete(b)); },
  discard(b) { assert(live.delete(b)); discarded.push(b); },
};
const modules = {
  '../../../../gpu/device.js': 'export const getDevice = () => linearCloneControl.device;',
  '../../../../memory/buffer-pool.js': `export const acquireBuffer = (...a) => linearCloneControl.acquire(...a);
    export const releaseBuffer = b => linearCloneControl.release(b);
    export const discardBuffer = b => linearCloneControl.discard(b);
    export const uploadData = (...a) => linearCloneControl.upload(...a);`,
};
const hooks = registerHooks({ resolve(specifier, context, next) {
  if (context.parentURL === target && modules[specifier]) return { url: 'data:text/javascript,' + encodeURIComponent(modules[specifier]), shortCircuit: true };
  return next(specifier, context);
} });
const fields = ['convWeight', 'dtBias', 'aLog', 'normWeight', 'convState', 'recurrentState'];
const bytesOf = values => new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
function source(components, resident) {
  const state = { layerIdx: 7, seqLen: 53, warnedSeqMismatch: true, convDim: 1, convKernelSize: 2,
    numVHeads: 1, headKDim: 1, headVDim: 2, normWeightOffset: false,
    convStatePrecision: components === 2 ? 'two-f32' : 'f32', convStateComponents: components,
    recurrentStatePrecision: components === 2 ? 'two-f32' : 'f32', recurrentStateComponents: components };
  for (const field of fields) {
    state[field] = new Float32Array(field.endsWith('State') ? 2 * components : 2).fill(-1);
    if (resident) {
      // GPU bytes deliberately differ from stale host mirrors, including paired low words.
      const data = new Float32Array(state[field].length).map((_, i) => i % 2 ? 2 ** -27 : i + 5);
      state[field + 'GPU'] = { size: data.byteLength + 64, bytes: new Uint8Array(data.byteLength + 64), __dopplerFakeGPUBuffer: true };
      state[field + 'GPU'].bytes.set(bytesOf(data));
    }
  }
  return { schemaVersion: 4, layers: new Map([[7, state]]) };
}
try {
  const { cloneLinearAttentionRuntimeOnGPU: clone } = await import(target);
  for (const components of [1, 2]) for (const resident of [false, true]) {
    const original = source(components, resident), layer = original.layers.get(7);
    submissions = 0; copies = [];
    const child = await clone(original), other = child.layers.get(7);
    assert.equal(submissions, 1); assert.equal(other.seqLen, 53); assert.equal(other.warnedSeqMismatch, true);
    for (const field of fields) {
      const expected = resident ? layer[field + 'GPU'].bytes.subarray(0, layer[field].byteLength) : bytesOf(layer[field]);
      assert.deepEqual(other[field + 'GPU'].bytes, expected);
      assert.notEqual(other[field], layer[field]);
      other[field + 'GPU'].bytes.fill(0);
      assert(expected.some(byte => byte !== 0), 'child mutation never changes parent');
      linearCloneControl.release(other[field + 'GPU']);
    }
    assert.equal(copies.length, resident ? 6 : 0);
    assert.equal(live.size, 0);
  }
  for (failAt of [0, 1, 5, 7]) {
    allocations = 0;
    const runtime = source(2, true); runtime.layers.set(8, source(1, true).layers.get(7));
    await assert.rejects(clone(runtime), /allocation failed/); assert.equal(live.size, 0);
  }
  failAt = -1;
  for (failure of ['copy', 'submit', 'completion', 'upload']) {
    allocations = 0; discarded = [];
    await assert.rejects(clone(source(2, failure !== 'upload')), /failed/);
    assert.equal(live.size, 0);
    assert.equal(discarded.length > 0, failure === 'completion', 'unknown completion quarantines every child allocation');
  }
  failure = null;
  const malformed = source(2, true); malformed.layers.get(7).recurrentStateGPU.size = 4;
  await assert.rejects(clone(malformed), /invalid recurrentState storage/); assert.equal(live.size, 0);
  await assert.rejects(clone({ schemaVersion: 3, layers: new Map() }), /schema version/);
  assert.equal((await clone(null)).layers.size, 0);
  assert.equal((await clone({ schemaVersion: 4, layers: new Map() })).layers.size, 0);
} finally { hooks.deregister(); delete globalThis.linearCloneControl; }
console.log('resident recurrent clone: exact GPU bytes, CPU-only state, paired low words, isolation and failure ownership passed');
