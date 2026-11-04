import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { acquireBuffer, releaseBuffer, uploadData, readBuffer } from '../../src/memory/buffer-pool.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { createGreedySpeculativeSession } from '../../src/gpu/greedy-speculative-frame.js';
import { runLinearAttentionCoreGPU } from '../../src/gpu/kernels/linear-attention-core.js';

assert.ok((await probeNodeGPU({ installFileFetchShim: true })).ready);
const device = getDevice(), errors = [], owned = [], cases = [];
device.addEventListener('uncapturederror', event => errors.push(event.error.message));
const upload = values => { const b = acquireBuffer(values.byteLength); owned.push(b); uploadData(b, values); return b; };
const values = (n, rate, scale) => Float32Array.from({ length: n }, (_, i) => Math.sin((i + 1) * rate) * scale);
const words = async (buffer, n) => new Uint32Array(await readBuffer(buffer, n * 4));
const NK = 2, NV = 4, K = 128, VD = 128, T = 4, H = 4, V = 32;
const Q = NK * K, D = NV * VD, C = 2 * Q + D;
const makeState = () => ({
  sourceLayout: 'gguf-v1', convDim: C, convKernelSize: 4, valueDim: D, numVHeads: NV, numKHeads: NK,
  headKDim: K, headVDim: VD, qSize: Q, kSize: Q, qRep: NV / NK, normMode: 'per_head', rmsNormEps: 1e-6,
  convWeightGPU: upload(values(C * 4, 0.011, 0.5)), dtBiasGPU: upload(values(NV, 0.19, 0.2)),
  aLogGPU: upload(Float32Array.from({ length: NV }, (_, i) => -0.25 - i * 0.07)),
  normWeightGPU: upload(Float32Array.from({ length: D }, (_, i) => 0.7 + (i % 13) * 0.04)),
  convStateGPU: upload(values(C * 4, 0.021, 0.05)), recurrentStateGPU: upload(values(NV * K * VD, 0.007, 0.005)),
  recurrentStatePrecision: 'f32', recurrentStateComponents: 1,
});
const inputs = [values(T * C, 0.019, 0.2), values(T * D, 0.029, 0.8), values(T * NV, 0.11, 1.2), values(T * NV, 0.09, 2)]
  .map((data, i) => createTensor(upload(data), 'f32', [T, [C, D, NV, NV][i]]));
const coreOptions = register => ({ numTokens: T, normGateActivation: 'silu', outputDtype: 'f32', qkL2NormEps: 1e-6,
  runtimeSession: { useLinearAttentionRegisterPrefill: register, useLinearAttentionRegisterDecode: register,
    useLinearAttentionFusedDecodeCore: false } });
const initialHidden = createTensor(upload(new Float32Array([-1, -2, -3, -4])), 'f32', [1, H]);
const targetValues = Float32Array.from({ length: T * H }, (_, i) => i + 0.5);
const targetHidden = upload(targetValues);
const clone = (rec, buffer, rows, width) => {
  const b = acquireBuffer(rows * width * 4);
  rec.getEncoder().copyBufferToBuffer(buffer, 0, b, 0, rows * width * 4);
  return createTensor(b, 'f32', [rows, width]);
};
const logits = (ids, nonfinite = false) => {
  const out = new Float32Array(ids.length * V).fill(-10);
  ids.forEach((id, row) => { out[row * V + id] = 10; });
  if (nonfinite) out[1] = NaN;
  return upload(out);
};
const draftLogits = [10, 11, 12].map(id => logits([id]));
let draftVocabSize = V;
let refillKind = 'hidden';
const sessionOptions = owner => ({ owner, hiddenSize: H, vocabSize: V, draftCount: 3, draftVocabSize, usePreparedDecodeWorkspace: true,
  refillShapes: refillKind === 'hidden' ? Array.from({ length: T }, () => ({ rows: 1, width: H })) : [{ rows: T, width: H }, { rows: T, width: H }],
  initialHidden, initialToken: 7, padTokenId: null, finitenessAbsThreshold: 3.4028234663852886e38 });
try {
  // Allocation failure must release every partially acquired raw buffer and
  // leave the session idle and disposable, before any recorded state mutation runs.
  await assert.rejects(() => createGreedySpeculativeSession({ ...sessionOptions({}), draftVocabSize: V + 1 }), /bounded/);
  for (const refillShapes of [undefined, [], [{ rows: 0, width: H }], [{ rows: T + 1, width: H }], [{ rows: T, width: NaN }]]) {
    await assert.rejects(() => createGreedySpeculativeSession({ ...sessionOptions({}), refillShapes }), /refill tensor shapes/);
  }
  const allocationSession = await createGreedySpeculativeSession(sessionOptions({}));
  const createBuffer = device.createBuffer, allocated = [], destroyed = new Set();
  device.createBuffer = function (descriptor) {
    if (descriptor.label?.includes('greedy_speculative_frame_feedback')) throw new Error('injected frame allocation failure');
    const buffer = createBuffer.call(this, descriptor), destroy = buffer.destroy.bind(buffer);
    allocated.push(buffer); buffer.destroy = () => { destroyed.add(buffer); destroy(); }; return buffer;
  };
  try {
    await assert.rejects(() => allocationSession.begin({ recordDraft() {}, recordTarget() {}, recordRefill() {}, setPrefix() {} }, 4), /allocation failure/);
    assert.ok(allocated.length > 0); assert.equal(destroyed.size, allocated.length);
  } finally { device.createBuffer = createBuffer; await allocationSession.dispose(); }
  for (refillKind of ['hidden', 'kv']) for (draftVocabSize of [16, V]) for (const register of [false, true]) {
    for (const spec of [
      { name: 'all', published: 4 }, { name: 'none', published: 0 }, { name: 'partial1', published: 1 },
      { name: 'partial2', published: 2 }, { name: 'limit1', published: 1, limit: 1 },
      { name: 'reject0', published: 1, mismatch: 0 }, { name: 'reject1', published: 2, mismatch: 1 },
      { name: 'reject2', published: 3, mismatch: 2 }, { name: 'nan', failure: true },
      { name: 'abort-record', failure: true }, { name: 'abort-settlement', failure: true },
      { name: 'callback-failure', failure: true }, { name: 'map-failure', failure: true },
      { name: 'refill-nan', failure: true }, { name: 'refill-shape', failure: true },
      { name: 'refill-count', failure: true },
    ]) {
      const reference = makeState(), actual = makeState(), owner = {}, controller = new AbortController();
      const session = await createGreedySpeculativeSession(sessionOptions(owner));
      await assert.rejects(() => createGreedySpeculativeSession(sessionOptions(owner)), /exclusive owner/);
      const ids = [10, 11, 12, 13]; if (spec.mismatch !== undefined) ids[spec.mismatch] = 29;
      const targetLogits = logits(ids, spec.name === 'nan'), continuationLogits = logits([10, 11, 12, 13]);
      const refillValues = targetValues.slice();
      if (spec.name === 'refill-nan') refillValues[0] = NaN;
      const refillSource = upload(refillValues);
      const seenInputs = upload(new Uint32Array(T));
      const seenHidden = upload(new Float32Array(H)); let prefix = -1, submissionCount = null;
      let lastRecorder, continuing = false;
      const program = {
        async recordDraft(rec, tokens, index, hidden, draft) {
          lastRecorder = rec;
          if (draft === 0) rec.getEncoder().copyBufferToBuffer(hidden.buffer, 0, seenHidden, 0, H * 4);
          return { hidden: clone(rec, initialHidden.buffer, 1, H), logits: clone(rec, draftLogits[draft], 1, draftVocabSize) };
        },
        async recordTarget(rec, tokens) {
          rec.getEncoder().copyBufferToBuffer(tokens, 0, seenInputs, 0, T * 4);
          const output = await runLinearAttentionCoreGPU(...inputs, actual, { ...coreOptions(register), recorder: rec });
          rec.trackTemporaryBuffer(output.buffer);
          if (!continuing && spec.name === 'abort-record') controller.abort(new Error('record cancelled'));
          return { hidden: clone(rec, targetHidden, T, H), logits: clone(rec, continuing ? continuationLogits : targetLogits, T, V) };
        },
        async recordRefill(rec, hidden) {
          if (refillKind === 'kv') {
            const rows = [clone(rec, continuing ? targetHidden : refillSource, T, H), clone(rec, targetHidden, T, H)];
            if (!continuing && spec.name === 'refill-shape') rows[0].shape = [1, T * H];
            if (!continuing && spec.name === 'refill-count') rows.push(clone(rec, targetHidden, T, H));
            return rows;
          }
          const rows = [];
          for (let i = 0; i < T; i++) {
            const b = acquireBuffer(H * 4);
            rec.getEncoder().copyBufferToBuffer(continuing ? hidden.buffer : refillSource, i * H * 4, b, 0, H * 4);
            rows.push(createTensor(b, 'f32', [1, H]));
          }
          if (!continuing && spec.name === 'refill-shape') rows[0].shape = [T, H];
          if (!continuing && spec.name === 'refill-count') rows.push(clone(rec, targetHidden, 1, H));
          return rows;
        },
        setPrefix(count) {
          prefix = count;
          if (!continuing && count > 0 && spec.name === 'abort-settlement') controller.abort(new Error('publication cancelled'));
          if (!continuing && count > 0 && spec.name === 'callback-failure') throw new Error('consumer callback failed');
        },
      };
      if (spec.name === 'map-failure') device.createBuffer = function (descriptor) {
        const buffer = createBuffer.call(this, descriptor);
        if (descriptor.label?.includes('greedy_speculative_frame_feedback')) {
          buffer.mapAsync = () => Promise.reject(new Error('injected frame map failure'));
        }
        return buffer;
      };
      try {
        if (spec.failure) {
          await assert.rejects(() => session.begin(program, 4, controller.signal), /validation|cancelled|callback failed|map failure|refill tensor|row geometry/);
          assert.equal(prefix, 0);
        } else {
          const transaction = await session.begin(program, spec.limit ?? 4, controller.signal);
          submissionCount = lastRecorder.getStats().submissionCount; assert.equal(submissionCount, 1);
          await assert.rejects(() => session.begin(program, 4), /idle live session/);
          await assert.rejects(() => session.dispose(), /Settle/);
          const expected = Math.min((spec.mismatch ?? 3) + 1, spec.limit ?? 4);
          assert.deepEqual(transaction.tokens, ids.slice(0, expected));
          assert.deepEqual(Array.from(await words(seenInputs, T)), [7, 10, 11, 12], 'GPU draft chain was overwritten');
          for (let i = 0; i < spec.published; i++) transaction.publish();
          const settled = transaction.settle(); assert.equal(transaction.settle(), settled); await settled;
          assert.equal(prefix, spec.published);
          assert.throws(() => transaction.publish(), /Invalid/);
        }
        const count = spec.failure ? 0 : spec.published;
        if (count > 0) {
          const output = await runLinearAttentionCoreGPU(...inputs, reference, { ...coreOptions(register), numTokens: count });
          owned.push(output.buffer);
        }
        assert.deepEqual(await words(actual.convStateGPU, C * 4), await words(reference.convStateGPU, C * 4));
        assert.deepEqual(await words(actual.recurrentStateGPU, NV * K * VD), await words(reference.recurrentStateGPU, NV * K * VD));
        // The next frame must consume precisely the retained hidden/token pair.
        device.createBuffer = createBuffer;
        continuing = true;
        const continuation = await session.begin(program, 4);
        const expectedHidden = count ? targetValues.slice((count - 1) * H, count * H) : [-1, -2, -3, -4];
        assert.deepEqual(Array.from(new Float32Array(await readBuffer(seenHidden, H * 4))), Array.from(expectedHidden));
        assert.equal((await words(seenInputs, T))[0], count ? ids[count - 1] : 7);
        await continuation.settle();
        assert.deepEqual(await words(actual.convStateGPU, C * 4), await words(reference.convStateGPU, C * 4));
        assert.deepEqual(await words(actual.recurrentStateGPU, NV * K * VD), await words(reference.recurrentStateGPU, NV * K * VD));
        cases.push({ refillKind, draftVocabSize, register, name: spec.name, published: count, exactStateAndCarryContinuation: true, submissionCount });
      } finally { device.createBuffer = createBuffer; await session.dispose(); await session.dispose(); }
    }
  }
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ passed: true, scope: 'GPU frame ownership, finite outputs and exact recurrent-prefix settlement; not model qualification', cases }));
} finally {
  await device.queue.onSubmittedWorkDone(); for (const b of owned) releaseBuffer(b); destroyDevice();
}
