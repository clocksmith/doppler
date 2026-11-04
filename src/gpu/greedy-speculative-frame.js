import { getDevice } from './device.js';
import { createCommandRecorder } from './command-recorder.js';
import { createTensor } from './tensor.js';
import { PreparedDecodeWorkspace } from './prepared-decode-workspace.js';
import { acquireBuffer, releaseBuffer, discardBuffer, uploadData } from '../memory/buffer-pool.js';
import { recordArgmax } from './kernels/sample.js';
import { recordCheckFinitenessF32 } from './kernels/check-finiteness-f32.js';
import { createLinearAttentionRecovery } from './kernels/linear-attention-recovery.js';
import { recordSpeculativeAcceptance, recordSpeculativeCarry } from './kernels/speculative-accept.js';
import { isGpuBufferInstance } from './weight-buffer.js';

const owners = new WeakSet();
/** @typedef {import('./command-recorder.js').CommandRecorder} CommandRecorder */
/** @typedef {import('./tensor.js').Tensor} Tensor */

/** @param {Tensor} tensor @param {number} rows @param {number} width */
function assertRows(tensor, rows, width) {
  if (tensor?.dtype !== 'f32' || tensor.precision != null || tensor.shape.length !== 2
    || tensor.shape[0] !== rows || tensor.shape[1] !== width || tensor.buffer.size < rows * width * 4
    || (tensor.buffer.usage & (GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC))
      !== (GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC)) {
    throw new Error('Greedy frame requires explicit scalar F32 row geometry and resident storage.');
  }
}

/** Allocate one recorder-owned lease; unwind partial allocation before returning.
 * @param {number} depth @param {number} hiddenSize @param {PreparedDecodeWorkspace | null} workspace */
async function createFrameResources(depth, hiddenSize, workspace) {
  const recorder = createCommandRecorder('greedy_speculative_frame', { recordLabels: false, recordDispatches: false });
  recorder.decodeWorkspace = workspace;
  const recovery = createLinearAttentionRecovery(recorder, depth + 1);
  /** @param {number} bytes @param {string} label @param {number} [usage] */
  const make = (bytes, label, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST) =>
    recorder.createTempBuffer(bytes, usage, label);
  try {
    const compactBytes = (depth + 6) * 4;
    return { recorder, recovery, make, compactBytes, drafts: make(depth * 4, 'drafts'),
      targets: make((depth + 1) * 4, 'targets'), inputs: make((depth + 1) * 4, 'inputs'),
      compact: make(compactBytes, 'acceptance'), status: make(16, 'finite'),
      savedHidden: make(hiddenSize * 4, 'saved_hidden'), savedToken: make(4, 'saved_token'),
      staging: make(compactBytes + 16, 'feedback', GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST) };
  } catch (error) {
    recovery.close();
    try { await recorder.abort(); }
    catch (cleanup) { throw new AggregateError([error, cleanup], 'Frame allocation and cleanup failed.'); }
    throw error;
  }
}

/** @type {typeof import('./greedy-speculative-frame.js').createGreedySpeculativeSession} */
export async function createGreedySpeculativeSession(options) {
  const device = getDevice(), { owner, hiddenSize: H, vocabSize: V, draftCount: D, draftVocabSize: DV } = options;
  if (!device || !owner || typeof owner !== 'object' || owners.has(owner)
    || ![H, V, D, DV].every(value => Number.isSafeInteger(value) && value > 0) || DV > V
    || D > 32 || (D + 1) * Math.max(H, V) * 4 > device.limits.maxStorageBufferBindingSize
    || !Number.isSafeInteger(options.initialToken) || options.initialToken < 0 || options.initialToken >= V
    || (options.padTokenId !== null && (!Number.isSafeInteger(options.padTokenId) || options.padTokenId < 0 || options.padTokenId >= V))
    || typeof options.usePreparedDecodeWorkspace !== 'boolean'
    || !Number.isFinite(options.finitenessAbsThreshold) || options.finitenessAbsThreshold <= 0
    || !Number.isFinite(Math.fround(options.finitenessAbsThreshold))) {
    throw new Error('Greedy frame requires an exclusive owner and bounded declared F32 geometry, seed and limits.');
  }
  assertRows(options.initialHidden, 1, H);
  if (!Array.isArray(options.refillShapes) || options.refillShapes.length < 1 || options.refillShapes.length > D + 1
    || options.refillShapes.some(shape => !shape || !Number.isSafeInteger(shape.rows) || shape.rows < 1 || shape.rows > D + 1
      || !Number.isSafeInteger(shape.width) || shape.width < 1 || !Number.isSafeInteger(shape.rows * shape.width * 4)
      || shape.rows * shape.width * 4 > device.limits.maxStorageBufferBindingSize)) {
    throw new Error('Greedy frame requires bounded declared auxiliary refill tensor shapes.');
  }
  const refillShapes = options.refillShapes.map(({ rows, width }) => Object.freeze({ rows, width }));
  const workspace = options.usePreparedDecodeWorkspace ? new PreparedDecodeWorkspace(device) : null;
  owners.add(owner);
  let carryHidden, carryToken, busy = false, disposed = false, poisoned = false;
  try {
    carryHidden = acquireBuffer(H * 4); carryToken = acquireBuffer(4);
    uploadData(carryToken, new Uint32Array([options.initialToken]));
    const recorder = createCommandRecorder('greedy_frame_initialize');
    try {
      recorder.getEncoder().copyBufferToBuffer(options.initialHidden.buffer, 0, carryHidden, 0, H * 4);
      await recorder.submitAndWait();
    } finally { await recorder.abort(); }
  } catch (error) {
    if (carryHidden) releaseBuffer(carryHidden);
    if (carryToken) releaseBuffer(carryToken);
    workspace?.release(); owners.delete(owner); throw error;
  }
  const hidden = carryHidden, token = carryToken;
  /** @type {import('./greedy-speculative-frame.js').GreedySpeculativeSession} */
  const session = {
    async begin(program, outputLimit, signal) {
      if (disposed || poisoned || busy || !Number.isSafeInteger(outputLimit) || outputLimit < 1 || outputLimit > D + 1
        || typeof program?.recordDraft !== 'function' || typeof program.recordTarget !== 'function'
        || typeof program.recordRefill !== 'function' || typeof program.setPrefix !== 'function') {
        throw new Error('Greedy frame requires an idle live session, bounded output and complete recording/settlement ports.');
      }
      signal?.throwIfAborted(); busy = true;
      let frame;
      try { frame = await createFrameResources(D, H, workspace); } catch (error) { busy = false; throw error; }
      const { recorder, recovery, make, compactBytes, drafts, targets, inputs, compact, status, savedHidden, savedToken, staging } = frame;
      let submitted = false, mapped = false;
      /** @type {Tensor} */
      let targetHidden;
      /** @param {CommandRecorder} rec @param {number} count */
      const recover = async (rec, count) => {
        if (count === 0) {
          recovery.recordRestore(rec);
          rec.getEncoder().copyBufferToBuffer(savedHidden, 0, hidden, 0, H * 4);
          rec.getEncoder().copyBufferToBuffer(savedToken, 0, token, 0, 4);
        } else {
          const acceptance = await recordSpeculativeAcceptance(rec, drafts, targets, {
            draftCount: D, outputLimit: count, vocabSize: V, outputBuffer: compact,
          });
          await recovery.recordRecover(rec, acceptance);
          await recordSpeculativeCarry(rec, acceptance, targetHidden, H, hidden, token);
        }
      };
      /** @param {number} count */
      const restore = async count => {
        const rec = createCommandRecorder('greedy_frame_settlement');
        const failures = [];
        try { await recover(rec, count); await rec.submitAndWait(); } catch (error) { failures.push(error); }
        try { await rec.abort(); } catch (error) { failures.push(error); }
        if (failures.length) throw new AggregateError(failures, 'Frame prefix restoration failed.');
      };
      const cleanup = async () => {
        try { recovery.close(); if (submitted) await recorder.completeDeferredCleanup(); else await recorder.abort(); }
        catch (error) { poisoned = true; throw error; }
        finally { busy = false; }
      };
      /** @param {Tensor} value */
      const take = value => {
        if (isGpuBufferInstance(value?.buffer) && value.buffer !== hidden && value.buffer !== token) recorder.trackTemporaryBuffer(value.buffer);
      };
      /** @param {Tensor} value @param {number} rows @param {number} width */
      const check = async (value, rows, width) => {
        take(value); assertRows(value, rows, width);
        if (value.buffer === hidden || value.buffer === token) throw new Error('Frame outputs must not transfer borrowed carry storage.');
        await recordCheckFinitenessF32(recorder, value.buffer, rows * width, status, 0, 0, options.finitenessAbsThreshold);
      };
      /** @param {Tensor} logits @param {number} rows @param {GPUBuffer} output */
      const select = async (logits, rows, output) => {
        const scratch = rows === 1 ? logits.buffer : make(V * 4, 'head_row');
        for (let index = 0; index < rows; index++) {
          if (rows > 1) recorder.getEncoder().copyBufferToBuffer(logits.buffer, index * V * 4, scratch, 0, V * 4);
          await recordArgmax(recorder, scratch, V, { padTokenId: options.padTokenId,
            logitSoftcap: 0, logitsDtype: 'f32', outputBuffer: output, outputIndex: index });
        }
      };
      try {
        const encoder = recorder.getEncoder();
        encoder.copyBufferToBuffer(hidden, 0, savedHidden, 0, H * 4);
        encoder.copyBufferToBuffer(token, 0, savedToken, 0, 4); encoder.clearBuffer(status);
        await recordCheckFinitenessF32(recorder, hidden, H, status, 0, 0, options.finitenessAbsThreshold);
        let previous = createTensor(hidden, 'f32', [1, H]);
        for (let index = 0; index < D; index++) {
          signal?.throwIfAborted();
          const row = await program.recordDraft(recorder, index ? drafts : token, index ? index - 1 : 0, previous, index);
          take(row.hidden); take(row.logits);
          await check(row.hidden, 1, H); await check(row.logits, 1, DV);
          await recordArgmax(recorder, row.logits.buffer, DV, { padTokenId: options.padTokenId,
            logitSoftcap: 0, logitsDtype: 'f32', outputBuffer: drafts, outputIndex: index });
          previous = row.hidden;
        }
        const inputEncoder = recorder.getEncoder();
        inputEncoder.copyBufferToBuffer(token, 0, inputs, 0, 4); inputEncoder.copyBufferToBuffer(drafts, 0, inputs, 4, D * 4);
        const result = await program.recordTarget(recorder, inputs, D + 1);
        take(result.hidden); take(result.logits);
        await check(result.hidden, D + 1, H); await check(result.logits, D + 1, V);
        targetHidden = result.hidden; await select(result.logits, D + 1, targets);
        const acceptance = await recordSpeculativeAcceptance(recorder, drafts, targets, {
          draftCount: D, outputLimit, vocabSize: V, outputBuffer: compact,
        });
        await recovery.recordRecover(recorder, acceptance);
        const refill = await program.recordRefill(recorder, result.hidden, targets);
        if (Array.isArray(refill)) for (const row of refill) take(row);
        if (!Array.isArray(refill) || refill.length !== refillShapes.length) throw new Error('Greedy frame requires every declared auxiliary refill tensor.');
        for (let i = 0; i < refill.length; i++) await check(refill[i], refillShapes[i].rows, refillShapes[i].width);
        await recordSpeculativeCarry(recorder, acceptance, result.hidden, H, hidden, token);
        const feedbackEncoder = recorder.getEncoder();
        feedbackEncoder.copyBufferToBuffer(compact, 0, staging, 0, compactBytes);
        feedbackEncoder.copyBufferToBuffer(status, 0, staging, compactBytes, 16);
        signal?.throwIfAborted(); recorder.submit({ cleanup: 'deferred' }); submitted = true;
        await staging.mapAsync(GPUMapMode.READ); mapped = true;
        const words = new Uint32Array(staging.getMappedRange()).slice(); staging.unmap(); mapped = false;
        signal?.throwIfAborted();
        const count = words[0], accepted = words[1], tokens = Object.freeze(Array.from(words.slice(5, 5 + count)));
        if (words[2] !== 0 || words[compactBytes / 4] !== 0 || count < 1 || count > outputLimit
          || accepted > D || accepted > count || words[3] !== count - 1 || words[4] >= V
          || tokens.length !== count || tokens.some(value => value >= V)) {
          throw new Error('Greedy frame acceptance or finite-output validation failed.');
        }
        await program.setPrefix(count); signal?.throwIfAborted();
        let published = 0;
        /** @type {Promise<void> | undefined} */
        let settlement;
        return {
          tokens, accepted,
          publish() {
            signal?.throwIfAborted();
            if (settlement || published >= count) throw new Error('Invalid greedy frame publication.');
            published++;
          },
          settle() {
            settlement ??= (async () => {
              try { if (published !== count) await restore(published); await program.setPrefix(published); }
              catch (error) { poisoned = true; throw error; }
              finally { await cleanup(); }
            })();
            return settlement;
          },
        };
      } catch (error) {
        const failures = [error];
        if (mapped) staging.unmap();
        try { if (submitted) { await device.queue.onSubmittedWorkDone(); await restore(0); } await program.setPrefix(0); }
        catch (failure) { poisoned = true; failures.push(failure); }
        try { await cleanup(); } catch (failure) { poisoned = true; failures.push(failure); }
        if (failures.length > 1) throw new AggregateError(failures, 'Greedy frame and rollback failed.');
        throw error;
      }
    },
    async dispose() {
      if (disposed) return;
      if (busy) throw new Error('Settle the outstanding greedy frame before disposing its session.');
      disposed = true;
      let lost = false;
      try { await device.queue.onSubmittedWorkDone(); }
      catch (error) { lost = true; throw error; }
      finally {
        const release = lost ? discardBuffer : releaseBuffer;
        release(hidden); release(token); workspace?.release(); owners.delete(owner);
      }
    },
  };
  return session;
}
