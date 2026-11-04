import { getDevice } from '../../../../gpu/device.js';
import { acquireBuffer, releaseBuffer, discardBuffer } from '../../../../memory/buffer-pool.js';
import { isGpuBuffer } from './plan.js';

// Caller serializes capture, use and restoration on this device's queue and
// retains the lease until all commands referencing it have settled.
/** @type {import('./state-checkpoint.js').createLinearAttentionCheckpoint} */
export function createLinearAttentionCheckpoint(runtime) {
  const device = getDevice();
  if (!device) throw new Error('Recurrent checkpoint requires a live GPU device.');
  /** @type {import('./state-checkpoint.js').SavedStateBuffer[]} */
  const copies = [];
  /** @type {import('./state-checkpoint.js').SavedStateCursor[]} */
  const cursors = [];
  let closed = false, captured = false;
  const close = (lost = false) => {
    if (closed) return;
    closed = true;
    for (const { saved } of copies) (lost ? discardBuffer : releaseBuffer)(saved);
  };
  try {
    for (const layer of runtime.layers.values()) {
      cursors.push({ layer, seqLen: layer.seqLen, warnedSeqMismatch: layer.warnedSeqMismatch });
      for (const source of [layer.convStateGPU, layer.recurrentStateGPU]) {
        if (!source || !isGpuBuffer(source)) throw new Error('Speculative checkpoint requires resident recurrent GPU state.');
        const saved = acquireBuffer(source.size, undefined, 'linear_speculative_saved');
        copies.push({ source, saved });
      }
    }
    const assertLive = () => {
      if (closed || getDevice() !== device) throw new Error('Recurrent checkpoint is closed or its device changed.');
      const layers = [...runtime.layers.values()];
      if (layers.length !== cursors.length || layers.some((layer, i) => layer !== cursors[i].layer
        || layer.convStateGPU !== copies[i * 2].source || layer.recurrentStateGPU !== copies[i * 2 + 1].source)) {
        throw new Error('Recurrent checkpoint state resources changed.');
      }
    };
    return { close,
      recordCapture(encoder) {
        assertLive();
        captured = false;
        for (const cursor of cursors) { cursor.seqLen = cursor.layer.seqLen; cursor.warnedSeqMismatch = cursor.layer.warnedSeqMismatch; }
        for (const { source, saved } of copies) encoder.copyBufferToBuffer(source, 0, saved, 0, source.size);
        captured = true;
      },
      recordRestore(encoder) {
        assertLive();
        if (!captured) throw new Error('Recurrent checkpoint has no recorded capture.');
        for (const { source, saved } of copies) encoder.copyBufferToBuffer(saved, 0, source, 0, source.size);
        for (const { layer, seqLen, warnedSeqMismatch } of cursors) Object.assign(layer, { seqLen, warnedSeqMismatch });
      },
    };
  } catch (error) {
    close(); throw error;
  }
}

/** @param {import('../linear-attention.js').LinearAttentionRuntime} runtime @param {boolean} waitForCapture */
async function captureLinearAttentionRuntime(runtime, waitForCapture) {
  const device = getDevice();
  if (!device) throw new Error('Recurrent checkpoint requires a live GPU device.');
  const checkpoint = createLinearAttentionCheckpoint(runtime);
  /** @param {string} label @param {(encoder: GPUCommandEncoder) => void} record @param {boolean} wait */
  const settle = async (label, record, wait) => {
    try {
      const encoder = device.createCommandEncoder({ label });
      record(encoder); device.queue.submit([encoder.finish()]);
      if (wait) await device.queue.onSubmittedWorkDone();
    } catch (error) {
      try { await device.queue.onSubmittedWorkDone(); checkpoint.close(); }
      catch (settlement) { checkpoint.close(true); throw new AggregateError([error, settlement], 'Recurrent checkpoint settlement failed.'); }
      throw error;
    }
  };
  await settle('linear_speculative_checkpoint', checkpoint.recordCapture, waitForCapture);
  return { close: checkpoint.close, restore: () => settle('linear_speculative_restore', checkpoint.recordRestore, true) };
}

/** @type {import('./state-checkpoint.js').checkpointLinearAttentionRuntime} */
export function checkpointLinearAttentionRuntime(runtime) { return captureLinearAttentionRuntime(runtime, true); }

/** @type {import('./state-checkpoint.js').queueLinearAttentionCheckpoint} */
export function queueLinearAttentionCheckpoint(runtime) { return captureLinearAttentionRuntime(runtime, false); }
