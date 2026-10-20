import { getDevice } from '../../../../gpu/device.js';
import { acquireBuffer, releaseBuffer, discardBuffer } from '../../../../memory/buffer-pool.js';
import { isGpuBuffer } from './plan.js';

// Same-device speculative rollback; static weights stay with their original owner.
/** @type {import('./state-checkpoint.js').checkpointLinearAttentionRuntime} */
export async function checkpointLinearAttentionRuntime(runtime) {
  const device = getDevice();
  if (!device) throw new Error('Recurrent checkpoint requires a live GPU device.');
  /** @type {import('./state-checkpoint.js').SavedStateBuffer[]} */
  const copies = [];
  /** @type {import('./state-checkpoint.js').SavedStateCursor[]} */
  const cursors = [];
  let closed = false;
  const close = (lost = false) => {
    if (closed) return;
    closed = true;
    for (const { saved } of copies) (lost ? discardBuffer : releaseBuffer)(saved);
  };
  try {
    const encoder = device.createCommandEncoder({ label: 'linear_speculative_checkpoint' });
    for (const layer of runtime.layers.values()) {
      cursors.push({ layer, seqLen: layer.seqLen, warnedSeqMismatch: layer.warnedSeqMismatch });
      for (const source of [layer.convStateGPU, layer.recurrentStateGPU]) {
        if (!source || !isGpuBuffer(source)) throw new Error('Speculative checkpoint requires resident recurrent GPU state.');
        const saved = acquireBuffer(source.size, undefined, 'linear_speculative_saved');
        copies.push({ source, saved }); encoder.copyBufferToBuffer(source, 0, saved, 0, source.size);
      }
    }
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    return { close,
      async restore() {
        if (closed) throw new Error('Recurrent checkpoint is closed.');
        const restore = device.createCommandEncoder({ label: 'linear_speculative_restore' });
        for (const { source, saved } of copies) restore.copyBufferToBuffer(saved, 0, source, 0, source.size);
        try {
          device.queue.submit([restore.finish()]);
          await device.queue.onSubmittedWorkDone();
        } catch (error) { close(true); throw error; }
        for (const { layer, seqLen, warnedSeqMismatch } of cursors) Object.assign(layer, { seqLen, warnedSeqMismatch });
      }
    };
  } catch (error) {
    try { await device.queue.onSubmittedWorkDone(); close(); }
    catch (settlement) { close(true); throw new AggregateError([error, settlement], 'Recurrent checkpoint cleanup failed.'); }
    throw error;
  }
}
