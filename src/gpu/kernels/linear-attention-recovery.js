import { acquireBuffer } from '../../memory/buffer-pool.js';
import { isGpuBufferInstance } from '../weight-buffer.js';

/** @typedef {import('./linear-attention-recovery.js').RecoveryEntry} RecoveryEntry */
/** @type {WeakMap<import('../command-recorder.js').CommandRecorder, {numTokens: number; entries: Map<object, RecoveryEntry>}>} */
const recordingWindows = new WeakMap();

/** @type {import('./linear-attention-recovery.js').isLinearAttentionRecoveryRecording} */
export function isLinearAttentionRecoveryRecording(recorder) {
  return recordingWindows.has(recorder);
}

/**
 * One explicit verification window. Snapshots and retained projection inputs remain
 * owned by its parent recorder until deferred cleanup, including recovery submits.
 * No private queue submit, host wait, projection recomputation or CPU tensor arithmetic.
 * @type {import('./linear-attention-recovery.js').createLinearAttentionRecovery}
 */
export function createLinearAttentionRecovery(recorder, numTokens) {
  if (!Number.isSafeInteger(numTokens) || numTokens < 2 || recordingWindows.has(recorder)
      || recorder.getStats().submitted) {
    throw new Error('[Linear recovery] One unsubmitted verification window with explicit token bounds is required.');
  }
  /** @type {Map<object, RecoveryEntry>} */
  const entries = new Map();
  const window = { numTokens, entries };
  recordingWindows.set(recorder, window);
  let closed = false;
  /** @param {import('../command-recorder.js').CommandRecorder} target */
  const assertOpen = (target) => {
    if (closed || target.device !== recorder.device || target.getStats().submitted) {
      throw new Error('[Linear recovery] Recovery requires an open same-device recording and a live parent lease.');
    }
  };
  return {
    get layerCount() { return entries.size; },
    async recordRecover(target, acceptance) {
      assertOpen(target);
      if (acceptance.maxTokens !== numTokens || !entries.size) {
        throw new Error('[Linear recovery] Acceptance must match a nonempty captured verification window.');
      }
      // Restore every initial state before the selected prefix consumes retained projections.
      const encoder = target.getEncoder();
      for (const entry of entries.values()) {
        for (const { source, saved } of entry.copies) encoder.copyBufferToBuffer(saved, 0, source, 0, source.size);
      }
      for (const entry of entries.values()) await entry.recover(target, acceptance);
    },
    recordRestore(target) {
      assertOpen(target);
      const encoder = target.getEncoder();
      for (const entry of entries.values()) {
        for (const { source, saved } of entry.copies) encoder.copyBufferToBuffer(saved, 0, source, 0, source.size);
      }
    },
    close() {
      closed = true;
      recordingWindows.delete(recorder);
      entries.clear();
      // Only the parent recorder releases resources, after submitted recovery work settles.
    },
  };
}

/** Called only by the core owner, before its first tentative state update.
 * @type {import('./linear-attention-recovery.js').captureLinearAttentionRecovery}
 */
export function captureLinearAttentionRecovery(recorder, state, numTokens, recover) {
  const window = recordingWindows.get(recorder);
  if (!window) return;
  if (recorder.getStats().submitted || numTokens !== window.numTokens || window.entries.has(state)
      || state.sourceLayout !== 'gguf-v1' || state.recurrentStatePrecision !== 'f32') {
    throw new Error('[Linear recovery] Each native F32 recurrent layer must occur exactly once in an unsubmitted window at the declared width.');
  }
  const sources = [state.convStateGPU, state.recurrentStateGPU];
  for (const source of sources) {
    if (!isGpuBufferInstance(source) || (source.usage & GPUBufferUsage.COPY_SRC) === 0
        || (source.usage & GPUBufferUsage.COPY_DST) === 0) {
      throw new Error('[Linear recovery] Resident copyable state buffers are required.');
    }
  }
  const copies = [];
  for (const source of sources) {
    const saved = acquireBuffer(source.size, undefined, 'linear_recovery_initial');
    recorder.trackTemporaryBuffer(saved);
    copies.push({ source, saved });
    recorder.getEncoder().copyBufferToBuffer(source, 0, saved, 0, source.size);
  }
  window.entries.set(state, { copies, recover });
}
