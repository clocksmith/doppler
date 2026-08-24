import { releaseBuffer } from '../../../../memory/buffer-pool.js';

export function releaseSharedAttentionState(sharedAttentionState, recorder = null) {
  if (!(sharedAttentionState instanceof Map) || sharedAttentionState.size === 0) {
    return;
  }
  const released = new Set();
  const releaseOnce = (buffer) => {
    if (!buffer || released.has(buffer)) {
      return;
    }
    released.add(buffer);
    if (recorder) {
      recorder.trackTemporaryBuffer(buffer);
      return;
    }
    releaseBuffer(buffer);
  };
  for (const entry of sharedAttentionState.values()) {
    releaseOnce(entry?.kTensor?.buffer ?? null);
    releaseOnce(entry?.vTensor?.buffer ?? null);
  }
  sharedAttentionState.clear();
}
