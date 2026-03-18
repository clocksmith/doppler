// =============================================================================
// Token Press — Queue Layer
// =============================================================================
// Manages the token record stream with forward/back navigation.
// Drains ONE token per flush tick to the renderer callback.
//
// Environment: browser-only (uses requestAnimationFrame).

export function createTokenQueue(options = {}) {
  const {
    topKSize = 10,
    onFlush = null,
  } = options;

  const pending = [];
  const committed = [];

  let playing = false;
  let rafId = null;
  let disposed = false;

  function notify() {
    if (onFlush) {
      onFlush({
        committed,
        cursor: committed.length,
        total: committed.length + pending.length,
        playing,
      });
    }
  }

  function tick() {
    if (disposed) return;

    // Drain exactly ONE token per frame tick
    if (playing && pending.length > 0) {
      committed.push(pending.shift());
      notify();
    }

    if (playing) {
      rafId = requestAnimationFrame(tick);
    }
  }

  function push(record) {
    if (disposed) return;
    pending.push({
      tokenId: record.tokenId,
      text: record.text ?? '',
      topK: Array.isArray(record.topK) ? record.topK.slice(0, topKSize) : [],
      confidence: Number.isFinite(record.confidence) ? record.confidence : 0,
    });
  }

  function play() {
    if (disposed || playing) return;
    playing = true;
    rafId = requestAnimationFrame(tick);
  }

  function pause() {
    playing = false;
    if (rafId) {
      cancelAnimationFrame(rafId);
      rafId = null;
    }
  }

  function stepForward() {
    if (disposed) return null;
    pause();
    if (pending.length === 0) return null;
    const record = pending.shift();
    committed.push(record);
    notify();
    return record;
  }

  // stepBack returns the removed record (caller is responsible for
  // rewinding pipeline state — KV cache, linear attention, etc.)
  function stepBack() {
    if (disposed || committed.length === 0) return null;
    pause();
    const removed = committed.pop();
    pending.unshift(removed);
    notify();
    return removed;
  }

  function clear() {
    pending.length = 0;
    committed.length = 0;
    notify();
  }

  function dispose() {
    disposed = true;
    pause();
    pending.length = 0;
    committed.length = 0;
  }

  return {
    push,
    play,
    pause,
    stepForward,
    stepBack,
    clear,
    dispose,
    get playing() { return playing; },
    get cursor() { return committed.length; },
    get total() { return committed.length + pending.length; },
    get committed() { return committed; },
    getState() {
      return { committed, cursor: committed.length, total: committed.length + pending.length, playing };
    },
  };
}
