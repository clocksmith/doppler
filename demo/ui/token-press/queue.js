// =============================================================================
// Token Press — Queue Layer
// =============================================================================
// Pure data layer. No DOM. Manages the stream of token records with their
// probability distributions, supports forward/back navigation, and drains
// to a renderer callback at frame rate.
//
// Usage:
//   const q = createTokenQueue({ trailSize: 8, onFlush: renderFn });
//   q.push({ tokenId: 42, text: 'blue', topK: [...], confidence: 0.92 });
//   q.play();           // drain at frame rate
//   q.pause();          // stop draining, keep queue
//   q.stepForward();    // drain one token
//   q.stepBack();       // rewind one token
//   q.dispose();        // cleanup

export function createTokenQueue(options = {}) {
  const {
    trailSize = 8,
    topKSize = 10,
    onFlush = null,
  } = options;

  // Pending tokens from GPU batches, not yet rendered
  const pending = [];

  // All committed tokens (rendered), supports back navigation
  const committed = [];

  // Playback state
  let playing = false;
  let rafId = null;
  let disposed = false;

  function getTrailWindow() {
    const start = Math.max(0, committed.length - trailSize);
    return committed.slice(start);
  }

  function getCursorPosition() {
    return committed.length;
  }

  function getTotalTokens() {
    return committed.length + pending.length;
  }

  function flush() {
    if (disposed) return;
    if (pending.length === 0 && !playing) return;

    let flushed = false;

    if (playing && pending.length > 0) {
      // In play mode, commit all pending tokens that arrived since last frame
      const batch = pending.splice(0);
      for (const record of batch) {
        committed.push(record);
      }
      flushed = true;
    }

    if (flushed && onFlush) {
      onFlush({
        trail: getTrailWindow(),
        cursor: getCursorPosition(),
        total: getTotalTokens(),
        playing,
      });
    }

    if (playing) {
      rafId = requestAnimationFrame(flush);
    }
  }

  function push(record) {
    if (disposed) return;
    const entry = {
      tokenId: record.tokenId,
      text: record.text ?? '',
      topK: Array.isArray(record.topK) ? record.topK.slice(0, topKSize) : [],
      confidence: Number.isFinite(record.confidence) ? record.confidence : 0,
      timestamp: record.timestamp ?? performance.now(),
    };
    pending.push(entry);

    // In step mode, don't auto-drain
    if (!playing && !rafId) return;
  }

  function pushBatch(records) {
    for (const r of records) {
      push(r);
    }
  }

  function play() {
    if (disposed || playing) return;
    playing = true;
    rafId = requestAnimationFrame(flush);
  }

  function pause() {
    playing = false;
    if (rafId) {
      cancelAnimationFrame(rafId);
      rafId = null;
    }
  }

  function stepForward() {
    if (disposed) return;
    pause();

    if (pending.length > 0) {
      committed.push(pending.shift());
    }

    if (onFlush) {
      onFlush({
        trail: getTrailWindow(),
        cursor: getCursorPosition(),
        total: getTotalTokens(),
        playing: false,
      });
    }
  }

  function stepBack() {
    if (disposed || committed.length === 0) return;
    pause();

    const removed = committed.pop();
    pending.unshift(removed);

    if (onFlush) {
      onFlush({
        trail: getTrailWindow(),
        cursor: getCursorPosition(),
        total: getTotalTokens(),
        playing: false,
      });
    }

    return removed;
  }

  function clear() {
    pending.length = 0;
    committed.length = 0;
    if (onFlush) {
      onFlush({
        trail: [],
        cursor: 0,
        total: 0,
        playing,
      });
    }
  }

  function dispose() {
    disposed = true;
    pause();
    pending.length = 0;
    committed.length = 0;
  }

  return {
    push,
    pushBatch,
    play,
    pause,
    stepForward,
    stepBack,
    clear,
    dispose,
    get playing() { return playing; },
    get cursor() { return getCursorPosition(); },
    get total() { return getTotalTokens(); },
    getTrailWindow,
    getCommitted: () => committed.slice(),
  };
}
