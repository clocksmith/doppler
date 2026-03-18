// =============================================================================
// Token Press — Integration Layer
// =============================================================================
// Wires the queue, renderer, and controls layers together. Provides a single
// entry point for the demo to attach token press visualization to a generation
// session.
//
// Architecture:
//   GPU batch → push(records) → queue → onFlush → renderer.render()
//                                  ↕
//                              controls (play/pause/step/back)
//
// Usage:
//   import { createTokenPress } from './token-press/index.js';
//
//   const press = createTokenPress(outputEl, controlsEl, { trailSize: 8 });
//
//   // During generation — call for each token after sampling:
//   press.pushToken({ tokenId, text, topK, confidence });
//
//   // Or batch:
//   press.pushBatch(records);
//
//   // Start auto-play (optional — can start in step mode):
//   press.play();
//
//   // Cleanup:
//   press.dispose();

import { createTokenQueue } from './queue.js';
import { createTokenPressRenderer } from './renderer.js';
import { createTokenPressControls } from './controls.js';

export function createTokenPress(outputEl, controlsEl, options = {}) {
  const { trailSize = 8, topKSize = 10, autoPlay = true } = options;

  const renderer = createTokenPressRenderer(outputEl, { trailSize });

  const queue = createTokenQueue({
    trailSize,
    topKSize,
    onFlush: (state) => {
      renderer.renderWithSettled(state, queue.getCommitted());
      if (controls) controls.refresh();
    },
  });

  const controls = createTokenPressControls(queue, controlsEl);

  function pushToken(record) {
    queue.push(record);
  }

  function pushBatch(records) {
    queue.pushBatch(records);
  }

  function play() {
    queue.play();
  }

  function pause() {
    queue.pause();
  }

  function clear() {
    queue.clear();
    renderer.clear();
    controls.refresh();
  }

  function dispose() {
    queue.dispose();
    renderer.dispose();
    controls.dispose();
  }

  if (autoPlay) {
    queue.play();
  }

  return {
    pushToken,
    pushBatch,
    play,
    pause,
    clear,
    dispose,
    get queue() { return queue; },
  };
}
