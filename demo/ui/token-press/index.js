// =============================================================================
// Token Press — Integration Layer
// =============================================================================
// Wires queue + renderer + controls + session (bridge) into a single unit.
//
// Usage:
//   import { createTokenPress } from './token-press/index.js';
//   import { createTokenPressSession } from './token-press/bridge.js';
//
//   const press = createTokenPress(outputEl, controlsEl, { trailSize: 8 });
//   const session = createTokenPressSession(pipeline, press, prompt, samplingOpts);
//
//   // Session drives generation; press drives visualization.
//   // Controls call session.stepForward/stepBack (real inference state).
//   press.attachSession(session);
//   press.play();

import { createTokenQueue } from './queue.js';
import { createTokenPressRenderer } from './renderer.js';
import { createTokenPressControls } from './controls.js';

export function createTokenPress(outputEl, controlsEl, options = {}) {
  const { trailSize = 8, topKSize = 10, autoPlay = false } = options;

  let session = null;
  let playLoopId = null;
  let playing = false;

  const renderer = createTokenPressRenderer(outputEl, { trailSize });

  const queue = createTokenQueue({
    topKSize,
    onFlush: (state) => {
      renderer.render(state);
      updateControls();
    },
  });

  function updateControls() {
    controls.update({
      playing,
      cursor: queue.cursor,
      total: queue.total,
      backDisabled: !session || !session.supportsStepBack,
      backReason: session?.stepBackReason ?? null,
    });
  }

  async function doStep() {
    if (!session || session.finished) return;
    await session.stepForward();
    // stepForward pushes to queue via press.pushToken,
    // queue.onFlush triggers render
  }

  async function doBack() {
    if (!session) return;
    await session.stepBack();
    // stepBack pops from queue, queue.onFlush triggers render
    updateControls();
  }

  async function playLoop() {
    while (playing && session && !session.finished) {
      await doStep();
      // Yield to the browser for one frame
      await new Promise(resolve => requestAnimationFrame(resolve));
    }
    if (playing) {
      playing = false;
      // Collapse remaining trail tokens into settled zone so they get
      // full tooltip spans and consistent styling.
      renderer.finalize(queue.getState());
      updateControls();
    }
  }

  function play() {
    if (playing) return;
    playing = true;
    updateControls();
    playLoop();
  }

  function pause() {
    playing = false;
    updateControls();
  }

  function toggle() {
    if (playing) pause(); else play();
  }

  const controls = createTokenPressControls(controlsEl, {
    onBack: doBack,
    onStep: doStep,
    onPlay: play,
    onPause: pause,
    onToggle: toggle,
  });

  function pushToken(record) {
    queue.push(record);
    // Session has already advanced — commit immediately so the token renders.
    queue.stepForward();
  }

  function attachSession(s) {
    session = s;
    updateControls();
  }

  function clear() {
    queue.clear();
    renderer.clear();
    updateControls();
  }

  function dispose() {
    playing = false;
    queue.dispose();
    renderer.dispose();
    controls.dispose();
    session = null;
  }

  updateControls();

  return {
    pushToken,
    attachSession,
    play,
    pause,
    clear,
    dispose,
    get queue() { return queue; },
    get playing() { return playing; },
  };
}
