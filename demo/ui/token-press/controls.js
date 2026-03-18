// =============================================================================
// Token Press — Controls Layer
// =============================================================================
// Play/pause/step UI controls and keyboard bindings for the token press.
// Thin wrapper — delegates all state to the queue layer.
//
// Usage:
//   const controls = createTokenPressControls(queue, containerEl);
//   controls.dispose();

export function createTokenPressControls(queue, container) {
  const bar = document.createElement('div');
  bar.className = 'tp-controls';

  const backBtn = document.createElement('button');
  backBtn.textContent = '◀ Back';
  backBtn.title = 'Step back one token (Left arrow)';

  const stepBtn = document.createElement('button');
  stepBtn.textContent = '▶ Step';
  stepBtn.title = 'Step forward one token (Right arrow / Space)';

  const playBtn = document.createElement('button');
  playBtn.textContent = '▶▶ Play';
  playBtn.title = 'Auto-play (Enter)';

  const pauseBtn = document.createElement('button');
  pauseBtn.textContent = '⏸ Pause';
  pauseBtn.title = 'Pause playback (Enter)';

  const position = document.createElement('span');
  position.className = 'tp-position';
  position.textContent = '0 / 0';

  bar.append(backBtn, stepBtn, playBtn, pauseBtn, position);
  container.append(bar);

  function updateState() {
    const isPlaying = queue.playing;
    playBtn.style.display = isPlaying ? 'none' : '';
    pauseBtn.style.display = isPlaying ? '' : 'none';
    backBtn.disabled = queue.cursor === 0;
    position.textContent = `${queue.cursor} / ${queue.total}`;
  }

  function onBack() {
    queue.stepBack();
    updateState();
  }

  function onStep() {
    queue.stepForward();
    updateState();
  }

  function onPlay() {
    queue.play();
    updateState();
  }

  function onPause() {
    queue.pause();
    updateState();
  }

  function onTogglePlay() {
    if (queue.playing) {
      onPause();
    } else {
      onPlay();
    }
  }

  backBtn.addEventListener('click', onBack);
  stepBtn.addEventListener('click', onStep);
  playBtn.addEventListener('click', onPlay);
  pauseBtn.addEventListener('click', onPause);

  function onKeydown(e) {
    // Only handle when not typing in an input/textarea
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    switch (e.key) {
      case 'ArrowLeft':
        e.preventDefault();
        onBack();
        break;
      case 'ArrowRight':
      case ' ':
        e.preventDefault();
        onStep();
        break;
      case 'Enter':
        e.preventDefault();
        onTogglePlay();
        break;
    }
  }

  document.addEventListener('keydown', onKeydown);

  // Initial state
  updateState();

  // Allow external update (called after queue flushes)
  function refresh() {
    updateState();
  }

  function dispose() {
    document.removeEventListener('keydown', onKeydown);
    backBtn.removeEventListener('click', onBack);
    stepBtn.removeEventListener('click', onStep);
    playBtn.removeEventListener('click', onPlay);
    pauseBtn.removeEventListener('click', onPause);
    bar.remove();
  }

  return {
    refresh,
    dispose,
  };
}
