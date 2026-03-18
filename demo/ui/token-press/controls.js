// =============================================================================
// Token Press — Controls Layer
// =============================================================================
// Play/pause/step UI and keyboard bindings.
// Delegates to a session (bridge) for forward/back, not just the queue.

export function createTokenPressControls(container, callbacks) {
  const { onBack, onStep, onPlay, onPause, onToggle } = callbacks;

  const bar = document.createElement('div');
  bar.className = 'tp-controls';

  const backBtn = document.createElement('button');
  backBtn.textContent = '\u25C0 Back';
  backBtn.title = 'Step back one token (Left arrow)';

  const stepBtn = document.createElement('button');
  stepBtn.textContent = '\u25B6 Step';
  stepBtn.title = 'Step forward one token (Right arrow / Space)';

  const playBtn = document.createElement('button');
  playBtn.textContent = '\u25B6\u25B6 Play';
  playBtn.title = 'Auto-play (Enter)';

  const pauseBtn = document.createElement('button');
  pauseBtn.textContent = '\u23F8 Pause';
  pauseBtn.title = 'Pause (Enter)';

  const position = document.createElement('span');
  position.className = 'tp-position';
  position.textContent = '0 / 0';

  bar.append(backBtn, stepBtn, playBtn, pauseBtn, position);
  container.append(bar);

  backBtn.addEventListener('click', onBack);
  stepBtn.addEventListener('click', onStep);
  playBtn.addEventListener('click', onPlay);
  pauseBtn.addEventListener('click', onPause);

  function onKeydown(e) {
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
        onToggle();
        break;
    }
  }

  document.addEventListener('keydown', onKeydown);

  function update(state) {
    const { playing, cursor, total, backDisabled, backReason } = state;
    playBtn.style.display = playing ? 'none' : '';
    pauseBtn.style.display = playing ? '' : 'none';
    backBtn.disabled = backDisabled || cursor === 0;
    backBtn.title = backReason || 'Step back one token (Left arrow)';
    position.textContent = `${cursor} / ${total}`;
  }

  function dispose() {
    document.removeEventListener('keydown', onKeydown);
    bar.remove();
  }

  return { update, dispose };
}
