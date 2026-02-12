import { state } from './state.js';
import { $, setText } from './dom.js';

const STATUS_CLASSES = ['status-success', 'status-warning', 'status-error', 'status-info'];

export function showErrorModal(message) {
  const modal = $('error-modal');
  const messageEl = $('error-message');
  if (!modal || !messageEl) return;
  setText(messageEl, message ? String(message) : 'Unknown error');
  modal.hidden = false;
}

export function hideErrorModal() {
  const modal = $('error-modal');
  if (!modal) return;
  modal.hidden = true;
}

export function setStatusIndicator(message, tone) {
  const indicator = $('status-indicator');
  if (!indicator) return;
  const textEl = indicator.querySelector('.status-text');
  const dot = indicator.querySelector('.status-dot');
  setText(textEl, message);
  indicator.classList.remove(...STATUS_CLASSES);
  if (tone) {
    indicator.classList.add(`status-${tone}`);
  }
  if (dot) {
    if (tone) {
      dot.classList.add('status-dot-filled');
    } else {
      dot.classList.remove('status-dot-filled');
    }
  }
}

export function updateStatusIndicator() {
  if (state.runLoading) {
    setStatusIndicator('Loading model', 'info');
    return;
  }
  if (state.diffusionLoading) {
    setStatusIndicator('Loading diffusion', 'info');
    return;
  }
  if (state.energyLoading) {
    setStatusIndicator('Loading energy', 'info');
    return;
  }
  if (state.convertActive) {
    setStatusIndicator('Converting', 'info');
    return;
  }
  if (state.runGenerating) {
    if (state.uiMode === 'embedding') {
      setStatusIndicator('Embedding', 'info');
      return;
    }
    setStatusIndicator('Generating', 'info');
    return;
  }
  if (state.diffusionGenerating) {
    setStatusIndicator('Generating', 'info');
    return;
  }
  if (state.energyGenerating) {
    setStatusIndicator('Running energy', 'info');
    return;
  }
  if (state.downloadActive) {
    setStatusIndicator('Downloading', 'info');
    return;
  }
  setStatusIndicator('Ready', 'success');
}

export function clampPercent(value) {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, value));
}

export function setBarWidth(id, percent) {
  const el = $(id);
  if (!el) return;
  el.style.width = `${clampPercent(percent)}%`;
}
