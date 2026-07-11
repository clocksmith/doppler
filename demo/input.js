import { state } from './ui/state.js';
import {
  appendConversationTurn,
  countConversationTurns,
  createConversationRequest,
  normalizeHistoryTurnLimit,
  trimConversationHistory,
} from './conversation.js';
import { renderChatMessages } from './output.js';

let examples = null;
let shuffleIndex = -1;
let imageData = null;
let onRun = null;

function $(id) { return document.getElementById(id); }

function syncConversationControls() {
  const historyToggle = $('history-toggle');
  const historyLimit = $('history-limit');
  const historyStatus = $('history-status');
  const clearButton = $('clear-history-btn');
  const turnCount = countConversationTurns(state.conversationHistory);
  const activeTurns = Math.min(turnCount, state.historyTurnLimit);

  if (historyToggle) historyToggle.checked = state.historyEnabled === true;
  if (historyLimit) historyLimit.value = String(state.historyTurnLimit);
  if (historyStatus) {
    if (turnCount === 0) {
      historyStatus.textContent = '0 turns';
    } else if (state.historyEnabled) {
      historyStatus.textContent = `${turnCount} saved · using ${activeTurns}`;
    } else {
      historyStatus.textContent = `${turnCount} saved · paused`;
    }
  }
  if (clearButton) clearButton.disabled = turnCount === 0;
}

function setupConversationControls() {
  $('history-toggle')?.addEventListener('change', (event) => {
    state.historyEnabled = event.target.checked === true;
    syncConversationControls();
  });
  $('history-limit')?.addEventListener('change', (event) => {
    state.historyTurnLimit = normalizeHistoryTurnLimit(event.target.value);
    syncConversationControls();
  });
  $('clear-history-btn')?.addEventListener('click', clearConversationHistory);
  syncConversationControls();
}

async function loadExamples() {
  try {
    const url = new URL('./examples.json', import.meta.url).toString();
    const res = await fetch(url);
    examples = await res.json();
  } catch {
    examples = { text: ['hello world'], image: [] };
  }
}

function shuffle() {
  if (!examples?.text?.length) return;
  shuffleIndex = (shuffleIndex + 1) % examples.text.length;
  const promptEl = $('prompt-input');
  if (promptEl) promptEl.value = examples.text[shuffleIndex];
}

function setupImageDrop() {
  const zone = $('image-drop');
  if (!zone) return;

  zone.addEventListener('dragover', (e) => {
    e.preventDefault();
    zone.classList.add('drag-over');
  });

  zone.addEventListener('dragleave', () => {
    zone.classList.remove('drag-over');
  });

  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const file = e.dataTransfer?.files?.[0];
    if (file && file.type.startsWith('image/')) {
      readImageFile(file);
    }
  });

  zone.addEventListener('click', () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.onchange = () => {
      const file = input.files?.[0];
      if (file) readImageFile(file);
    };
    input.click();
  });
}

function readImageFile(file) {
  const reader = new FileReader();
  reader.onload = () => {
    imageData = reader.result;
    const zone = $('image-drop');
    if (zone) {
      zone.innerHTML = '';
      zone.classList.add('has-image');
      const img = document.createElement('img');
      img.src = imageData;
      img.alt = 'Attached image';
      zone.appendChild(img);
    }
  };
  reader.readAsDataURL(file);
}

export function clearImage() {
  imageData = null;
  const zone = $('image-drop');
  if (zone) {
    zone.textContent = 'Drop image here';
    zone.classList.remove('has-image');
  }
}

export function getPrompt() {
  return ($('prompt-input')?.value ?? '').trim();
}

export function setPromptValue(value) {
  const promptEl = $('prompt-input');
  if (!promptEl) {
    return;
  }
  promptEl.value = typeof value === 'string' ? value : String(value ?? '');
  promptEl.focus();
}

export function clearPrompt() {
  const promptEl = $('prompt-input');
  if (promptEl) promptEl.value = '';
}

export function getImage() {
  return imageData;
}

export function buildConversationRequest(prompt, options = {}) {
  return createConversationRequest(state.conversationHistory, prompt, {
    historyEnabled: state.historyEnabled,
    turnLimit: state.historyTurnLimit,
    templateType: options.templateType ?? null,
    translation: options.translation,
  });
}

export function recordConversationTurn(request, output) {
  state.conversationHistory = appendConversationTurn(state.conversationHistory, request, output);
  const visibleMessages = request?.historyEnabled === true
    ? state.conversationHistory
    : [
      { role: 'user', content: request?.currentPrompt ?? '' },
      { role: 'assistant', content: output ?? '' },
    ];
  renderChatMessages(visibleMessages);
  syncConversationControls();
}

export function clearConversationHistory() {
  state.conversationHistory = [];
  renderChatMessages([]);
  syncConversationControls();
}

export function restoreConversationHistory(messages, options = {}) {
  state.conversationHistory = trimConversationHistory(messages);
  if (typeof options.historyEnabled === 'boolean') {
    state.historyEnabled = options.historyEnabled;
  }
  if (options.turnLimit != null) {
    state.historyTurnLimit = normalizeHistoryTurnLimit(options.turnLimit);
  }
  renderChatMessages(state.conversationHistory);
  syncConversationControls();
}

export function resetConversationForModel(modelId) {
  const nextModelId = typeof modelId === 'string' && modelId.trim() ? modelId : null;
  if (state.conversationModelId === nextModelId) return;
  state.conversationModelId = nextModelId;
  syncConversationControls();
}

export function setRunHandler(handler) {
  onRun = handler;
}

export async function initInput() {
  await loadExamples();

  $('shuffle-btn')?.addEventListener('click', shuffle);

  $('run-btn')?.addEventListener('click', () => {
    if (onRun) onRun();
  });

  $('prompt-input')?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (onRun) onRun();
    }
  });

  setupImageDrop();
  setupConversationControls();

  // Start with a random example
  if (examples?.text?.length) {
    shuffleIndex = Math.floor(Math.random() * examples.text.length) - 1;
    shuffle();
  }
}

export function setRunEnabled(enabled) {
  const btn = $('run-btn');
  if (btn) btn.disabled = !enabled;
}

export function setGenerating(active) {
  const runBtn = $('run-btn');
  const stopBtn = $('stop-btn');
  if (runBtn) runBtn.hidden = active;
  if (stopBtn) stopBtn.hidden = !active;
}
