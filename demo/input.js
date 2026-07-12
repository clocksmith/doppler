import { state } from './ui/state.js';
import {
  appendConversationTurn,
  createConversationRequest,
  normalizeConversationHistory,
} from './conversation.js';
import { clearOutput, renderChatMessages } from './output.js';

let examples = null;
let shuffleIndex = -1;
let imageData = null;
let onRun = null;

function $(id) { return document.getElementById(id); }

function syncClearChatButton() {
  const clearButton = $('clear-history-btn');
  if (clearButton) clearButton.disabled = state.conversationHistory.length === 0;
}

function setupConversationActions() {
  $('clear-history-btn')?.addEventListener('click', clearConversationHistory);
  syncClearChatButton();
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
  if (promptEl) {
    promptEl.value = examples.text[shuffleIndex];
    syncSendButton();
  }
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
      zone.classList.add('has-image');
      zone.setAttribute('aria-pressed', 'true');
      zone.title = `Attached: ${file.name}`;
    }
  };
  reader.readAsDataURL(file);
}

export function clearImage() {
  imageData = null;
  const zone = $('image-drop');
  if (zone) {
    zone.textContent = 'Attach image';
    zone.classList.remove('has-image');
    zone.setAttribute('aria-pressed', 'false');
    zone.title = 'Attach an image, or drop an image on this button';
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
  syncSendButton();
}

export function clearPrompt() {
  const promptEl = $('prompt-input');
  if (promptEl) {
    promptEl.value = '';
    syncSendButton();
  }
}

export function getImage() {
  return imageData;
}

export function buildConversationRequest(prompt, options = {}) {
  return createConversationRequest(state.conversationHistory, prompt, {
    templateType: options.templateType ?? null,
    translation: options.translation,
  });
}

export function recordConversationTurn(request, output) {
  state.conversationHistory = appendConversationTurn(state.conversationHistory, request, output);
  renderChatMessages(state.conversationHistory);
  syncClearChatButton();
}

export function clearConversationHistory() {
  state.conversationHistory = [];
  clearOutput();
  syncClearChatButton();
}

export function restoreConversationHistory(messages) {
  state.conversationHistory = normalizeConversationHistory(messages);
  renderChatMessages(state.conversationHistory);
  syncClearChatButton();
}

export function resetConversationForModel(modelId) {
  const nextModelId = typeof modelId === 'string' && modelId.trim() ? modelId : null;
  if (state.conversationModelId === nextModelId) return;
  state.conversationModelId = nextModelId;
  syncClearChatButton();
}

export function setRunHandler(handler) {
  onRun = handler;
}

export function isSendReady({ pipeline, prompt, generating = false, prefilling = false }) {
  return pipeline != null
    && typeof prompt === 'string'
    && prompt.trim().length > 0
    && generating !== true
    && prefilling !== true;
}

export function syncSendButton(options = {}) {
  const btn = $('run-btn');
  const prompt = getPrompt();
  const generating = options.generating ?? state.generating;
  const prefilling = options.prefilling ?? state.prefilling;
  const ready = isSendReady({
    pipeline: state.pipeline,
    prompt,
    generating,
    prefilling,
  });
  if (!btn) return ready;

  btn.disabled = !ready;
  if (state.pipeline == null) {
    btn.title = 'Load a model to send';
  } else if (!prompt) {
    btn.title = 'Enter a message to send';
  } else if (generating || prefilling) {
    btn.title = 'Generation is in progress';
  } else {
    btn.title = 'Send message';
  }
  return ready;
}

function submitIfReady() {
  if (syncSendButton() && onRun) {
    onRun();
  }
}

export async function initInput() {
  await loadExamples();

  $('shuffle-btn')?.addEventListener('click', shuffle);

  $('run-btn')?.addEventListener('click', submitIfReady);

  $('prompt-input')?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submitIfReady();
    }
  });
  $('prompt-input')?.addEventListener('input', () => syncSendButton());

  setupImageDrop();
  setupConversationActions();

  // Start with a random example
  if (examples?.text?.length) {
    shuffleIndex = Math.floor(Math.random() * examples.text.length) - 1;
    shuffle();
  }
  syncSendButton();
}

export function setGenerating(active) {
  const runBtn = $('run-btn');
  const stopBtn = $('stop-btn');
  if (runBtn) runBtn.hidden = active;
  if (stopBtn) stopBtn.hidden = !active;
  syncSendButton({ generating: active, prefilling: active });
}
