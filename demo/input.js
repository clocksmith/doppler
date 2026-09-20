import { state } from './ui/state.js';
import {
  appendConversationTurn,
  createConversationRequest,
  normalizeConversationHistory,
} from './conversation.js';
import { clearOutput, renderChatMessages, setPhase } from './output.js';
import { syncModelControls } from './models.js';

let examples = null;
let shuffleIndex = -1;
let onRun = null;

function $(id) { return document.getElementById(id); }

function syncClearChatButton() {
  const clearButton = $('clear-history-btn');
  if (clearButton) clearButton.disabled = state.generating || state.prefilling
    || state.modelBusy || state.settingsBusy
    || (state.conversationHistory.length === 0 && !state.lastRun && !state.model);
}

function setupConversationActions() {
  $('clear-history-btn')?.addEventListener('click', () => { void clearConversation(); });
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

export function buildConversationRequest(prompt, options = {}) {
  return createConversationRequest(state.conversationHistory, prompt, {
    templateType: options.templateType ?? null,
    translation: options.translation,
  });
}

export function recordConversationTurn(request, output, { render = true, quality = null } = {}) {
  const previousLength = state.conversationHistory.length;
  state.conversationHistory = appendConversationTurn(state.conversationHistory, request, output);
  if (state.conversationHistory.length > previousLength) {
    const index = state.conversationHistory.length - 1;
    state.conversationAnnotations[index] = {
      text: state.conversationHistory[index].content,
      quality,
    };
  }
  if (render) renderChatMessages(state.conversationHistory);
  syncClearChatButton();
}

export async function clearConversation() {
  if (state.generating || state.prefilling || state.modelBusy || state.settingsBusy) return;
  state.modelBusy = true;
  syncModelControls();
  syncClearChatButton();
  try {
    if (state.model) {
      if (typeof state.model.resetGenerationState !== 'function') {
        throw new Error('This runtime cannot reset model state. Reload the updated demo.');
      }
      await state.model.resetGenerationState();
    }
    clearConversationHistory();
    clearPrompt();
    setPhase('Conversation and model state cleared');
  } catch (error) {
    setPhase(`Clear failed: ${error?.message || error}`);
  } finally {
    state.modelBusy = false;
    syncModelControls();
    syncClearChatButton();
  }
}

export function clearConversationHistory() {
  if (state.generating || state.prefilling) return;
  state.conversationHistory = [];
  state.conversationAnnotations = [];
  state.lastRun = null;
  state.lastImportedReport = null;
  state.lastInspection = null;
  state.lastInferenceStats = null;
  delete globalThis.__DOPPLER_DEMO_EVIDENCE__;
  $('xray-container')?.replaceChildren();
  const exportButton = $('export-btn');
  if (exportButton) exportButton.disabled = true;
  clearOutput();
  syncClearChatButton();
}

export function restoreConversationHistory(messages) {
  state.conversationHistory = normalizeConversationHistory(messages);
  state.conversationAnnotations = [];
  renderChatMessages(state.conversationHistory);
  syncClearChatButton();
}

export function resetConversationForModel(modelId) {
  const nextModelId = typeof modelId === 'string' && modelId.trim() ? modelId : null;
  if (state.conversationModelId === nextModelId) return;
  state.conversationModelId = nextModelId;
  clearConversationHistory();
}

export function setRunHandler(handler) {
  onRun = handler;
}

export function isSendReady({ model, pipeline, prompt, generating = false, prefilling = false }) {
  const activeModel = model ?? pipeline ?? null;
  return activeModel != null
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
  const ready = !state.modelBusy && !state.settingsBusy && isSendReady({
    model: state.model,
    prompt,
    generating,
    prefilling,
  });
  if (!btn) return ready;

  btn.disabled = !ready;
  if (state.model == null) {
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
  for (const control of document.querySelectorAll(
    '#set-profile, #set-max-tokens, #settings-panel input:not(:disabled), #xray-toggle-all, #set-word-quality, #import-btn'
  )) {
    if (active) control.dataset.runLocked = 'true';
    control.disabled = active;
  }
  if (!active) {
    for (const control of document.querySelectorAll('[data-run-locked]')) {
      control.disabled = false;
      delete control.dataset.runLocked;
    }
  }
  syncClearChatButton();
  syncSendButton({ generating: active, prefilling: active });
}
