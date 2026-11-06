import { state } from './ui/state.js';
import {
  appendConversationTurn,
  createConversationRequest,
  normalizeConversationHistory,
} from './conversation.js';
import { clearOutput, renderChatMessages, setPhase } from './output.js';
import { ensurePromptModel, syncModelControls } from './models.js';

let examples = null;
// Shuffled decks of prompts, one per category plus 'all' for Surprise. A deck
// deals every prompt once before it is reshuffled.
const promptDecks = new Map();
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

function isPromptList(value) {
  return Array.isArray(value) && value.length > 0
    && value.every((prompt) => typeof prompt === 'string' && prompt.trim());
}

function validateExamples(value) {
  const { illustrative, categories } = value ?? {};
  if (typeof illustrative?.prompt !== 'string' || !illustrative.prompt.trim()
    || typeof illustrative.answer !== 'string' || !illustrative.answer.trim()) {
    throw new Error('demo/examples.json needs an illustrative prompt and answer.');
  }
  if (!Array.isArray(categories) || categories.length === 0
    || categories.some((entry) => typeof entry?.label !== 'string' || !entry.label.trim() || !isPromptList(entry.prompts))) {
    throw new Error('demo/examples.json needs categories, each with a label and a non-empty prompt bank.');
  }
  return value;
}

// The suggestion row is optional: without a valid bank it stays hidden rather
// than offering invented prompts.
async function loadExamples() {
  try {
    const url = new URL('./examples.json', import.meta.url).toString();
    const res = await fetch(url);
    if (!res.ok) throw new Error(`demo/examples.json: HTTP ${res.status}`);
    examples = validateExamples(await res.json());
  } catch (error) {
    examples = null;
    console.warn('[demo] Prompt suggestions unavailable:', error?.message ?? error);
  }
}

function shuffled(prompts) {
  const deck = [...prompts];
  for (let index = deck.length - 1; index > 0; index--) {
    const other = Math.floor(Math.random() * (index + 1));
    [deck[index], deck[other]] = [deck[other], deck[index]];
  }
  return deck;
}

// Deals the next prompt from a deck, never the one already in the composer
// while another remains.
function dealPrompt(key, prompts) {
  let deck = promptDecks.get(key);
  if (!deck?.length) {
    deck = shuffled(prompts);
    promptDecks.set(key, deck);
  }
  let prompt = deck.pop();
  if (prompt === getPrompt() && deck.length) {
    const next = deck.pop();
    deck.unshift(prompt);
    prompt = next;
  }
  return prompt;
}

function allPrompts() {
  return [...new Set(examples.categories.flatMap((entry) => entry.prompts))];
}

/** The first-visit example answer and its prompt, or null without a valid bank. */
export function getIllustrativeExample() {
  return examples?.illustrative ?? null;
}

// Preloads the composer without taking focus, so it never replaces typed or
// launched text and does not raise a mobile keyboard on load.
export function preloadPrompt(value) {
  const promptEl = $('prompt-input');
  if (!promptEl || promptEl.value.trim() || typeof value !== 'string') return;
  promptEl.value = value;
  syncSendButton();
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

  const canLoadModel = !state.model && Boolean(prompt) && !state.modelBusy && !state.settingsBusy && !generating && !prefilling;
  btn.disabled = !ready && !canLoadModel;
  btn.textContent = state.modelBusy ? 'Loading…' : 'Send';
  if (state.model == null) {
    btn.title = 'Load the model and send this prompt locally';
  } else if (!prompt) {
    btn.title = 'Enter a message to send';
  } else if (generating || prefilling) {
    btn.title = 'Generation is in progress';
  } else {
    btn.title = 'Send message';
  }
  return ready;
}

async function submitIfReady() {
  const prompt = getPrompt();
  if (!prompt || state.modelBusy || state.settingsBusy || state.generating || state.prefilling) return;
  try {
    if (!state.model) {
      setPhase('Loading model…');
      await ensurePromptModel();
      // Keep edits made during loading; never send a different prompt implicitly.
      if (getPrompt() !== prompt) {
        setPhase('Ready');
        return;
      }
    }
    if (syncSendButton() && onRun) await onRun();
  } catch (error) {
    setPhase(`Could not load model: ${error.message}`);
    syncSendButton();
  }
}

export async function initInput() {
  await loadExamples();
  const suggestions = $('prompt-suggestions');
  if (suggestions) suggestions.hidden = !examples;
  for (const entry of examples?.categories ?? []) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'btn btn-ghost btn-small';
    button.textContent = entry.label;
    button.title = `Try another ${entry.label.toLowerCase()} prompt`;
    button.addEventListener('click', () => setPromptValue(dealPrompt(entry.label, entry.prompts)));
    suggestions?.insertBefore(button, $('shuffle-btn'));
  }

  $('shuffle-btn')?.addEventListener('click', () => {
    if (examples) setPromptValue(dealPrompt('all', allPrompts()));
  });

  $('run-btn')?.addEventListener('click', submitIfReady);

  $('prompt-input')?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submitIfReady();
    }
  });
  $('prompt-input')?.addEventListener('input', () => syncSendButton());

  setupConversationActions();
  syncSendButton();
}

export function setGenerating(active) {
  const runBtn = $('run-btn');
  const stopBtn = $('stop-btn');
  if (runBtn) runBtn.hidden = active;
  if (stopBtn) stopBtn.hidden = !active;
  for (const control of document.querySelectorAll(
'#set-profile, #set-max-tokens, #settings-panel input:not(:disabled), #xray-toggle-all, #set-word-quality, #import-btn, #shuffle-btn, #prompt-suggestions button'
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
