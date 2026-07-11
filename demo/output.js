import { state } from './ui/state.js';

function $(id) { return document.getElementById(id); }

function setText(id, text) {
  const el = $(id);
  if (el) el.textContent = text;
}

function scrollChatToLatest() {
  const surface = document.querySelector('.chat-surface');
  if (surface) surface.scrollTop = surface.scrollHeight;
}

function createChatMessage(message) {
  const article = document.createElement('article');
  const role = message?.role === 'user' ? 'user' : 'assistant';
  article.className = `chat-message chat-message--${role}`;

  const label = document.createElement('div');
  label.className = 'chat-role';
  label.textContent = role === 'user' ? 'You' : 'Doppler';
  article.appendChild(label);

  const body = document.createElement('div');
  body.className = 'chat-message-text';
  body.textContent = typeof message?.content === 'string' ? message.content : '';
  article.appendChild(body);
  return article;
}

function resetLiveAssistant() {
  const liveMessage = $('live-assistant-message');
  const output = $('output-text');
  if (output) output.textContent = '';
  if (liveMessage) liveMessage.hidden = true;
  showTokenPress(false);
}

export function renderChatMessages(messages) {
  resetLiveAssistant();
  const thread = $('chat-thread');
  if (!thread) return;
  thread.innerHTML = '';
  const visibleMessages = Array.isArray(messages) ? messages : [];
  if (visibleMessages.length === 0) {
    const empty = document.createElement('p');
    empty.id = 'chat-empty';
    empty.className = 'chat-empty';
    empty.textContent = 'Start a conversation.';
    thread.appendChild(empty);
    return;
  }
  for (const message of visibleMessages) {
    thread.appendChild(createChatMessage(message));
  }
  scrollChatToLatest();
}

export function beginChatTurn(messages) {
  renderChatMessages(messages);
  const liveMessage = $('live-assistant-message');
  const output = $('output-text');
  if (output) output.textContent = '';
  if (liveMessage) liveMessage.hidden = false;
  scrollChatToLatest();
}

export function renderImportedChat(output, prompt = null) {
  const messages = [];
  if (typeof prompt === 'string' && prompt.trim()) {
    messages.push({ role: 'user', content: prompt.trim() });
  }
  if (typeof output === 'string' && output) {
    messages.push({ role: 'assistant', content: output });
  }
  resetLiveAssistant();
  renderChatMessages(messages);
}

export function setPhase(label) {
  setText('output-phase', label);
}

export function setTokSec(value) {
  if (!state.liveTokSec) return;
  setText('output-toks', value != null ? `${value.toFixed(1)} tok/s` : '');
}

export function clearTokSec() {
  setText('output-toks', '');
}

export function setPrefillProgress(percent) {
  const bar = $('output-prefill-bar');
  if (bar) bar.style.width = `${Math.min(100, Math.max(0, percent))}%`;
}

export function appendToken(text) {
  const el = $('output-text');
  const liveMessage = $('live-assistant-message');
  if (liveMessage) liveMessage.hidden = false;
  if (el) {
    el.textContent += text;
    scrollChatToLatest();
  }
}

export function clearOutput() {
  resetLiveAssistant();
  renderChatMessages([]);
  setPrefillProgress(0);
  setPhase('');
  clearTokSec();
}

export function showTokenPress(show) {
  const plain = $('output-text');
  const tpOut = $('token-press-output');
  const tpCtrl = $('token-press-controls');
  const liveMessage = $('live-assistant-message');
  if (liveMessage && show) liveMessage.hidden = false;
  if (plain) plain.hidden = show;
  if (tpOut) tpOut.hidden = !show;
  if (tpCtrl) tpCtrl.hidden = !show;
}

export function setFinalStats(stats) {
  if (!stats) return;
  const parts = [];
  if (stats.totalTokens != null) parts.push(`${stats.totalTokens} tokens`);
  if (stats.prefillMs != null) parts.push(`prefill ${stats.prefillMs.toFixed(0)}ms`);
  if (stats.decodeMs != null) parts.push(`decode ${stats.decodeMs.toFixed(0)}ms`);
  if (stats.tokPerSec != null) parts.push(`${stats.tokPerSec.toFixed(1)} tok/s`);
  setText('output-toks', parts.join(' · '));
}
