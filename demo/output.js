import { state } from './ui/state.js';
import {
  appendTokenInspectorToken,
  renderTokenInspector,
  setTokenInspectorActive,
} from './ui/token-inspector/index.js';
import { renderChatMarkdown } from './ui/chat-markdown.js';

function $(id) { return document.getElementById(id); }

function setText(id, text) {
  const el = $(id);
  if (el) el.textContent = text;
}

function scrollChatToLatest() {
  const surface = document.querySelector('.chat-surface');
  if (surface) surface.scrollTop = surface.scrollHeight;
}

function createChatMessage(message, quality = null) {
  const article = document.createElement('article');
  const role = message?.role === 'user' ? 'user' : 'assistant';
  article.className = `chat-message chat-message--${role}`;

  const label = document.createElement('div');
  label.className = 'chat-role';
  label.textContent = role === 'user' ? 'You' : 'Doppler';
  article.appendChild(label);

  const body = document.createElement('div');
  body.className = 'chat-message-text';
  const content = typeof message?.content === 'string' ? message.content : '';
  if (role === 'assistant') renderChatMarkdown(body, content, quality);
  else body.textContent = content;
  article.appendChild(body);
  return article;
}

function createEmptyState() {
  const empty = document.createElement('div');
  empty.id = 'chat-empty';
  empty.className = 'chat-empty';

  const mark = document.createElement('span');
  mark.className = 'chat-empty-mark';
  mark.setAttribute('aria-hidden', 'true');
  mark.textContent = 'D';

  const heading = document.createElement('strong');
  heading.textContent = state.model ? 'Start a conversation.' : 'Load a model to begin.';

  empty.append(mark, heading);
  return empty;
}

function resetLiveAssistant() {
  const liveMessage = $('live-assistant-message');
  const output = $('output-text');
  if (output) {
    output.textContent = '';
    output.classList.remove('chat-markdown');
  }
  $('word-quality-output')?.replaceChildren();
  $('word-quality-legend')?.setAttribute('hidden', '');
  renderTokenInspection([]);
  if (liveMessage) liveMessage.hidden = true;
  showWordQuality(false);
  const inspectorView = $('token-inspector-view');
  if (inspectorView) inspectorView.hidden = true;
  setTokenInspectorActive(false);
}

export function renderChatMessages(messages, annotations = state.conversationAnnotations) {
  resetLiveAssistant();
  const thread = $('chat-thread');
  if (!thread) return;
  thread.innerHTML = '';
  const visibleMessages = Array.isArray(messages) ? messages : [];
  if (visibleMessages.length === 0) {
    thread.appendChild(createEmptyState());
    return;
  }
  for (const [index, message] of visibleMessages.entries()) {
    const annotation = annotations[index];
    const quality = state.wordQualityEnabled && annotation?.text === message.content
      ? annotation.quality : null;
    thread.appendChild(createChatMessage(message, quality));
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
  renderChatMessages(messages, []);
}

export function setPhase(label) {
  setText('output-phase', label);
}

export function clearTokSec() {
  setText('output-toks', '');
}

export function setPrefillProgress(percent) {
  const bar = $('output-prefill-bar');
  if (bar) bar.style.width = `${Math.min(100, Math.max(0, percent))}%`;
}

export function createOutputStream(decodeTokenIds, signal) {
  const output = $('output-text');
  const surface = document.querySelector('.chat-surface');
  const liveMessage = $('live-assistant-message');
  const textNode = document.createTextNode('');
  output.classList.remove('chat-markdown');
  output.replaceChildren(textNode);
  renderTokenInspection([]);
  setTokenInspectorActive(state.tokenInspectorActive);
  liveMessage?.setAttribute('aria-busy', 'true');
  const tokenIds = [];
  let frame = null;
  let closed = false;
  let dirty = false;

  function updateText(text) {
    const previous = textNode.data;
    if (text === previous) return;
    const follow = surface && surface.scrollHeight - surface.scrollTop - surface.clientHeight <= 2;
    if (text.startsWith(previous)) {
      textNode.appendData(text.slice(previous.length));
    } else {
      // Tokenizer decoding can revise an unfinished byte sequence or whitespace.
      let prefix = 0;
      while (prefix < previous.length && prefix < text.length && previous[prefix] === text[prefix]) prefix++;
      textNode.replaceData(prefix, previous.length - prefix, text.slice(prefix));
    }
    if (follow) surface.scrollTop = surface.scrollHeight;
  }

  function flush() {
    frame = null;
    if (!dirty) return;
    dirty = false;
    // Decode together, once per paint, so split Unicode and tokenizer spacing
    // stay consistent with the final receipt. Never decode each ID separately.
    updateText(decodeTokenIds(tokenIds).replace(/\uFFFD+$/u, ''));
  }

  function finish(finalText) {
    if (closed) return textNode.data;
    closed = true;
    signal?.removeEventListener('abort', onAbort);
    if (frame !== null) cancelAnimationFrame(frame);
    frame = null;
    try {
      if (typeof finalText === 'string') updateText(finalText);
      else flush();
      renderChatMarkdown(output, textNode.data);
    } finally {
      liveMessage?.setAttribute('aria-busy', 'false');
    }
    return textNode.data;
  }

  function onAbort() { finish(); }
  signal?.addEventListener('abort', onAbort, { once: true });
  if (signal?.aborted) finish();

  return {
    push(tokenId, token = null) {
      if (closed) return;
      tokenIds.push(tokenId);
      if (token) {
        appendTokenInspectorToken(token, $('token-stream-container'), $('token-inspector-card-container'));
        if (state.tokenInspectorActive) showTokenInspectorView(true);
      }
      dirty = true;
      if (frame === null) frame = requestAnimationFrame(flush);
    },
    finish,
  };
}

export function clearOutput() {
  resetLiveAssistant();
  renderChatMessages([]);
  setPrefillProgress(0);
  setPhase('');
  clearTokSec();
}

export function showWordQuality(show) {
  const plain = $('output-text');
  const qualityOutput = $('word-quality-output');
  const liveMessage = $('live-assistant-message');
  const hasTokens = Boolean($('token-stream-container')?.childElementCount);
  const tokensVisible = state.tokenInspectorActive && hasTokens;
  if (liveMessage && show) liveMessage.hidden = false;
  if (plain) plain.hidden = show || tokensVisible;
  if (qualityOutput) qualityOutput.hidden = !show || tokensVisible;
  const legend = $('word-quality-legend');
  if (legend) legend.hidden = !show || tokensVisible;
}

export function renderWordQuality(quality, text = state.lastInspection?.outputText ?? '') {
  const output = $('word-quality-output');
  if (output) renderChatMarkdown(output, text, quality);
}

export function showTokenInspectorView(show) {
  const plain = $('output-text');
  const qualityOutput = $('word-quality-output');
  const legend = $('word-quality-legend');
  const inspectorView = $('token-inspector-view');
  const hasTokens = Boolean($('token-stream-container')?.childElementCount);
  const visible = Boolean(show && hasTokens);
  if (inspectorView) inspectorView.hidden = !visible;
  setTokenInspectorActive(visible);
  if (visible) {
    if (plain) plain.hidden = true;
    if (qualityOutput) qualityOutput.hidden = true;
    if (legend) legend.hidden = true;
    return;
  }
  const showQuality = state.wordQualityEnabled && Boolean(qualityOutput?.childElementCount);
  if (plain) plain.hidden = showQuality;
  if (qualityOutput) qualityOutput.hidden = !showQuality;
  if (legend) legend.hidden = !showQuality;
}

export function renderTokenInspection(tokens) {
  const streamContainer = $('token-stream-container');
  const cardContainer = $('token-inspector-card-container');
  renderTokenInspector(tokens, streamContainer, cardContainer);
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
