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
  if (output) output.textContent = '';
  if (liveMessage) liveMessage.hidden = true;
  showWordQuality(false);
}

export function renderChatMessages(messages) {
  resetLiveAssistant();
  const thread = $('chat-thread');
  if (!thread) return;
  thread.innerHTML = '';
  const visibleMessages = Array.isArray(messages) ? messages : [];
  if (visibleMessages.length === 0) {
    thread.appendChild(createEmptyState());
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
  output.replaceChildren(textNode);
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
    } finally {
      liveMessage?.setAttribute('aria-busy', 'false');
    }
    return textNode.data;
  }

  function onAbort() { finish(); }
  signal?.addEventListener('abort', onAbort, { once: true });
  if (signal?.aborted) finish();

  return {
    push(tokenId) {
      if (closed) return;
      tokenIds.push(tokenId);
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
  if (liveMessage && show) liveMessage.hidden = false;
  if (plain) plain.hidden = show;
  if (qualityOutput) qualityOutput.hidden = !show;
}

export function renderWordQuality(quality) {
  const output = $('word-quality-output');
  if (!output) return;
  output.replaceChildren();
  const words = Array.isArray(quality?.words) ? quality.words : [];
  for (const word of words) {
    const span = document.createElement('span');
    span.className = 'word-quality';
    span.textContent = word.text;
    const height = Number.isFinite(word.rollingPerplexity)
      ? Math.min(1, Math.log1p(word.rollingPerplexity) / 8)
      : 0;
    span.style.setProperty('--word-surprisal', String(height));
    span.title = [
      `Summed word surprisal: ${Number.isFinite(word.summedSurprisal) ? word.summedSurprisal.toFixed(4) : 'unavailable'}`,
      `Rolling perplexity (${word.rollingWindow.size} ${word.rollingWindow.unit}): ${Number.isFinite(word.rollingPerplexity) ? word.rollingPerplexity.toFixed(4) : 'unavailable'}`,
      `Cumulative sequence perplexity: ${Number.isFinite(word.cumulativePerplexity) ? word.cumulativePerplexity.toFixed(4) : 'unavailable'}`,
      `Subword tokens: ${word.tokenCount}`,
    ].join('\n');
    output.append(span, document.createTextNode(' '));
  }
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
