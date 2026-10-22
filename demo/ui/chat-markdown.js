// Chat Markdown uses DOM construction only. HTML and image embeds stay inert.
// Source offsets keep recorded word-quality data attached to the right text.
/**
 * @param {HTMLElement} container
 * @param {string} source
 * @param {import('./chat-markdown.js').ChatMarkdownQuality | null} quality
 */
export function renderChatMarkdown(container, source, quality = null) {
  const text = String(source ?? '').replace(/\r\n?/g, '\n');
  /** @type {Array<{ start: number, end: number, word: NonNullable<import('./chat-markdown.js').ChatMarkdownQuality['words']>[number] }>} */
  const ranges = [];
  let cursor = 0;
  for (const word of quality?.words || []) {
    if (!word.text) continue;
    const start = text.indexOf(word.text, cursor);
    if (start < 0) continue;
    ranges.push({ start, end: start + word.text.length, word });
    cursor = start + word.text.length;
  }

  /** @param {HTMLElement} parent @param {string} value @param {number} offset */
  function appendText(parent, value, offset) {
    let position = 0;
    for (const range of ranges) {
      if (range.end <= offset) continue;
      if (range.start >= offset + value.length) break;
      const start = Math.max(0, range.start - offset);
      const end = Math.min(value.length, range.end - offset);
      parent.append(document.createTextNode(value.slice(position, start)));
      const span = document.createElement('span');
      const word = range.word;
      span.className = 'word-quality';
      span.textContent = value.slice(start, end);
      const perplexity = word.rollingPerplexity;
      const available = typeof perplexity === 'number' && Number.isFinite(perplexity);
      span.classList.toggle('word-quality--unavailable', !available);
      span.style.setProperty('--word-surprisal', String(
        available ? Math.min(1, Math.log1p(perplexity) / 8) : 0
      ));
      /** @param {number | null} v */
      const number = (v) => typeof v === 'number' && Number.isFinite(v) ? v.toFixed(3) : 'unavailable';
      span.dataset.quality = [
        'Perplexity measures model surprise, not factual accuracy.',
        'Rolling perplexity: ' + number(word.rollingPerplexity),
        'Window: ' + (word.rollingWindow?.size ?? '?') + ' ' + (word.rollingWindow?.unit ?? 'words'),
        'Summed surprisal: ' + number(word.summedSurprisal),
        'Sequence perplexity: ' + number(word.cumulativePerplexity),
      ].join('\n');
      parent.append(span);
      position = end;
    }
    parent.append(document.createTextNode(value.slice(position)));
  }

  /** @param {HTMLElement} parent @param {string} value @param {number} offset */
  function inline(parent, value, offset) {
    const pattern = /`([^`\n]+)`|\*\*([^*\n]+)\*\*|__([^_\n]+)__|\*([^*\n]+)\*|_([^_\n]+)_|\[([^\]\n]+)\]\(([^)\s]+)\)/g;
    let previous = 0;
    for (const match of value.matchAll(pattern)) {
      appendText(parent, value.slice(previous, match.index), offset + previous);
      let tag, content, prefix;
      if (match[1] != null) { tag = 'code'; content = match[1]; prefix = 1; }
      else if (match[2] != null || match[3] != null) { tag = 'strong'; content = match[2] ?? match[3]; prefix = 2; }
      else if (match[4] != null || match[5] != null) { tag = 'em'; content = match[4] ?? match[5]; prefix = 1; }
      else { tag = 'a'; content = match[6]; prefix = 1; }
      const element = document.createElement(tag);
      if (tag === 'a') {
        try {
          const url = new URL(match[7], document.baseURI);
          if (!['https:', 'http:', 'mailto:'].includes(url.protocol)) throw new Error('Unsupported link');
          element.setAttribute('href', url.href);
          element.setAttribute('target', '_blank');
          element.setAttribute('rel', 'noopener noreferrer');
        } catch {
          appendText(parent, match[0], offset + match.index);
          previous = match.index + match[0].length;
          continue;
        }
      }
      appendText(element, content, offset + match.index + prefix);
      parent.append(element);
      previous = match.index + match[0].length;
    }
    appendText(parent, value.slice(previous), offset + previous);
  }

  let offset = 0;
  const lines = text.split('\n').map((value) => {
    const row = { value, offset };
    offset += value.length + 1;
    return row;
  });
  const fragment = document.createDocumentFragment();
  const listPattern = /^(\s*)([-+*]|\d+[.)])\s+(.*)$/;
  /** @param {string} value */
  const beginsBlock = (value) => /^\s*$|^ {0,3}(?:#{1,6}\s|`{3,}|~{3,}|> ?|(?:-{3,}|\*{3,}|_{3,})\s*$)/.test(value)
    || listPattern.test(value);
  let index = 0;
  while (index < lines.length) {
    const row = lines[index];
    if (!row.value.trim()) { index++; continue; }
    const fence = row.value.match(/^ {0,3}(`{3,}|~{3,})(.*)$/);
    if (fence) {
      const pre = document.createElement('pre');
      const code = document.createElement('code');
      const language = fence[2].trim().split(/\s+/)[0];
      if (/^[\w+-]+$/.test(language)) code.dataset.language = language;
      const start = ++index;
      const close = new RegExp('^ {0,3}' + fence[1][0] + '{' + fence[1].length + ',}\\s*$');
      while (index < lines.length && !close.test(lines[index].value)) index++;
      if (start < index) {
        const end = lines[index - 1].offset + lines[index - 1].value.length;
        appendText(code, text.slice(lines[start].offset, end), lines[start].offset);
      }
      pre.append(code);
      fragment.append(pre);
      if (index < lines.length) index++;
      continue;
    }
    const heading = row.value.match(/^ {0,3}(#{1,6})\s+(.*)$/);
    if (heading) {
      const node = document.createElement('h' + Math.min(6, heading[1].length + 1));
      inline(node, heading[2], row.offset + row.value.length - heading[2].length);
      fragment.append(node);
      index++;
      continue;
    }
    if (/^ {0,3}(?:-{3,}|\*{3,}|_{3,})\s*$/.test(row.value)) {
      fragment.append(document.createElement('hr'));
      index++;
      continue;
    }
    const list = row.value.match(listPattern);
    if (list) {
      const ordered = /^\d/.test(list[2]);
      const node = document.createElement(ordered ? 'ol' : 'ul');
      if (ordered) node.setAttribute('start', String(Number.parseInt(list[2], 10)));
      while (index < lines.length) {
        const item = lines[index].value.match(listPattern);
        if (!item || /^\d/.test(item[2]) !== ordered) break;
        const li = document.createElement('li');
        inline(li, item[3], lines[index].offset + lines[index].value.length - item[3].length);
        node.append(li);
        index++;
      }
      fragment.append(node);
      continue;
    }
    const quote = row.value.match(/^ {0,3}> ?(.*)$/);
    if (quote) {
      const node = document.createElement('blockquote');
      inline(node, quote[1], row.offset + row.value.length - quote[1].length);
      fragment.append(node);
      index++;
      continue;
    }
    const start = index++;
    while (index < lines.length && !beginsBlock(lines[index].value)) index++;
    const end = lines[index - 1].offset + lines[index - 1].value.length;
    const paragraph = document.createElement('p');
    inline(paragraph, text.slice(lines[start].offset, end), lines[start].offset);
    fragment.append(paragraph);
  }
  container.classList.add('chat-markdown');
  container.replaceChildren(fragment);
  bindQualityTips(container);
}

/** @type {HTMLElement | null} */
let qualityTip = null;
/** @type {HTMLElement | null} */
let pinnedWord = null;

/** @param {HTMLElement} word */
function showQualityTip(word) {
  if (!qualityTip) {
    qualityTip = document.createElement('div');
    qualityTip.className = 'word-quality-tip';
    qualityTip.setAttribute('role', 'tooltip');
    qualityTip.hidden = true;
    document.body.append(qualityTip);
  }
  qualityTip.textContent = word.dataset.quality || '';
  qualityTip.hidden = false;
  const rect = word.getBoundingClientRect();
  const tip = qualityTip.getBoundingClientRect();
  const left = Math.min(Math.max(8, rect.left), window.innerWidth - tip.width - 8);
  const below = rect.bottom + 8 + tip.height <= window.innerHeight;
  qualityTip.style.left = `${left}px`;
  qualityTip.style.top = `${below ? rect.bottom + 8 : Math.max(8, rect.top - tip.height - 8)}px`;
}

function hideQualityTip() {
  if (qualityTip) qualityTip.hidden = true;
  pinnedWord = null;
}

// Hover shows a word's perplexity at once; a click keeps it open until the
// next click or Escape. One shared tip serves every rendered answer.
/** @param {HTMLElement} container */
function bindQualityTips(container) {
  if (container.dataset.qualityTips) return;
  container.dataset.qualityTips = 'bound';
  /** @param {Event} event */
  const wordFor = (event) => /** @type {HTMLElement | null} */ (
    event.target instanceof Element ? event.target.closest('.word-quality[data-quality]') : null
  );
  container.addEventListener('pointerover', (event) => {
    const word = wordFor(event);
    if (word && !pinnedWord) showQualityTip(word);
  });
  container.addEventListener('pointerout', (event) => {
    const word = wordFor(event);
    if (word && !pinnedWord && !word.contains(/** @type {Node | null} */ (event.relatedTarget))) hideQualityTip();
  });
  container.addEventListener('click', (event) => {
    const word = wordFor(event);
    if (!word) return;
    if (pinnedWord === word) { hideQualityTip(); return; }
    showQualityTip(word);
    pinnedWord = word;
  });
  if (!document.body.dataset.qualityTipsDismiss) {
    document.body.dataset.qualityTipsDismiss = 'bound';
    document.addEventListener('keydown', (event) => { if (event.key === 'Escape') hideQualityTip(); });
    document.addEventListener('click', (event) => {
      if (pinnedWord && !(event.target instanceof Element && event.target.closest('.word-quality'))) hideQualityTip();
    });
    window.addEventListener('scroll', () => { if (!pinnedWord) hideQualityTip(); }, true);
  }
}
