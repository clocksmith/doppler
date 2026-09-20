// Inline token telemetry keeps the generated answer readable. Selecting an
// underlined token reveals its evidence without duplicating the whole answer.
let currentTokens = [];
let selectedIndex = -1;
let inspectorActive = false;
let boundContainer = null;
let boundCardContainer = null;

function pct(value) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : 'Unavailable';
}

function button(text, action, disabled = false) {
  const element = document.createElement('button');
  element.type = 'button';
  element.className = 'token-inspector-nav-btn';
  element.textContent = text;
  element.disabled = disabled;
  element.addEventListener('click', action);
  return element;
}

function tokenButton(token, index) {
  const element = document.createElement('button');
  element.type = 'button';
  element.className = 'token-chip';
  element.dataset.tokenIndex = String(index);
  element.tabIndex = index === selectedIndex ? 0 : -1;
  element.setAttribute('aria-pressed', String(index === selectedIndex));
  element.textContent = String(token?.text ?? '') || '\u200b';
  const surprise = Number.isFinite(token?.surprisal)
    ? Math.min(1, Math.max(0, token.surprisal / 8))
    : 0.5;
  element.style.setProperty('--token-surprisal', String(surprise));
  element.classList.toggle('token-chip--unavailable', !Number.isFinite(token?.probability));
  element.title = [
    `Token ${index + 1} · ID ${token?.tokenId ?? 'unavailable'}`,
    `Probability ${pct(token?.probability)}`,
    'Select for alternatives',
  ].join('\n');
  element.setAttribute('aria-label', `${element.title}: ${JSON.stringify(token?.text ?? '')}`);
  element.addEventListener('click', () => selectToken(index, { focus: true }));
  return element;
}

function syncSelection() {
  boundContainer?.querySelectorAll('.token-chip').forEach((element, index) => {
    const selected = index === selectedIndex;
    element.classList.toggle('is-selected', selected);
    element.setAttribute('aria-pressed', String(selected));
    element.tabIndex = selected ? 0 : -1;
  });
}

export function isTokenInspectorActive() { return inspectorActive; }

export function setTokenInspectorActive(active) {
  inspectorActive = Boolean(active);
  if (boundContainer) boundContainer.hidden = !inspectorActive || currentTokens.length === 0;
  if (boundCardContainer) {
    boundCardContainer.hidden = !inspectorActive || selectedIndex < 0 || currentTokens.length === 0;
  }
}

export function selectToken(index, { focus = false } = {}) {
  if (!currentTokens.length || !Number.isFinite(index)) return;
  selectedIndex = Math.max(0, Math.min(currentTokens.length - 1, Math.trunc(index)));
  syncSelection();
  renderTokenCard();
  setTokenInspectorActive(inspectorActive);
  if (focus) boundContainer?.querySelector('[aria-pressed="true"]')?.focus({ preventScroll: true });
}

function renderTokenCard() {
  if (!boundCardContainer) return;
  const token = currentTokens[selectedIndex];
  if (!token) {
    boundCardContainer.replaceChildren();
    return;
  }

  const card = document.createElement('aside');
  card.className = 'token-inspector-card';
  card.setAttribute('aria-label', `Evidence for token ${selectedIndex + 1}`);
  const header = document.createElement('div');
  header.className = 'token-inspector-card-header';
  const identity = document.createElement('div');
  identity.className = 'token-inspector-primary';
  const badge = document.createElement('code');
  badge.className = 'token-inspector-badge';
  badge.textContent = JSON.stringify(token.text ?? '');
  const id = document.createElement('span');
  id.className = 'token-inspector-id';
  id.textContent = `Token ${selectedIndex + 1}/${currentTokens.length} · ID ${token.tokenId ?? 'unavailable'}`;
  identity.append(badge, id);
  const metrics = document.createElement('span');
  metrics.className = 'token-inspector-metrics';
  metrics.textContent = `Probability ${pct(token.probability)} · Surprisal ${
    Number.isFinite(token.surprisal) ? `${token.surprisal.toFixed(2)} nats` : 'unavailable'
  }`;
  header.append(identity, metrics);
  card.append(header);

  const alternatives = Array.isArray(token.topCandidates) ? token.topCandidates : [];
  if (alternatives.length) {
    const details = document.createElement('details');
    details.className = 'token-candidates';
    const summary = document.createElement('summary');
    summary.textContent = 'Top alternatives';
    const list = document.createElement('div');
    list.className = 'token-candidates-list';
    for (const [rank, candidate] of alternatives.entries()) {
      const row = document.createElement('div');
      row.className = `token-candidate-row${candidate.tokenId === token.tokenId ? ' is-winner' : ''}`;
      const position = document.createElement('span');
      position.textContent = String(rank + 1);
      const text = document.createElement('code');
      text.className = 'token-candidate-text';
      text.textContent = JSON.stringify(candidate.text ?? '');
      const track = document.createElement('span');
      track.className = 'token-candidate-bar-bg';
      const fill = document.createElement('span');
      fill.className = 'token-candidate-bar-fill';
      fill.style.width = `${Math.max(0, Math.min(100, (candidate.probability ?? 0) * 100))}%`;
      track.append(fill);
      const value = document.createElement('span');
      value.className = 'token-candidate-pct';
      value.textContent = pct(candidate.probability);
      row.append(position, text, track, value);
      list.append(row);
    }
    details.append(summary, list);
    card.append(details);
  }

  const nav = document.createElement('div');
  nav.className = 'token-inspector-nav';
  nav.append(
    button('Previous', () => selectToken(selectedIndex - 1, { focus: true }), selectedIndex === 0),
    button('Next', () => selectToken(selectedIndex + 1, { focus: true }), selectedIndex === currentTokens.length - 1)
  );
  card.append(nav);
  boundCardContainer.replaceChildren(card);
}

function bindContainers(streamContainer, cardContainer) {
  boundContainer = streamContainer ?? boundContainer;
  boundCardContainer = cardContainer ?? boundCardContainer;
  if (!boundContainer) return;
  boundContainer.onkeydown = (event) => {
    if (!inspectorActive || !event.target.closest('.token-chip') || event.altKey || event.ctrlKey || event.metaKey) return;
    const targets = {
      ArrowLeft: selectedIndex - 1,
      ArrowRight: selectedIndex + 1,
      Home: 0,
      End: currentTokens.length - 1,
    };
    if (!(event.key in targets)) return;
    event.preventDefault();
    selectToken(targets[event.key], { focus: true });
  };
}

export function appendTokenInspectorToken(token, streamContainer = null, cardContainer = null) {
  bindContainers(streamContainer, cardContainer);
  if (!boundContainer || !token) return;
  const index = currentTokens.length;
  currentTokens.push(token);
  boundContainer.append(tokenButton(token, index));
  setTokenInspectorActive(inspectorActive);
}

export function renderTokenInspector(tokens, streamContainer, cardContainer) {
  bindContainers(streamContainer, cardContainer);
  currentTokens = Array.isArray(tokens) ? tokens : [];
  selectedIndex = -1;
  boundContainer?.replaceChildren(...currentTokens.map(tokenButton));
  boundCardContainer?.replaceChildren();
  setTokenInspectorActive(inspectorActive);
}
