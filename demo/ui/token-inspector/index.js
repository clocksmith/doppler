// Token inspection stays separate from the readable answer.
const PAGE_SIZE = 48;
let currentTokens = [];
let selectedIndex = 0;
let inspectorActive = false;
let boundContainer = null;
let boundCardContainer = null;

function pct(value) {
  return Number.isFinite(value) ? (value * 100).toFixed(1) + '%' : 'Unavailable';
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

export function isTokenInspectorActive() { return inspectorActive; }

export function setTokenInspectorActive(active) {
  inspectorActive = Boolean(active);
  if (boundContainer) boundContainer.hidden = !inspectorActive || !currentTokens.length;
  if (boundCardContainer) boundCardContainer.hidden = !inspectorActive || !currentTokens.length;
}

export function selectToken(index, { focus = false } = {}) {
  if (!currentTokens.length || !Number.isFinite(index)) return;
  selectedIndex = Math.max(0, Math.min(currentTokens.length - 1, Math.trunc(index)));
  renderPage();
  renderTokenCard();
  if (focus) boundContainer?.querySelector('[aria-pressed="true"]')?.focus({ preventScroll: true });
}

function renderPage() {
  if (!boundContainer) return;
  const start = Math.floor(selectedIndex / PAGE_SIZE) * PAGE_SIZE;
  const end = Math.min(currentTokens.length, start + PAGE_SIZE);
  const toolbar = document.createElement('div');
  toolbar.className = 'token-page-toolbar';
  const label = document.createElement('span');
  label.textContent = 'Tokens ' + (start + 1) + '-' + end + ' of ' + currentTokens.length;
  label.setAttribute('role', 'status');
  const actions = document.createElement('div');
  actions.className = 'token-inspector-nav-btns';
  actions.append(
    button('Previous page', () => selectToken(start - PAGE_SIZE), start === 0),
    button('Next page', () => selectToken(end), end === currentTokens.length)
  );
  toolbar.append(label, actions);
  const list = document.createElement('div');
  list.className = 'token-chip-list';
  list.setAttribute('role', 'group');
  list.setAttribute('aria-label', 'Output tokens. Use arrow keys to change selection.');
  for (let index = start; index < end; index++) {
    const token = currentTokens[index];
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'token-chip';
    chip.classList.toggle('is-selected', index === selectedIndex);
    chip.dataset.tokenIndex = String(index);
    chip.tabIndex = index === selectedIndex ? 0 : -1;
    chip.setAttribute('aria-pressed', String(index === selectedIndex));
    const display = token.text === '\n' ? '\\n'
      : token.text === '' ? '[empty]'
        : String(token.text ?? '').replaceAll('\n', '\\n').replaceAll('\t', '\\t');
    chip.textContent = display || '[empty]';
    chip.title = 'Token ' + (index + 1) + ', ID ' + token.tokenId + ', probability ' + pct(token.probability);
    chip.setAttribute('aria-label', chip.title + ': ' + JSON.stringify(token.text));
    const surprise = Number.isFinite(token.probability) && token.probability > 0
      ? Math.min(1, -Math.log(token.probability) / 8) : 1;
    chip.style.setProperty('--token-surprisal', String(surprise));
    chip.addEventListener('click', () => selectToken(index, { focus: true }));
    list.append(chip);
  }
  boundContainer.replaceChildren(toolbar, list);
}

function renderTokenCard() {
  if (!boundCardContainer) return;
  const token = currentTokens[selectedIndex];
  if (!token) { boundCardContainer.replaceChildren(); return; }
  const card = document.createElement('div');
  card.className = 'token-inspector-card';
  const header = document.createElement('div');
  header.className = 'token-inspector-card-header';
  const identity = document.createElement('div');
  identity.className = 'token-inspector-primary';
  const badge = document.createElement('code');
  badge.className = 'token-inspector-badge';
  badge.textContent = JSON.stringify(token.text ?? '');
  const id = document.createElement('span');
  id.className = 'token-inspector-id';
  id.textContent = 'Token ' + (selectedIndex + 1) + '/' + currentTokens.length + ' - ID ' + token.tokenId;
  identity.append(badge, id);
  const metrics = document.createElement('span');
  metrics.className = 'token-inspector-metrics';
  metrics.textContent = 'Probability ' + pct(token.probability) + ' / Surprisal '
    + (Number.isFinite(token.surprisal) ? token.surprisal.toFixed(2) + ' nats' : 'unavailable');
  header.append(identity, metrics);
  card.append(header);
  const title = document.createElement('strong');
  title.className = 'token-candidates-title';
  title.textContent = 'Top alternatives';
  const candidates = document.createElement('div');
  candidates.className = 'token-candidates-list';
  for (const [rank, candidate] of (token.topCandidates || []).entries()) {
    const row = document.createElement('div');
    const selected = candidate.tokenId === token.tokenId;
    row.className = 'token-candidate-row' + (selected ? ' is-winner' : '');
    const position = document.createElement('span');
    position.textContent = String(rank + 1);
    const text = document.createElement('code');
    text.className = 'token-candidate-text';
    text.textContent = JSON.stringify(candidate.text ?? '');
    text.title = 'ID ' + candidate.tokenId + (selected ? ' (sampled)' : '');
    const track = document.createElement('span');
    track.className = 'token-candidate-bar-bg';
    const fill = document.createElement('span');
    fill.className = 'token-candidate-bar-fill';
    fill.style.width = Math.max(0, Math.min(100, (candidate.probability ?? 0) * 100)) + '%';
    track.append(fill);
    const value = document.createElement('span');
    value.className = 'token-candidate-pct';
    value.textContent = pct(candidate.probability);
    row.append(position, text, track, value);
    candidates.append(row);
  }
  if (!candidates.childElementCount) candidates.textContent = 'No candidate probabilities recorded.';
  card.append(title, candidates);
  const nav = document.createElement('div');
  nav.className = 'token-inspector-nav';
  const jumpLabel = document.createElement('label');
  jumpLabel.textContent = 'Go to token ';
  const jump = document.createElement('input');
  jump.type = 'number';
  jump.min = '1';
  jump.max = String(currentTokens.length);
  jump.step = '1';
  jump.value = String(selectedIndex + 1);
  jump.addEventListener('change', () => {
    if (jump.checkValidity() && jump.value !== '') selectToken(jump.valueAsNumber - 1, { focus: true });
  });
  jumpLabel.append(jump);
  const actions = document.createElement('div');
  actions.className = 'token-inspector-nav-btns';
  actions.append(
    button('Previous', () => selectToken(selectedIndex - 1, { focus: true }), selectedIndex === 0),
    button('Next', () => selectToken(selectedIndex + 1, { focus: true }), selectedIndex === currentTokens.length - 1)
  );
  nav.append(jumpLabel, actions);
  card.append(nav);
  boundCardContainer.replaceChildren(card);
}

export function renderTokenInspector(tokens, streamContainer, cardContainer) {
  currentTokens = Array.isArray(tokens) ? tokens : [];
  selectedIndex = 0;
  boundContainer = streamContainer;
  boundCardContainer = cardContainer;
  streamContainer?.replaceChildren();
  cardContainer?.replaceChildren();
  if (!streamContainer) return;
  streamContainer.onkeydown = (event) => {
    if (!inspectorActive || !event.target.closest('.token-chip') || event.altKey || event.ctrlKey || event.metaKey) return;
    const targets = {
      ArrowLeft: selectedIndex - 1,
      ArrowRight: selectedIndex + 1,
      Home: 0,
      End: currentTokens.length - 1,
      PageUp: selectedIndex - PAGE_SIZE,
      PageDown: selectedIndex + PAGE_SIZE,
    };
    if (!(event.key in targets)) return;
    event.preventDefault();
    selectToken(targets[event.key], { focus: true });
  };
  if (currentTokens.length) selectToken(0);
  setTokenInspectorActive(inspectorActive);
}
