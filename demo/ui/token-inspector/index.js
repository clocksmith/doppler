import { groupTokenFragments } from '../../branches.js';

// Every recorded token is underlined. Selecting one shows its alternatives;
// picking an alternative continues a live answer from that exact token.
let currentTokens = [];
let selectedIndex = -1;
let inspectorActive = false;
let boundContainer = null;
let boundCardContainer = null;
let chipIndices = [];
let renderFrame = null;
let options = {
  decodeTokenIds: null,
  forks: new Map(),
  branchable: false,
  unavailableReason: 'Only a live answer can branch. Recorded and illustrative examples cannot.',
  onBranch: null,
  onSwitch: null,
};

function pct(value) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : 'Unavailable';
}

function button(text, action, { disabled = false, className = 'btn token-inspector-nav-btn', label = null } = {}) {
  const element = document.createElement('button');
  element.type = 'button';
  element.className = className;
  element.textContent = text;
  element.disabled = disabled;
  if (label) element.setAttribute('aria-label', label);
  element.addEventListener('click', action);
  return element;
}

function tokenButton(token, index, text) {
  const element = document.createElement('button');
  element.type = 'button';
  element.className = 'token-chip';
  element.dataset.tokenIndex = String(index);
  element.tabIndex = -1;
  element.setAttribute('aria-pressed', 'false');
  const whitespace = !text.trim();
  element.classList.toggle('token-chip--space', whitespace);
  element.textContent = whitespace && text.includes('\n') ? `↵${text}` : text;
  const probability = Number.isFinite(token?.probability) && token.probability >= 0 && token.probability <= 1 ? token.probability : null;
  element.style.setProperty('--token-surprisal', String(probability === null ? .5 : 1 - probability));
  // The center retains this token's measured likelihood. Shared boundary colors
  // interpolate neighboring values for a continuous underline, not three bins.
  const neighbor = offset => {
    const p = currentTokens[index + offset]?.probability;
    return Number.isFinite(p) && p >= 0 && p <= 1 ? p : probability ?? .5;
  };
  element.style.setProperty('--token-gradient-start', String(1 - (neighbor(-1) + (probability ?? .5)) / 2));
  element.style.setProperty('--token-gradient-end', String(1 - (neighbor(1) + (probability ?? .5)) / 2));
  element.classList.toggle('token-chip--unavailable', probability === null);
  element.classList.toggle('token-chip--chosen', token?.chosen === true);
  const description = [
    `Branch point · token ${index + 1}`,
    `The model gave this token ${pct(probability)}`,
    'Click to see alternatives and branch. Likelihood, not factual accuracy.',
  ].join('\n');
  element.title = description;
  element.setAttribute('aria-label', `${description}: ${JSON.stringify(text)}`);
  element.addEventListener('focus', () => {
    if (selectedIndex !== index) selectToken(index);
  });
  element.addEventListener('click', () => selectToken(index, { focus: true }));
  return element;
}

function forkSwitcher(fork) {
  const wrapper = document.createElement('span');
  wrapper.className = 'branch-switch';
  wrapper.dataset.forkIndex = String(fork.position);
  const count = fork.members.length;
  const label = document.createElement('span');
  label.className = 'branch-switch-count';
  label.textContent = `${fork.active + 1}/${count}`;
  const go = (offset) => () => options.onSwitch?.(fork.members[(fork.active + offset + count) % count]);
  wrapper.append(
    button('‹', go(-1), { className: 'branch-switch-btn', label: `Previous branch at token ${fork.position + 1}` }),
    label,
    button('›', go(1), { className: 'branch-switch-btn', label: `Next branch at token ${fork.position + 1}` })
  );
  return wrapper;
}

function syncSelection() {
  boundContainer?.querySelectorAll('.token-chip').forEach((element) => {
    const selected = Number(element.dataset.tokenIndex) === selectedIndex;
    element.classList.toggle('is-selected', selected);
    element.setAttribute('aria-pressed', String(selected));
    element.tabIndex = selected || (selectedIndex < 0 && Number(element.dataset.tokenIndex) === chipIndices[0]) ? 0 : -1;
  });
}

function renderStream() {
  renderFrame = null;
  if (!boundContainer) return;
  const groups = groupTokenFragments(currentTokens, options.decodeTokenIds);
  const forksByGroup = new Map();
  for (const [position, fork] of options.forks) {
    const group = groups.find((entry) => entry.start <= position && position < entry.end);
    if (group) forksByGroup.set(group.start, { position, ...fork });
  }
  chipIndices = [];
  const nodes = [];
  for (const group of groups) {
    const fork = forksByGroup.get(group.start);
    if (fork) nodes.push(forkSwitcher(fork));
    if (!group.text) continue;
    chipIndices.push(group.start);
    nodes.push(tokenButton(currentTokens[group.start], group.start, group.text));
  }
  boundContainer.replaceChildren(...nodes);
  if (!chipIndices.includes(selectedIndex)) selectedIndex = -1;
  syncSelection();
  renderTokenCard();
  setTokenInspectorActive(inspectorActive);
}

function scheduleRender() {
  if (renderFrame !== null) return;
  if (typeof requestAnimationFrame === 'function') {
    renderFrame = requestAnimationFrame(renderStream);
  } else {
    renderStream();
  }
}

export function isTokenInspectorActive() { return inspectorActive; }

export function getTokenInspectorTokenCount() { return currentTokens.length; }

export function setTokenInspectorActive(active) {
  inspectorActive = Boolean(active);
  if (boundContainer) boundContainer.hidden = !inspectorActive || currentTokens.length === 0;
  if (boundCardContainer) {
    boundCardContainer.hidden = !inspectorActive || selectedIndex < 0 || currentTokens.length === 0;
  }
}

export function setTokenInspectorOptions(next = {}) {
  options = { ...options, ...next };
  if (renderFrame !== null && typeof cancelAnimationFrame === 'function') cancelAnimationFrame(renderFrame);
  renderStream();
}

export function getTokenInspectorSelection() {
  return selectedIndex;
}

export function selectToken(index, { focus = false } = {}) {
  if (!currentTokens.length || !Number.isFinite(index)) return;
  const target = chipIndices.includes(index) ? index : chipIndices.find((entry) => entry >= index) ?? chipIndices.at(-1);
  if (target === undefined) return;
  selectedIndex = target;
  syncSelection();
  renderTokenCard();
  setTokenInspectorActive(inspectorActive);
  if (focus) boundContainer?.querySelector('[aria-pressed="true"]')?.focus({ preventScroll: true });
}

function stepSelection(offset) {
  const position = chipIndices.indexOf(selectedIndex);
  const next = chipIndices[Math.max(0, Math.min(chipIndices.length - 1, position + offset))];
  if (next !== undefined) selectToken(next, { focus: true });
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
  card.setAttribute('aria-label', `Alternatives for token ${selectedIndex + 1}`);
  const header = document.createElement('div');
  header.className = 'token-inspector-card-header';
  const identity = document.createElement('div');
  identity.className = 'token-inspector-primary';
  const badge = document.createElement('code');
  badge.className = 'token-inspector-badge';
  badge.textContent = JSON.stringify(token.text ?? '');
  const id = document.createElement('span');
  id.className = 'token-inspector-id';
  id.textContent = `token ${selectedIndex + 1} / ${currentTokens.length}`;
  id.title = `Token ID ${token.tokenId ?? 'unavailable'}`;
  identity.append(badge, id);
  const metrics = document.createElement('span');
  metrics.className = 'token-inspector-metrics';
  metrics.textContent = `${pct(token.probability)} likely`;
  metrics.title = `Surprisal: ${Number.isFinite(token.surprisal) ? token.surprisal.toFixed(2) : 'unavailable'} nats`;
  header.append(identity, metrics);
  card.append(header);

  const alternatives = Array.isArray(token.topCandidates) ? token.topCandidates : [];
  const list = document.createElement('div');
  list.className = 'token-candidates-list';
  for (const [rank, candidate] of alternatives.entries()) {
    const current = candidate.tokenId === token.tokenId;
    const row = document.createElement('div');
    row.className = `token-candidate-row${current ? ' is-winner' : ''}`;
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
    const action = current
      ? Object.assign(document.createElement('span'), { className: 'token-candidate-current', textContent: 'Shown' })
      : button('Branch', () => options.onBranch?.(selectedIndex, candidate.tokenId), {
        disabled: !options.branchable || typeof options.onBranch !== 'function',
        className: 'btn btn-small token-candidate-branch',
        label: `Branch with ${JSON.stringify(candidate.text ?? '')} at token ${selectedIndex + 1}`,
      });
    row.append(position, text, track, value, action);
    list.append(row);
  }
  if (alternatives.length) card.append(list);
  const note = document.createElement('p');
  note.className = 'token-inspector-note';
  note.textContent = options.branchable
    ? 'Pick an alternative to continue the answer from here. Percentages are model likelihoods, not confidence that a statement is true.'
    : options.unavailableReason;
  card.append(note);

  const position = chipIndices.indexOf(selectedIndex);
  const nav = document.createElement('div');
  nav.className = 'token-inspector-nav';
  nav.append(
    button('Previous', () => stepSelection(-1), { disabled: position <= 0 }),
    button('Next', () => stepSelection(1), { disabled: position < 0 || position >= chipIndices.length - 1 })
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
    const steps = { ArrowLeft: -1, ArrowRight: 1, Home: -Infinity, End: Infinity };
    if (!(event.key in steps)) return;
    event.preventDefault();
    stepSelection(steps[event.key]);
  };
}

export function appendTokenInspectorToken(token, streamContainer = null, cardContainer = null) {
  bindContainers(streamContainer, cardContainer);
  if (!boundContainer || !token) return;
  currentTokens.push(token);
  scheduleRender();
}

export function renderTokenInspector(tokens, streamContainer, cardContainer) {
  bindContainers(streamContainer, cardContainer);
  currentTokens = Array.isArray(tokens) ? [...tokens] : [];
  selectedIndex = -1;
  if (renderFrame !== null && typeof cancelAnimationFrame === 'function') cancelAnimationFrame(renderFrame);
  renderStream();
}

// Re-renders a changed token list (a growing branch) without clearing the selection.
export function updateTokenInspector(tokens) {
  currentTokens = Array.isArray(tokens) ? [...tokens] : [];
  scheduleRender();
}
