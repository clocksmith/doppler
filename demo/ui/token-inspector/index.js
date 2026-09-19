// Doppler Interactive Token Generation Inspector Component

let currentTokens = [];
let selectedIndex = 0;
let inspectorActive = false;
let boundContainer = null;
let boundCardContainer = null;

function formatPct(val) {
  if (typeof val !== 'number' || !Number.isFinite(val)) return '—';
  return `${(val * 100).toFixed(1)}%`;
}

function formatSurprisal(val) {
  if (typeof val !== 'number' || !Number.isFinite(val)) return '—';
  return `${val.toFixed(2)} nats`;
}

function getConfidenceClass(probability) {
  if (typeof probability !== 'number') return 'conf-med';
  if (probability >= 0.85) return 'conf-high';
  if (probability >= 0.55) return 'conf-med';
  if (probability >= 0.25) return 'conf-low';
  return 'conf-rare';
}

function formatDisplayToken(text) {
  if (!text) return '∅';
  // Represent leading spaces visibly
  if (text.startsWith(' ')) {
    return `·${text.slice(1)}`;
  }
  if (text === '\n') return '↵\\n';
  return text;
}

export function isTokenInspectorActive() {
  return inspectorActive;
}

export function setTokenInspectorActive(active) {
  inspectorActive = Boolean(active);
  if (boundContainer) {
    boundContainer.hidden = !inspectorActive;
  }
  if (boundCardContainer) {
    boundCardContainer.hidden = !inspectorActive || !currentTokens.length;
  }
}

export function selectToken(index) {
  if (!currentTokens.length) return;
  selectedIndex = Math.max(0, Math.min(currentTokens.length - 1, index));
  
  if (boundContainer) {
    const chips = boundContainer.querySelectorAll('.token-chip');
    chips.forEach((chip, idx) => {
      chip.classList.toggle('is-selected', idx === selectedIndex);
      if (idx === selectedIndex) {
        chip.scrollIntoView({ block: 'nearest', inline: 'nearest', behavior: 'smooth' });
      }
    });
  }

  renderTokenCard();
}

function renderTokenCard() {
  if (!boundCardContainer) return;
  const token = currentTokens[selectedIndex];
  if (!token) {
    boundCardContainer.replaceChildren();
    return;
  }

  const card = document.createElement('div');
  card.className = 'token-inspector-card';

  // Card Header
  const header = document.createElement('div');
  header.className = 'token-inspector-card-header';

  const primary = document.createElement('div');
  primary.className = 'token-inspector-primary';

  const badge = document.createElement('span');
  badge.className = 'token-inspector-badge';
  badge.textContent = JSON.stringify(token.text);

  const idInfo = document.createElement('span');
  idInfo.className = 'token-inspector-id';
  idInfo.textContent = `ID #${token.tokenId ?? '—'} · Step ${selectedIndex + 1}/${currentTokens.length}`;

  primary.append(badge, idInfo);

  const metrics = document.createElement('div');
  metrics.className = 'token-inspector-metrics';

  const confPill = document.createElement('div');
  confPill.className = 'token-metric-pill';
  confPill.innerHTML = `
    <span class="token-metric-label">Confidence</span>
    <span class="token-metric-val" style="color: ${token.probability >= 0.8 ? '#059669' : (token.probability >= 0.5 ? '#2563eb' : '#d97706')}">${formatPct(token.probability)}</span>
  `;

  const surprisalPill = document.createElement('div');
  surprisalPill.className = 'token-metric-pill';
  surprisalPill.innerHTML = `
    <span class="token-metric-label">Surprisal</span>
    <span class="token-metric-val">${formatSurprisal(token.surprisal)}</span>
  `;

  metrics.append(confPill, surprisalPill);
  header.append(primary, metrics);
  card.append(header);

  // Candidates Distribution Section
  const candidatesSection = document.createElement('div');
  candidatesSection.className = 'token-inspector-candidates';

  const candTitle = document.createElement('div');
  candTitle.className = 'token-candidates-title';
  candTitle.innerHTML = `
    <span>Alternative Candidates Evaluated at this Step</span>
    <span style="font-weight: 500; font-size: 10px; color: #64748b;">Top ${token.topCandidates?.length || 0} distribution</span>
  `;
  candidatesSection.append(candTitle);

  const candList = document.createElement('div');
  candList.className = 'token-candidates-list';

  const candidates = Array.isArray(token.topCandidates) && token.topCandidates.length > 0
    ? token.topCandidates
    : [{ tokenId: token.tokenId, text: token.text, probability: token.probability ?? 1, logit: 0 }];

  candidates.forEach((cand, rankIdx) => {
    const isWinner = cand.tokenId === token.tokenId || cand.text === token.text;
    const row = document.createElement('div');
    row.className = `token-candidate-row${isWinner ? ' is-winner' : ''}`;

    const rank = document.createElement('span');
    rank.className = 'token-candidate-rank';
    rank.textContent = `#${rankIdx + 1}`;

    const text = document.createElement('span');
    text.className = 'token-candidate-text';
    text.textContent = JSON.stringify(cand.text);
    text.title = `Token ID #${cand.tokenId} ${isWinner ? '(Sampled)' : ''}`;

    const barBg = document.createElement('div');
    barBg.className = 'token-candidate-bar-bg';
    const barFill = document.createElement('div');
    barFill.className = 'token-candidate-bar-fill';
    const pct = Math.max(0, Math.min(100, (cand.probability ?? 0) * 100));
    barFill.style.width = `${pct}%`;
    barBg.append(barFill);

    const pctLabel = document.createElement('span');
    pctLabel.className = 'token-candidate-pct';
    pctLabel.textContent = formatPct(cand.probability);

    row.append(rank, text, barBg, pctLabel);
    candList.append(row);
  });

  candidatesSection.append(candList);
  card.append(candidatesSection);

  // Navigation footer
  const nav = document.createElement('div');
  nav.className = 'token-inspector-nav';

  const hint = document.createElement('span');
  hint.className = 'token-inspector-hint';
  hint.textContent = 'Use ← / → arrow keys to step through tokens';

  const navBtns = document.createElement('div');
  navBtns.className = 'token-inspector-nav-btns';

  const prevBtn = document.createElement('button');
  prevBtn.type = 'button';
  prevBtn.className = 'token-inspector-nav-btn';
  prevBtn.textContent = '← Prev';
  prevBtn.disabled = selectedIndex <= 0;
  prevBtn.addEventListener('click', () => selectToken(selectedIndex - 1));

  const nextBtn = document.createElement('button');
  nextBtn.type = 'button';
  nextBtn.className = 'token-inspector-nav-btn';
  nextBtn.textContent = 'Next →';
  nextBtn.disabled = selectedIndex >= currentTokens.length - 1;
  nextBtn.addEventListener('click', () => selectToken(selectedIndex + 1));

  navBtns.append(prevBtn, nextBtn);
  nav.append(hint, navBtns);
  card.append(nav);

  boundCardContainer.replaceChildren(card);
}

export function renderTokenInspector(tokens, streamContainer, cardContainer) {
  currentTokens = Array.isArray(tokens) ? tokens : [];
  boundContainer = streamContainer;
  boundCardContainer = cardContainer;
  selectedIndex = 0;

  if (!streamContainer) return;
  streamContainer.replaceChildren();

  if (!currentTokens.length) {
    if (cardContainer) cardContainer.replaceChildren();
    return;
  }

  currentTokens.forEach((token, idx) => {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = `token-chip ${getConfidenceClass(token.probability)}`;
    chip.dataset.tokenIndex = String(idx);
    chip.title = `Token #${token.tokenId}: "${token.text}" · ${formatPct(token.probability)} confidence`;

    // Whitespace handling
    if (token.text && token.text.startsWith(' ')) {
      const ws = document.createElement('span');
      ws.className = 'token-ws';
      ws.textContent = '·';
      chip.append(ws);
      chip.append(document.createTextNode(token.text.slice(1)));
    } else {
      chip.textContent = token.text || '';
    }

    chip.addEventListener('click', () => {
      selectToken(idx);
    });

    streamContainer.append(chip);
  });

  selectToken(0);
}

// Global keyboard arrow navigation listener
if (typeof window !== 'undefined') window.addEventListener('keydown', (e) => {
  if (!inspectorActive || !currentTokens.length) return;
  // Ignore if typing in input
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName)) return;

  if (e.key === 'ArrowLeft') {
    e.preventDefault();
    selectToken(selectedIndex - 1);
  } else if (e.key === 'ArrowRight') {
    e.preventDefault();
    selectToken(selectedIndex + 1);
  }
});
