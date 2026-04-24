import {
  buildModeScoreMaps,
  sortTokenIdsByScore,
} from '../../../src/tooling/precision-replay-math.js';
import { setPromptValue } from '../../input.js';

const DATA_ROOT = new URL('../../data/f16-precision-collapse/', import.meta.url);
const DISPLAY_MODES = Object.freeze([
  { id: 'exact', label: 'Exact' },
  { id: 'f32_forward', label: 'F32' },
  { id: 'f16_forward', label: 'F16' },
]);
const MAX_TABLE_ROWS = 12;

const replayState = {
  manifest: null,
  curated: null,
  selectedPromptId: null,
  selectedMode: 'f32_forward',
  slices: new Map(),
};

function $(id) {
  return document.getElementById(id);
}

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

function formatFixed(value, digits = 6) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return 'n/a';
  }
  return value.toFixed(digits);
}

function formatCompact(value, digits = 6) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return 'n/a';
  }
  const abs = Math.abs(value);
  if (abs >= 1 || abs === 0) {
    return value.toFixed(Math.min(digits, 4));
  }
  return value.toExponential(3);
}

async function fetchJson(relativePath) {
  const response = await fetch(new URL(relativePath, DATA_ROOT));
  if (!response.ok) {
    throw new Error(`precision replay data fetch failed for ${relativePath} (${response.status})`);
  }
  return response.json();
}

function getSelectedPrompt() {
  return replayState.curated?.prompts?.find((prompt) => prompt.id === replayState.selectedPromptId) ?? null;
}

async function loadSlice(prompt) {
  if (!prompt?.sliceArtifact) {
    return null;
  }
  if (replayState.slices.has(prompt.sliceArtifact)) {
    return replayState.slices.get(prompt.sliceArtifact);
  }
  const slice = await fetchJson(`curated/${prompt.sliceArtifact}`);
  replayState.slices.set(prompt.sliceArtifact, slice);
  return slice;
}

function renderStatus(text) {
  const statusEl = $('precision-replay-status');
  if (statusEl) {
    statusEl.textContent = text;
  }
}

function renderStats() {
  const statsEl = $('precision-replay-stats');
  if (!statsEl || !replayState.manifest || !replayState.curated) {
    return;
  }
  const broad = replayState.manifest.broad.aggregate;
  const curated = replayState.curated.aggregate;
  const cards = [
    {
      label: 'Broad winner flips',
      value: `${broad.f16VsF32FlipCount} / ${broad.promptCount}`,
      detail: `${broad.persistentFlipCount} persisted through 8 steps`,
    },
    {
      label: 'Top-64 inversions',
      value: `${broad.totalF16VsF32Inversions.toLocaleString()}`,
      detail: 'f16 forward vs f32 forward',
    },
    {
      label: 'Curated watched-pair swaps',
      value: `${curated.watchPairSwapCount}`,
      detail: `${curated.f16VsF32FlipCount} curated top-1 flips`,
    },
    {
      label: 'Replay max abs error',
      value: formatCompact(broad.maxReplayAbsError, 6),
      detail: 'live logits vs replayed f32 slice',
    },
  ];
  statsEl.innerHTML = cards.map((card) => `
    <article class="precision-replay-stat-card">
      <div class="type-label">${escapeHtml(card.label)}</div>
      <div class="precision-replay-stat-value">${escapeHtml(card.value)}</div>
      <div class="precision-replay-stat-detail">${escapeHtml(card.detail)}</div>
    </article>
  `).join('');
}

function renderModeGroup() {
  const groupEl = $('precision-replay-mode-group');
  if (!groupEl) {
    return;
  }
  groupEl.innerHTML = DISPLAY_MODES.map(({ id, label }) => `
    <button
      type="button"
      class="precision-replay-mode-btn${replayState.selectedMode === id ? ' is-active' : ''}"
      data-precision-mode="${id}"
    >${escapeHtml(label)}</button>
  `).join('');
  for (const button of groupEl.querySelectorAll('[data-precision-mode]')) {
    button.addEventListener('click', async () => {
      replayState.selectedMode = button.dataset.precisionMode;
      renderModeGroup();
      await renderSelectedPrompt();
    });
  }
}

function renderBroadFlips() {
  const broadEl = $('precision-replay-broad-flips');
  if (!broadEl || !replayState.manifest) {
    return;
  }
  const flipExamples = replayState.manifest.broad.flipExamples ?? [];
  broadEl.innerHTML = flipExamples.map((flip) => `
    <article class="precision-replay-mini-card">
      <div class="precision-replay-mini-id">${escapeHtml(flip.id)}</div>
      <div class="precision-replay-mini-text">${escapeHtml(flip.text)}</div>
      <div class="precision-replay-mini-line">F32 winner: <code>${escapeHtml(flip.modes.f32_forward.winnerText)}</code></div>
      <div class="precision-replay-mini-line">F16 winner: <code>${escapeHtml(flip.modes.f16_forward.winnerText)}</code></div>
    </article>
  `).join('');
}

function renderPromptSelect() {
  const selectEl = $('precision-replay-prompt-select');
  if (!selectEl || !replayState.curated) {
    return;
  }
  const prompts = replayState.curated.prompts ?? [];
  selectEl.innerHTML = prompts.map((prompt) => `
    <option value="${escapeHtml(prompt.id)}">${escapeHtml(prompt.id)}</option>
  `).join('');
  if (!replayState.selectedPromptId) {
    replayState.selectedPromptId = replayState.manifest?.curated?.defaultPromptId ?? prompts[0]?.id ?? null;
  }
  selectEl.value = replayState.selectedPromptId ?? '';
}

function createDecodeLookup(slice) {
  const byId = new Map(slice.candidateRows.map((entry) => [entry.tokenId, entry.text]));
  return (tokenId) => byId.get(tokenId) ?? `[${tokenId}]`;
}

function renderWinnerCards(prompt) {
  const winnersEl = $('precision-replay-winners');
  if (!winnersEl) {
    return;
  }
  winnersEl.innerHTML = DISPLAY_MODES.map(({ id, label }) => {
    const summary = prompt.modes[id];
    return `
      <article class="precision-replay-card precision-replay-winner-card${replayState.selectedMode === id ? ' is-selected-mode' : ''}">
        <div class="type-label">${escapeHtml(label)}</div>
        <div class="precision-replay-winner-token"><code>${escapeHtml(summary.winnerText)}</code></div>
        <div class="precision-replay-winner-score">score ${escapeHtml(formatFixed(summary.winnerScore))}</div>
        <div class="precision-replay-winner-gap">gap ${escapeHtml(formatFixed(summary.winnerGap))}</div>
      </article>
    `;
  }).join('');
}

function renderWatchPairs(prompt) {
  const watchEl = $('precision-replay-watch-pairs');
  if (!watchEl) {
    return;
  }
  const presentPairs = (prompt.watchPairs ?? []).filter((pair) => pair.present);
  if (presentPairs.length === 0) {
    watchEl.innerHTML = '';
    return;
  }
  watchEl.innerHTML = presentPairs.map((pair) => `
    <article class="precision-replay-card">
      <div class="type-label">Watched pair</div>
      <div class="precision-replay-watch-title"><code>${escapeHtml(pair.left.decodedText)}</code> vs <code>${escapeHtml(pair.right.decodedText)}</code></div>
      <div class="precision-replay-watch-line">Exact: <code>${escapeHtml(pair.modes.exact.winnerText)}</code> (${escapeHtml(formatFixed(pair.modes.exact.gap))})</div>
      <div class="precision-replay-watch-line">F32: <code>${escapeHtml(pair.modes.f32_forward.winnerText)}</code> (${escapeHtml(formatFixed(pair.modes.f32_forward.gap))})</div>
      <div class="precision-replay-watch-line">F16: <code>${escapeHtml(pair.modes.f16_forward.winnerText)}</code> (${escapeHtml(formatFixed(pair.modes.f16_forward.gap))})</div>
    </article>
  `).join('');
}

function renderBranch(prompt) {
  const branchEl = $('precision-replay-branch');
  if (!branchEl) {
    return;
  }
  if (!prompt.branchComparison || !prompt.branches) {
    branchEl.innerHTML = `
      <article class="precision-replay-card">
        <div class="type-label">Forced branch</div>
        <div class="precision-replay-branch-meta">No winner split for this curated prompt, so there is no forced-branch comparison here.</div>
      </article>
    `;
    return;
  }
  branchEl.innerHTML = `
    <article class="precision-replay-card">
      <div class="type-label">Forced branch</div>
      <div class="precision-replay-branch-meta">Differing steps: ${escapeHtml(String(prompt.branchComparison.differingStepCount))}</div>
      <div class="precision-replay-branch-meta">Healed at: ${escapeHtml(String(prompt.branchComparison.healedAtStep ?? 'none'))}</div>
      <div class="precision-replay-branch-meta">Persists through end: ${escapeHtml(String(prompt.branchComparison.persistsThroughEnd))}</div>
      <div class="precision-replay-branch-grid">
        <div>
          <div class="type-label">F32 branch</div>
          <pre class="precision-replay-branch-text">${escapeHtml(prompt.branches.f32_forward.decodedText ?? '')}</pre>
        </div>
        <div>
          <div class="type-label">F16 branch</div>
          <pre class="precision-replay-branch-text">${escapeHtml(prompt.branches.f16_forward.decodedText ?? '')}</pre>
        </div>
      </div>
    </article>
  `;
}

function renderCandidateTable(prompt, slice) {
  const tableBodyEl = $('precision-replay-table-body');
  if (!tableBodyEl) {
    return;
  }
  const hidden = Float32Array.from(slice.embedding);
  const rows = new Map(slice.candidateRows.map((entry) => [entry.tokenId, Float32Array.from(entry.row)]));
  const scores = buildModeScoreMaps(hidden, rows);
  const decode = createDecodeLookup(slice);
  const tokenIds = slice.candidateRows.map((entry) => entry.tokenId);
  const ranked = sortTokenIdsByScore(tokenIds, scores[replayState.selectedMode]).slice(0, MAX_TABLE_ROWS);
  const selectedWinnerId = prompt.modes[replayState.selectedMode].winnerTokenId;
  const f32WinnerId = prompt.modes.f32_forward.winnerTokenId;
  const f16WinnerId = prompt.modes.f16_forward.winnerTokenId;

  tableBodyEl.innerHTML = ranked.map((tokenId, index) => {
    const rowClasses = [
      tokenId === selectedWinnerId ? 'is-selected-mode' : '',
      tokenId === f32WinnerId ? 'is-f32-winner' : '',
      tokenId === f16WinnerId ? 'is-f16-winner' : '',
    ].filter(Boolean).join(' ');
    return `
      <tr class="${rowClasses}">
        <td>${index + 1}</td>
        <td><code>${escapeHtml(decode(tokenId))}</code></td>
        <td>${escapeHtml(formatFixed(scores.exact.get(tokenId)))}</td>
        <td>${escapeHtml(formatFixed(scores.f32_forward.get(tokenId)))}</td>
        <td>${escapeHtml(formatFixed(scores.f16_forward.get(tokenId)))}</td>
        <td>${escapeHtml(formatFixed((scores.f16_forward.get(tokenId) ?? 0) - (scores.f32_forward.get(tokenId) ?? 0)))}</td>
      </tr>
    `;
  }).join('');
}

async function renderSelectedPrompt() {
  const prompt = getSelectedPrompt();
  if (!prompt) {
    return;
  }
  const promptEl = $('precision-replay-prompt');
  if (promptEl) {
    promptEl.textContent = prompt.text;
  }
  renderWinnerCards(prompt);
  renderWatchPairs(prompt);
  renderBranch(prompt);
  const slice = await loadSlice(prompt);
  if (slice) {
    renderCandidateTable(prompt, slice);
  }
}

function setPanelOpen(open) {
  const panel = $('precision-replay-panel');
  if (panel) {
    panel.hidden = !open;
  }
}

export async function initPrecisionReplay() {
  const toggleEl = $('precision-replay-toggle');
  const selectEl = $('precision-replay-prompt-select');
  const usePromptBtn = $('precision-replay-use-prompt');
  if (!toggleEl || !selectEl || !usePromptBtn) {
    return;
  }

  renderStatus('Loading checked-in evidence...');
  try {
    replayState.manifest = await fetchJson('manifest.json');
    replayState.curated = await fetchJson('curated/summary.json');
    replayState.selectedPromptId = replayState.manifest?.curated?.defaultPromptId ?? replayState.curated?.prompts?.[0]?.id ?? null;

    renderStats();
    renderPromptSelect();
    renderModeGroup();
    renderBroadFlips();
    await renderSelectedPrompt();
    renderStatus(`${replayState.manifest.broad.aggregate.f16VsF32FlipCount} / ${replayState.manifest.broad.aggregate.promptCount} broad prompts flipped on this checked-in run.`);
  } catch (error) {
    toggleEl.disabled = true;
    renderStatus(`Precision replay unavailable: ${error.message}`);
    return;
  }

  toggleEl.addEventListener('change', () => {
    setPanelOpen(toggleEl.checked);
  });

  selectEl.addEventListener('change', async () => {
    replayState.selectedPromptId = selectEl.value;
    await renderSelectedPrompt();
  });

  usePromptBtn.addEventListener('click', () => {
    const prompt = getSelectedPrompt();
    if (prompt) {
      setPromptValue(prompt.text);
    }
  });
}
