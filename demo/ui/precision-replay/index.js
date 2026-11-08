import {
  buildModeScoreMaps,
  sortTokenIdsByScore,
} from 'doppler-gpu/tooling/evidence';
import { setPromptValue } from '../../input.js';

const DATA_ROOT = new URL('../../data/f16-precision-collapse/', import.meta.url);
const DISPLAY_MODES = Object.freeze([
  { id: 'exact', label: 'Exact' },
  { id: 'f32_forward', label: '32-bit' },
  { id: 'f16_forward', label: '16-bit' },
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
      label: 'Next token changed',
      value: `${broad.f16VsF32FlipCount} / ${broad.promptCount}`,
      detail: `${broad.persistentFlipCount} still differed 8 tokens later`,
    },
    {
      label: 'Swaps among the top 64',
      value: `${broad.totalF16VsF32Inversions.toLocaleString()}`,
      detail: 'pairs of candidates whose order flipped, 16-bit vs 32-bit',
    },
    {
      label: 'Hand-picked pairs that swapped',
      value: `${curated.watchPairSwapCount}`,
      detail: `${curated.f16VsF32FlipCount} hand-picked prompts changed their next token`,
    },
    {
      label: 'Largest replay error',
      value: formatCompact(broad.maxReplayAbsError, 6),
      detail: 'saved scores vs scores recomputed here in 32-bit',
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
      aria-pressed="${replayState.selectedMode === id}"
    >${escapeHtml(label)}</button>
  `).join('');
  for (const button of groupEl.querySelectorAll('[data-precision-mode]')) {
    button.addEventListener('click', async () => {
      replayState.selectedMode = button.dataset.precisionMode;
      for (const control of groupEl.querySelectorAll('[data-precision-mode]')) {
        const active = control.dataset.precisionMode === replayState.selectedMode;
        control.classList.toggle('is-active', active);
        control.setAttribute('aria-pressed', String(active));
      }
      await renderSelectedPrompt().catch((error) => renderStatus(`Recorded examples unavailable: ${error.message}`));
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
      <div class="precision-replay-mini-line">32-bit picked: <code>${escapeHtml(flip.modes.f32_forward.winnerText)}</code></div>
      <div class="precision-replay-mini-line">16-bit picked: <code>${escapeHtml(flip.modes.f16_forward.winnerText)}</code></div>
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
    <option value="${escapeHtml(prompt.id)}">${escapeHtml(prompt.id.replaceAll('-', ' '))}</option>
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

function visibleToken(text) {
  if (text === '') return 'Empty token';
  if (/^\n+$/.test(text)) return text.length === 1 ? 'Line break' : `${text.length} line breaks`;
  if (/^ +$/.test(text)) return text.length === 1 ? 'Space' : `${text.length} spaces`;
  return JSON.stringify(text);
}

function renderWinnerCards(prompt) {
  const winnersEl = $('precision-replay-winners');
  if (!winnersEl) return;
  const same = prompt.modes.f32_forward.winnerTokenId === prompt.modes.f16_forward.winnerTokenId;
  const outcome = $('precision-replay-outcome');
  if (outcome) outcome.textContent = same
    ? '32-bit and 16-bit math picked the same next token.'
    : '16-bit math picked a different next token than 32-bit.';
  winnersEl.innerHTML = DISPLAY_MODES.filter(({ id }) => id !== 'exact').map(({ id, label }) => `
    <article class="precision-replay-card precision-replay-winner-card">
      <div class="type-label">${label}</div>
      <div class="precision-replay-winner-token">${escapeHtml(visibleToken(prompt.modes[id].winnerText))}</div>
    </article>
  `).join('');
  const scores = $('precision-replay-winner-scores');
  if (scores) scores.innerHTML = DISPLAY_MODES.map(({ id, label }) => `
    <p>${label}: ${escapeHtml(visibleToken(prompt.modes[id].winnerText))} · score ${formatFixed(prompt.modes[id].winnerScore)} · ahead of the runner-up by ${formatFixed(prompt.modes[id].winnerGap)}</p>
  `).join('');
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
      <div class="type-label">Hand-picked pair</div>
      <div class="precision-replay-watch-title"><code>${escapeHtml(pair.left.decodedText)}</code> vs <code>${escapeHtml(pair.right.decodedText)}</code></div>
      <div class="precision-replay-watch-line">Exact: <code>${escapeHtml(pair.modes.exact.winnerText)}</code> (${escapeHtml(formatFixed(pair.modes.exact.gap))})</div>
      <div class="precision-replay-watch-line">32-bit: <code>${escapeHtml(pair.modes.f32_forward.winnerText)}</code> (${escapeHtml(formatFixed(pair.modes.f32_forward.gap))})</div>
      <div class="precision-replay-watch-line">16-bit: <code>${escapeHtml(pair.modes.f16_forward.winnerText)}</code> (${escapeHtml(formatFixed(pair.modes.f16_forward.gap))})</div>
    </article>
  `).join('');
}

function renderBranch(prompt) {
  const branchEl = $('precision-replay-branch');
  if (!branchEl) {
    return;
  }
  if (!prompt.branchComparison || !prompt.branches) {
    branchEl.replaceChildren();
    return;
  }
  branchEl.innerHTML = `
    <article class="precision-replay-card">
      <div class="type-label">Continuing both versions</div>
      <div class="precision-replay-branch-meta">Tokens that differ: ${escapeHtml(String(prompt.branchComparison.differingStepCount))}</div>
      <div class="precision-replay-branch-meta">Back in agreement at token: ${escapeHtml(String(prompt.branchComparison.healedAtStep ?? 'never'))}</div>
      <div class="precision-replay-branch-meta">Still different at the end: ${prompt.branchComparison.persistsThroughEnd ? 'yes' : 'no'}</div>
      <div class="precision-replay-branch-grid">
        <div>
          <div class="type-label">32-bit continues</div>
          <pre class="precision-replay-branch-text">${escapeHtml(prompt.branches.f32_forward.decodedText ?? '')}</pre>
        </div>
        <div>
          <div class="type-label">16-bit continues</div>
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
  tableBodyEl.dataset.promptId = prompt.id;
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
  const table = $('precision-replay-table-body');
  table?.replaceChildren();
  if (table) delete table.dataset.promptId;
  const slice = await loadSlice(prompt);
  if (slice && getSelectedPrompt()?.id === prompt.id) {
    renderCandidateTable(prompt, slice);
  }
}

export async function initPrecisionReplay() {
  const toggleEl = $('precision-replay');
  const selectEl = $('precision-replay-prompt-select');
  const usePromptBtn = $('precision-replay-use-prompt');
  if (!toggleEl || !selectEl || !usePromptBtn) {
    return;
  }

  $('precision-replay-panel').hidden = false;

  let loading = null;
  const ensureLoaded = async () => {
    if (replayState.manifest && replayState.curated) return;
    if (loading) return loading;
    renderStatus('Loading examples…');
    loading = (async () => {
      replayState.manifest = await fetchJson('manifest.json');
      replayState.curated = await fetchJson('curated/summary.json');
      replayState.selectedPromptId = replayState.manifest?.curated?.defaultPromptId
        ?? replayState.curated?.prompts?.[0]?.id
        ?? null;
      renderStats();
      renderPromptSelect();
      renderModeGroup();
      renderBroadFlips();
      await renderSelectedPrompt();
      const aggregate = replayState.manifest.broad.aggregate;
      renderStatus(`In ${aggregate.f16VsF32FlipCount} of ${aggregate.promptCount} saved prompts, 16-bit math picked a different next token.`);
    })();
    try {
      await loading;
    } catch (error) {
      replayState.manifest = null;
      replayState.curated = null;
      renderStatus(`Recorded examples unavailable: ${error.message}`);
      throw error;
    } finally {
      loading = null;
    }
  };

  renderStatus('');
  toggleEl.addEventListener('toggle', async () => {
    if (toggleEl.open) {
      await ensureLoaded().catch(() => {});
    }
  });

  selectEl.addEventListener('change', async () => {
    replayState.selectedPromptId = selectEl.value;
    await renderSelectedPrompt().catch((error) => renderStatus(`Recorded examples unavailable: ${error.message}`));
  });

  usePromptBtn.addEventListener('click', () => {
    const prompt = getSelectedPrompt();
    if (prompt) {
      setPromptValue(prompt.text);
    }
  });
}
