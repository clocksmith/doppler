// =============================================================================
// Xray — Debug visualization panels for Doppler inference internals.
// =============================================================================
// Toggle via:
//   URL flag:  ?xray=decode,kv,kernel,gpu,exec,mem,batch  or  ?xray=all
//   Checkbox:  each panel has an individual checkbox in the UI
//
// URL and checkboxes are bidirectionally synced:
//   - URL ?xray= seeds checkbox state on init
//   - Toggling a checkbox updates the URL via replaceState
//   - Sharing/bookmarking a URL preserves the exact panel selection
//
// Each panel reads from state.lastInferenceStats / state.lastMemoryStats
// and pipeline.getBatchingStats() / pipeline.getBufferPool().getLabelStats().

import { formatBytes } from 'doppler-gpu/tooling';
import { buildKernelPathBuilderRuntimeOverlay } from '../../../src/tooling/kernel-path-builder/runtime-overlay.js';
import { loadKernelPathBuilderIndex } from '../kernel-path-builder/index.js';
import { state } from '../state.js';
import { $ } from '../dom.js';

// ---------------------------------------------------------------------------
// Panel registry
// ---------------------------------------------------------------------------

const PANELS = {
  decode: { label: 'Decode Waterfall', render: renderDecodeWaterfall },
  kv: { label: 'KV Cache', render: renderKVCache },
  kernel: { label: 'Kernel Timing', render: renderKernelTiming },
  gpu: { label: 'GPU Pipeline', render: renderGPUPipeline },
  exec: { label: 'Execution Plan', render: renderExecPlan },
  mem: { label: 'Memory Breakdown', render: renderMemoryBreakdown },
  batch: { label: 'Batching Stats', render: renderBatchingStats },
};

// ---------------------------------------------------------------------------
// Module state
// ---------------------------------------------------------------------------

let panelEls = {};
let initialized = false;
let onChangeCallback = null;
let hoverCardEl = null;
let detailDrawerEl = null;
let detailDrawerBodyEl = null;
let detailDrawerTitleEl = null;
let detailDrawerSubtitleEl = null;
let builderIndexRequested = false;
let selectedExplainKey = null;

const MAX_WATERFALL_STEPS = 32;
const MAX_MEMORY_ROWS = 12;
const MAX_KERNEL_ROWS = 24;

// ---------------------------------------------------------------------------
// URL ↔ checkbox sync
// ---------------------------------------------------------------------------

function parseXrayFlags() {
  if (typeof window === 'undefined') return new Set();
  const query = new URLSearchParams(window.location.search);
  const hashRaw = (window.location.hash || '').replace(/^#/, '').replace(/^\?/, '');
  const hash = new URLSearchParams(hashRaw);
  const raw = hash.get('xray') ?? query.get('xray') ?? null;
  if (!raw) return new Set();
  const trimmed = raw.trim().toLowerCase();
  if (trimmed === 'all') return new Set(Object.keys(PANELS));
  const keys = trimmed.split(',').map(s => s.trim()).filter(s => s in PANELS);
  return new Set(keys);
}

function pushXrayToUrl(active) {
  if (typeof window === 'undefined') return;
  const url = new URL(window.location.href);
  if (active.size === 0) {
    url.searchParams.delete('xray');
  } else if (active.size === Object.keys(PANELS).length) {
    url.searchParams.set('xray', 'all');
  } else {
    url.searchParams.set('xray', [...active].join(','));
  }
  window.history.replaceState(null, '', url.toString());
}

// ---------------------------------------------------------------------------
// Active panel resolution (reads checkboxes as live source of truth)
// ---------------------------------------------------------------------------

function getActivePanels() {
  const active = new Set();
  for (const key of Object.keys(PANELS)) {
    const cb = $(`xray-toggle-${key}`);
    if (cb?.checked) active.add(key);
  }
  return active;
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

export function initXray({ onChange } = {}) {
  if (initialized) return;
  initialized = true;
  onChangeCallback = onChange || null;

  const urlFlags = parseXrayFlags();

  // Seed checkboxes from URL flags
  for (const key of Object.keys(PANELS)) {
    const cb = $(`xray-toggle-${key}`);
    if (cb) {
      cb.checked = urlFlags.has(key);
      cb.addEventListener('change', () => syncXrayState());
    }
  }

  // Build panel DOM for all panels (hidden by default, toggled by checkbox)
  const container = $('xray-container');
  if (!container) return;

  panelEls = {};
  for (const key of Object.keys(PANELS)) {
    const section = document.createElement('div');
    section.className = 'xray-section';
    section.id = `xray-${key}`;
    section.hidden = true;

    const header = document.createElement('div');
    header.className = 'xray-section-header';
    header.textContent = PANELS[key].label;
    section.appendChild(header);

    const content = document.createElement('div');
    content.className = 'xray-content';
    section.appendChild(content);

    container.appendChild(section);
    panelEls[key] = { section, content };
  }

  const detailDrawer = document.createElement('div');
  detailDrawer.className = 'xray-detail-drawer';
  detailDrawer.hidden = true;
  detailDrawer.innerHTML = `
    <div class="xray-detail-header">
      <div class="xray-detail-heading">
        <div class="xray-detail-title"></div>
        <div class="xray-detail-subtitle"></div>
      </div>
      <button class="btn btn-small xray-detail-close" type="button">Close</button>
    </div>
    <div class="xray-detail-body"></div>
  `;
  container.appendChild(detailDrawer);
  detailDrawerEl = detailDrawer;
  detailDrawerBodyEl = detailDrawer.querySelector('.xray-detail-body');
  detailDrawerTitleEl = detailDrawer.querySelector('.xray-detail-title');
  detailDrawerSubtitleEl = detailDrawer.querySelector('.xray-detail-subtitle');
  detailDrawer.querySelector('.xray-detail-close')?.addEventListener('click', () => {
    selectedExplainKey = null;
    closeExplainDrawer();
  });

  hoverCardEl = document.createElement('div');
  hoverCardEl.className = 'xray-hover-card';
  hoverCardEl.hidden = true;
  document.body.appendChild(hoverCardEl);

  syncXrayState();
}

function syncXrayState() {
  const container = $('xray-container');
  if (!container) return;

  const active = getActivePanels();
  for (const key of Object.keys(PANELS)) {
    const entry = panelEls[key];
    if (entry) entry.section.hidden = !active.has(key);
  }
  container.hidden = active.size === 0;
  pushXrayToUrl(active);
  ensureKernelBuilderIndexForXray(active);
  if (onChangeCallback) onChangeCallback();
}

export function isXrayEnabled() {
  return getActivePanels().size > 0;
}

const PROFILING_PANELS = new Set(['decode', 'kernel', 'gpu']);

export function isXrayProfilingNeeded() {
  const active = getActivePanels();
  for (const key of PROFILING_PANELS) {
    if (active.has(key)) return true;
  }
  return false;
}

export function resetXray() {
  for (const entry of Object.values(panelEls)) {
    entry.content.innerHTML = '';
  }
  hideHoverCard();
  selectedExplainKey = null;
  closeExplainDrawer();
}

export function updateXrayPanels(pipeline) {
  const active = getActivePanels();
  if (active.size === 0) return;

  const stats = state.lastInferenceStats || {};
  const memStats = state.lastMemoryStats || {};

  for (const key of active) {
    const entry = panelEls[key];
    if (!entry) continue;
    try {
      PANELS[key].render(entry.content, stats, memStats, pipeline);
    } catch {
      // silent — debug panels should not break the demo
    }
  }
  refreshExplainDrawer(stats, memStats, pipeline);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function fmtMs(v) {
  if (!Number.isFinite(v)) return '--';
  return v < 1 ? `${(v * 1000).toFixed(0)}us` : `${v.toFixed(1)}ms`;
}

function barRow(label, value, maxValue, colorClass, valueLabel) {
  const pct = maxValue > 0 ? Math.min(100, (value / maxValue) * 100) : 0;
  const row = document.createElement('div');
  row.className = 'xray-bar-row';
  row.innerHTML = `
    <span class="xray-bar-label" title="${label}">${label}</span>
    <span class="xray-bar-track"><span class="xray-bar ${colorClass}" style="width:${pct.toFixed(1)}%"></span></span>
    <span class="xray-bar-value">${valueLabel ?? fmtMs(value)}</span>
  `;
  return row;
}

function emptyMsg(text) {
  const el = document.createElement('div');
  el.className = 'xray-empty';
  el.textContent = text;
  return el;
}

function appendKeyValueList(el, pairs) {
  const grid = document.createElement('div');
  grid.className = 'xray-kv-list';

  for (const [k, v] of pairs) {
    const key = document.createElement('span');
    key.className = 'xray-kv-key';
    key.textContent = k;
    const val = document.createElement('span');
    val.className = 'xray-kv-val';
    val.textContent = v;
    grid.appendChild(key);
    grid.appendChild(val);
  }

  el.appendChild(grid);
}

function appendSubheader(el, text) {
  const header = document.createElement('div');
  header.className = 'xray-subheader';
  header.textContent = text;
  el.appendChild(header);
}

function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function normalizeTimingLabel(value) {
  return normalizeText(value).toLowerCase();
}

function isTokenPressMode() {
  if (state.generating) {
    const toggle = $('set-token-press');
    return toggle?.checked === true;
  }
  return state.lastRun?.mode === 'token-press';
}

export function getXrayRuntimeNoticeText(options = {}) {
  const tokenPressEnabled = options.tokenPressEnabled === true;
  const traceEnabled = options.traceEnabled === true;
  const profilingEnabled = options.profilingEnabled === true;
  if (tokenPressEnabled && profilingEnabled) {
    return 'Token Press is active. Decode, Kernel, and GPU X-Ray panels add per-step profiling; Batch stays unavailable.';
  }
  if (tokenPressEnabled) {
    return 'Token Press is a separate generation mode. It runs step-by-step decode, so Batch stats stay unavailable.';
  }
  if (profilingEnabled) {
    return 'Decode, Kernel, and GPU X-Ray panels request extra profiling. Trace logging is separate.';
  }
  if (traceEnabled) {
    return 'Trace logging is active. It affects runtime logs, not X-Ray panel selection.';
  }
  return null;
}

function formatBatchGuardReason(reason) {
  if (!reason) return null;
  const known = {
    batch_size_1: 'batch size is 1',
    bdpa_paged_layout: 'BDPA paged KV layout',
    command_batching_disabled: 'command batching disabled',
    cpu_weights: 'CPU weights active',
    finiteness_fallback_window: 'finiteness fallback window active',
    multi_token_decode_disabled: 'multi-token decode disabled',
    no_gpu: 'GPU decode unavailable',
    no_gpu_sampling: 'GPU sampling unavailable',
  };
  return known[reason] ?? reason.replaceAll('_', ' ');
}

function formatObservedDecodeLabel(stats) {
  const decodeMode = stats?.decodeMode ?? null;
  if (isTokenPressMode()) {
    return 'Token Press stepwise decode';
  }
  if (decodeMode === 'batched_gpu') {
    return 'Generate batched GPU decode';
  }
  if (decodeMode === 'single_token') {
    return 'Generate single-token decode';
  }
  if (decodeMode === 'self_speculation') {
    return 'Generate self-speculation decode';
  }
  return 'Generate';
}

function fmtPct(value) {
  if (!Number.isFinite(value)) return '--';
  return `${(value * 100).toFixed(0)}%`;
}

function formatExplainTime(valueMs) {
  return Number.isFinite(valueMs) ? fmtMs(valueMs) : '--';
}

function formatLayers(value) {
  if (Array.isArray(value)) return value.join(', ');
  if (value === 'all') return 'all';
  return value == null ? '--' : String(value);
}

function buildRepoSourceHref(relativePath) {
  const normalized = normalizeText(relativePath).replace(/^\/+/u, '');
  if (!normalized) return null;
  return `https://github.com/clocksmith/doppler/blob/main/${normalized}`;
}

function findBuilderModelForPipeline(pipeline) {
  const modelId = normalizeText(pipeline?.manifest?.modelId || state.activeModelId);
  const models = Array.isArray(state.kernelPathBuilderIndex?.models) ? state.kernelPathBuilderIndex.models : [];
  if (!modelId || models.length === 0) return null;
  return models.find((entry) => entry?.modelId === modelId) || null;
}

function findExecutionStepById(model, stepId) {
  if (!model || !stepId) return null;
  const sections = model.execution?.sections ?? {};
  for (const sectionSteps of Object.values(sections)) {
    for (const step of Array.isArray(sectionSteps) ? sectionSteps : []) {
      if (step?.id === stepId) return step;
    }
  }
  return null;
}

function buildXrayRuntimeOverlay(pipeline, stats, memStats) {
  const model = findBuilderModelForPipeline(pipeline);
  if (!model) return null;
  const report = {
    modelId: model.modelId,
    timestamp: new Date().toISOString(),
    runtimeProfile: state.diagnosticsRuntimeProfileId ?? null,
    metrics: {
      modelLoadMs: state.lastReport?.metrics?.modelLoadMs ?? null,
      firstTokenMs: stats.ttftMs ?? stats.prefillTimeMs ?? null,
      prefillTokensPerSec: Number.isFinite(stats.prefillTokens) && Number.isFinite(stats.prefillTimeMs) && stats.prefillTimeMs > 0
        ? stats.prefillTokens / (stats.prefillTimeMs / 1000)
        : null,
      decodeTokensPerSec: state.lastMetrics?.liveTokensPerSec ?? state.lastMetrics?.tokensPerSec ?? null,
      gpu: {
        decodeRecordMs: stats.decodeRecordMs ?? null,
        decodeSubmitWaitMs: stats.decodeSubmitWaitMs ?? null,
        decodeReadbackWaitMs: stats.decodeReadbackWaitMs ?? null,
      },
      decodeProfileSteps: Array.isArray(stats.decodeProfileSteps) ? stats.decodeProfileSteps : [],
      executionPlan: stats.executionPlan ?? null,
      kernelPathId: stats.kernelPathId ?? null,
      kernelPathSource: stats.kernelPathSource ?? null,
    },
    memory: memStats ?? {},
  };
  const overlay = buildKernelPathBuilderRuntimeOverlay(model, report);
  if (!overlay) return null;
  return { model, overlay };
}

function buildExplainIndex(context) {
  const steps = Array.isArray(context?.overlay?.stepTimings) ? context.overlay.stepTimings : [];
  const map = new Map();
  for (const stepTiming of steps) {
    const labels = Array.isArray(stepTiming.labels) ? stepTiming.labels : [];
    for (const labelEntry of labels) {
      const key = normalizeTimingLabel(labelEntry?.label);
      if (!key) continue;
      const bucket = map.get(key) ?? [];
      bucket.push({
        stepTiming,
        step: findExecutionStepById(context.model, stepTiming.stepId),
        labelTotalMs: Number(labelEntry?.totalMs),
      });
      map.set(key, bucket);
    }
  }
  return map;
}

function resolveExplainMetadata(label, context, currentMs = null) {
  const normalized = normalizeTimingLabel(label);
  if (!normalized) return null;
  const candidates = buildExplainIndex(context).get(normalized) ?? [];
  const best = candidates
    .slice()
    .sort((left, right) => (right.labelTotalMs ?? right.stepTiming.totalMs ?? 0) - (left.labelTotalMs ?? left.stepTiming.totalMs ?? 0))[0] ?? null;
  if (!best) {
    return {
      key: normalized,
      label,
      currentMs,
      model: context?.model ?? null,
      overlay: context?.overlay ?? null,
      matched: false,
      kernelHref: null,
      configHref: context?.model?.configPath ? buildRepoSourceHref(context.model.configPath) : null,
    };
  }
  const step = best.step ?? {};
  const kernelFile = normalizeText(step.kernel || best.stepTiming.kernel);
  return {
    key: normalized,
    label,
    currentMs,
    matched: true,
    model: context.model,
    overlay: context.overlay,
    stepId: best.stepTiming.stepId,
    section: best.stepTiming.section,
    phase: best.stepTiming.phase || normalizeText(step.phase),
    op: best.stepTiming.op || normalizeText(step.op),
    kernel: kernelFile,
    entry: best.stepTiming.entry || normalizeText(step.entry) || 'main',
    weights: normalizeText(step.weights),
    layers: step.layers ?? null,
    digest: normalizeText(step.digest),
    labelTotalMs: best.labelTotalMs,
    stepTotalMs: best.stepTiming.totalMs ?? null,
    kernelHref: kernelFile ? buildRepoSourceHref(`src/gpu/kernels/${kernelFile}`) : null,
    configHref: context.model?.configPath ? buildRepoSourceHref(context.model.configPath) : null,
  };
}

function ensureKernelBuilderIndexForXray(activePanels = getActivePanels()) {
  if (builderIndexRequested) return;
  if (!activePanels.has('decode') && !activePanels.has('kernel')) return;
  builderIndexRequested = true;
  loadKernelPathBuilderIndex()
    .then(() => {
      if (isXrayEnabled()) {
        updateXrayPanels(state.activePipeline);
      }
    })
    .catch(() => {});
}

function buildHoverCardHtml(metadata) {
  const lines = [
    `<div class="xray-hover-title">${escapeHtml(metadata.label)}</div>`,
  ];
  if (metadata.matched) {
    lines.push(`<div class="xray-hover-line"><span>step</span><strong>${escapeHtml(metadata.stepId || '--')}</strong></div>`);
    lines.push(`<div class="xray-hover-line"><span>phase/op</span><strong>${escapeHtml(`${metadata.phase || '--'} / ${metadata.op || '--'}`)}</strong></div>`);
    lines.push(`<div class="xray-hover-line"><span>kernel</span><strong>${escapeHtml(`${metadata.kernel || '--'}#${metadata.entry || 'main'}`)}</strong></div>`);
    if (metadata.weights) lines.push(`<div class="xray-hover-line"><span>weights</span><strong>${escapeHtml(metadata.weights)}</strong></div>`);
    if (metadata.layers != null) lines.push(`<div class="xray-hover-line"><span>layers</span><strong>${escapeHtml(formatLayers(metadata.layers))}</strong></div>`);
  } else {
    lines.push('<div class="xray-hover-empty">No execution-step mapping available.</div>');
  }
  if (Number.isFinite(metadata.currentMs)) {
    lines.push(`<div class="xray-hover-line"><span>this sample</span><strong>${escapeHtml(formatExplainTime(metadata.currentMs))}</strong></div>`);
  }
  return lines.join('');
}

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function showHoverCard(metadata, clientX, clientY) {
  if (!hoverCardEl || !metadata) return;
  hoverCardEl.innerHTML = buildHoverCardHtml(metadata);
  hoverCardEl.hidden = false;
  hoverCardEl.style.left = `${Math.round(clientX + 14)}px`;
  hoverCardEl.style.top = `${Math.round(clientY + 14)}px`;
}

function hideHoverCard() {
  if (!hoverCardEl) return;
  hoverCardEl.hidden = true;
}

function renderExplainDrawer(metadata) {
  if (!detailDrawerEl || !detailDrawerBodyEl || !detailDrawerTitleEl || !detailDrawerSubtitleEl || !metadata) {
    return;
  }
  detailDrawerTitleEl.textContent = metadata.label;
  detailDrawerSubtitleEl.textContent = metadata.matched
    ? `${metadata.phase || '--'} / ${metadata.op || '--'}`
    : 'No execution-step mapping available';
  const statsPairs = [
    ['This sample', formatExplainTime(metadata.currentMs)],
    ['Mapped total', formatExplainTime(metadata.labelTotalMs)],
    ['Step total', formatExplainTime(metadata.stepTotalMs)],
    ['Step id', metadata.stepId ?? '--'],
    ['Kernel', metadata.kernel ? `${metadata.kernel}#${metadata.entry || 'main'}` : '--'],
    ['Layers', formatLayers(metadata.layers)],
    ['Weights', metadata.weights || '--'],
    ['Config', metadata.model?.configPath || '--'],
  ];
  const links = [];
  if (metadata.kernelHref) {
    links.push(`<a class="link-secondary type-caption" href="${escapeHtml(metadata.kernelHref)}" target="_blank" rel="noopener">Kernel Source</a>`);
  }
  if (metadata.configHref) {
    links.push(`<a class="link-secondary type-caption" href="${escapeHtml(metadata.configHref)}" target="_blank" rel="noopener">Model Config</a>`);
  }
  detailDrawerBodyEl.innerHTML = `
    <div class="xray-detail-links">${links.join('')}</div>
    <div class="xray-kv-list">
      ${statsPairs.map(([k, v]) => `
        <span class="xray-kv-key">${escapeHtml(k)}</span>
        <span class="xray-kv-val">${escapeHtml(v)}</span>
      `).join('')}
    </div>
  `;
  detailDrawerEl.hidden = false;
}

function closeExplainDrawer() {
  if (!detailDrawerEl) return;
  detailDrawerEl.hidden = true;
}

function refreshExplainDrawer(stats, memStats, pipeline) {
  if (!selectedExplainKey) return;
  const context = buildXrayRuntimeOverlay(pipeline, stats, memStats);
  if (!context) {
    closeExplainDrawer();
    return;
  }
  const metadata = resolveExplainMetadata(selectedExplainKey.label, context, selectedExplainKey.currentMs ?? null);
  if (!metadata) {
    closeExplainDrawer();
    return;
  }
  metadata.currentMs = selectedExplainKey.currentMs ?? metadata.currentMs ?? null;
  renderExplainDrawer(metadata);
}

function attachExplainInteraction(target, metadata) {
  if (!target || !metadata) return;
  target.classList.add('xray-explainable');
  target.tabIndex = 0;
  target.addEventListener('mouseenter', (event) => {
    showHoverCard(metadata, event.clientX, event.clientY);
  });
  target.addEventListener('mousemove', (event) => {
    showHoverCard(metadata, event.clientX, event.clientY);
  });
  target.addEventListener('mouseleave', hideHoverCard);
  target.addEventListener('focus', () => {
    const rect = target.getBoundingClientRect();
    showHoverCard(metadata, rect.left + 12, rect.bottom + 8);
  });
  target.addEventListener('blur', hideHoverCard);
  target.addEventListener('click', () => {
    selectedExplainKey = {
      label: metadata.label,
      currentMs: metadata.currentMs ?? null,
    };
    renderExplainDrawer(metadata);
  });
  target.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      selectedExplainKey = {
        label: metadata.label,
        currentMs: metadata.currentMs ?? null,
      };
      renderExplainDrawer(metadata);
    }
  });
}

function classifyKernel(name) {
  const n = (name || '').toLowerCase();
  if (
    n.includes('attn')
    || n.includes('attention')
    || n.includes('rope')
    || n.includes('qkv')
    || n.includes('q_proj')
    || n.includes('k_proj')
    || n.includes('v_proj')
    || n.includes('o_proj')
  ) return 'attention';
  if (
    n.includes('ffn')
    || n.includes('mlp')
    || n.includes('gate')
    || n.includes('expert')
    || n.includes('up_proj')
    || n.includes('down_proj')
  ) return 'ffn';
  if (n.includes('embed')) return 'embed';
  if (n.includes('norm') || n.includes('rms')) return 'norm';
  if (n.includes('sample') || n.includes('logit') || n.includes('softmax') || n.includes('softcap')) return 'sample';
  if (n.includes('matmul') || n.includes('gemm') || n.includes('linear')) return 'matmul';
  return 'other';
}

// ---------------------------------------------------------------------------
// A. Decode Waterfall
// ---------------------------------------------------------------------------

function renderDecodeWaterfall(el, stats, memStats, pipeline) {
  el.innerHTML = '';
  const steps = stats.decodeProfileSteps;
  if (!Array.isArray(steps) || steps.length === 0) {
    el.appendChild(emptyMsg('Waiting for generation.'));
    return;
  }
  const explainContext = buildXrayRuntimeOverlay(pipeline, stats, memStats);

  const visible = steps.slice(-MAX_WATERFALL_STEPS);
  const maxTotal = Math.max(...visible.map(s => s.totalMs || 0), 0.001);

  const container = document.createElement('div');
  container.className = 'xray-waterfall';

  for (const step of visible) {
    const row = document.createElement('div');
    row.className = 'xray-waterfall-row';

    const stepLabel = document.createElement('span');
    stepLabel.className = 'xray-waterfall-step';
    stepLabel.textContent = step.step ?? '';
    row.appendChild(stepLabel);

    const segments = document.createElement('div');
    segments.className = 'xray-waterfall-segments';

    const timings = step.timings || {};
    const totalMs = step.totalMs || 0;

    for (const [name, ms] of Object.entries(timings)) {
      if (!Number.isFinite(ms) || ms <= 0) continue;
      const seg = document.createElement('div');
      seg.className = `xray-waterfall-segment xray-bar--${classifyKernel(name)}`;
      const widthPct = (ms / maxTotal) * 100;
      seg.style.width = `${widthPct.toFixed(1)}%`;
      seg.title = `${name}: ${fmtMs(ms)}`;
      if (explainContext) {
        const metadata = resolveExplainMetadata(name, explainContext, ms);
        if (metadata) attachExplainInteraction(seg, metadata);
      }
      segments.appendChild(seg);
    }
    row.appendChild(segments);

    const total = document.createElement('span');
    total.className = 'xray-waterfall-total';
    total.textContent = fmtMs(totalMs);
    row.appendChild(total);

    container.appendChild(row);
  }

  if (steps.length > MAX_WATERFALL_STEPS) {
    const note = document.createElement('div');
    note.className = 'xray-empty';
    note.textContent = `Showing last ${MAX_WATERFALL_STEPS} of ${steps.length} steps`;
    el.appendChild(note);
  }

  el.appendChild(container);

  // Color legend
  const legend = document.createElement('div');
  legend.className = 'xray-legend';
  const categories = [
    ['attention', 'Attn'], ['ffn', 'FFN'], ['embed', 'Embed'],
    ['norm', 'Norm'], ['sample', 'Sample'], ['matmul', 'Matmul'], ['other', 'Other'],
  ];
  for (const [cls, label] of categories) {
    legend.innerHTML += `<span><span class="xray-legend-dot xray-bar--${cls}"></span>${label}</span>`;
  }
  el.appendChild(legend);
}

// ---------------------------------------------------------------------------
// B. KV Cache Timeline
// ---------------------------------------------------------------------------

function renderKVCache(el, stats, memStats) {
  el.innerHTML = '';
  const kvStats = memStats?.kvCache || null;
  if (!kvStats) {
    el.appendChild(emptyMsg('No KV cache data yet.'));
    return;
  }

  const seqLen = kvStats.seqLen ?? 0;
  const maxSeqLen = Math.max(kvStats.maxSeqLen ?? 1, 1);
  const allocated = kvStats.allocated ?? 0;
  const used = kvStats.used ?? 0;
  const efficiency = kvStats.efficiency;
  const theoretical = kvStats.theoretical ?? allocated;
  const utilPct = maxSeqLen > 0 ? ((seqLen / maxSeqLen) * 100) : 0;
  const usedPct = allocated > 0 ? ((used / allocated) * 100) : 0;
  const headroomTokens = Math.max(0, maxSeqLen - seqLen);
  const bytesPerToken = seqLen > 0 ? (used / seqLen) : null;

  el.appendChild(barRow(
    'Seq util',
    seqLen,
    maxSeqLen,
    'xray-bar--kv',
    `${seqLen} / ${maxSeqLen} (${utilPct.toFixed(0)}%)`
  ));
  el.appendChild(barRow(
    'Bytes used',
    used,
    Math.max(allocated, used, 1),
    'xray-bar--mem',
    `${formatBytes(used)} / ${formatBytes(allocated)} (${usedPct.toFixed(0)}%)`
  ));
  if (Number.isFinite(efficiency)) {
    el.appendChild(barRow(
      'Efficiency',
      efficiency,
      1,
      'xray-bar--sample',
      fmtPct(efficiency)
    ));
  }

  const summaryPairs = [
    ['Layout', String(kvStats.layout ?? 'unknown')],
    ['Headroom', `${headroomTokens} tokens`],
    ['Bytes / token', bytesPerToken == null ? '--' : formatBytes(bytesPerToken)],
    ['Theoretical', formatBytes(theoretical)],
  ];
  appendKeyValueList(el, summaryPairs);

  if (Number.isFinite(kvStats.windowSize) || Number.isFinite(kvStats.totalTokensSeen)) {
    appendSubheader(el, 'Window');
    const windowSize = Math.max(kvStats.windowSize ?? maxSeqLen, 1);
    const totalTokensSeen = Math.max(kvStats.totalTokensSeen ?? seqLen, seqLen);
    const retainedTokens = Math.min(seqLen, windowSize);
    const evictedTokens = Math.max(0, totalTokensSeen - retainedTokens);
    const windowMax = Math.max(totalTokensSeen, windowSize, retainedTokens, evictedTokens, 1);
    el.appendChild(barRow(
      'Retained',
      retainedTokens,
      windowMax,
      'xray-bar--kv',
      `${retainedTokens} tokens`
    ));
    el.appendChild(barRow(
      'Evicted',
      evictedTokens,
      windowMax,
      'xray-bar--other',
      `${evictedTokens} tokens`
    ));
    appendKeyValueList(el, [
      ['Window size', `${windowSize} tokens`],
      ['Tokens seen', `${totalTokensSeen}`],
      ['Churn', totalTokensSeen > 0 ? fmtPct(evictedTokens / totalTokensSeen) : '--'],
    ]);
  }

  if (kvStats.layout === 'tiered' && kvStats.hot && kvStats.cold) {
    appendSubheader(el, 'Tiering');
    const hotUsed = kvStats.hot.used ?? 0;
    const coldUsed = kvStats.cold.used ?? 0;
    const tierMax = Math.max(hotUsed + coldUsed, 1);
    el.appendChild(barRow(
      'Hot bytes',
      hotUsed,
      tierMax,
      'xray-bar--kv',
      formatBytes(hotUsed)
    ));
    el.appendChild(barRow(
      'Cold bytes',
      coldUsed,
      tierMax,
      'xray-bar--mem',
      formatBytes(coldUsed)
    ));
    appendKeyValueList(el, [
      ['Hot window', `${kvStats.hot.maxSeqLen ?? '--'} tokens`],
      ['Hot efficiency', fmtPct(kvStats.hot.efficiency)],
      ['Cold layout', String(kvStats.cold.layout ?? 'unknown')],
      ['Cold efficiency', fmtPct(kvStats.cold.efficiency)],
    ]);
  }

  if (kvStats.layout !== 'tiered' && !Number.isFinite(kvStats.windowSize)) {
    const meta = document.createElement('div');
    meta.className = 'xray-kv-meta';
    meta.innerHTML = '<span>Snapshot</span><span>Monotonic growth only for this cache layout; showing efficiency and headroom instead of a history sparkline.</span>';
    el.appendChild(meta);
  }
}

// ---------------------------------------------------------------------------
// C. Kernel Timing Breakdown
// ---------------------------------------------------------------------------

function renderKernelTiming(el, stats, memStats, pipeline) {
  el.innerHTML = '';
  const steps = stats.decodeProfileSteps;
  if (!Array.isArray(steps) || steps.length === 0) {
    el.appendChild(emptyMsg('Waiting for generation.'));
    return;
  }
  const explainContext = buildXrayRuntimeOverlay(pipeline, stats, memStats);

  const agg = {};
  for (const step of steps) {
    const timings = step.timings || {};
    for (const [name, ms] of Object.entries(timings)) {
      if (!Number.isFinite(ms) || ms <= 0) continue;
      if (!agg[name]) agg[name] = { total: 0, count: 0 };
      agg[name].total += ms;
      agg[name].count += 1;
    }
  }

  const sorted = Object.entries(agg).sort((a, b) => b[1].total - a[1].total);
  if (sorted.length === 0) {
    el.appendChild(emptyMsg('No kernel timings recorded.'));
    return;
  }

  const grandTotal = sorted.reduce((s, [, v]) => s + v.total, 0);
  const maxVal = sorted[0][1].total;
  const visible = sorted.slice(0, MAX_KERNEL_ROWS);
  const hidden = sorted.slice(MAX_KERNEL_ROWS);
  const hiddenTotal = hidden.reduce((sum, [, value]) => sum + value.total, 0);

  // Stacked overview bar
  const stack = document.createElement('div');
  stack.className = 'xray-gpu-stack';
  for (const [name, { total }] of sorted) {
    const pct = grandTotal > 0 ? (total / grandTotal) * 100 : 0;
    if (pct < 1) continue;
    const seg = document.createElement('div');
    seg.className = `xray-gpu-segment xray-bar--${classifyKernel(name)}`;
    seg.style.width = `${pct.toFixed(1)}%`;
    seg.title = `${name}: ${fmtMs(total)} (${pct.toFixed(0)}%)`;
    seg.textContent = pct > 12 ? name.split('/').pop() : '';
    stack.appendChild(seg);
  }
  el.appendChild(stack);

  // Individual bars
  for (const [name, { total, count }] of visible) {
    const pctLabel = grandTotal > 0 ? `${((total / grandTotal) * 100).toFixed(0)}%` : '';
    const label = `${name} (${count}x)`;
    const valueLabel = `${fmtMs(total)} ${pctLabel}`;
    const row = barRow(label, total, maxVal, `xray-bar--${classifyKernel(name)}`, valueLabel);
    if (explainContext) {
      const metadata = resolveExplainMetadata(name, explainContext, total);
      if (metadata) attachExplainInteraction(row, metadata);
    }
    el.appendChild(row);
  }
  if (hidden.length > 0) {
    const pctLabel = grandTotal > 0 ? `${((hiddenTotal / grandTotal) * 100).toFixed(0)}%` : '';
    el.appendChild(
      barRow(
        `${hidden.length} smaller kernels`,
        hiddenTotal,
        maxVal,
        'xray-bar--other',
        `${fmtMs(hiddenTotal)} ${pctLabel}`
      )
    );
  }

  const legend = document.createElement('div');
  legend.className = 'xray-legend';
  const limitLabel = hidden.length > 0
    ? `Showing top ${visible.length} of ${sorted.length} kernels`
    : `${sorted.length} kernels`;
  legend.innerHTML = `<span>${limitLabel}, ${steps.length} steps, ${fmtMs(grandTotal)} total</span>`;
  el.appendChild(legend);
}

// ---------------------------------------------------------------------------
// D. GPU Pipeline
// ---------------------------------------------------------------------------

function renderGPUPipeline(el, stats) {
  el.innerHTML = '';
  const record = stats.decodeRecordMs ?? 0;
  const submit = stats.decodeSubmitWaitMs ?? 0;
  const readback = stats.decodeReadbackWaitMs ?? 0;

  if (record === 0 && submit === 0 && readback === 0) {
    el.appendChild(emptyMsg('Waiting for generation.'));
    return;
  }

  const total = record + submit + readback;

  // Stacked bar showing proportions
  const phases = [
    { label: 'Record', ms: record, cls: 'xray-bar--record' },
    { label: 'Submit/Wait', ms: submit, cls: 'xray-bar--submit' },
    { label: 'Readback', ms: readback, cls: 'xray-bar--readback' },
  ];

  const stack = document.createElement('div');
  stack.className = 'xray-gpu-stack';
  for (const p of phases) {
    if (p.ms <= 0) continue;
    const seg = document.createElement('div');
    seg.className = `xray-gpu-segment ${p.cls}`;
    seg.style.width = `${((p.ms / total) * 100).toFixed(1)}%`;
    seg.textContent = p.ms > total * 0.15 ? fmtMs(p.ms) : '';
    seg.title = `${p.label}: ${fmtMs(p.ms)} (${((p.ms / total) * 100).toFixed(0)}%)`;
    stack.appendChild(seg);
  }
  el.appendChild(stack);

  const phaseLegend = document.createElement('div');
  phaseLegend.className = 'xray-legend';
  phaseLegend.innerHTML = `
    <span><span class="xray-legend-dot xray-bar--record"></span>Record</span>
    <span><span class="xray-legend-dot xray-bar--submit"></span>Submit/Wait</span>
    <span><span class="xray-legend-dot xray-bar--readback"></span>Readback</span>
  `;
  el.appendChild(phaseLegend);

  // Breakdown bars
  const maxVal = Math.max(record, submit, readback, 0.001);
  for (const p of phases) {
    el.appendChild(barRow(p.label, p.ms, maxVal, p.cls));
  }

  // Legend + total
  const legend = document.createElement('div');
  legend.className = 'xray-legend';
  legend.innerHTML = `<span>Total: ${fmtMs(total)}</span>`;
  el.appendChild(legend);
}

// ---------------------------------------------------------------------------
// E. Execution Plan
// ---------------------------------------------------------------------------

function renderExecPlan(el, stats) {
  el.innerHTML = '';
  const plan = stats.executionPlan;
  const kpId = stats.kernelPathId ?? '--';
  const kpSrc = stats.kernelPathSource ?? 'none';

  const pairs = [
    ['Kernel Path', kpId],
    ['Source', kpSrc],
  ];

  if (plan) {
    const primary = plan.primary;
    if (primary) {
      if (primary.activationDtype) pairs.push(['Activation dtype', primary.activationDtype]);
      if (Number.isFinite(primary.batchSize)) pairs.push(['Configured decode batch', String(primary.batchSize)]);
      if (Number.isFinite(primary.readbackInterval)) pairs.push(['Readback interval', String(primary.readbackInterval)]);
      if (primary.stopCheckMode) pairs.push(['Stop check mode', primary.stopCheckMode]);
      pairs.push(['Command batching', primary.disableCommandBatching === true ? 'disabled' : 'enabled']);
    }
    if (plan.activePlanIdAtStart) pairs.push(['Plan at start', plan.activePlanIdAtStart]);
    if (plan.finalActivePlanId && plan.finalActivePlanId !== plan.activePlanIdAtStart) {
      pairs.push(['Final plan', plan.finalActivePlanId]);
    }
  }
  pairs.push(['Observed decode', formatObservedDecodeLabel(stats)]);
  const batchGuardReason = formatBatchGuardReason(stats.batchGuardReason);
  if (batchGuardReason) {
    pairs.push(['Batch guard', batchGuardReason]);
  }

  appendKeyValueList(el, pairs);

  const transitions = plan?.transitions;
  if (Array.isArray(transitions) && transitions.length > 0) {
    const header = document.createElement('div');
    header.className = 'xray-section-header';
    header.textContent = 'Transitions';
    header.style.marginTop = '6px';
    el.appendChild(header);

    for (const t of transitions) {
      const entry = document.createElement('div');
      entry.className = 'xray-transition';
      entry.innerHTML = `
        <span class="xray-transition-step">step ${t.step ?? '?'}</span>
        <span>${t.from ?? '?'} → ${t.to ?? '?'}</span>
        <span class="xray-transition-reason">${t.reason ?? ''}</span>
      `;
      el.appendChild(entry);
    }
  } else {
    const note = document.createElement('div');
    note.className = 'xray-empty';
    note.style.marginTop = '4px';
    note.textContent = 'No transitions (stable plan)';
    el.appendChild(note);
  }
}

// ---------------------------------------------------------------------------
// F. Memory Breakdown (full label stats)
// ---------------------------------------------------------------------------

function renderMemoryBreakdown(el, stats, memStats, pipeline) {
  el.innerHTML = '';
  const pool = pipeline?.getBufferPool?.();
  const labelStats = typeof pool?.getLabelStats === 'function' ? pool.getLabelStats() : null;

  if (!labelStats || labelStats.length === 0) {
    el.appendChild(emptyMsg('No tracked buffers.'));
    return;
  }

  const sorted = [...labelStats].sort((a, b) => (b.bytes || 0) - (a.bytes || 0));
  const visible = sorted.slice(0, MAX_MEMORY_ROWS);
  const hidden = sorted.slice(MAX_MEMORY_ROWS);
  const hiddenBytes = hidden.reduce((sum, entry) => sum + (entry.bytes || 0), 0);
  const hiddenCount = hidden.reduce((sum, entry) => sum + (entry.count || 0), 0);
  if (hidden.length > 0) {
    visible.push({
      label: `${hidden.length} smaller labels`,
      bytes: hiddenBytes,
      count: hiddenCount,
    });
  }
  const maxBytes = visible[0]?.bytes || 1;

  for (const entry of visible) {
    const bytes = entry.bytes || 0;
    const count = entry.count || 0;
    const label = `${entry.label || 'unlabeled'} (${count})`;
    el.appendChild(barRow(label, bytes, maxBytes, `xray-bar--${classifyKernel(entry.label || '')}`, formatBytes(bytes)));
  }

  const note = document.createElement('div');
  note.className = 'xray-empty';
  note.style.marginTop = '4px';
  note.textContent = `${sorted.length} labels, ${sorted.reduce((sum, entry) => sum + (entry.count || 0), 0)} buffers, ${formatBytes(sorted.reduce((sum, entry) => sum + (entry.bytes || 0), 0))} total`;
  el.appendChild(note);
}

// ---------------------------------------------------------------------------
// G. Batching Stats
// ---------------------------------------------------------------------------

function renderBatchingStats(el, stats, memStats, pipeline) {
  el.innerHTML = '';
  const summaryPairs = [
    ['Observed decode', formatObservedDecodeLabel(stats)],
  ];
  const batchGuardReason = formatBatchGuardReason(stats.batchGuardReason);
  if (batchGuardReason) {
    summaryPairs.push(['Batch guard', batchGuardReason]);
  }
  appendKeyValueList(el, summaryPairs);

  if (isTokenPressMode()) {
    const note = emptyMsg('Batching stats are unavailable in Token Press mode. Token Press drives per-step decode rather than the normal run-ahead generate path.');
    note.style.marginTop = '6px';
    el.appendChild(note);
    return;
  }
  const bs = typeof pipeline?.getBatchingStats === 'function' ? pipeline.getBatchingStats() : null;

  if (!bs) {
    const note = emptyMsg('No batching data.');
    note.style.marginTop = '6px';
    el.appendChild(note);
    return;
  }

  const pairs = [
    ['Batched calls', String(bs.batchedForwardCalls ?? 0)],
    ['Unbatched calls', String(bs.unbatchedForwardCalls ?? 0)],
    ['Batched time', fmtMs(bs.totalBatchedTimeMs)],
    ['Unbatched time', fmtMs(bs.totalUnbatchedTimeMs)],
    ['GPU submissions', String(bs.gpuSubmissions ?? 0)],
  ];
  const detailList = document.createElement('div');
  detailList.style.marginTop = '6px';
  el.appendChild(detailList);
  appendKeyValueList(detailList, pairs);

  const batched = bs.batchedForwardCalls ?? 0;
  const unbatched = bs.unbatchedForwardCalls ?? 0;
  const maxCalls = Math.max(batched, unbatched, 1);
  if (batched > 0 || unbatched > 0) {
    const sep = document.createElement('div');
    sep.style.marginTop = '6px';
    el.appendChild(sep);
    el.appendChild(barRow('Batched', batched, maxCalls, 'xray-bar--batched', String(batched)));
    el.appendChild(barRow('Unbatched', unbatched, maxCalls, 'xray-bar--unbatched', String(unbatched)));
  }
}
