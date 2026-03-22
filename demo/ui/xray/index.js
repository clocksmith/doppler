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

import { formatBytes } from '@simulatte/doppler';
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
let kvSnapshots = [];
let initialized = false;

const MAX_WATERFALL_STEPS = 32;
const MAX_KV_SNAPSHOTS = 50;

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

export function initXray() {
  if (initialized) return;
  initialized = true;

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
  kvSnapshots = [];
  for (const entry of Object.values(panelEls)) {
    entry.content.innerHTML = '';
  }
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

function classifyKernel(name) {
  const n = (name || '').toLowerCase();
  if (n.includes('attn') || n.includes('attention') || n.includes('rope') || n.includes('qkv')) return 'attention';
  if (n.includes('ffn') || n.includes('mlp') || n.includes('gate') || n.includes('expert')) return 'ffn';
  if (n.includes('embed')) return 'embed';
  if (n.includes('norm') || n.includes('rms')) return 'norm';
  if (n.includes('sample') || n.includes('logit') || n.includes('softmax') || n.includes('softcap')) return 'sample';
  return 'other';
}

// ---------------------------------------------------------------------------
// A. Decode Waterfall
// ---------------------------------------------------------------------------

function renderDecodeWaterfall(el, stats) {
  el.innerHTML = '';
  const steps = stats.decodeProfileSteps;
  if (!Array.isArray(steps) || steps.length === 0) {
    el.appendChild(emptyMsg('No profile steps — enable profiling in runtime config.'));
    return;
  }

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
    ['norm', 'Norm'], ['sample', 'Sample'], ['other', 'Other'],
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
  if (kvStats) {
    kvSnapshots.push({
      seqLen: kvStats.seqLen ?? 0,
      maxSeqLen: kvStats.maxSeqLen ?? 1,
      allocated: kvStats.allocated ?? 0,
      used: kvStats.used ?? 0,
    });
    if (kvSnapshots.length > MAX_KV_SNAPSHOTS) kvSnapshots.shift();
  }

  if (kvSnapshots.length === 0) {
    el.appendChild(emptyMsg('No KV cache data yet.'));
    return;
  }

  const maxSeq = Math.max(...kvSnapshots.map(s => s.maxSeqLen), 1);

  const timeline = document.createElement('div');
  timeline.className = 'xray-kv-timeline';

  for (const snap of kvSnapshots) {
    const bar = document.createElement('div');
    bar.className = 'xray-kv-bar';
    const pct = (snap.seqLen / maxSeq) * 100;
    bar.style.height = `${Math.max(2, pct).toFixed(1)}%`;
    bar.title = `seq ${snap.seqLen} / ${snap.maxSeqLen}`;
    timeline.appendChild(bar);
  }
  el.appendChild(timeline);

  const last = kvSnapshots[kvSnapshots.length - 1];
  const utilPct = last.maxSeqLen > 0 ? ((last.seqLen / last.maxSeqLen) * 100).toFixed(0) : 0;
  const meta = document.createElement('div');
  meta.className = 'xray-kv-meta';
  meta.innerHTML = `<span>seq ${last.seqLen} / ${last.maxSeqLen} (${utilPct}%)</span><span>${Number.isFinite(last.allocated) ? formatBytes(last.allocated) : '--'}</span>`;
  el.appendChild(meta);
}

// ---------------------------------------------------------------------------
// C. Kernel Timing Breakdown
// ---------------------------------------------------------------------------

function renderKernelTiming(el, stats) {
  el.innerHTML = '';
  const steps = stats.decodeProfileSteps;
  if (!Array.isArray(steps) || steps.length === 0) {
    el.appendChild(emptyMsg('No profile steps — enable profiling in runtime config.'));
    return;
  }

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
  for (const [name, { total, count }] of sorted) {
    const pctLabel = grandTotal > 0 ? `${((total / grandTotal) * 100).toFixed(0)}%` : '';
    const label = `${name} (${count}x)`;
    const valueLabel = `${fmtMs(total)} ${pctLabel}`;
    el.appendChild(barRow(label, total, maxVal, `xray-bar--${classifyKernel(name)}`, valueLabel));
  }

  const legend = document.createElement('div');
  legend.className = 'xray-legend';
  legend.innerHTML = `<span>${sorted.length} kernels, ${steps.length} steps, ${fmtMs(grandTotal)} total</span>`;
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
    el.appendChild(emptyMsg('No GPU timing data.'));
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

  const grid = document.createElement('div');
  grid.className = 'xray-kv-list';

  const pairs = [
    ['Kernel Path', kpId],
    ['Source', kpSrc],
  ];

  if (plan) {
    const primary = plan.primary;
    if (primary) {
      if (primary.activationDtype) pairs.push(['Activation dtype', primary.activationDtype]);
      if (Number.isFinite(primary.batchSize)) pairs.push(['Batch size', String(primary.batchSize)]);
    }
    if (plan.activePlanIdAtStart) pairs.push(['Plan at start', plan.activePlanIdAtStart]);
    if (plan.finalActivePlanId && plan.finalActivePlanId !== plan.activePlanIdAtStart) {
      pairs.push(['Final plan', plan.finalActivePlanId]);
    }
  }

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
  const maxBytes = sorted[0]?.bytes || 1;
  let totalBytes = 0;
  let totalCount = 0;

  for (const entry of sorted) {
    const bytes = entry.bytes || 0;
    const count = entry.count || 0;
    totalBytes += bytes;
    totalCount += count;
    const label = `${entry.label || 'unlabeled'} (${count})`;
    el.appendChild(barRow(label, bytes, maxBytes, 'xray-bar--mem', formatBytes(bytes)));
  }

  const note = document.createElement('div');
  note.className = 'xray-empty';
  note.style.marginTop = '4px';
  note.textContent = `${sorted.length} labels, ${totalCount} buffers, ${formatBytes(totalBytes)} total`;
  el.appendChild(note);
}

// ---------------------------------------------------------------------------
// G. Batching Stats
// ---------------------------------------------------------------------------

function renderBatchingStats(el, stats, memStats, pipeline) {
  el.innerHTML = '';
  const bs = typeof pipeline?.getBatchingStats === 'function' ? pipeline.getBatchingStats() : null;

  if (!bs) {
    el.appendChild(emptyMsg('No batching data.'));
    return;
  }

  const grid = document.createElement('div');
  grid.className = 'xray-kv-list';
  const pairs = [
    ['Batched calls', String(bs.batchedForwardCalls ?? 0)],
    ['Unbatched calls', String(bs.unbatchedForwardCalls ?? 0)],
    ['Batched time', fmtMs(bs.totalBatchedTimeMs)],
    ['Unbatched time', fmtMs(bs.totalUnbatchedTimeMs)],
    ['GPU submissions', String(bs.gpuSubmissions ?? 0)],
  ];
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
