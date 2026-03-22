import { state } from '../state.js';
import { $, setText } from '../dom.js';
import { buildKernelPathBuilderRuntimeOverlay } from '../../../src/tooling/kernel-path-builder/runtime-overlay.js';

const KERNEL_PATH_BUILDER_INDEX_URL = typeof window === 'object' && window.location?.origin
  ? new URL('/demo/data/kernel-path-builder-index.json', window.location.origin).toString()
  : new URL('../../data/kernel-path-builder-index.json', import.meta.url).toString();

function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function formatNumber(value, digits = 2) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return '--';
  return numeric.toFixed(digits);
}

function formatInteger(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return '--';
  return String(Math.round(numeric));
}

function formatList(values) {
  if (!Array.isArray(values) || values.length === 0) {
    return 'None';
  }
  return values.join(', ');
}

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function getBuilderPayload() {
  return state.kernelPathBuilderIndex && typeof state.kernelPathBuilderIndex === 'object'
    ? state.kernelPathBuilderIndex
    : null;
}

function getBuilderModels() {
  const payload = getBuilderPayload();
  return Array.isArray(payload?.models) ? payload.models : [];
}

function findBuilderModel(modelId) {
  const normalized = normalizeText(modelId);
  if (!normalized) return null;
  return getBuilderModels().find((entry) => entry?.modelId === normalized) || null;
}

function ensureSelectedBuilderModelId() {
  const models = getBuilderModels();
  if (models.length === 0) {
    state.kernelPathBuilderModelId = null;
    return null;
  }
  const current = findBuilderModel(state.kernelPathBuilderModelId);
  if (current) {
    return current.modelId;
  }
  const activeModelMatch = findBuilderModel(state.activeModelId);
  if (activeModelMatch) {
    state.kernelPathBuilderModelId = activeModelMatch.modelId;
    return activeModelMatch.modelId;
  }
  const lastReportMatch = findBuilderModel(state.lastReport?.modelId);
  if (lastReportMatch) {
    state.kernelPathBuilderModelId = lastReportMatch.modelId;
    return lastReportMatch.modelId;
  }
  state.kernelPathBuilderModelId = models[0].modelId;
  return state.kernelPathBuilderModelId;
}

function populateBuilderModelSelect() {
  const select = $('kernel-builder-model-select');
  if (!(select instanceof HTMLSelectElement)) return;
  const models = getBuilderModels();
  const selectedId = ensureSelectedBuilderModelId();
  select.innerHTML = '';
  if (models.length === 0) {
    const option = document.createElement('option');
    option.value = '';
    option.textContent = 'No builder models indexed';
    select.appendChild(option);
    return;
  }
  for (const model of models) {
    const option = document.createElement('option');
    option.value = model.modelId;
    option.textContent = model.modelId;
    option.selected = model.modelId === selectedId;
    select.appendChild(option);
  }
}

function buildRuntimeOverlay(model) {
  const report = state.kernelPathBuilderOverlayReport;
  if (!report || typeof report !== 'object') {
    return null;
  }
  const overlay = buildKernelPathBuilderRuntimeOverlay(model, report);
  if (!overlay) {
    return null;
  }
  return {
    ...overlay,
    source: normalizeText(state.kernelPathBuilderOverlaySource) || overlay.source || 'report',
  };
}

function renderSummary(model, payload, overlay) {
  const summary = $('kernel-builder-summary');
  if (!summary) return;
  if (!model) {
    summary.innerHTML = '<div class="kernel-builder-empty type-caption">No kernel-path builder model selected.</div>';
    return;
  }
  const exactPathId = Array.isArray(model.runtime?.exactKernelPathIds) && model.runtime.exactKernelPathIds.length > 0
    ? model.runtime.exactKernelPathIds[0]
    : null;
  const loweredPathId = Array.isArray(model.runtime?.kernelPathIds) && model.runtime.kernelPathIds.length > 0
    ? model.runtime.kernelPathIds[0]
    : null;
  const sharedPathModels = loweredPathId
    ? (payload?.reverseIndexes?.kernelPaths?.[loweredPathId] || []).filter((entry) => entry !== model.modelId)
    : [];
  const sharedGraphModels = (payload?.reverseIndexes?.executionGraphs?.[model.execution?.signature] || [])
    .filter((entry) => entry !== model.modelId);
  const overlayTransitions = Array.isArray(overlay?.executionPlan?.transitions)
    ? overlay.executionPlan.transitions.length
    : 0;
  const sourceLabels = Array.isArray(model.sources)
    ? model.sources.map((source) => source.sourceKind || source.sourceLabel || 'source')
    : [];

  summary.innerHTML = [
    {
      label: 'Model',
      value: escapeHtml(model.modelId),
      detail: escapeHtml(model.configPath),
    },
    {
      label: 'Execution Signature',
      value: escapeHtml(model.execution?.signature || '--'),
      detail: `Layer count: ${escapeHtml(formatInteger(model.layerPattern?.layerCount))}`,
    },
    {
      label: 'Lowering',
      value: escapeHtml(model.runtime?.actualLowering || '--'),
      detail: `Inline enabled: ${model.runtime?.inlineKernelPathEnabled ? 'yes' : 'no'}`,
    },
    {
      label: 'Lowered Path',
      value: escapeHtml(loweredPathId || '--'),
      detail: exactPathId ? `Exact match: ${escapeHtml(exactPathId)}` : 'Proposal-only or execution-graph-only',
    },
    {
      label: 'Shared Graph Models',
      value: escapeHtml(formatInteger(sharedGraphModels.length)),
      detail: escapeHtml(formatList(sharedGraphModels.slice(0, 4))),
    },
    {
      label: 'Shared Lowered Models',
      value: escapeHtml(formatInteger(sharedPathModels.length)),
      detail: loweredPathId ? escapeHtml(formatList(sharedPathModels.slice(0, 4))) : 'No lowered path group',
    },
    {
      label: 'Session Dtypes',
      value: escapeHtml(`${model.session?.compute?.activationDtype || '--'} / ${model.session?.kvDtype || '--'}`),
      detail: `math=${escapeHtml(model.session?.compute?.mathDtype || '--')} output=${escapeHtml(model.session?.compute?.outputDtype || '--')}`,
    },
    {
      label: 'Sources',
      value: escapeHtml(formatInteger(model.sources?.length || 0)),
      detail: model.sourceConsistency?.ok === false
        ? `Drift across ${escapeHtml(formatInteger(model.sourceConsistency.comparedSources))} sources`
        : escapeHtml(formatList(sourceLabels)),
    },
    {
      label: 'Overlay',
      value: overlay ? escapeHtml(overlay.source) : 'None',
      detail: overlay
        ? `report=${escapeHtml(overlay.modelId || '--')} match=${overlay.matchesSelectedModel ? 'yes' : 'no'} transitions=${escapeHtml(formatInteger(overlayTransitions))}`
        : 'Use the latest diagnostics report or load JSON.',
    },
  ]
    .map((card) => `
      <article class="kernel-builder-card">
        <span class="kernel-builder-card-label type-caption">${card.label}</span>
        <strong class="kernel-builder-card-value">${card.value}</strong>
        <span class="kernel-builder-card-detail type-caption">${card.detail}</span>
      </article>
    `)
    .join('');
}

function resolveStepIo(step, model) {
  const aDtype = model.session?.compute?.activationDtype || '--';
  const kvDtype = model.session?.kvDtype || '--';
  const outDtype = model.session?.compute?.outputDtype || aDtype;
  const op = step.op || '';
  const isKvWrite = op === 'k_proj' || op === 'v_proj' || op === 'rope_k';
  const isAttention = op === 'attention';
  const isEmbed = op === 'embed';
  const isFinalNorm = op === 'final_norm';
  const isLmHead = op === 'lm_head';
  const inputDtype = isEmbed ? 'int32' : isAttention ? `${aDtype}+${kvDtype}` : aDtype;
  let outputDtype = aDtype;
  if (isKvWrite) outputDtype = kvDtype;
  else if (isLmHead) outputDtype = outDtype;
  else if (isFinalNorm) outputDtype = aDtype;
  else if (isEmbed) outputDtype = aDtype;
  const hasWeights = !!step.weights;
  const inputLabel = hasWeights ? `act + weights` : `act`;
  const outputLabel = isKvWrite ? `kv` : `act`;
  return { inputDtype, outputDtype, inputLabel, outputLabel };
}

function renderExecution(model, overlay) {
  const container = $('kernel-builder-execution');
  if (!container) return;
  if (!model) {
    container.innerHTML = '<div class="kernel-builder-empty type-caption">No execution graph available.</div>';
    return;
  }
  const phases = [
    ['preLayer', 'pre-layer'],
    ['prefill', 'prefill'],
    ['decode', 'decode'],
    ['postLayer', 'post-layer'],
  ];
  const allCards = [];
  for (const [key, phaseLabel] of phases) {
    const steps = Array.isArray(model.execution?.sections?.[key]) ? model.execution.sections[key] : [];
    for (const step of steps) {
      const stepTiming = overlay?.stepTimingsById?.[step.id] ?? null;
      const io = resolveStepIo(step, model);
      const timingHtml = stepTiming
        ? `<span class="kernel-builder-step-timing">${escapeHtml(formatNumber(stepTiming.totalMs, 3))}ms</span>`
        : '';
      allCards.push(`
        <article class="kernel-builder-step-card">
          <div class="kernel-builder-step-card-header">
            <span class="kernel-builder-step-card-phase type-caption">${escapeHtml(phaseLabel)}</span>
            ${timingHtml}
          </div>
          <strong class="kernel-builder-step-card-op">${escapeHtml(step.op || '--')}</strong>
          <span class="kernel-builder-step-card-kernel type-caption">${escapeHtml(step.kernel || '--')}#${escapeHtml(step.entry || 'main')}</span>
          <div class="kernel-builder-step-card-io">
            <div class="kernel-builder-step-card-io-col">
              <span class="kernel-builder-step-card-io-label type-caption">in</span>
              <span class="kernel-builder-step-card-io-dtype">${escapeHtml(io.inputDtype)}</span>
              <span class="kernel-builder-step-card-io-buf type-caption">${escapeHtml(io.inputLabel)}</span>
            </div>
            <span class="kernel-builder-step-card-io-arrow">\u2192</span>
            <div class="kernel-builder-step-card-io-col">
              <span class="kernel-builder-step-card-io-label type-caption">out</span>
              <span class="kernel-builder-step-card-io-dtype">${escapeHtml(io.outputDtype)}</span>
              <span class="kernel-builder-step-card-io-buf type-caption">${escapeHtml(io.outputLabel)}</span>
            </div>
          </div>
          <div class="kernel-builder-step-card-meta type-caption">
            ${step.weights ? `<span>weights: ${escapeHtml(step.weights)}</span>` : ''}
            <span>layers: ${escapeHtml(Array.isArray(step.layers) ? step.layers.join(', ') : String(step.layers ?? 'all'))}</span>
          </div>
        </article>
      `);
    }
  }
  if (allCards.length === 0) {
    allCards.push('<div class="kernel-builder-empty type-caption">No execution steps.</div>');
  }

  const factRows = Array.isArray(model.customRuntimeFacts) && model.customRuntimeFacts.length > 0
    ? model.customRuntimeFacts.map((fact) => `
        <article class="kernel-builder-fact">
          <strong>${escapeHtml(fact.label || fact.id || 'fact')}</strong>
          <div class="type-caption">${escapeHtml(fact.summary || '')}</div>
          <div class="type-caption">Refs: ${escapeHtml((fact.sourceRefs || []).join(', '))}</div>
        </article>
      `).join('')
    : '<div class="type-caption">No custom runtime facts recorded for this model.</div>';
  const overlaySummary = overlay
    ? `
      <div class="kernel-builder-fact">
        <strong>Runtime Overlay</strong>
        <div class="type-caption">kernelPath=${escapeHtml(overlay.kernelPathId || '--')} source=${escapeHtml(overlay.kernelPathSource || '--')}</div>
        <div class="type-caption">unmatched timers=${escapeHtml(formatInteger(overlay.unmatchedTimingLabels?.length || 0))}</div>
      </div>
    `
    : '';

  container.innerHTML = `
    <div class="kernel-builder-step-grid">${allCards.join('')}</div>
    <div class="kernel-builder-facts">
      <h4 class="type-caption">Custom Runtime Facts</h4>
      ${factRows}
      ${overlaySummary}
    </div>
  `;
}

function renderMatches(model) {
  const container = $('kernel-builder-matches');
  if (!container) return;
  if (!model) {
    container.innerHTML = '<div class="kernel-builder-empty type-caption">No match data available.</div>';
    return;
  }
  if (model.candidate?.error) {
    container.innerHTML = `
      <article class="kernel-builder-match is-error">
        <strong>Candidate synthesis failed</strong>
        <div class="type-caption">${escapeHtml(model.candidate.error)}</div>
      </article>
    `;
    return;
  }
  const matches = Array.isArray(model.candidate?.closestMatches) ? model.candidate.closestMatches : [];
  if (matches.length === 0) {
    container.innerHTML = '<div class="kernel-builder-empty type-caption">No candidate matches recorded.</div>';
    return;
  }
  container.innerHTML = matches.map((match) => {
    const detailItems = Array.isArray(match.mismatchDetails) ? match.mismatchDetails.slice(0, 6) : [];
    return `
      <article class="kernel-builder-match ${match.exact ? 'is-exact' : ''}">
        <div class="kernel-builder-match-top">
          <strong>${escapeHtml(match.id)}</strong>
          <span class="type-caption">mismatches: ${escapeHtml(formatInteger(match.mismatchCount))}</span>
        </div>
        <div class="type-caption">status: ${escapeHtml(match.status || '--')} ${match.statusReason ? `· ${escapeHtml(match.statusReason)}` : ''}</div>
        <div class="type-caption">${escapeHtml(match.notes || '')}</div>
        <ul class="kernel-builder-reason-list">
          ${detailItems.length > 0
            ? detailItems.map((detail) => `
                <li>
                  <strong>${escapeHtml(detail.label || detail.code || 'Mismatch')}</strong>
                  <div class="type-caption">${escapeHtml(detail.category || 'shape')} ${detail.phase ? `· ${escapeHtml(detail.phase)}` : ''}</div>
                  ${detail.repairHint ? `<div class="type-caption">${escapeHtml(detail.repairHint)}</div>` : ''}
                </li>
              `).join('')
            : '<li>Exact structural match.</li>'}
        </ul>
      </article>
    `;
  }).join('');
}

function renderProposal(model, overlay) {
  const proposalPre = $('kernel-builder-proposal-json');
  const overlayEl = $('kernel-builder-overlay');
  const proposal = model?.candidate?.proposal ?? null;
  const verification = proposal?.verification ?? null;
  if (proposalPre) {
    proposalPre.textContent = proposal
      ? JSON.stringify(proposal, null, 2)
      : 'No proposal available for this model.';
  }
  if (!overlayEl) return;
  const verificationChecks = Array.isArray(verification?.checks) ? verification.checks : [];
  const verificationErrors = Array.isArray(verification?.errors) ? verification.errors : [];
  const proposalHtml = `
    <div class="kernel-builder-overlay-summary">
      <div class="kernel-builder-overlay-row"><span class="type-caption">Proposal</span><strong>${escapeHtml(proposal?.kind || '--')}</strong></div>
      <div class="kernel-builder-overlay-row"><span class="type-caption">Verification</span><strong>${proposal ? (verification?.ok === true ? 'pass' : 'fail') : 'none'}</strong></div>
      <div class="kernel-builder-overlay-row"><span class="type-caption">Compiled plan</span><strong>${escapeHtml(verification?.compiledPlan?.primaryPlanId || '--')}</strong></div>
      <div class="kernel-builder-overlay-row"><span class="type-caption">Proposal path</span><strong>${escapeHtml(proposal?.selectedKernelPathId || proposal?.path?.id || '--')}</strong></div>
      <div class="kernel-builder-overlay-row"><span class="type-caption">Verification checks</span><strong>${escapeHtml(formatInteger(verificationChecks.filter((entry) => entry?.ok === true).length))}/${escapeHtml(formatInteger(verificationChecks.length))}</strong></div>
    </div>
    <div class="kernel-builder-overlay-timers">
      <h4 class="type-caption">Verification errors</h4>
      <ul class="kernel-builder-reason-list">
        ${verificationErrors.length > 0
          ? verificationErrors.map((error) => `<li>${escapeHtml(error)}</li>`).join('')
          : '<li>No verification errors recorded.</li>'}
      </ul>
    </div>
  `;
  if (!overlay) {
    overlayEl.innerHTML = `
      ${proposalHtml}
      <div class="kernel-builder-overlay-timers">
        <h4 class="type-caption">Runtime Overlay</h4>
        <ul class="kernel-builder-reason-list">
          <li>No runtime overlay loaded. Run Diagnostics or load a report JSON.</li>
        </ul>
      </div>
    `;
    return;
  }
  const memoryKv = overlay.memory?.kvCache && typeof overlay.memory.kvCache === 'object'
    ? overlay.memory.kvCache
    : null;
  const topTimers = Array.isArray(overlay.topDecodeTimers) ? overlay.topDecodeTimers : [];
  const executionPlan = overlay.executionPlan && typeof overlay.executionPlan === 'object'
    ? overlay.executionPlan
    : null;
  overlayEl.innerHTML = `
    ${proposalHtml}
    <div class="kernel-builder-overlay-summary">
      <div class="kernel-builder-overlay-row"><span class="type-caption">Report</span><strong>${escapeHtml(overlay.modelId || '--')}</strong></div>
      <div class="kernel-builder-overlay-row"><span class="type-caption">Timestamp</span><strong>${escapeHtml(overlay.timestamp || '--')}</strong></div>
      <div class="kernel-builder-overlay-row"><span class="type-caption">Profile</span><strong>${escapeHtml(overlay.runtimeProfile || '--')}</strong></div>
      <div class="kernel-builder-overlay-row"><span class="type-caption">Match selected model</span><strong>${overlay.matchesSelectedModel ? 'yes' : 'no'}</strong></div>
      <div class="kernel-builder-overlay-row"><span class="type-caption">Load ms</span><strong>${escapeHtml(formatNumber(overlay.modelLoadMs, 2))}</strong></div>
      <div class="kernel-builder-overlay-row"><span class="type-caption">TTFT ms</span><strong>${escapeHtml(formatNumber(overlay.firstTokenMs, 2))}</strong></div>
      <div class="kernel-builder-overlay-row"><span class="type-caption">Prefill tok/s</span><strong>${escapeHtml(formatNumber(overlay.prefillTokensPerSec, 2))}</strong></div>
      <div class="kernel-builder-overlay-row"><span class="type-caption">Decode tok/s</span><strong>${escapeHtml(formatNumber(overlay.decodeTokensPerSec, 2))}</strong></div>
      <div class="kernel-builder-overlay-row"><span class="type-caption">GPU waits</span><strong>record=${escapeHtml(formatNumber(overlay.gpu?.decodeRecordMs, 2))} submit=${escapeHtml(formatNumber(overlay.gpu?.decodeSubmitWaitMs, 2))} readback=${escapeHtml(formatNumber(overlay.gpu?.decodeReadbackWaitMs, 2))}</strong></div>
      <div class="kernel-builder-overlay-row"><span class="type-caption">Tracked memory</span><strong>${escapeHtml(formatInteger(overlay.memory?.used))}</strong></div>
      <div class="kernel-builder-overlay-row"><span class="type-caption">KV layout</span><strong>${escapeHtml(memoryKv?.layout || '--')}</strong></div>
      <div class="kernel-builder-overlay-row"><span class="type-caption">Execution plan</span><strong>${escapeHtml(executionPlan?.finalActivePlanId || executionPlan?.primary?.id || '--')}</strong></div>
      <div class="kernel-builder-overlay-row"><span class="type-caption">Plan transitions</span><strong>${escapeHtml(formatInteger(executionPlan?.transitions?.length || 0))}</strong></div>
      <div class="kernel-builder-overlay-row"><span class="type-caption">Kernel path</span><strong>${escapeHtml(overlay.kernelPathId || '--')} (${escapeHtml(overlay.kernelPathSource || '--')})</strong></div>
    </div>
    <div class="kernel-builder-overlay-timers">
      <h4 class="type-caption">Top decode timers</h4>
      <ul class="kernel-builder-reason-list">
        ${topTimers.length > 0
          ? topTimers.map((timer) => `<li>${escapeHtml(timer.label)} · ${escapeHtml(formatNumber(timer.totalMs, 3))}ms</li>`).join('')
          : '<li>No decodeProfileSteps in this report.</li>'}
      </ul>
    </div>
    <div class="kernel-builder-overlay-timers">
      <h4 class="type-caption">Unmatched timing labels</h4>
      <ul class="kernel-builder-reason-list">
        ${Array.isArray(overlay.unmatchedTimingLabels) && overlay.unmatchedTimingLabels.length > 0
          ? overlay.unmatchedTimingLabels.slice(0, 6).map((entry) => `<li>${escapeHtml(entry.label)} · ${escapeHtml(formatNumber(entry.totalMs, 3))}ms</li>`).join('')
          : '<li>All timing labels mapped to execution steps.</li>'}
      </ul>
    </div>
  `;
}

export function renderKernelPathBuilderView() {
  populateBuilderModelSelect();
  const payload = getBuilderPayload();
  const status = $('kernel-builder-status');
  if (state.kernelPathBuilderLoading) {
    setText(status, 'Loading kernel-path builder index...');
    renderSummary(null, null, null);
    renderExecution(null);
    renderMatches(null);
    renderProposal(null, null);
    return;
  }
  if (state.kernelPathBuilderError) {
    setText(status, `Kernel-path builder index unavailable: ${state.kernelPathBuilderError}`);
    renderSummary(null, null, null);
    renderExecution(null);
    renderMatches(null);
    renderProposal(null, null);
    return;
  }
  const model = findBuilderModel(ensureSelectedBuilderModelId());
  const overlay = buildRuntimeOverlay(model);
  if (!payload) {
    setText(status, 'Kernel-path builder index not loaded.');
  } else {
    setText(
      status,
      `Indexed ${payload.stats?.models ?? 0} models, ${payload.stats?.registryKernelPaths ?? 0} kernel paths, ` +
      `${payload.stats?.exactKernelPathMatches ?? 0} exact structural matches.`
    );
  }
  renderSummary(model, payload, overlay);
  renderExecution(model, overlay);
  renderMatches(model);
  renderProposal(model, overlay);
}

export async function loadKernelPathBuilderIndex(options = {}) {
  const force = options?.force === true;
  if (!force && getBuilderPayload()) {
    renderKernelPathBuilderView();
    return getBuilderPayload();
  }
  state.kernelPathBuilderLoading = true;
  state.kernelPathBuilderError = null;
  renderKernelPathBuilderView();
  try {
    const response = await fetch(KERNEL_PATH_BUILDER_INDEX_URL, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    state.kernelPathBuilderIndex = await response.json();
    ensureSelectedBuilderModelId();
    return state.kernelPathBuilderIndex;
  } catch (error) {
    state.kernelPathBuilderIndex = null;
    state.kernelPathBuilderError = error instanceof Error ? error.message : String(error);
    return null;
  } finally {
    state.kernelPathBuilderLoading = false;
    renderKernelPathBuilderView();
  }
}

export function selectKernelPathBuilderModel(modelId) {
  state.kernelPathBuilderModelId = normalizeText(modelId) || null;
  renderKernelPathBuilderView();
}

export function applyLatestKernelPathBuilderReport() {
  if (!state.lastReport) {
    state.kernelPathBuilderError = 'No diagnostics report has been produced in this session.';
    renderKernelPathBuilderView();
    return;
  }
  state.kernelPathBuilderOverlayReport = state.lastReport;
  state.kernelPathBuilderOverlaySource = 'latest-diagnostics';
  state.kernelPathBuilderError = null;
  renderKernelPathBuilderView();
}

export async function applyKernelPathBuilderReportFile(file) {
  if (!file) return;
  const text = await file.text();
  const parsed = JSON.parse(text);
  state.kernelPathBuilderOverlayReport = parsed;
  state.kernelPathBuilderOverlaySource = file.name || 'report-file';
  state.kernelPathBuilderError = null;
  renderKernelPathBuilderView();
}

export function clearKernelPathBuilderReport() {
  state.kernelPathBuilderOverlayReport = null;
  state.kernelPathBuilderOverlaySource = null;
  renderKernelPathBuilderView();
}
