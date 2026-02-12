import { getRuntimeConfig, loadRuntimePreset } from '@doppler/core';
import { state } from '../state.js';
import { $, setHidden, setText } from '../dom.js';
import {
  BENCH_INTENTS,
  DEFAULT_RUNTIME_PRESET,
  DIAGNOSTICS_DEFAULTS,
  DIAGNOSTICS_SUITE_INFO,
  DIAGNOSTICS_SUITE_ORDER,
  RUNTIME_PRESET_REGISTRY,
} from '../constants.js';
import {
  getModelTypeForId,
  isModeModelSelectable,
  normalizeModelType,
} from '../models/utils.js';

function getDiagnosticsRequiredModelType(suite) {
  const key = String(suite || 'inference').trim().toLowerCase();
  if (key === 'kernels') return null;
  if (key === 'diffusion') return 'diffusion';
  if (key === 'energy') return 'energy';
  return 'text';
}

function suiteRequiresPrompt(suite) {
  const key = String(suite || '').trim().toLowerCase();
  return key === 'inference' || key === 'debug' || key === 'bench' || key === 'diffusion';
}

function suiteRequiresMaxTokens(suite, modelType) {
  const key = String(suite || '').trim().toLowerCase();
  if (normalizeModelType(modelType) === 'embedding') return false;
  return key === 'inference' || key === 'debug' || key === 'bench';
}

function isSuiteCompatibleModelType(modelType, suite) {
  const normalized = normalizeModelType(modelType);
  const required = getDiagnosticsRequiredModelType(suite);
  if (!required) return true;
  if (!normalized || normalized === 'unknown') return false;
  if (required === 'text') {
    return normalized !== 'diffusion' && normalized !== 'energy';
  }
  return normalized === required;
}

function formatDiagnosticsModelTypeLabel(requiredType) {
  if (requiredType === 'diffusion') return 'diffusion';
  if (requiredType === 'energy') return 'energy';
  return 'text (non-diffusion, non-energy)';
}

export function storeDiagnosticsSelection(mode, updates) {
  if (!mode) return;
  state.diagnosticsSelections[mode] = {
    ...(state.diagnosticsSelections[mode] || {}),
    ...(updates || {}),
  };
}

function getDiagnosticsSuiteOrder() {
  if (Array.isArray(DIAGNOSTICS_SUITE_ORDER) && DIAGNOSTICS_SUITE_ORDER.length > 0) {
    return DIAGNOSTICS_SUITE_ORDER.slice();
  }
  return Object.keys(DIAGNOSTICS_SUITE_INFO);
}

function getDiagnosticsBasePresetIds() {
  const ids = RUNTIME_PRESET_REGISTRY
    .filter((entry) => entry.base && typeof entry.id === 'string' && entry.id.length > 0)
    .map((entry) => entry.id);
  if (ids.length === 0) {
    return [DEFAULT_RUNTIME_PRESET];
  }
  if (ids.includes(DEFAULT_RUNTIME_PRESET)) {
    return [DEFAULT_RUNTIME_PRESET, ...ids.filter((id) => id !== DEFAULT_RUNTIME_PRESET)];
  }
  return ids;
}

function getDiagnosticsPresetOrderForSuite(suite, presetIds, mode, modelType) {
  if (!Array.isArray(presetIds) || presetIds.length === 0) return [];
  const key = String(suite || '').trim().toLowerCase();
  const normalizedModelType = normalizeModelType(modelType);
  const isEmbeddingTarget = mode === 'embedding' || normalizedModelType === 'embedding';
  const pushUnique = (list, value) => {
    if (presetIds.includes(value) && !list.includes(value)) list.push(value);
  };

  if (isEmbeddingTarget) {
    const ordered = [];
    if (key === 'inference' || key === 'debug') {
      pushUnique(ordered, 'modes/embedding');
      pushUnique(ordered, 'modes/debug');
      return ordered.length > 0 ? ordered : presetIds.slice(0, 1);
    }
    if (key === 'bench') {
      pushUnique(ordered, 'modes/embedding-bench');
      pushUnique(ordered, 'modes/bench');
      pushUnique(ordered, 'modes/embedding');
      pushUnique(ordered, 'modes/debug');
      return ordered.length > 0 ? ordered : presetIds.slice(0, 1);
    }
  }

  if (key === 'inference' || key === 'debug' || key === 'energy' || key === 'kernels') {
    const ordered = [];
    pushUnique(ordered, 'modes/debug');
    return ordered.length > 0 ? ordered : presetIds.slice(0, 1);
  }

  if (key === 'bench' || key === 'diffusion') {
    const ordered = [];
    pushUnique(ordered, 'modes/bench');
    pushUnique(ordered, 'modes/debug');
    return ordered.length > 0 ? ordered : presetIds.slice(0, 1);
  }

  return presetIds.slice(0, 1);
}

function formatRuntimePresetShortLabel(presetId) {
  if (typeof presetId !== 'string' || presetId.length === 0) return 'default';
  return presetId.startsWith('modes/') ? presetId.slice('modes/'.length) : presetId;
}

function encodeDiagnosticsProfileId(suite, presetId) {
  return `${suite}|${presetId}`;
}

function decodeDiagnosticsProfileId(profileId) {
  if (typeof profileId !== 'string' || profileId.length === 0) return null;
  const splitAt = profileId.indexOf('|');
  if (splitAt <= 0 || splitAt >= profileId.length - 1) return null;
  const suite = profileId.slice(0, splitAt).trim().toLowerCase();
  const preset = profileId.slice(splitAt + 1).trim();
  if (!suite || !preset) return null;
  return { suite, preset };
}

function getDiagnosticsProfileLabel(suite, presetId) {
  const key = `${suite}|${presetId}`;
  const labels = {
    'inference|modes/debug': 'Quick Check',
    'inference|modes/embedding': 'Embedding Check',
    'debug|modes/debug': 'Trace Debug',
    'debug|modes/embedding': 'Embedding Debug',
    'bench|modes/bench': 'Performance Bench',
    'bench|modes/embedding-bench': 'Embedding Bench',
    'bench|modes/debug': 'Bench with Traces',
    'diffusion|modes/bench': 'Image Bench',
    'diffusion|modes/debug': 'Image Bench (Debug)',
    'energy|modes/debug': 'Energy Run',
    'kernels|modes/debug': 'Kernel Validation',
  };
  if (labels[key]) return labels[key];
  return `${suite} · ${formatRuntimePresetShortLabel(presetId)}`;
}

function buildDiagnosticsProfileEntries(suites, mode, modelType) {
  const entries = [];
  const basePresetIds = getDiagnosticsBasePresetIds();
  for (const suite of suites) {
    const orderedPresets = getDiagnosticsPresetOrderForSuite(suite, basePresetIds, mode, modelType);
    for (const preset of orderedPresets) {
      entries.push({
        id: encodeDiagnosticsProfileId(suite, preset),
        suite,
        preset,
        label: getDiagnosticsProfileLabel(suite, preset),
      });
    }
  }
  return entries;
}

function updateDiagnosticsProfileOptions(mode, modelId, modelType) {
  const profileSelect = $('diagnostics-profile');
  if (!profileSelect) return null;

  const order = getDiagnosticsSuiteOrder();
  const isKernelsMode = mode === 'kernels';
  const availableSuites = [];
  if (isKernelsMode) {
    availableSuites.push('kernels');
  }

  if (!isKernelsMode && !modelId) {
    profileSelect.innerHTML = '';
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = 'Select an active model';
    profileSelect.appendChild(opt);
    profileSelect.disabled = true;
    profileSelect.value = '';
    return null;
  }

  if (!isKernelsMode && modelId && !modelType) {
    profileSelect.innerHTML = '';
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = 'Loading model type...';
    profileSelect.appendChild(opt);
    profileSelect.disabled = true;
    profileSelect.value = '';
    return null;
  }

  if (!isKernelsMode) {
    const normalizedModelType = normalizeModelType(modelType);
    if (normalizedModelType === 'unknown') {
      profileSelect.innerHTML = '';
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = 'Model type unavailable (manifest unreadable)';
      profileSelect.appendChild(opt);
      profileSelect.disabled = true;
      profileSelect.value = '';
      return null;
    }
    const compatibleSuites = order.filter((suite) => suite !== 'kernels' && isSuiteCompatibleModelType(modelType, suite));
    availableSuites.push(...compatibleSuites);
  }

  profileSelect.innerHTML = '';
  if (!availableSuites.length) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = 'No compatible profiles';
    profileSelect.appendChild(opt);
    profileSelect.disabled = true;
    profileSelect.value = '';
    return null;
  }

  const entries = buildDiagnosticsProfileEntries(availableSuites, mode, modelType);
  for (const entry of entries) {
    const opt = document.createElement('option');
    opt.value = entry.id;
    opt.textContent = entry.label;
    profileSelect.appendChild(opt);
  }

  const selection = state.diagnosticsSelections[mode] || {};
  const currentValue = profileSelect.value || '';
  const currentPair = decodeDiagnosticsProfileId(currentValue);
  const storedPair = selection.profile ? decodeDiagnosticsProfileId(selection.profile) : null;
  const explicitPair = selection.suite && selection.preset
    ? { suite: String(selection.suite).trim().toLowerCase(), preset: String(selection.preset).trim() }
    : null;

  const hasEntry = (pair) => Boolean(pair && entries.some((entry) => entry.suite === pair.suite && entry.preset === pair.preset));
  let targetPair = null;
  if (hasEntry(currentPair)) targetPair = currentPair;
  if (!targetPair && hasEntry(storedPair)) targetPair = storedPair;
  if (!targetPair && hasEntry(explicitPair)) targetPair = explicitPair;
  if (!targetPair) {
    const defaultSuite = getDiagnosticsDefaultSuite(mode);
    const fallbackSuite = availableSuites.includes(defaultSuite) ? defaultSuite : availableSuites[0];
    const orderedPresets = getDiagnosticsPresetOrderForSuite(
      fallbackSuite,
      getDiagnosticsBasePresetIds(),
      mode,
      modelType
    );
    targetPair = { suite: fallbackSuite, preset: orderedPresets[0] };
  }

  const chosen = entries.find((entry) => entry.suite === targetPair.suite && entry.preset === targetPair.preset) || entries[0];
  profileSelect.disabled = false;
  profileSelect.value = chosen.id;
  storeDiagnosticsSelection(mode, {
    profile: chosen.id,
    suite: chosen.suite,
    preset: chosen.preset,
  });
  return chosen;
}

function updateDiagnosticsSummary({ suite, modelId, modelType, runtimePreset, intent }) {
  const summaryEl = $('diagnostics-summary');
  if (!summaryEl) return;
  const parts = [];
  if (suite && runtimePreset) {
    parts.push(`Profile: ${getDiagnosticsProfileLabel(suite, runtimePreset)}`);
  } else if (suite) {
    parts.push(`Suite: ${suite}`);
  }
  if (modelId) {
    const typeLabel = modelType ? normalizeModelType(modelType) : null;
    parts.push(typeLabel ? `Model: ${modelId} (${typeLabel})` : `Model: ${modelId}`);
  } else if (suite === 'kernels') {
    parts.push('Model: none');
  }
  if (intent) parts.push(`Intent: ${intent}`);
  summaryEl.textContent = parts.join(' · ');
  summaryEl.hidden = parts.length === 0;
}

export function syncDiagnosticsModeUI(mode) {
  const titleEl = $('diagnostics-title');
  const suiteField = $('diagnostics-suite-field');
  const modeTabs = document.querySelectorAll('.diagnostics-mode-tab');
  if (titleEl) {
    titleEl.textContent = mode === 'kernels' ? 'Kernel Diagnostics' : 'Inference Diagnostics';
  }
  if (suiteField) {
    suiteField.hidden = false;
  }
  if (modeTabs.length) {
    modeTabs.forEach((button) => {
      const target = button.dataset.diagnosticsMode;
      const isActive = target === mode;
      button.classList.toggle('is-active', isActive);
      button.setAttribute('aria-pressed', isActive ? 'true' : 'false');
    });
  }
}

function isPlainObject(value) {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function mergeRuntimeOverrides(base, override) {
  if (override === undefined) return base;
  if (override === null) return null;
  if (!isPlainObject(base) || !isPlainObject(override)) {
    return override;
  }
  const merged = { ...base };
  for (const [key, value] of Object.entries(override)) {
    if (value === undefined) continue;
    merged[key] = mergeRuntimeOverrides(base[key], value);
  }
  return merged;
}

export function normalizeRuntimeConfig(raw) {
  if (!raw || typeof raw !== 'object') return null;
  if (raw.runtime && typeof raw.runtime === 'object') return raw.runtime;
  if (raw.shared || raw.loading || raw.inference || raw.emulation) return raw;
  return null;
}

export function getDiagnosticsDefaultSuite(mode) {
  return DIAGNOSTICS_DEFAULTS[mode]?.suite || 'inference';
}

function getDiagnosticsDefaultPreset(mode) {
  return DIAGNOSTICS_DEFAULTS[mode]?.preset || DEFAULT_RUNTIME_PRESET;
}

export function getDiagnosticsRuntimeConfig() {
  return state.diagnosticsRuntimeConfig || getRuntimeConfig();
}

export async function refreshDiagnosticsRuntimeConfig(presetId) {
  const targetPreset = presetId || DEFAULT_RUNTIME_PRESET;
  const { runtime } = await loadRuntimePreset(targetPreset);
  const mergedOverride = getMergedRuntimeOverride();
  const mergedRuntime = mergedOverride ? mergeRuntimeOverrides(runtime, mergedOverride) : runtime;
  state.diagnosticsRuntimeConfig = mergedRuntime;
  state.diagnosticsRuntimePresetId = targetPreset;
  return mergedRuntime;
}

export async function syncDiagnosticsDefaultsForMode(mode) {
  if (mode !== 'run' && mode !== 'embedding' && mode !== 'diffusion' && mode !== 'energy' && mode !== 'diagnostics' && mode !== 'kernels') {
    return;
  }
  const profileSelect = $('diagnostics-profile');
  const presetSelect = $('runtime-preset');
  const selections = state.diagnosticsSelections[mode] || {};
  const targetProfile = selections.profile || '';
  if (presetSelect) {
    const targetPreset = selections.preset || getDiagnosticsDefaultPreset(mode);
    presetSelect.value = targetPreset;
  }
  if (profileSelect && targetProfile) {
    profileSelect.value = targetProfile;
  }
  updateDiagnosticsGuidance();
  await applySelectedRuntimePreset();
}

function formatDiagnosticsDuration(ms) {
  if (!Number.isFinite(ms)) return '';
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

function formatDiagnosticsSummary(result) {
  if (!result) return '';
  const parts = [];
  if (Number.isFinite(result.passed)) {
    parts.push(`Passed ${result.passed}`);
  }
  if (Number.isFinite(result.failed)) {
    parts.push(`Failed ${result.failed}`);
  }
  if (Number.isFinite(result.skipped) && result.skipped > 0) {
    parts.push(`Skipped ${result.skipped}`);
  }
  const duration = formatDiagnosticsDuration(result.duration);
  if (duration) {
    parts.push(`Duration ${duration}`);
  }
  return parts.join(' · ');
}

function formatDiagnosticsMetricsLine(result, suite) {
  const metrics = result?.metrics;
  if (!metrics || !suite) return '';
  if (suite === 'bench') {
    if (Number.isFinite(metrics.medianEmbeddingMs)) {
      const dim = Number.isFinite(metrics.embeddingDim) ? metrics.embeddingDim : '--';
      const invalidRuns = Number.isFinite(metrics.invalidRuns) ? metrics.invalidRuns : 0;
      const validRuns = Number.isFinite(metrics.validRuns) ? metrics.validRuns : '--';
      const p95 = Number.isFinite(metrics.p95EmbeddingMs) ? `${metrics.p95EmbeddingMs}ms` : '--';
      const minMs = Number.isFinite(metrics.minEmbeddingMs) ? `${metrics.minEmbeddingMs}ms` : '--';
      const maxMs = Number.isFinite(metrics.maxEmbeddingMs) ? `${metrics.maxEmbeddingMs}ms` : '--';
      const avgTokens = Number.isFinite(metrics.avgEmbeddingTokens) ? metrics.avgEmbeddingTokens.toFixed(1) : '--';
      return `Embedding dim ${dim} • Median ${metrics.medianEmbeddingMs}ms • P95 ${p95} • Range ${minMs}-${maxMs} • Tokens ${avgTokens} • Invalid ${invalidRuns} • Valid ${validRuns}`;
    }
    if (Number.isFinite(metrics.medianTokensPerSec)) {
      const prefill = Number.isFinite(metrics.medianPrefillMs) ? `${metrics.medianPrefillMs}ms` : '--';
      const ttft = Number.isFinite(metrics.medianTtftMs) ? `${metrics.medianTtftMs}ms` : '--';
      const decode = Number.isFinite(metrics.medianDecodeTokensPerSec)
        ? `${metrics.medianDecodeTokensPerSec} tok/s`
        : '--';
      return `Median ${metrics.medianTokensPerSec} tok/s • Avg ${metrics.avgTokensPerSec ?? '--'} tok/s • Prefill ${prefill} • TTFT ${ttft} • Decode ${decode}`;
    }
  }
  if (suite === 'energy') {
    if (Number.isFinite(metrics.energy)) {
      return `Energy ${metrics.energy} • Steps ${metrics.steps ?? '--'}`;
    }
  }
  if (suite === 'diffusion') {
    if (Number.isFinite(metrics.width) && Number.isFinite(metrics.height)) {
      return `Resolution ${metrics.width}×${metrics.height} • Steps ${metrics.steps ?? '--'}`;
    }
  }
  if (suite === 'inference' || suite === 'debug') {
    if (Number.isFinite(metrics.embeddingDim)) {
      const embedMs = Number.isFinite(metrics.embeddingMs) ? `${metrics.embeddingMs}ms` : '--';
      const loadMs = Number.isFinite(metrics.modelLoadMs) ? `${metrics.modelLoadMs}ms` : '--';
      const nonFinite = Number.isFinite(metrics.nonFiniteValues) ? metrics.nonFiniteValues : 0;
      const tokens = Number.isFinite(metrics.embeddingTokens) ? metrics.embeddingTokens : '--';
      const l2Norm = Number.isFinite(metrics.embeddingL2Norm) ? metrics.embeddingL2Norm.toFixed(4) : '--';
      const maxAbs = Number.isFinite(metrics.embeddingMaxAbs) ? metrics.embeddingMaxAbs.toFixed(4) : '--';
      return `Embedding dim ${metrics.embeddingDim} • Tokens ${tokens} • Embed ${embedMs} • Load ${loadMs} • L2 ${l2Norm} • MaxAbs ${maxAbs} • Non-finite ${nonFinite}`;
    }
    if (Number.isFinite(metrics.tokensGenerated)) {
      const prefill = Number.isFinite(metrics.prefillMs) ? `${metrics.prefillMs}ms` : '--';
      const ttft = Number.isFinite(metrics.ttftMs) ? `${metrics.ttftMs}ms` : '--';
      const decode = Number.isFinite(metrics.decodeTokensPerSec)
        ? `${metrics.decodeTokensPerSec} tok/s`
        : '--';
      return `Tokens ${metrics.tokensGenerated} • ${metrics.tokensPerSec ?? '--'} tok/s • Prefill ${prefill} • TTFT ${ttft} • Decode ${decode}`;
    }
  }
  return '';
}

function buildDiagnosticsJsonPayload(result) {
  if (!result || typeof result !== 'object') return null;
  if (result.report && typeof result.report === 'object') {
    return result.report;
  }
  return {
    suite: result.suite ?? null,
    duration: result.duration ?? null,
    metrics: result.metrics ?? null,
    output: result.output ?? null,
    memoryStats: result.memoryStats ?? null,
  };
}

export function clearDiagnosticsOutput() {
  const container = $('diagnostics-output');
  const textEl = $('diagnostics-output-text');
  const canvas = $('diagnostics-output-canvas');
  const jsonWrap = $('diagnostics-output-json-wrap');
  const jsonEl = $('diagnostics-output-json');
  if (textEl) textEl.textContent = 'No output yet.';
  if (jsonEl) jsonEl.textContent = 'No JSON yet.';
  if (jsonWrap) {
    jsonWrap.open = false;
    setHidden(jsonWrap, true);
  }
  if (canvas) {
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    canvas.hidden = true;
  }
  if (container) container.hidden = false;
}

function drawDiagnosticsCanvas(output) {
  const canvas = $('diagnostics-output-canvas');
  if (!canvas || !output) return;
  canvas.width = output.width;
  canvas.height = output.height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const imageData = new ImageData(output.pixels, output.width, output.height);
  ctx.putImageData(imageData, 0, 0);
  canvas.hidden = false;
}

export function renderDiagnosticsOutput(result, suite, captureOutput) {
  const container = $('diagnostics-output');
  if (!container) return;
  container.hidden = false;
  const textEl = $('diagnostics-output-text');
  const canvas = $('diagnostics-output-canvas');
  const jsonWrap = $('diagnostics-output-json-wrap');
  const jsonEl = $('diagnostics-output-json');
  if (canvas) {
    canvas.hidden = true;
  }
  const jsonPayload = buildDiagnosticsJsonPayload(result);
  if (jsonWrap && jsonEl && jsonPayload) {
    jsonEl.textContent = JSON.stringify(jsonPayload, null, 2);
    setHidden(jsonWrap, false);
  } else if (jsonWrap) {
    jsonWrap.open = false;
    setHidden(jsonWrap, true);
  }
  const output = result?.output ?? null;
  const summary = formatDiagnosticsSummary(result);
  const metricsLine = formatDiagnosticsMetricsLine(result, suite);
  const prefix = [summary, metricsLine].filter(Boolean).join('\n');
  if (suite === 'diffusion') {
    if (output && typeof output === 'object' && output.pixels) {
      if (textEl) textEl.textContent = prefix;
      drawDiagnosticsCanvas(output);
      return;
    }
    if (textEl) {
      const fallback = captureOutput ? 'No diffusion output captured.' : 'Output capture disabled.';
      textEl.textContent = [prefix, fallback].filter(Boolean).join('\n');
    }
    return;
  }
  if ((suite === 'inference' || suite === 'debug') && typeof output === 'string' && output.length > 0) {
    if (textEl) {
      const body = prefix ? `${prefix}\n\n${output}` : output;
      textEl.textContent = body;
    }
    return;
  }
  if ((suite === 'inference' || suite === 'debug') && output && typeof output === 'object') {
    if (textEl) {
      const body = JSON.stringify(output, null, 2);
      textEl.textContent = prefix ? `${prefix}\n\n${body}` : body;
    }
    return;
  }
  if (textEl) {
    let fallback = '';
    if (suite !== 'bench' && suite !== 'energy' && suite !== 'kernels') {
      fallback = captureOutput ? 'No output captured.' : 'Output capture disabled.';
    }
    textEl.textContent = [prefix, fallback].filter(Boolean).join('\n');
  }
}

export function updateDiagnosticsStatus(message, isError = false) {
  const status = $('diagnostics-status');
  if (!status) return;
  status.textContent = message;
  status.dataset.state = isError ? 'error' : 'ready';
}

export function updateDiagnosticsReport(text) {
  const report = $('diagnostics-report');
  if (!report) return;
  const next = text ? String(text) : '';
  report.textContent = next;
  report.hidden = next.length === 0;
}

function getDiagnosticsSuiteInfo(suite) {
  const key = String(suite || 'inference').trim().toLowerCase();
  return DIAGNOSTICS_SUITE_INFO[key] || DIAGNOSTICS_SUITE_INFO.inference;
}

export function updateDiagnosticsGuidance() {
  const profileSelect = $('diagnostics-profile');
  const presetSelect = $('runtime-preset');
  const modelSelect = $('diagnostics-model');
  const intentEl = $('diagnostics-intent');
  const suiteHelp = $('diagnostics-suite-help');
  const requirements = $('diagnostics-requirements');
  const runBtn = $('diagnostics-run-btn');
  const verifyBtn = $('diagnostics-verify-btn');
  if (!profileSelect || !intentEl || !suiteHelp || !requirements) return;

  const mode = state.uiMode;
  const modelId = modelSelect?.value || '';
  const modelType = modelId ? (state.modelTypeCache[modelId] || null) : null;
  const normalizedModelType = modelType ? normalizeModelType(modelType) : null;
  const resolvedProfile = updateDiagnosticsProfileOptions(mode, modelId, normalizedModelType);
  const suite = resolvedProfile?.suite || getDiagnosticsDefaultSuite(mode);
  if (presetSelect && resolvedProfile?.preset && presetSelect.value !== resolvedProfile.preset) {
    presetSelect.value = resolvedProfile.preset;
    if (state.diagnosticsRuntimePresetId !== resolvedProfile.preset) {
      void applySelectedRuntimePreset();
    }
  }
  const info = getDiagnosticsSuiteInfo(suite);
  const runtimeConfig = getDiagnosticsRuntimeConfig();
  const intent = runtimeConfig?.shared?.tooling?.intent ?? null;
  const requiredModelType = getDiagnosticsRequiredModelType(suite);
  const runtimePreset = presetSelect?.value || DEFAULT_RUNTIME_PRESET;
  const needsMaxTokens = suiteRequiresMaxTokens(suite, normalizedModelType);

  intentEl.textContent = intent || 'unset';

  const requirementHints = [];
  if (info.requiresModel) {
    const typeLabel = formatDiagnosticsModelTypeLabel(requiredModelType);
    requirementHints.push(`${typeLabel} model`);
  }
  if (info.requiresBenchIntent) {
    requirementHints.push('intent investigate/calibrate');
  }
  if (suiteRequiresPrompt(suite)) {
    requirementHints.push('prompt');
  }
  if (needsMaxTokens) {
    requirementHints.push('maxTokens');
  }

  const hintSuffix = requirementHints.length ? `Requires: ${requirementHints.join(', ')}.` : '';
  const presetHint = runtimePreset ? `Preset: ${formatRuntimePresetShortLabel(runtimePreset)}.` : '';
  suiteHelp.textContent = [info.description, presetHint, hintSuffix].filter(Boolean).join(' ');

  const missing = [];
  if (!intent) {
    missing.push('intent');
  } else if (info.requiresBenchIntent && !BENCH_INTENTS.has(intent)) {
    missing.push('bench intent');
  }
  if (modelId && normalizedModelType == null) {
    getModelTypeForId(modelId)
      .then((resolved) => {
        if (resolved != null) updateDiagnosticsGuidance();
      })
      .catch(() => {});
  }

  if (info.requiresModel && !modelId) {
    missing.push('model');
  } else if (info.requiresModel && modelId && requiredModelType) {
    if (!normalizedModelType || normalizedModelType === 'unknown') {
      missing.push('model type');
    } else if (!isSuiteCompatibleModelType(normalizedModelType, suite)) {
      missing.push(formatDiagnosticsModelTypeLabel(requiredModelType));
    }
  }

  const promptValue = runtimeConfig?.inference?.prompt;
  if (suiteRequiresPrompt(suite) && (!promptValue || !String(promptValue).trim())) {
    missing.push('prompt');
  }
  const maxTokensValue = runtimeConfig?.inference?.batching?.maxTokens;
  if (needsMaxTokens && !Number.isFinite(maxTokensValue)) {
    missing.push('maxTokens');
  }

  if (missing.length > 0) {
    requirements.textContent = `Needs: ${missing.join(', ')}`;
  } else {
    const ready = info.requiresModel ? `Ready with ${modelId}.` : 'Ready.';
    requirements.textContent = ready;
  }

  updateDiagnosticsSummary({ suite, modelId, modelType, runtimePreset, intent });

  const intentOk = Boolean(intent) && (!info.requiresBenchIntent || BENCH_INTENTS.has(intent));
  const modelOk = !info.requiresModel
    || (Boolean(modelId) && normalizedModelType != null && normalizedModelType !== 'unknown'
      && (!requiredModelType || isSuiteCompatibleModelType(normalizedModelType, suite)));
  const promptOk = !suiteRequiresPrompt(suite) || (promptValue && String(promptValue).trim());
  const maxTokensOk = !needsMaxTokens || Number.isFinite(maxTokensValue);
  const canVerify = intentOk;
  const canRun = intentOk && modelOk && promptOk && maxTokensOk;
  if (verifyBtn) verifyBtn.disabled = !canVerify;
  if (runBtn) runBtn.disabled = !canRun;
}

export function selectDiagnosticsModel(modelId) {
  const modelSelect = $('diagnostics-model');
  if (!modelSelect) return;
  modelSelect.value = modelId;
  state.activeModelId = modelId || null;
  if (isModeModelSelectable(state.uiMode)) {
    state.modeModelId[state.uiMode] = modelId || null;
  }
  updateDiagnosticsGuidance();
}

export function updateRuntimeConfigStatus(presetId) {
  const status = $('runtime-config-status');
  if (!status) return;
  const presetLabel = presetId || DEFAULT_RUNTIME_PRESET;
  if (state.runtimeOverride) {
    const labels = [];
    if (state.runtimeOverrideLabel) {
      labels.push(state.runtimeOverrideLabel);
    }
    const overrideLabel = labels.length ? labels.join(' + ') : 'custom';
    status.textContent = `Preset: ${presetLabel} - Override: ${overrideLabel}`;
    return;
  }
  status.textContent = `Preset: ${presetLabel}`;
}

async function setRuntimeOverride(runtime, label) {
  state.runtimeOverrideBase = runtime;
  state.runtimeOverrideLabel = label || null;
  await applySelectedRuntimePreset();
}

export async function handleRuntimeConfigFile(file) {
  if (!file) return;
  try {
    const text = await file.text();
    const json = JSON.parse(text);
    const runtime = normalizeRuntimeConfig(json);
    if (!runtime) {
      throw new Error('Runtime config file is missing runtime fields');
    }
    await setRuntimeOverride(runtime, file.name);
    const presetSelect = $('runtime-config-preset');
    if (presetSelect) {
      presetSelect.value = '';
    }
  } catch (error) {
    updateDiagnosticsStatus(`Runtime config error: ${error.message}`, true);
  }
}

export async function applyRuntimeConfigPreset(presetId) {
  if (!presetId) {
    state.runtimeOverrideBase = null;
    state.runtimeOverrideLabel = null;
    await applySelectedRuntimePreset();
    return;
  }
  try {
    const { runtime } = await loadRuntimePreset(presetId);
    await setRuntimeOverride(runtime, presetId);
  } catch (error) {
    updateDiagnosticsStatus(`Runtime config preset error: ${error.message}`, true);
  }
}

export function getMergedRuntimeOverride() {
  return state.runtimeOverrideBase;
}

export async function applySelectedRuntimePreset() {
  const presetSelect = $('runtime-preset');
  if (!presetSelect) return;
  const presetId = presetSelect.value || DEFAULT_RUNTIME_PRESET;
  if (!presetSelect.value) {
    presetSelect.value = presetId;
  }
  const mergedOverride = getMergedRuntimeOverride();
  state.runtimeOverride = mergedOverride;
  updateRuntimeConfigStatus(presetId);
  try {
    await refreshDiagnosticsRuntimeConfig(presetId);
    updateDiagnosticsGuidance();
  } catch (error) {
    updateDiagnosticsStatus(`Preset error: ${error.message}`, true);
  }
}
