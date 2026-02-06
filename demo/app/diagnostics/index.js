import { getRuntimeConfig } from '../../../src/config/runtime.js';
import { loadRuntimePreset } from '../../../src/inference/browser-harness.js';
import { state } from '../state.js';
import { $, setHidden, setText } from '../dom.js';
import {
  BENCH_INTENTS,
  DEFAULT_RUNTIME_PRESET,
  DIAGNOSTICS_DEFAULTS,
  DIAGNOSTICS_SUITE_INFO,
  DIAGNOSTICS_SUITE_ORDER,
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

function suiteRequiresMaxTokens(suite) {
  const key = String(suite || '').trim().toLowerCase();
  return key === 'inference' || key === 'debug' || key === 'bench';
}

function isSuiteCompatibleModelType(modelType, suite) {
  const normalized = normalizeModelType(modelType);
  const required = getDiagnosticsRequiredModelType(suite);
  if (!required) return true;
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

function updateDiagnosticsSuiteOptions(mode, modelId, modelType) {
  const suiteSelect = $('diagnostics-suite');
  if (!suiteSelect) return null;

  const order = getDiagnosticsSuiteOrder();
  const isKernelsMode = mode === 'kernels';
  if (isKernelsMode) {
    suiteSelect.innerHTML = '';
    const opt = document.createElement('option');
    opt.value = 'kernels';
    opt.textContent = 'kernels';
    suiteSelect.appendChild(opt);
    suiteSelect.disabled = true;
    suiteSelect.value = 'kernels';
    storeDiagnosticsSelection(mode, { suite: 'kernels' });
    return 'kernels';
  }

  if (!modelId) {
    suiteSelect.innerHTML = '';
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = 'Select an active model';
    suiteSelect.appendChild(opt);
    suiteSelect.disabled = true;
    suiteSelect.value = '';
    return '';
  }

  if (modelId && !modelType) {
    suiteSelect.innerHTML = '';
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = 'Loading model type...';
    suiteSelect.appendChild(opt);
    suiteSelect.disabled = true;
    suiteSelect.value = '';
    return '';
  }

  const available = order.filter((suite) => suite !== 'kernels' && isSuiteCompatibleModelType(modelType, suite));
  suiteSelect.innerHTML = '';
  if (!available.length) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = 'No compatible suites';
    suiteSelect.appendChild(opt);
    suiteSelect.disabled = true;
    suiteSelect.value = '';
    return '';
  }

  const storedSuite = state.diagnosticsSelections[mode]?.suite || '';
  const defaultSuite = getDiagnosticsDefaultSuite(mode);
  const currentValue = suiteSelect.value || '';
  for (const suite of available) {
    const opt = document.createElement('option');
    opt.value = suite;
    opt.textContent = suite;
    suiteSelect.appendChild(opt);
  }

  let target = currentValue;
  if (!available.includes(target)) target = storedSuite;
  if (!available.includes(target)) target = defaultSuite;
  if (!available.includes(target)) target = available[0];
  suiteSelect.disabled = false;
  suiteSelect.value = target;
  storeDiagnosticsSelection(mode, { suite: target });
  return target;
}

function updateDiagnosticsSummary({ suite, modelId, modelType, runtimePreset, intent }) {
  const summaryEl = $('diagnostics-summary');
  if (!summaryEl) return;
  const parts = [];
  if (suite) parts.push(`Suite: ${suite}`);
  if (modelId) {
    const typeLabel = modelType ? normalizeModelType(modelType) : null;
    parts.push(typeLabel ? `Model: ${modelId} (${typeLabel})` : `Model: ${modelId}`);
  } else if (suite === 'kernels') {
    parts.push('Model: none');
  }
  if (runtimePreset) parts.push(`Preset: ${runtimePreset}`);
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
    suiteField.hidden = mode === 'kernels';
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
  if (mode !== 'run' && mode !== 'diffusion' && mode !== 'energy' && mode !== 'diagnostics' && mode !== 'kernels') {
    return;
  }
  const suiteSelect = $('diagnostics-suite');
  const presetSelect = $('runtime-preset');
  const selections = state.diagnosticsSelections[mode] || {};
  const targetSuite = selections.suite || getDiagnosticsDefaultSuite(mode);
  if (suiteSelect && targetSuite) {
    suiteSelect.value = targetSuite;
  }
  if (presetSelect) {
    const targetPreset = selections.preset || DEFAULT_RUNTIME_PRESET;
    presetSelect.value = targetPreset;
  }
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
    if (Number.isFinite(metrics.medianTokensPerSec)) {
      return `Median ${metrics.medianTokensPerSec} tok/s • Avg ${metrics.avgTokensPerSec ?? '--'} tok/s`;
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
    if (Number.isFinite(metrics.tokensGenerated)) {
      return `Tokens ${metrics.tokensGenerated} • ${metrics.tokensPerSec ?? '--'} tok/s`;
    }
  }
  return '';
}

export function clearDiagnosticsOutput() {
  const container = $('diagnostics-output');
  const textEl = $('diagnostics-output-text');
  const canvas = $('diagnostics-output-canvas');
  if (textEl) textEl.textContent = 'No output yet.';
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
  if (canvas) {
    canvas.hidden = true;
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
  const suiteSelect = $('diagnostics-suite');
  const modelSelect = $('diagnostics-model');
  const intentEl = $('diagnostics-intent');
  const suiteHelp = $('diagnostics-suite-help');
  const requirements = $('diagnostics-requirements');
  const runBtn = $('diagnostics-run-btn');
  const verifyBtn = $('diagnostics-verify-btn');
  if (!suiteSelect || !intentEl || !suiteHelp || !requirements) return;

  const mode = state.uiMode;
  const modelId = modelSelect?.value || '';
  const modelType = modelId ? state.modelTypeCache[modelId] : null;
  const resolvedSuite = updateDiagnosticsSuiteOptions(mode, modelId, modelType);
  const suite = resolvedSuite || suiteSelect.value || getDiagnosticsDefaultSuite(mode);
  const info = getDiagnosticsSuiteInfo(suite);
  const runtimeConfig = getDiagnosticsRuntimeConfig();
  const intent = runtimeConfig?.shared?.tooling?.intent ?? null;
  const requiredModelType = getDiagnosticsRequiredModelType(suite);
  const runtimePreset = $('runtime-preset')?.value || DEFAULT_RUNTIME_PRESET;

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
  if (suiteRequiresMaxTokens(suite)) {
    requirementHints.push('maxTokens');
  }

  const hintSuffix = requirementHints.length ? `Requires: ${requirementHints.join(', ')}.` : '';
  suiteHelp.textContent = [info.description, hintSuffix].filter(Boolean).join(' ');

  const missing = [];
  if (!intent) {
    missing.push('intent');
  } else if (info.requiresBenchIntent && !BENCH_INTENTS.has(intent)) {
    missing.push('bench intent');
  }
  if (modelId && !modelType) {
    getModelTypeForId(modelId)
      .then(() => updateDiagnosticsGuidance())
      .catch(() => {});
  }

  if (info.requiresModel && !modelId) {
    missing.push('model');
  } else if (info.requiresModel && modelId && requiredModelType) {
    if (modelType) {
      if (!isSuiteCompatibleModelType(modelType, suite)) {
        missing.push(formatDiagnosticsModelTypeLabel(requiredModelType));
      }
    } else {
      missing.push('model type');
    }
  }

  const promptValue = runtimeConfig?.inference?.prompt;
  if (suiteRequiresPrompt(suite) && (!promptValue || !String(promptValue).trim())) {
    missing.push('prompt');
  }
  const maxTokensValue = runtimeConfig?.inference?.batching?.maxTokens;
  if (suiteRequiresMaxTokens(suite) && !Number.isFinite(maxTokensValue)) {
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
    || (Boolean(modelId) && (!requiredModelType || (modelType && isSuiteCompatibleModelType(modelType, suite))));
  const promptOk = !suiteRequiresPrompt(suite) || (promptValue && String(promptValue).trim());
  const maxTokensOk = !suiteRequiresMaxTokens(suite) || Number.isFinite(maxTokensValue);
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
