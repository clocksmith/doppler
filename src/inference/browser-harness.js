

import { initializeInference, parseRuntimeOverridesFromURL } from './test-harness.js';
import { saveReport } from '../storage/reports.js';
import { setRuntimeConfig } from '../config/runtime.js';

function resolveRuntime(options) {
  if (options.runtime) return options.runtime;
  if (options.searchParams) return parseRuntimeOverridesFromURL(options.searchParams);
  return parseRuntimeOverridesFromURL();
}

function normalizePresetPath(value) {
  const trimmed = String(value || '').replace(/^[./]+/, '');
  if (!trimmed) return null;
  return trimmed.endsWith('.json') ? trimmed : `${trimmed}.json`;
}

function resolveRuntimeFromConfig(config) {
  if (!config || typeof config !== 'object') return null;
  if (config.runtime && typeof config.runtime === 'object') return config.runtime;
  if (config.shared || config.loading || config.inference || config.emulation) return config;
  return null;
}

export async function loadRuntimeConfigFromUrl(url, options = {}) {
  if (!url) {
    throw new Error('runtime config url is required');
  }
  const response = await fetch(url, { signal: options.signal });
  if (!response.ok) {
    throw new Error(`Failed to load runtime config: ${response.status}`);
  }
  const config = await response.json();
  const runtime = resolveRuntimeFromConfig(config);
  if (!runtime) {
    throw new Error('Runtime config is missing runtime fields');
  }
  return { config, runtime };
}

export async function applyRuntimeConfigFromUrl(url, options = {}) {
  const { runtime } = await loadRuntimeConfigFromUrl(url, options);
  setRuntimeConfig(runtime);
  return runtime;
}

export async function loadRuntimePreset(presetId, options = {}) {
  const baseUrl = options.baseUrl || '/doppler/src/config/presets/runtime';
  const normalized = normalizePresetPath(presetId);
  if (!normalized) {
    throw new Error('runtime preset id is required');
  }
  const url = `${baseUrl.replace(/\/$/, '')}/${normalized}`;
  return loadRuntimeConfigFromUrl(url, options);
}

export async function applyRuntimePreset(presetId, options = {}) {
  const { runtime } = await loadRuntimePreset(presetId, options);
  setRuntimeConfig(runtime);
  return runtime;
}

export async function initializeBrowserHarness(options = {}) {
  const { modelUrl, onProgress, log } = options;
  if (!modelUrl) {
    throw new Error('modelUrl is required');
  }

  const runtime = resolveRuntime(options);
  const result = await initializeInference(modelUrl, {
    runtime,
    onProgress,
    log,
  });

  return { ...result, runtime };
}

export async function saveBrowserReport(modelId, report, options = {}) {
  return saveReport(modelId, report, options);
}

export async function runBrowserHarness(options = {}) {
  const harness = await initializeBrowserHarness(options);
  const modelId = options.modelId || harness.manifest?.modelId || 'unknown';

  let report = options.report || null;
  if (!report && typeof options.buildReport === 'function') {
    report = await options.buildReport(harness);
  }
  if (!report) {
    report = {
      modelId,
      timestamp: new Date().toISOString(),
    };
  }

  const reportInfo = await saveReport(modelId, report, { timestamp: options.timestamp });
  return { ...harness, report, reportInfo };
}
