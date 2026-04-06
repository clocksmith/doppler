import { parseRuntimeOverridesFromURL } from './test-harness.js';
import { getRuntimeConfig, setRuntimeConfig } from '../config/runtime.js';
import {
  setActiveKernelPath,
  getActiveKernelPath,
  getActiveKernelPathSource,
  getActiveKernelPathPolicy,
} from '../config/kernel-path-loader.js';
import { mergeRuntimeValues } from '../config/runtime-merge.js';
import {
  applyOrderedRuntimeInputs,
  resolveRuntimeFromConfig,
} from '../tooling/runtime-input-composition.js';

export function parseReportTimestamp(rawTimestamp, label = 'timestamp') {
  if (rawTimestamp == null) {
    return null;
  }

  if (rawTimestamp instanceof Date) {
    const timestamp = rawTimestamp.getTime();
    if (!Number.isFinite(timestamp)) {
      throw new Error(`Invalid ${label}: not a valid Date.`);
    }
    return rawTimestamp.toISOString();
  }

  if (typeof rawTimestamp === 'number') {
    if (!Number.isFinite(rawTimestamp)) {
      throw new Error(`Invalid ${label}: must be a finite epoch timestamp.`);
    }
    return new Date(rawTimestamp).toISOString();
  }

  if (typeof rawTimestamp === 'string') {
    const trimmed = rawTimestamp.trim();
    if (trimmed.length === 0) {
      return null;
    }
    const numericCandidate = Number(trimmed);
    if (Number.isFinite(numericCandidate)) {
      return new Date(numericCandidate).toISOString();
    }
    const parsed = new Date(trimmed);
    if (Number.isNaN(parsed.getTime())) {
      throw new Error(`Invalid ${label}: expected ISO-8601 string or epoch milliseconds.`);
    }
    return parsed.toISOString();
  }

  throw new Error(`Invalid ${label}: expected Date, ISO-8601 string, epoch milliseconds, or nullish.`);
}

export function resolveReportTimestamp(rawTimestamp, label, fallbackTimestamp = null) {
  const parsed = parseReportTimestamp(rawTimestamp, label);
  return parsed ?? (fallbackTimestamp == null ? new Date().toISOString() : String(fallbackTimestamp));
}

export function resolveRuntime(options) {
  if (options.runtime) return options.runtime;
  if (options.searchParams) return parseRuntimeOverridesFromURL(options.searchParams);
  const runtimeConfig = cloneRuntimeConfig(getRuntimeConfig());
  const runtime = typeof globalThis.location === 'undefined'
    ? parseRuntimeOverridesFromURL(new URLSearchParams())
    : parseRuntimeOverridesFromURL();
  if (runtimeConfig) {
    runtime.runtimeConfig = runtime.runtimeConfig
      ? mergeRuntimeValues(runtimeConfig, runtime.runtimeConfig)
      : runtimeConfig;
  }
  return runtime;
}

function normalizeProfilePath(value) {
  const trimmed = String(value || '').replace(/^[./]+/, '');
  if (!trimmed) return null;
  return trimmed.endsWith('.json') ? trimmed : `${trimmed}.json`;
}

function resolveProfileBaseUrl() {
  try {
    return new URL('../config/runtime', import.meta.url).toString().replace(/\/$/, '');
  } catch {
    if (typeof globalThis.location !== 'undefined' && globalThis.location?.href) {
      return new URL('/src/config/runtime/', globalThis.location.href).toString().replace(/\/$/, '');
    }
    return '/src/config/runtime';
  }
}

export function cloneRuntimeConfig(runtimeConfig) {
  if (!runtimeConfig) return null;
  if (typeof structuredClone === 'function') {
    return structuredClone(runtimeConfig);
  }
  return JSON.parse(JSON.stringify(runtimeConfig));
}

export function snapshotRuntimeState() {
  return {
    runtimeConfig: cloneRuntimeConfig(getRuntimeConfig()),
    activeKernelPath: getActiveKernelPath(),
    activeKernelPathSource: getActiveKernelPathSource(),
    activeKernelPathPolicy: getActiveKernelPathPolicy(),
  };
}

export function restoreRuntimeState(snapshot) {
  if (!snapshot) {
    return;
  }
  setRuntimeConfig(snapshot.runtimeConfig);
  setActiveKernelPath(
    snapshot.activeKernelPath,
    snapshot.activeKernelPathSource || 'none',
    snapshot.activeKernelPathPolicy ?? null
  );
}

export async function runWithRuntimeIsolationForSuite(run) {
  const snapshot = snapshotRuntimeState();
  try {
    return await run();
  } finally {
    restoreRuntimeState(snapshot);
  }
}

export function sanitizeReportOutput(output) {
  if (output == null) return null;
  if (typeof output !== 'object') return output;
  if (ArrayBuffer.isView(output)) {
    return {
      type: output.constructor?.name || 'TypedArray',
      length: Number.isFinite(output.length) ? output.length : null,
    };
  }
  if (
    Number.isFinite(output?.width)
    && Number.isFinite(output?.height)
    && ArrayBuffer.isView(output?.pixels)
  ) {
    const { pixels, ...rest } = output;
    return {
      ...rest,
      width: output.width,
      height: output.height,
      pixels: {
        type: pixels.constructor?.name || 'TypedArray',
        length: Number.isFinite(pixels.length) ? pixels.length : null,
      },
    };
  }
  return output;
}

function normalizeExtends(value) {
  if (Array.isArray(value)) {
    return value.map((entry) => String(entry || '').trim()).filter(Boolean);
  }
  if (typeof value === 'string') {
    const trimmed = value.trim();
    return trimmed ? [trimmed] : [];
  }
  return [];
}

function normalizeExtendsPath(value) {
  const trimmed = String(value || '').trim();
  if (!trimmed) return null;
  return trimmed.endsWith('.json') ? trimmed : `${trimmed}.json`;
}

function resolveAbsoluteUrl(target, base) {
  try {
    if (base) {
      return new URL(target, base).toString();
    }
    if (typeof globalThis.location !== 'undefined' && globalThis.location?.href) {
      return new URL(target, globalThis.location.href).toString();
    }
    return new URL(target, import.meta.url).toString();
  } catch {
    return target;
  }
}

function isAbsoluteUrl(value) {
  return /^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(value);
}

function joinUrl(base, path) {
  if (!base) return path;
  if (isAbsoluteUrl(base)) {
    return new URL(path, base.endsWith('/') ? base : `${base}/`).toString();
  }
  const normalizedBase = base.replace(/\/$/, '');
  const normalizedPath = path.replace(/^\//, '');
  return `${normalizedBase}/${normalizedPath}`;
}

function resolveExtendCandidates(ref, context) {
  const normalized = normalizeExtendsPath(ref);
  if (!normalized) return [];
  if (isAbsoluteUrl(normalized) || normalized.startsWith('/')) {
    return [normalized];
  }
  if (normalized.startsWith('./') || normalized.startsWith('../')) {
    return [resolveAbsoluteUrl(normalized, context.sourceUrl)];
  }
  if (normalized.includes('/')) {
    return [joinUrl(context.profileBaseUrl, normalized)];
  }
  const candidates = [];
  if (context.profileBaseUrl) {
    candidates.push(joinUrl(context.profileBaseUrl, normalized));
    candidates.push(joinUrl(context.profileBaseUrl, `profiles/${normalized}`));
  }
  if (context.sourceUrl) {
    const sourceDir = resolveAbsoluteUrl('./', context.sourceUrl);
    candidates.push(resolveAbsoluteUrl(normalized, sourceDir));
  }
  return [...new Set(candidates)];
}

async function fetchRuntimeConfig(url, options = {}) {
  const response = await fetch(url, { signal: options.signal });
  if (!response.ok) {
    const error = new Error(`Failed to load runtime config: ${response.status}`);
    error.code = response.status === 404 ? 'runtime_config_not_found' : 'runtime_config_fetch_failed';
    throw error;
  }
  return response.json();
}

async function resolveRuntimeConfigExtends(config, context) {
  const runtime = resolveRuntimeFromConfig(config);
  if (!runtime) {
    throw new Error('Runtime config is missing runtime fields');
  }

  const extendsRefs = normalizeExtends(config.extends);
  let mergedRuntime = null;
  let mergedConfig = null;

  for (const ref of extendsRefs) {
    const base = await loadRuntimeConfigFromRef(ref, context);
    mergedRuntime = mergedRuntime ? mergeRuntimeValues(mergedRuntime, base.runtime) : base.runtime;
    mergedConfig = mergedConfig ? mergeRuntimeValues(mergedConfig, base.config) : base.config;
  }

  const combinedRuntime = mergedRuntime ? mergeRuntimeValues(mergedRuntime, runtime) : runtime;
  const combinedConfig = mergedConfig ? mergeRuntimeValues(mergedConfig, config) : { ...config };
  const resolved = { ...combinedConfig, runtime: combinedRuntime };
  if (resolved.extends !== undefined) {
    delete resolved.extends;
  }
  return { config: resolved, runtime: combinedRuntime };
}

async function loadRuntimeConfigChain(url, options = {}, stack = []) {
  const profileBaseUrl = options.profileBaseUrl || options.baseUrl || resolveProfileBaseUrl();
  const resolvedUrl = resolveAbsoluteUrl(url);
  if (stack.includes(resolvedUrl)) {
    throw new Error(`Runtime config extends cycle: ${[...stack, resolvedUrl].join(' -> ')}`);
  }
  const config = await fetchRuntimeConfig(resolvedUrl, options);
  return resolveRuntimeConfigExtends(config, {
    ...options,
    sourceUrl: resolvedUrl,
    profileBaseUrl,
    stack: [...stack, resolvedUrl],
  });
}

export async function loadRuntimeConfigFromRef(ref, context) {
  const candidates = resolveExtendCandidates(ref, context);
  if (!candidates.length) {
    throw new Error(`Runtime config extends is invalid: ${ref}`);
  }
  let lastError = null;
  for (const candidate of candidates) {
    try {
      return await loadRuntimeConfigChain(candidate, context, context.stack ?? []);
    } catch (error) {
      if (error?.code === 'runtime_config_not_found') {
        lastError = error;
        continue;
      }
      throw error;
    }
  }
  if (lastError) {
    throw lastError;
  }
  throw new Error(`Runtime config extends not found: ${ref}`);
}

export async function loadRuntimeConfigFromUrl(url, options = {}) {
  if (!url) {
    throw new Error('runtime config url is required');
  }
  return loadRuntimeConfigChain(url, options);
}

export async function applyRuntimeConfigFromUrl(url, options = {}) {
  const { runtime } = await loadRuntimeConfigFromUrl(url, options);
  const mergedRuntime = mergeRuntimeValues(getRuntimeConfig(), runtime);
  setRuntimeConfig(mergedRuntime);
  return mergedRuntime;
}

export async function loadRuntimeProfile(profileId, options = {}) {
  const baseUrl = options.baseUrl || resolveProfileBaseUrl();
  const normalized = normalizeProfilePath(profileId);
  if (!normalized) {
    throw new Error('runtime profile id is required');
  }
  const url = `${baseUrl.replace(/\/$/, '')}/${normalized}`;
  return loadRuntimeConfigFromUrl(url, { ...options, profileBaseUrl: baseUrl });
}

export async function applyRuntimeProfile(profileId, options = {}) {
  const { runtime } = await loadRuntimeProfile(profileId, options);
  const mergedRuntime = mergeRuntimeValues(getRuntimeConfig(), runtime);
  setRuntimeConfig(mergedRuntime);
  return mergedRuntime;
}

export function normalizeRuntimeConfigChain(value) {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map((entry) => typeof entry === 'string' ? entry.trim() : '')
    .filter(Boolean);
}

export async function applyRuntimeForRun(run, options = {}) {
  const configChain = normalizeRuntimeConfigChain(
    run.configChain
    ?? run.runtime?.configChain
    ?? options.runtime?.configChain
  );
  await applyOrderedRuntimeInputs({
    getRuntimeConfig,
    setRuntimeConfig,
  }, {
    configChain,
    runtimeProfile: run.runtimeProfile ?? null,
    runtimeConfigUrl: run.runtimeConfigUrl ?? null,
    runtimeConfig: run.runtimeConfig ?? null,
  }, {
    loadRuntimeConfigFromRef: (ref, runtimeOptions) => loadRuntimeConfigFromRef(ref, runtimeOptions),
    applyRuntimeProfile,
    applyRuntimeConfigFromUrl,
  }, options);
}

export function normalizeManifest(manifest) {
  if (!manifest || typeof manifest !== 'object') {
    throw new Error('Harness manifest must be an object.');
  }
  const runs = Array.isArray(manifest.runs) ? manifest.runs : [];
  if (!runs.length) {
    throw new Error('Harness manifest must include at least one run.');
  }
  return {
    defaults: manifest.defaults ?? {},
    runs,
    reportModelId: manifest.reportModelId ?? manifest.id ?? 'manifest',
    report: manifest.report ?? null,
  };
}

export function mergeRunDefaults(defaults, run) {
  return {
    ...defaults,
    ...run,
    configChain: run.configChain ?? defaults.configChain ?? null,
    runtimeProfile: run.runtimeProfile ?? defaults.runtimeProfile ?? null,
    runtimeConfigUrl: run.runtimeConfigUrl ?? defaults.runtimeConfigUrl ?? null,
    runtimeConfig: run.runtimeConfig ?? defaults.runtimeConfig ?? null,
    mode: run.mode ?? defaults.mode ?? run.command ?? defaults.command ?? null,
    workload: run.workload ?? run.suite ?? defaults.workload ?? defaults.suite ?? 'inference',
  };
}

export function summarizeManifestRuns(results) {
  let passedRuns = 0;
  let failedRuns = 0;
  let durationMs = 0;
  for (const result of results) {
    const failures = (result.results || []).filter((entry) => !entry.passed && !entry.skipped);
    if (failures.length > 0) {
      failedRuns += 1;
    } else {
      passedRuns += 1;
    }
    durationMs += result.duration || 0;
  }
  return {
    totalRuns: results.length,
    passedRuns,
    failedRuns,
    durationMs,
  };
}
