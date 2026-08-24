import { mergeRuntimeValues } from '../../config/runtime-merge.js';
import { getRuntimeConfig, setRuntimeConfig } from '../../config/runtime.js';
import { validateRuntimeProfileMetadata } from '../../config/schema/runtime-profile.schema.js';
import { resolveCommandRuntimeContract } from '../../tooling/command-request-normalization.js';
import {
  resolveOrderedRuntimeInputs,
  resolveRuntimeFromConfig,
} from '../../tooling/runtime-input-composition.js';
import {
  assertRuntimeInputIntentCompatibility,
  stripLegacyRuntimeIntent,
} from '../../tooling/runtime-intent.js';
import { parseRuntimeOverridesFromURL } from '../test-harness.js';
import { cloneRuntimeConfig } from './runtime-isolation.js';

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
    return new URL('../../config/runtime', import.meta.url).toString().replace(/\/$/, '');
  } catch {
    if (typeof globalThis.location !== 'undefined' && globalThis.location?.href) {
      return new URL('/src/config/runtime/', globalThis.location.href).toString().replace(/\/$/, '');
    }
    return '/src/config/runtime';
  }
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
  const mergedRuntime = mergeRuntimeValues(getRuntimeConfig(), stripLegacyRuntimeIntent(runtime));
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
  const loaded = await loadRuntimeConfigFromUrl(url, { ...options, profileBaseUrl: baseUrl });
  validateRuntimeProfileMetadata(loaded.config, `runtime profile "${profileId}"`);
  return loaded;
}

export async function applyRuntimeProfile(profileId, options = {}) {
  const { runtime } = await loadRuntimeProfile(profileId, options);
  const normalizedRuntime = stripLegacyRuntimeIntent(runtime);
  const mergedRuntime = mergeRuntimeValues(getRuntimeConfig(), normalizedRuntime);
  setRuntimeConfig(mergedRuntime);
  return mergedRuntime;
}

function normalizeRuntimeConfigChain(value) {
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
  const resolved = await resolveOrderedRuntimeInputs(getRuntimeConfig(), {
    configChain,
    runtimeProfile: run.runtimeProfile ?? null,
    runtimeConfigUrl: run.runtimeConfigUrl ?? null,
    runtimeConfig: run.runtimeConfig ?? null,
  }, {
    loadRuntimeConfigFromRef: (ref, runtimeOptions) => loadRuntimeConfigFromRef(ref, runtimeOptions),
    loadRuntimeProfile,
    loadRuntimeConfigFromUrl,
  }, options);
  const commandIntent = run.command
    ? resolveCommandRuntimeContract(run.command).intent
    : null;
  if (commandIntent !== null) {
    assertRuntimeInputIntentCompatibility(commandIntent, resolved.documents);
  }
  if (resolved.documents.length > 0) {
    setRuntimeConfig(stripLegacyRuntimeIntent(resolved.runtime));
  }
}
