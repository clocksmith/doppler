#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { runNodeCommand } from '../tooling/node-command-runner.js';
import { runBrowserCommandInNode } from '../tooling/node-browser-command-runner.js';
import {
  TOOLING_COMMANDS,
  normalizeToolingCommandRequest,
} from '../tooling/command-api.js';
import { createToolingErrorEnvelope } from '../tooling/command-envelope.js';
import { buildHfResolveBaseUrl } from '../utils/hf-resolve-url.js';
import { DEFAULT_EXTERNAL_MODELS_ROOT } from '../tooling/hf-registry-utils.js';

const NODE_WEBGPU_INCOMPLETE_MESSAGE = 'node command: WebGPU runtime is incomplete in Node';
const CLI_POLICY_PATH = fileURLToPath(new URL('./config/doppler-cli-policy.json', import.meta.url));
const DEFAULT_EXTERNAL_RDRR_ROOT = path.join(DEFAULT_EXTERNAL_MODELS_ROOT, 'rdrr');
const DEFAULT_CLI_POLICY = {
  defaults: {
    surface: {
      default: 'auto',
      allowed: ['auto', 'node', 'browser'],
    },
    bench: {
      cacheMode: 'warm',
    },
    cacheMode: null,
    loadMode: null,
    benchmark: {
      saveDir: './benchmarks/vendors/results',
    },
  },
  surfaceFallback: {
    enabled: true,
    from: 'auto',
    to: 'browser',
    errorFragments: [NODE_WEBGPU_INCOMPLETE_MESSAGE],
  },
};

const COMMON_CLI_FLAGS = Object.freeze([
  'config',
  'surface',
  'pretty',
  'json',
  'help',
  'h',
  'runtime-config',
]);

function asStringOrNull(value) {
  if (value === undefined || value === null) return null;
  if (typeof value === 'string') {
    const trimmed = value.trim();
    if (!trimmed.length) return null;
    return trimmed;
  }
  return String(value);
}

function usage() {
  return [
    'Usage:',
    '  doppler convert --config <path|url|json> [--surface auto|node]',
    '  doppler debug --config <path|url|json> [--runtime-config <path|url|json>] [--surface auto|node|browser]',
    '  doppler bench --config <path|url|json> [--runtime-config <path|url|json>] [--surface auto|node|browser]',
    '  doppler verify --config <path|url|json> [--runtime-config <path|url|json>] [--surface auto|node|browser]',
    '  doppler lora --config <path|url|json> [--surface auto|node]',
    '  doppler distill --config <path|url|json> [--surface auto|node]',
    '',
    'Flags:',
    '  --config <path|url|json>        Required command config payload (file path, URL, or JSON object string).',
    '  --runtime-config <value>        Compatibility runtime override alias (JSON object, URL, or file path).',
    '  --surface <auto|node|browser>   Optional execution surface override.',
    '  --pretty                        Print human-readable summary instead of JSON',
    '  --help, -h                      Show this help message',
    '',
    'Command Config Contract:',
    '  The config payload must be a JSON object and may include:',
    '    - request: tooling command request fields (workload, modelId, training fields, convertPayload, etc).',
    '      May also include `runtimeProfile`, `runtimeConfigUrl`, and `runtimeConfig`.',
    '      Unknown top-level keys are disallowed when `request` is used as the envelope key.',
    '    - run: CLI-only run controls (surface, browser options, and bench save/compare/manifest settings).',
    '    - runtimeProfile: optional runtime profile id.',
    '    - runtimeConfigUrl: optional runtime override URL or local JSON path.',
    '    - runtimeConfig: optional inline runtime override object.',
    '',
    'Example:',
    '  doppler verify --config \'{"request":{"workload":"inference","modelId":"gemma-3-270m-it-f16-af32"}}\'',
  ].join('\n');
}

function parseArgs(argv) {
  const out = {
    command: null,
    flags: {},
  };

  if (!argv.length) return out;
  out.command = asStringOrNull(argv[0]);

  for (let i = 1; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === '-h') {
      out.flags.h = true;
      continue;
    }
    if (token.startsWith('-') && !token.startsWith('--')) {
      throw new Error(`Unsupported short flag "${token}". Use long-form flags (for example --help).`);
    }
    if (!token.startsWith('--')) {
      throw new Error('Positional arguments are not supported. Use --config for command payloads.');
    }

    const key = token.slice(2);
    if (
      key === 'json'
      || key === 'pretty'
      || key === 'help'
      || key === 'h'
    ) {
      out.flags[key] = true;
      continue;
    }

    const value = argv[i + 1];
    if (value === undefined) {
      throw new Error(`Missing value for --${key}`);
    }

    if (value.startsWith('--')) {
      throw new Error(`Missing value for --${key}`);
    }

    out.flags[key] = value;
    i += 1;
  }

  return out;
}

function levenshteinDistance(a, b) {
  const source = String(a ?? '');
  const target = String(b ?? '');
  if (source === target) return 0;
  if (source.length === 0) return target.length;
  if (target.length === 0) return source.length;

  const previous = new Array(target.length + 1);
  const current = new Array(target.length + 1);

  for (let i = 0; i <= target.length; i += 1) {
    previous[i] = i;
  }

  for (let i = 1; i <= source.length; i += 1) {
    current[0] = i;
    for (let j = 1; j <= target.length; j += 1) {
      const cost = source[i - 1] === target[j - 1] ? 0 : 1;
      current[j] = Math.min(
        previous[j] + 1,
        current[j - 1] + 1,
        previous[j - 1] + cost
      );
    }
    for (let j = 0; j <= target.length; j += 1) {
      previous[j] = current[j];
    }
  }

  return previous[target.length];
}

function findClosestFlag(flag, allowedFlags) {
  const normalizedFlag = String(flag ?? '').trim();
  if (!normalizedFlag) return null;

  let candidate = null;
  let distance = Infinity;
  for (const allowedFlag of allowedFlags) {
    const nextDistance = levenshteinDistance(normalizedFlag, allowedFlag);
    if (nextDistance < distance) {
      candidate = allowedFlag;
      distance = nextDistance;
    }
  }
  return distance <= 3 ? candidate : null;
}

function validateCommandFlags(parsed) {
  const command = parsed?.command;
  if (!command || !TOOLING_COMMANDS.includes(command)) {
    return;
  }

  const allowedFlags = new Set(COMMON_CLI_FLAGS);
  const keys = Object.keys(parsed.flags || {});
  for (const key of keys) {
    if (allowedFlags.has(key)) {
      continue;
    }

    const suggestion = findClosestFlag(key, allowedFlags);
    if (suggestion) {
      throw new Error(`Unknown flag --${key} for "${command}". Did you mean --${suggestion}?`);
    }
    throw new Error(`Unknown flag --${key} for "${command}".`);
  }
}

function parseJsonObjectFlag(value, label) {
  if (asStringOrNull(value) === null) return null;
  try {
    const parsed = JSON.parse(value);
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      throw new Error('value must be a JSON object');
    }
    return parsed;
  } catch (error) {
    throw new Error(`Invalid ${label}: ${error.message}`);
  }
}

function parseRuntimeConfigUrl(value) {
  const normalized = asStringOrNull(value);
  if (normalized === null) return null;
  return isAbsoluteUrl(normalized)
    ? normalized
    : pathToFileURL(path.resolve(normalized)).href;
}

function isAbsoluteUrl(value) {
  const normalized = asStringOrNull(value);
  if (normalized === null) return false;
  try {
    const parsed = new URL(normalized);
    return typeof parsed.protocol === 'string' && parsed.protocol.length > 0;
  } catch {
    return false;
  }
}

function parseUnifiedRuntimeConfig(value) {
  const normalized = asStringOrNull(value);
  if (normalized === null) return null;
  if (normalized.startsWith('{')) {
    return {
      runtimeProfile: null,
      runtimeConfigUrl: null,
      runtimeConfig: parseJsonObjectFlag(normalized, '--runtime-config'),
    };
  }
  if (isAbsoluteUrl(normalized)) {
    return {
      runtimeProfile: null,
      runtimeConfigUrl: normalized,
      runtimeConfig: null,
    };
  }
  return {
    runtimeProfile: null,
    runtimeConfigUrl: parseRuntimeConfigUrl(normalized),
    runtimeConfig: null,
  };
}

function resolveRuntimeConfigFlags(parsed) {
  const unifiedRaw = asStringOrNull(parsed.flags['runtime-config']);
  return parseUnifiedRuntimeConfig(unifiedRaw);
}

async function readJsonObjectFile(filePath, label) {
  const resolved = path.resolve(String(filePath));
  let raw;
  try {
    raw = await fs.readFile(resolved, 'utf8');
  } catch (error) {
    throw new Error(`${label} not found or unreadable: ${resolved}`);
  }
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch (error) {
    throw new Error(`${label} must contain valid JSON: ${error.message}`);
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error(`${label} must be a JSON object.`);
  }
  return parsed;
}

async function readJsonObjectUrl(rawUrl, label) {
  let response;
  try {
    response = await fetch(rawUrl, {
      headers: {
        Connection: 'close',
      },
      redirect: 'follow',
      signal: AbortSignal.timeout(30000),
    });
  } catch (error) {
    throw new Error(`${label} URL request failed: ${error?.message || String(error)}`);
  }
  if (!response.ok) {
    throw new Error(`${label} URL request failed: HTTP ${response.status}`);
  }
  let raw;
  try {
    raw = await response.text();
  } catch (error) {
    throw new Error(`${label} URL request failed: ${error?.message || String(error)}`);
  }
  return parseJsonObjectFlag(raw, label);
}

async function readJsonObjectInput(inputValue, label) {
  const normalized = asStringOrNull(inputValue);
  if (normalized === null) {
    throw new Error(`${label} is required.`);
  }
  if (normalized.startsWith('{')) {
    return parseJsonObjectFlag(normalized, label);
  }
  if (isAbsoluteUrl(normalized)) {
    return readJsonObjectUrl(normalized, label);
  }
  return readJsonObjectFile(normalized, label);
}

function resolveCommandConfigFlag(parsed) {
  const config = asStringOrNull(parsed.flags.config);
  if (!config) {
    throw new Error('command requires --config <path|url|json>.');
  }
  return config;
}

function parseBooleanFlag(value, label) {
  const normalizedInput = asStringOrNull(value);
  if (normalizedInput === null) return null;
  if (typeof value === 'boolean') return value;
  if (typeof value === 'string') {
    const normalized = normalizedInput.toLowerCase();
    if (normalized === 'true') return true;
    if (normalized === 'false') return false;
  }
  throw new Error(`${label} must be true or false`);
}

function parseNumberFlag(value, label) {
  const normalized = asStringOrNull(value);
  if (normalized === null) return null;
  const parsed = Number(normalized);
  if (!Number.isFinite(parsed)) {
    throw new Error(`${label} must be a number`);
  }
  return parsed;
}

function parseBrowserArgs(value) {
  if (value === undefined || value === null) return [];
  return Array.isArray(value) ? value.map((item) => String(item)) : [String(value)];
}

async function pathExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

function resolveStaticRootDir(browserOptions = {}) {
  const configured = asStringOrNull(browserOptions.staticRootDir);
  if (configured) {
    return path.resolve(String(configured));
  }
  return process.cwd();
}

function resolveRdrrRoot(options = {}) {
  return path.resolve(asStringOrNull(options.rdrrRoot) || DEFAULT_EXTERNAL_RDRR_ROOT);
}

async function findResolvableModelCandidate(candidates) {
  const discoveredManifestCandidates = [];

  for (const candidate of candidates) {
    if (!await pathExists(candidate.manifestPath)) {
      continue;
    }
    discoveredManifestCandidates.push(candidate);

    const modelDir = path.dirname(candidate.manifestPath);
    try {
      const files = await fs.readdir(modelDir, { withFileTypes: true });
      const hasShards = files.some((entry) =>
        entry.isFile() && /^shard_\d+\.bin$/u.test(entry.name)
      );
      if (hasShards) {
        return { candidate, discoveredManifestCandidates };
      }
    } catch {
      return { candidate, discoveredManifestCandidates };
    }
  }

  return { candidate: null, discoveredManifestCandidates };
}

async function resolveExternalModelDirectory(rdrrRoot, modelId) {
  const directModelDir = path.join(rdrrRoot, modelId);
  const directManifestPath = path.join(directModelDir, 'manifest.json');
  if (await pathExists(directManifestPath)) {
    return {
      modelDir: directModelDir,
      manifestPath: directManifestPath,
    };
  }

  let entries = [];
  try {
    entries = await fs.readdir(rdrrRoot, { withFileTypes: true });
  } catch {
    return null;
  }

  const matches = [];
  for (const entry of entries) {
    if (!entry.isDirectory()) {
      continue;
    }
    const manifestPath = path.join(rdrrRoot, entry.name, 'manifest.json');
    if (!await pathExists(manifestPath)) {
      continue;
    }
    let manifest = null;
    try {
      manifest = JSON.parse(await fs.readFile(manifestPath, 'utf8'));
    } catch {
      continue;
    }
    if (manifest?.modelId !== modelId) {
      continue;
    }
    matches.push({
      modelDir: path.join(rdrrRoot, entry.name),
      manifestPath,
    });
  }

  if (matches.length > 1) {
    const matchPaths = matches.map((match) => match.modelDir).join(', ');
    throw new Error(
      `Model "${modelId}" matched multiple external directories. ` +
      `Disambiguate by setting request.modelUrl in --config. Matches: ${matchPaths}`
    );
  }

  return matches[0] || null;
}

const CATALOG_PATH = fileURLToPath(new URL('../../models/catalog.json', import.meta.url));

async function resolveCatalogEntry(modelId) {
  let catalog;
  try {
    catalog = JSON.parse(await fs.readFile(CATALOG_PATH, 'utf8'));
  } catch {
    return null;
  }
  if (!Array.isArray(catalog?.models)) {
    return null;
  }
  return catalog.models.find((entry) => (
    entry.modelId === modelId
    || (Array.isArray(entry.aliases) && entry.aliases.includes(modelId))
  )) || null;
}

function buildCatalogModelUrl(entry) {
  if (!entry?.hf?.repoId || !entry?.hf?.path) {
    return null;
  }
  return buildHfResolveBaseUrl(entry.hf);
}

export async function resolveBrowserModelUrl(request, browserOptions = {}) {
  if (request.modelUrl || !request.modelId) {
    return request;
  }

  const modelId = String(request.modelId);
  const encodedModelId = encodeURIComponent(modelId);

  if (asStringOrNull(browserOptions.baseUrl)) {
    return {
      ...request,
      modelUrl: `/models/${encodedModelId}`,
    };
  }

  const staticRootDir = resolveStaticRootDir(browserOptions);
  const externalModel = await resolveExternalModelDirectory(resolveRdrrRoot(browserOptions), modelId);
  const candidates = [
    {
    modelUrl: `/models/local/${encodedModelId}`,
    manifestPath: path.join(staticRootDir, 'models', 'local', modelId, 'manifest.json'),
    },
    {
    modelUrl: `/models/${encodedModelId}`,
    manifestPath: path.join(staticRootDir, 'models', modelId, 'manifest.json'),
    },
    {
      modelUrl: `/models/external/${encodeURIComponent(path.basename(externalModel?.modelDir || modelId))}`,
      manifestPath: externalModel?.manifestPath || path.join(resolveRdrrRoot(browserOptions), modelId, 'manifest.json'),
    },
  ];

  const { candidate, discoveredManifestCandidates } = await findResolvableModelCandidate(candidates);
  if (candidate) {
    return {
      ...request,
      modelUrl: candidate.modelUrl,
    };
  }

  if (discoveredManifestCandidates.length > 0) {
    const paths = discoveredManifestCandidates
      .map((candidate) => candidate.modelUrl)
      .join(', ');
    throw new Error(
      `Model "${modelId}" was found, but no shard files (shard_*.bin) are present. ` +
      `Checked: ${paths}. Add shard files beside the manifest, or set request.modelUrl in --config to a complete model directory.`
    );
  }

  const catalogEntry = await resolveCatalogEntry(modelId);
  if (catalogEntry) {
    const hfUrl = buildCatalogModelUrl(catalogEntry);
    if (hfUrl) {
      return {
        ...request,
        modelUrl: hfUrl,
      };
    }
  }

  return {
    ...request,
    modelUrl: `/models/${encodedModelId}`,
  };
}

export async function resolveNodeModelUrl(request, options = {}) {
  if (request.modelUrl || !request.modelId) {
    return request;
  }

  const modelId = String(request.modelId);
  const staticRootDir = resolveStaticRootDir(options);
  const rdrrRoot = resolveRdrrRoot(options);
  const externalModel = await resolveExternalModelDirectory(rdrrRoot, modelId);
  const localCandidates = [
    {
      modelDir: path.join(staticRootDir, 'models', 'local', modelId),
      manifestPath: path.join(staticRootDir, 'models', 'local', modelId, 'manifest.json'),
    },
    {
      modelDir: path.join(staticRootDir, 'models', modelId),
      manifestPath: path.join(staticRootDir, 'models', modelId, 'manifest.json'),
    },
  ];
  const candidates = [
    ...localCandidates,
    {
      modelDir: externalModel?.modelDir || path.join(rdrrRoot, modelId),
      manifestPath: externalModel?.manifestPath || path.join(rdrrRoot, modelId, 'manifest.json'),
    },
  ];
  const { candidate, discoveredManifestCandidates } =
    await findResolvableModelCandidate(candidates);

  if (candidate) {
    return {
      ...request,
      modelUrl: pathToFileURL(candidate.modelDir).href.replace(/\/$/, ''),
    };
  }

  if (discoveredManifestCandidates.length > 0) {
    const paths = discoveredManifestCandidates
      .map((candidate) => candidate.modelDir)
      .join(', ');
    throw new Error(
      `Model "${modelId}" was found, but no shard files (shard_*.bin) are present. ` +
      `Checked: ${paths}. Add shard files beside the manifest, or set request.modelUrl to a complete model directory.`
    );
  }

  const catalogEntry = await resolveCatalogEntry(modelId);
  if (catalogEntry) {
    const hfUrl = buildCatalogModelUrl(catalogEntry);
    if (hfUrl) {
      return {
        ...request,
        modelUrl: hfUrl,
      };
    }
    throw new Error(
      `Model "${modelId}" found in catalog as "${catalogEntry.modelId}" but has no HF source configured.`
    );
  }

  throw new Error(
    `Model "${modelId}" not found. Searched local: ${rdrrRoot}. Not in catalog. `
    + 'Set request.modelUrl to a file:// or https:// URL, '
    + 'or set DOPPLER_EXTERNAL_MODELS_ROOT to the parent of your rdrr/ folder.'
  );
}

function parseSurface(value, command, policy = DEFAULT_CLI_POLICY) {
  const normalizedInput = asStringOrNull(value);
  const normalizedSurface = policy.defaults && policy.defaults.surface && policy.defaults.surface.default
    ? policy.defaults.surface.default
    : 'auto';
  const allowedSurfaces = policy.defaults && Array.isArray(policy.defaults.surface?.allowed)
    ? policy.defaults.surface.allowed
    : ['auto', 'node', 'browser'];
  const normalized = String(normalizedInput === null ? normalizedSurface : normalizedInput).trim().toLowerCase();
  if (!allowedSurfaces.includes(normalized)) {
    throw new Error('--surface must be one of auto, node, browser');
  }
  if (command === 'convert' && normalized === 'browser') {
    throw new Error('convert is not supported on browser relay. Use --surface node or --surface auto.');
  }
  if ((command === 'lora' || command === 'distill') && normalized === 'browser') {
    throw new Error(`${command} is not supported on browser relay. Use --surface node or --surface auto.`);
  }
  return normalized;
}

function isPlainObject(value) {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

const CONFIG_ENVELOPE_KNOWN_KEYS = new Set([
  'request',
  'run',
  'runtimeProfile',
  'runtimeConfigUrl',
  'runtimeConfig',
]);

function buildRuntimeOverridesFromObject(source = {}, sourceLabel = '--config') {
  const normalizedRuntimeProfile = asStringOrNull(source.runtimeProfile);
  const normalizedRuntimeConfigUrl = asStringOrNull(source.runtimeConfigUrl) == null
    ? null
    : parseRuntimeConfigUrl(source.runtimeConfigUrl);

  const hasInlineRuntimeConfig = Object.prototype.hasOwnProperty.call(source, 'runtimeConfig');
  let runtimeConfig = null;
  if (hasInlineRuntimeConfig) {
    if (!isPlainObject(source.runtimeConfig)) {
      throw new Error(`${sourceLabel} runtimeConfig must be a JSON object when provided.`);
    }
    runtimeConfig = source.runtimeConfig;
  }

  return {
    runtimeProfile: normalizedRuntimeProfile,
    runtimeConfigUrl: normalizedRuntimeConfigUrl,
    runtimeConfig,
  };
}

function resolveConfigEnvelope(configPayload) {
  if (!isPlainObject(configPayload)) {
    throw new Error('--config must resolve to a JSON object.');
  }
  if (configPayload.request !== undefined && !isPlainObject(configPayload.request)) {
    throw new Error('--config field "request" must be a JSON object when provided.');
  }
  if (configPayload.run !== undefined && !isPlainObject(configPayload.run)) {
    throw new Error('--config field "run" must be a JSON object when provided.');
  }
  if (configPayload.request !== undefined) {
    const topLevelUnknown = Object.keys(configPayload).filter(
      (key) => !CONFIG_ENVELOPE_KNOWN_KEYS.has(key)
    );
    if (topLevelUnknown.length > 0) {
      throw new Error(
        `--config has unknown top-level keys for request envelope: ${topLevelUnknown.join(', ')}.`
      );
    }
  }

  const hasRequestEnvelope = isPlainObject(configPayload.request);
  const requestPayload = hasRequestEnvelope
    ? { ...configPayload.request }
    : { ...configPayload };
  const topLevelRuntimeConfig = hasRequestEnvelope
    ? buildRuntimeOverridesFromObject(configPayload, '--config')
    : {
      runtimeProfile: null,
      runtimeConfigUrl: null,
      runtimeConfig: null,
    };
  const requestRuntimeConfig = buildRuntimeOverridesFromObject(requestPayload, '--config.request');
  if (hasRequestEnvelope) {
    if (
      topLevelRuntimeConfig.runtimeProfile != null
      && requestRuntimeConfig.runtimeProfile != null
    ) {
      throw new Error(
        'Cannot set runtimeProfile in both --config payload top-level and --config.request.'
      );
    }
    if (
      topLevelRuntimeConfig.runtimeConfigUrl != null
      && requestRuntimeConfig.runtimeConfigUrl != null
    ) {
      throw new Error(
        'Cannot set runtimeConfigUrl in both --config payload top-level and --config.request.'
      );
    }
    if (
      topLevelRuntimeConfig.runtimeConfig != null
      && requestRuntimeConfig.runtimeConfig != null
    ) {
      throw new Error(
        'Cannot set runtimeConfig in both --config payload top-level and --config.request.'
      );
    }
  }

  const requestInput = {
    ...requestPayload,
    runtimeProfile: (
      topLevelRuntimeConfig.runtimeProfile !== null
      ? topLevelRuntimeConfig.runtimeProfile
      : requestRuntimeConfig.runtimeProfile
    ),
    runtimeConfigUrl: (
      topLevelRuntimeConfig.runtimeConfigUrl !== null
      ? topLevelRuntimeConfig.runtimeConfigUrl
      : requestRuntimeConfig.runtimeConfigUrl
    ),
    runtimeConfig: (
      topLevelRuntimeConfig.runtimeConfig !== null
      ? topLevelRuntimeConfig.runtimeConfig
      : requestRuntimeConfig.runtimeConfig
    ),
  };
  return {
    request: requestInput,
    run: isPlainObject(configPayload.run) ? configPayload.run : {},
  };
}

function applyRuntimeFlagOverride(requestInput, runtimeOverride) {
  if (!runtimeOverride) {
    return;
  }
  const hasInlineRuntime = (
    requestInput.runtimeProfile != null
    || requestInput.runtimeConfigUrl != null
    || requestInput.runtimeConfig != null
  );
  if (hasInlineRuntime) {
    throw new Error(
      '--runtime-config cannot be combined with runtimeProfile/runtimeConfigUrl/runtimeConfig values inside --config.'
    );
  }
  requestInput.runtimeProfile = runtimeOverride.runtimeProfile;
  requestInput.runtimeConfigUrl = runtimeOverride.runtimeConfigUrl;
  requestInput.runtimeConfig = runtimeOverride.runtimeConfig;
}

function resolveBenchRunOptions(runConfig, policy = DEFAULT_CLI_POLICY) {
  const benchConfig = isPlainObject(runConfig?.bench) ? runConfig.bench : {};
  const configuredSaveDir = asStringOrNull(benchConfig.saveDir);
  const defaultSaveDir = asStringOrNull(policy?.defaults?.benchmark?.saveDir);
  return {
    shouldSave: benchConfig.save === true,
    saveDir: configuredSaveDir === null
      ? defaultSaveDir || './benchmarks/vendors/results'
      : configuredSaveDir,
    comparePath: asStringOrNull(benchConfig.compare),
    manifestPath: asStringOrNull(benchConfig.manifest),
  };
}

function normalizeModelUrl(value) {
  if (typeof value !== 'string' || !value.trim()) {
    return value;
  }
  const trimmed = value.trim();
  if (/^[a-z][a-z0-9+.-]*:\/\//u.test(trimmed)) {
    return trimmed;
  }
  if (trimmed.startsWith('/') || trimmed.startsWith('.')) {
    return pathToFileURL(path.resolve(trimmed)).href.replace(/\/$/, '');
  }
  return trimmed;
}

function resolveSurfaceForCommand(command, parsed, runConfig, policy = DEFAULT_CLI_POLICY) {
  const fromCli = asStringOrNull(parsed.flags.surface);
  const fromRun = asStringOrNull(runConfig?.surface);
  return parseSurface(fromCli ?? fromRun ?? null, command, policy);
}

export async function buildRequest(parsed, policy = DEFAULT_CLI_POLICY) {
  const command = parsed.command;
  if (!command || !TOOLING_COMMANDS.includes(command)) {
    throw new Error(`Unsupported command "${command || ''}"`);
  }

  const configValue = resolveCommandConfigFlag(parsed);
  const configPayload = await readJsonObjectInput(configValue, '--config');
  const envelope = resolveConfigEnvelope(configPayload);
  const runtimeOverride = resolveRuntimeConfigFlags(parsed);

  const requestInput = { ...envelope.request };
  if (requestInput.command != null && requestInput.command !== command) {
    throw new Error(
      `--config request command mismatch: CLI command is "${command}" but config command is "${requestInput.command}".`
    );
  }
  requestInput.command = command;
  if (requestInput.modelUrl != null) {
    requestInput.modelUrl = normalizeModelUrl(requestInput.modelUrl);
  }

  applyRuntimeFlagOverride(requestInput, runtimeOverride);

  const surfaceFromCli = asStringOrNull(parsed.flags.surface) !== null;
  const surface = resolveSurfaceForCommand(command, parsed, envelope.run, policy);

  return {
    request: normalizeToolingCommandRequest(requestInput),
    runConfig: envelope.run,
    surface,
    surfaceFromCli,
    benchRunOptions: resolveBenchRunOptions(envelope.run, policy),
  };
}

function buildNodeRunOptions(jsonOutput) {
  return {
    onProgress(progress) {
      if (jsonOutput) return;
      if (!progress?.message) return;
      if (Number.isFinite(progress.current) && Number.isFinite(progress.total)) {
        console.error(`[progress] ${progress.current}/${progress.total} ${progress.message}`);
      } else {
        console.error(`[progress] ${progress.stage ?? 'run'} ${progress.message}`);
      }
    },
  };
}

function buildBrowserRunOptions(runConfig, jsonOutput, request = {}) {
  const browser = isPlainObject(runConfig?.browser) ? runConfig.browser : {};

  const headed = parseBooleanFlag(browser.headed, 'run.browser.headed') === true;
  const explicitHeadless = parseBooleanFlag(browser.headless, 'run.browser.headless');
  if (headed && explicitHeadless !== null) {
    throw new Error('run.browser.headed is mutually exclusive with run.browser.headless.');
  }

  const options = {
    channel: asStringOrNull(browser.channel),
    executablePath: asStringOrNull(browser.executablePath),
    runnerPath: asStringOrNull(browser.runnerPath),
    staticRootDir: asStringOrNull(browser.staticRootDir),
    rdrrRoot: asStringOrNull(browser.rdrrRoot),
    baseUrl: asStringOrNull(browser.baseUrl),
    browserArgs: parseBrowserArgs(browser.browserArgs),
    headless: headed ? false : (explicitHeadless ?? true),
  };
  const rdrrRoot = resolveRdrrRoot(options);
  options.staticMounts = [
    {
      urlPrefix: '/models/external',
      rootDir: rdrrRoot,
    },
  ];

  const port = parseNumberFlag(browser.port, 'run.browser.port');
  if (port !== null) {
    options.port = port;
  }

  const timeoutMs = parseNumberFlag(browser.timeoutMs, 'run.browser.timeoutMs');
  if (timeoutMs !== null) {
    options.timeoutMs = timeoutMs;
  }

  const opfsCache = parseBooleanFlag(browser.opfsCache, 'run.browser.opfsCache');
  if (opfsCache === false) {
    options.opfsCache = false;
  }

  const userDataDir = asStringOrNull(browser.userDataDir);
  if (userDataDir) {
    options.userDataDir = userDataDir;
  }

  if (request.cacheMode === 'cold') {
    options.wipeCacheBeforeLaunch = true;
  }

  const streamConsole = parseBooleanFlag(browser.console, 'run.browser.console');
  const shouldStreamConsole = streamConsole === true;
  if (shouldStreamConsole && !jsonOutput) {
    options.onConsole = ({ type, text }) => {
      console.error(`[browser:${type}] ${text}`);
    };
  }

  return options;
}

function isNodeWebGPUFallbackCandidate(error, fallbackPolicy = DEFAULT_CLI_POLICY.surfaceFallback) {
  const message = error?.message || String(error || '');
  const fallbackSignatures = Array.isArray(fallbackPolicy?.errorFragments) && fallbackPolicy.errorFragments.length > 0
    ? fallbackPolicy.errorFragments
    : [NODE_WEBGPU_INCOMPLETE_MESSAGE];
  return fallbackSignatures.some((signature) => message.includes(signature));
}

function isTrainingCommandFlow(request) {
  if (!request || typeof request !== 'object') return false;
  if (request.workload === 'training') return true;
  if (request.command === 'lora' || request.command === 'distill') return true;
  return request.command === 'bench' && (request.workload === 'training' || request.workloadType === 'training');
}

function resolveErrorSurface(error, fallbackSurface = null) {
  return (
    asStringOrNull(fallbackSurface)
    || asStringOrNull(error?.surface)
    || asStringOrNull(error?.details?.surface)
    || null
  );
}

export function createCliToolingErrorEnvelope(error, context = {}) {
  return createToolingErrorEnvelope(error, {
    surface: resolveErrorSurface(error, context.surface),
    request: context.request ?? null,
  });
}

export function finalizeCliCommandResponse(response, request) {
  if (!isPlainObject(response) || !Object.prototype.hasOwnProperty.call(response, 'request')) {
    return response;
  }
  return {
    ...response,
    request,
  };
}

async function runCommandOnSurface(request, surface, runConfig, jsonOutput) {
  if (surface === 'node') {
    const nodeRequest = await resolveNodeModelUrl(request);
    if (!jsonOutput) {
      console.error('[surface] running on: node');
      if (nodeRequest.modelUrl && nodeRequest.modelUrl !== request.modelUrl) {
        console.error(`[surface] node resolved modelUrl=${nodeRequest.modelUrl}`);
      }
    }
    const response = await runNodeCommand(nodeRequest, buildNodeRunOptions(jsonOutput));
    return finalizeCliCommandResponse(response, request);
  }

  const browserOptions = buildBrowserRunOptions(runConfig, jsonOutput, request);
  const browserRequest = await resolveBrowserModelUrl(request, browserOptions);

  if (!jsonOutput) {
    const mode = browserOptions.headless === false ? 'headed' : 'headless';
    console.error(`[surface] running on: browser (${mode})`);
    if (browserRequest.modelUrl && browserRequest.modelUrl !== request.modelUrl) {
      console.error(`[surface] browser resolved modelUrl=${browserRequest.modelUrl}`);
    }
  }

  const response = await runBrowserCommandInNode(browserRequest, browserOptions);
  return finalizeCliCommandResponse(response, request);
}

async function runWithAutoSurface(request, runConfig, jsonOutput, policy = DEFAULT_CLI_POLICY) {
  if (request.command === 'convert') {
    return runCommandOnSurface(request, 'node', runConfig, jsonOutput);
  }
  const fallbackPolicy = policy?.surfaceFallback || { enabled: false };

  try {
    return await runCommandOnSurface(request, 'node', runConfig, jsonOutput);
  } catch (error) {
    if (!fallbackPolicy.enabled || !isNodeWebGPUFallbackCandidate(error, fallbackPolicy)) {
      throw error;
    }
    if (isTrainingCommandFlow(request)) {
      const downgradeError = new Error(
        (request.command === 'lora' || request.command === 'distill')
          ? 'Training command auto-surface downgrade is blocked. Re-run with --surface node after fixing Node WebGPU support.'
          : 'Training command auto-surface downgrade is blocked. Re-run with --surface node after fixing Node WebGPU support, or explicitly choose --surface browser.'
      );
      downgradeError.code = 'training_surface_downgrade_blocked';
      downgradeError.surface = 'node';
      downgradeError.command = request.command;
      downgradeError.workload = request.workload;
      downgradeError.workloadType = request.workloadType || null;
      downgradeError.fromSurface = 'node';
      downgradeError.toSurface = fallbackPolicy.to || 'browser';
      throw downgradeError;
    }
    if (fallbackPolicy.to !== 'browser') {
      throw error;
    }
    if (!jsonOutput) {
      console.error('[surface] node WebGPU unavailable, falling back to browser');
    }
    return runCommandOnSurface(request, 'browser', runConfig, jsonOutput);
  }
}

function toSummary(result) {
  if (!result || typeof result !== 'object') {
    return 'ok';
  }

  if (result.manifest?.modelId) {
    const contractStatus = result.executionContractArtifact?.ok === true
      ? ' contract=pass'
      : result.executionContractArtifact
        ? ' contract=fail'
        : '';
    return `converted ${result.manifest.modelId} (${result.tensorCount} tensors, ${result.shardCount} shards)${contractStatus}`;
  }

  if (result.kind === 'lora' || result.kind === 'distill') {
    const workloadId = result.workloadId || 'unknown';
    const action = result.action || 'run';
    const runRoot = result.runRoot || 'n/a';
    return `${result.kind} ${action} workload=${workloadId} runRoot=${runRoot}`;
  }

  const suite = result.suite || result.report?.suite || 'suite';
  const modelId = result.modelId || result.report?.modelId || 'unknown';
  const passed = Number.isFinite(result.passed) ? result.passed : null;
  const failed = Number.isFinite(result.failed) ? result.failed : null;
  const duration = Number.isFinite(result.duration) ? `${result.duration.toFixed(1)}ms` : 'n/a';
  if (passed !== null && failed !== null) {
    return `${suite} model=${modelId} passed=${passed} failed=${failed} duration=${duration}`;
  }
  return `${suite} model=${modelId}`;
}

function formatNumber(value, digits = 2) {
  return Number.isFinite(value) ? Number(value).toFixed(digits) : 'n/a';
}

function formatMs(value) {
  return Number.isFinite(value) ? `${Number(value).toFixed(1)}ms` : 'n/a';
}

function quoteOneLine(value) {
  const s = String(value ?? '').replace(/\s+/g, ' ').trim();
  if (!s) return '""';
  const clipped = s.length > 120 ? `${s.slice(0, 117)}...` : s;
  return JSON.stringify(clipped);
}

function quoteOneLineOrStructured(value) {
  if (typeof value === 'string') return quoteOneLine(value);
  if (value == null) return null;
  try {
    return quoteOneLine(JSON.stringify(value));
  } catch {
    return quoteOneLine(String(value));
  }
}

function compactTimestamp() {
  return new Date().toISOString().replace(/[-:]/g, '').replace(/\.\d+Z$/, '');
}

async function saveBenchResult(result, saveDir) {
  await fs.mkdir(saveDir, { recursive: true });
  const modelId = String(result?.modelId || 'unknown').replace(/[^a-zA-Z0-9_-]/g, '_');
  const ts = compactTimestamp();
  const filename = `${modelId}_${ts}.json`;
  const filePath = path.join(saveDir, filename);
  const json = JSON.stringify(result, null, 2);
  await fs.writeFile(filePath, json, 'utf-8');
  await fs.writeFile(path.join(saveDir, 'latest.json'), json, 'utf-8');
  return filePath;
}

async function loadBaseline(comparePath, saveDir) {
  const resolved = comparePath === 'last'
    ? path.join(saveDir, 'latest.json')
    : path.resolve(comparePath);
  try {
    const raw = await fs.readFile(resolved, 'utf-8');
    return JSON.parse(raw);
  } catch (error) {
    console.error(`[compare] failed to load baseline from ${resolved}: ${error.message}`);
    return null;
  }
}

function normalizeBenchMetrics(result) {
  const m = result?.metrics;
  if (!m) return m;
  return {
    decodeTokensPerSec: m.decodeTokensPerSec,
    prefillTokensPerSec: m.prefillTokensPerSec,
    firstTokenMs: m.firstTokenMs,
    firstResponseMs: m.firstResponseMs,
    prefillMs: m.prefillMs,
    decodeMs: m.decodeMs,
    totalRunMs: m.totalRunMs,
    modelLoadMs: m.modelLoadMs,
    decodeMsPerTokenP50: m.decodeMsPerTokenP50,
    decodeMsPerTokenP95: m.decodeMsPerTokenP95,
    decodeMsPerTokenP99: m.decodeMsPerTokenP99,
  };
}

function compareBenchResults(current, baseline) {
  const cm = normalizeBenchMetrics(current);
  const bm = normalizeBenchMetrics(baseline);
  if (!cm || !bm) {
    console.error('[compare] missing metrics in current or baseline result');
    return { regressions: [], improvements: [] };
  }

  const isCrossEngine = (current?.env?.library) !== (baseline?.env?.library);
  const regressions = [];
  const improvements = [];

  const metrics = [
    { label: 'decode tok/s', cur: cm.decodeTokensPerSec, base: bm.decodeTokensPerSec, higherBetter: true },
    { label: 'prefill tok/s', cur: cm.prefillTokensPerSec, base: bm.prefillTokensPerSec, higherBetter: true },
    { label: 'first token', cur: cm.firstTokenMs, base: bm.firstTokenMs, higherBetter: false },
    { label: 'prefill ms', cur: cm.prefillMs, base: bm.prefillMs, higherBetter: false },
    { label: 'decode ms', cur: cm.decodeMs, base: bm.decodeMs, higherBetter: false },
    { label: 'first response', cur: cm.firstResponseMs, base: bm.firstResponseMs, higherBetter: false },
    { label: 'total run', cur: cm.totalRunMs, base: bm.totalRunMs, higherBetter: false },
    { label: 'model load', cur: cm.modelLoadMs, base: bm.modelLoadMs, higherBetter: false },
  ];

  // GPU phase metrics only available in Doppler-vs-Doppler comparisons
  const cg = current?.metrics?.gpu;
  const bg = baseline?.metrics?.gpu;
  if (cg && bg) {
    metrics.push(
      { label: 'gpu record ms', cur: cg.decodeRecordMs?.median, base: bg.decodeRecordMs?.median, higherBetter: false },
      { label: 'gpu submit_wait', cur: cg.decodeSubmitWaitMs?.median, base: bg.decodeSubmitWaitMs?.median, higherBetter: false },
      { label: 'gpu readback', cur: cg.decodeReadbackWaitMs?.median, base: bg.decodeReadbackWaitMs?.median, higherBetter: false },
    );
  }

  const curLabel = isCrossEngine ? (current?.env?.library || 'current') : 'current';
  const baseLabel = isCrossEngine ? (baseline?.env?.library || 'baseline') : 'baseline';
  const baseModelId = baseline.modelId || 'unknown';
  console.log(`[compare] vs ${baseLabel} model=${baseModelId}`);
  console.log(`[compare] ${'metric'.padEnd(20)} ${baseLabel.padStart(14)} ${curLabel.padStart(14)} ${'delta'.padStart(10)}`);

  for (const m of metrics) {
    if (!Number.isFinite(m.cur) || !Number.isFinite(m.base) || m.base === 0) continue;
    const deltaPct = ((m.cur - m.base) / Math.abs(m.base)) * 100;
    const sign = deltaPct >= 0 ? '+' : '';
    const deltaStr = `${sign}${deltaPct.toFixed(1)}%`;
    const isRegression = m.higherBetter ? deltaPct < -10 : deltaPct > 10;
    const isImprovement = m.higherBetter ? deltaPct > 10 : deltaPct < -10;
    const flag = isRegression ? ' !!REGRESSION' : isImprovement ? ' *improved' : '';
    console.log(`[compare] ${m.label.padEnd(20)} ${formatNumber(m.base, 1).padStart(14)} ${formatNumber(m.cur, 1).padStart(14)} ${deltaStr.padStart(10)}${flag}`);
    if (isRegression) regressions.push(m.label);
    if (isImprovement) improvements.push(m.label);
  }

  if (regressions.length) {
    console.log(`[compare] ${regressions.length} regression(s) detected (>10% threshold)`);
  }
  return { regressions, improvements };
}

async function loadManifest(manifestPath) {
  const raw = await fs.readFile(path.resolve(manifestPath), 'utf-8');
  const manifest = JSON.parse(raw);
  if (!manifest.runs || !Array.isArray(manifest.runs) || manifest.runs.length === 0) {
    throw new Error('manifest must have a non-empty "runs" array');
  }
  return manifest;
}

function mergeRunConfig(base, ...overrides) {
  const merged = isPlainObject(base) ? { ...base } : {};
  for (const source of overrides) {
    if (!isPlainObject(source)) {
      continue;
    }
    for (const [key, value] of Object.entries(source)) {
      if ((key === 'browser' || key === 'bench') && isPlainObject(value)) {
        merged[key] = {
          ...(isPlainObject(merged[key]) ? merged[key] : {}),
          ...value,
        };
        continue;
      }
      merged[key] = value;
    }
  }
  return merged;
}

async function runManifestSweep(manifest, commandContext, jsonOutput, policy = DEFAULT_CLI_POLICY) {
  const defaults = manifest.defaults || {};
  const results = [];

  for (let i = 0; i < manifest.runs.length; i++) {
    const run = manifest.runs[i];
    const label = run.label || run.modelId || run.request?.modelId || `run-${i}`;
    if (!jsonOutput) {
      console.error(`[sweep] (${i + 1}/${manifest.runs.length}) ${label}`);
    }

    const requestInput = {
      ...commandContext.request,
      ...(isPlainObject(defaults.request) ? defaults.request : {}),
      ...(isPlainObject(run.request) ? run.request : {}),
      command: commandContext.request.command,
    };
    const modelId = asStringOrNull(run.modelId) || asStringOrNull(defaults.modelId);
    if (modelId) {
      requestInput.modelId = modelId;
    }
    const modelUrl = asStringOrNull(run.modelUrl) || asStringOrNull(defaults.modelUrl);
    if (modelUrl) {
      requestInput.modelUrl = modelUrl;
    }
    const runtimeProfile = asStringOrNull(run.runtimeProfile)
      || asStringOrNull(defaults.runtimeProfile);
    if (runtimeProfile) {
      requestInput.runtimeProfile = runtimeProfile;
    }

    const mergedRunConfig = mergeRunConfig(commandContext.runConfig, defaults.run, run.run);
    let request = null;
    let surface = commandContext.surface;
    try {
      request = normalizeToolingCommandRequest(requestInput);
      surface = commandContext.surfaceFromCli
        ? commandContext.surface
        : resolveSurfaceForCommand(
          request.command,
          { flags: { surface: null } },
          mergedRunConfig,
          policy
        );
      const response = surface === 'auto'
        ? await runWithAutoSurface(request, mergedRunConfig, jsonOutput, policy)
        : await runCommandOnSurface(request, surface, mergedRunConfig, jsonOutput);
      results.push({ label, response, error: null });
    } catch (error) {
      results.push({
        label,
        response: null,
        error: createCliToolingErrorEnvelope(error, {
          surface: surface === 'auto' ? null : surface,
          request,
        }),
      });
      if (!jsonOutput) {
        console.error(`[sweep] ${label} FAILED: ${error.message}`);
      }
    }
  }

  return results;
}

function printManifestSummary(results) {
  const completed = results.filter((r) => r.response && !r.error);
  const failed = results.filter((r) => r.error);
  console.log(`[sweep] ${completed.length} completed, ${failed.length} failed`);

  for (const r of results) {
    if (r.error) {
      console.log(`  ${r.label.padEnd(30)} FAILED`);
      continue;
    }
    const m = r.response?.result?.metrics;
    if (!m) {
      console.log(`  ${r.label.padEnd(30)} no metrics`);
      continue;
    }
    console.log(
      `  ${r.label.padEnd(30)} ` +
      `${formatNumber(m.decodeTokensPerSec)} decode tok/s  ` +
      `prefill=${formatNumber(m.prefillTokensPerSec)}  ` +
      `first=${formatMs(m.firstTokenMs)}`
    );
  }
}

function formatMB(bytes) {
  return Number.isFinite(bytes) ? `${(bytes / (1024 * 1024)).toFixed(1)}MB` : 'n/a';
}

function printDeviceInfo(result) {
  const info = result?.deviceInfo;
  if (!info) return;
  const ai = info.adapterInfo;
  if (ai) {
    console.log(`[device] vendor=${ai.vendor || 'unknown'} arch=${ai.architecture || 'unknown'} device=${ai.device || 'unknown'}`);
  }
  console.log(
    `[device] f16=${info.hasF16 ? 'yes' : 'no'} subgroups=${info.hasSubgroups ? 'yes' : 'no'} timestamp_query=${info.hasTimestampQuery ? 'yes' : 'no'}`
  );
}

function printGpuPhases(metrics) {
  const gpu = metrics?.gpu;
  if (!gpu) return;
  const rm = gpu.decodeRecordMs?.median;
  const sw = gpu.decodeSubmitWaitMs?.median;
  const rw = gpu.decodeReadbackWaitMs?.median;
  if (Number.isFinite(rm) || Number.isFinite(sw) || Number.isFinite(rw)) {
    console.log(`[gpu] decode record=${formatMs(rm)} submit_wait=${formatMs(sw)} readback_wait=${formatMs(rw)} (median)`);
  }
  const pm = gpu.prefillMs?.median;
  const dm = gpu.decodeMs?.median;
  if (Number.isFinite(pm) || Number.isFinite(dm)) {
    console.log(`[gpu] prefill=${formatMs(pm)} decode=${formatMs(dm)} (median gpu time)`);
  }
}

function printMemoryReport(result) {
  const mem = result?.memoryStats;
  if (!mem) return;
  const parts = [`used=${formatMB(mem.used)}`];
  if (mem.pool && Number.isFinite(mem.pool.currentBytesAllocated)) {
    parts.push(`pool=${formatMB(mem.pool.currentBytesAllocated)}`);
  }
  if (mem.kvCache) {
    parts.push(`kv_cache=${formatMB(mem.kvCache.allocated)}`);
    if (Number.isFinite(mem.kvCache.seqLen) && Number.isFinite(mem.kvCache.maxSeqLen)) {
      parts.push(`(seq=${mem.kvCache.seqLen}/${mem.kvCache.maxSeqLen})`);
    }
  }
  console.log(`[memory] ${parts.join(' ')}`);
}

function printExecutionContractSummary(result) {
  const artifact = result?.metrics?.executionContractArtifact;
  if (!artifact || typeof artifact !== 'object') return;
  const checks = Array.isArray(artifact.checks) ? artifact.checks : [];
  const passedChecks = checks.filter((entry) => entry?.ok === true).length;
  const session = artifact.session && typeof artifact.session === 'object'
    ? artifact.session
    : null;
  const attentionPhases = artifact.steps?.attentionPhases && typeof artifact.steps.attentionPhases === 'object'
    ? artifact.steps.attentionPhases
    : null;
  const parts = [
    `status=${artifact.ok === true ? 'pass' : 'fail'}`,
    checks.length > 0 ? `checks=${passedChecks}/${checks.length}` : 'checks=n/a',
  ];
  if (session?.layout) {
    parts.push(`layout=${session.layout}`);
  }
  if (attentionPhases) {
    parts.push(
      `attn(prefill=${attentionPhases.prefill ?? 'n/a'},decode=${attentionPhases.decode ?? 'n/a'},both=${attentionPhases.both ?? 'n/a'})`
    );
  }
  console.log(`[contract] ${parts.join(' ')}`);
  if (artifact.ok !== true && Array.isArray(artifact.errors)) {
    for (const error of artifact.errors.slice(0, 3)) {
      console.log(`[contract] error=${quoteOneLine(error)}`);
    }
  }
}

function printSimpleArtifactSummary(label, artifact) {
  if (!artifact || typeof artifact !== 'object') return;
  const checks = Array.isArray(artifact.checks) ? artifact.checks : [];
  const passedChecks = checks.filter((entry) => entry?.ok === true).length;
  console.log(
    `[${label}] status=${artifact.ok === true ? 'pass' : 'fail'} ` +
    `checks=${checks.length > 0 ? `${passedChecks}/${checks.length}` : 'n/a'}`
  );
  if (artifact.ok !== true && Array.isArray(artifact.errors)) {
    for (const error of artifact.errors.slice(0, 2)) {
      console.log(`[${label}] error=${quoteOneLine(error)}`);
    }
  }
}

function printConvertContractSummary(result) {
  const artifact = result?.executionContractArtifact;
  if (!artifact || typeof artifact !== 'object') return;
  const checks = Array.isArray(artifact.checks) ? artifact.checks : [];
  const passedChecks = checks.filter((entry) => entry?.ok === true).length;
  const session = artifact.session && typeof artifact.session === 'object'
    ? artifact.session
    : null;
  console.log(
    `[contract] status=${artifact.ok === true ? 'pass' : 'fail'} ` +
    `checks=${checks.length > 0 ? `${passedChecks}/${checks.length}` : 'n/a'} ` +
    `layout=${session?.layout ?? 'n/a'}`
  );
  if (artifact.ok !== true && Array.isArray(artifact.errors)) {
    for (const error of artifact.errors.slice(0, 3)) {
      console.log(`[contract] error=${quoteOneLine(error)}`);
    }
  }
  printSimpleArtifactSummary('layer-pattern', result?.layerPatternContractArtifact);
  printSimpleArtifactSummary('required-inference', result?.requiredInferenceFieldsArtifact);
}

function printConvertReportSummary(result) {
  const reportInfo = result?.reportInfo;
  if (!reportInfo || typeof reportInfo !== 'object') return;
  if (typeof reportInfo.path !== 'string' || reportInfo.path.length === 0) return;
  console.log(`[report] ${reportInfo.path}`);
}

function printMetricsSummary(result) {
  if (!result || typeof result !== 'object') return;
  if (result.kind === 'distill') {
    const stageCount = Array.isArray(result.stageResults) ? result.stageResults.length : 0;
    console.log(
      `[metrics] kind=distill action=${result.action || 'run'} stages=${stageCount} runRoot=${quoteOneLine(result.runRoot)}`
    );
    return;
  }
  if (result.kind === 'lora') {
    const exportCount = Array.isArray(result.exports) ? result.exports.length : 0;
    console.log(
      `[metrics] kind=lora action=${result.action || 'run'} exports=${exportCount} runRoot=${quoteOneLine(result.runRoot)}`
    );
    return;
  }
  const suite = String(result.suite || '');
  const metrics = result.metrics;
  if (!metrics || typeof metrics !== 'object') return;

  if (suite === 'inference' || suite === 'debug') {
    const prompt = quoteOneLine(metrics.prompt);
    console.log(`[metrics] prompt=${prompt}`);
    console.log(
      `[metrics] load=${formatMs(metrics.modelLoadMs)} ` +
      `prefillTokens=${Number.isFinite(metrics.prefillTokens) ? Math.round(metrics.prefillTokens) : 'n/a'} ` +
      `decodeTokens=${Number.isFinite(metrics.decodeTokens) ? Math.round(metrics.decodeTokens) : 'n/a'} ` +
      `maxTokens=${Number.isFinite(metrics.maxTokens) ? Math.round(metrics.maxTokens) : 'n/a'}`
    );
    console.log(
      `[metrics] first=${formatMs(metrics.firstTokenMs)} prefill=${formatMs(metrics.prefillMs)} ` +
      `decode=${formatMs(metrics.decodeMs)} total=${formatMs(metrics.totalRunMs)}`
    );
    console.log(
      `[metrics] tok/s=${formatNumber(metrics.decodeTokensPerSec)} ` +
      `prefill=${formatNumber(metrics.prefillTokensPerSec)} ` +
      `decode=${formatNumber(metrics.decodeTokensPerSec)}`
    );
    if (typeof result.output === 'string' && result.output.length > 0) {
      console.log(`[output] ${quoteOneLine(result.output)}`);
    }
    printExecutionContractSummary(result);
    return;
  }

  if (suite === 'bench') {
    if (Number.isFinite(metrics.embeddingDim) || Number.isFinite(metrics.avgEmbeddingMs)) {
      console.log(`[metrics] prompt=${quoteOneLine(metrics.prompt)}`);
      console.log(
        `[metrics] load=${formatMs(metrics.modelLoadMs)} runs=${Number.isFinite(metrics.warmupRuns) ? metrics.warmupRuns : 'n/a'}+${Number.isFinite(metrics.timedRuns) ? metrics.timedRuns : 'n/a'}`
      );
      console.log(
        `[metrics] embedding dim=${Number.isFinite(metrics.embeddingDim) ? Math.round(metrics.embeddingDim) : 'n/a'} ` +
        `median=${formatMs(metrics.medianEmbeddingMs)} avg=${formatMs(metrics.avgEmbeddingMs)} ` +
        `eps=${formatNumber(metrics.avgEmbeddingsPerSec)}`
      );
      printExecutionContractSummary(result);
        return;
    }

    console.log(`[metrics] prompt=${quoteOneLine(metrics.prompt)}`);
    console.log(
      `[metrics] load=${formatMs(metrics.modelLoadMs)} runs=${Number.isFinite(metrics.warmupRuns) ? metrics.warmupRuns : 'n/a'}+${Number.isFinite(metrics.timedRuns) ? metrics.timedRuns : 'n/a'} ` +
      `maxTokens=${Number.isFinite(metrics.maxTokens) ? Math.round(metrics.maxTokens) : 'n/a'}`
    );
    console.log(
      `[metrics] tokens prefill(avg)=${Number.isFinite(metrics.avgPrefillTokens) ? Math.round(metrics.avgPrefillTokens) : 'n/a'} ` +
      `decode(avg)=${Number.isFinite(metrics.avgDecodeTokens) ? Math.round(metrics.avgDecodeTokens) : 'n/a'} ` +
      `generated(avg)=${Number.isFinite(metrics.avgTokensGenerated) ? Math.round(metrics.avgTokensGenerated) : 'n/a'}`
    );
    console.log(
      `[metrics] decode tok/s=${formatNumber(metrics.decodeTokensPerSec)} avg=${formatNumber(metrics.avgDecodeTokensPerSec)} ` +
      `prefill=${formatNumber(metrics.prefillTokensPerSec)} avg=${formatNumber(metrics.avgPrefillTokensPerSec)}`
    );
    console.log(
      `[metrics] latency first=${formatMs(metrics.firstTokenMs)} ` +
      `prefill=${formatMs(metrics.prefillMs)} decode=${formatMs(metrics.decodeMs)}`
    );
    printExecutionContractSummary(result);
    printDeviceInfo(result);
    printGpuPhases(metrics);
    printMemoryReport(result);
    const samplePrompt = quoteOneLineOrStructured(metrics.promptInput);
    if (samplePrompt !== null) {
      console.log(`[sample] prompt=${samplePrompt}`);
    }
    if (typeof metrics.generatedText === 'string' && metrics.generatedText.length > 0) {
      console.log(`[sample] text=${quoteOneLine(metrics.generatedText)}`);
    }
    return;
  }

  if (suite === 'training') {
    const selectedTests = Array.isArray(metrics.selectedTests)
      ? metrics.selectedTests.length
      : 'n/a';
    const availableTests = Array.isArray(metrics.availableTests)
      ? metrics.availableTests.length
      : 'n/a';
    const stage = typeof metrics.trainingStage === 'string' ? metrics.trainingStage : 'n/a';
    console.log(
      `[metrics] training tests=${Number.isFinite(metrics.testsRun) ? metrics.testsRun : 'n/a'} ` +
      `selected=${selectedTests} available=${availableTests} stage=${stage}`
    );
  }
}

async function main() {
  const argv = process.argv.slice(2);
  const jsonOutputRequested = !argv.includes('--pretty');
  const errorContext = {
    surface: null,
    request: null,
  };

  try {
    if (!argv.length || argv[0] === '--help' || argv[0] === '-h') {
      console.log(usage());
      return;
    }

    const cliPolicy = await readJsonObjectFile(CLI_POLICY_PATH, '--cli-policy');
    const parsed = parseArgs(argv);
    if (parsed.flags.help === true || parsed.flags.h === true) {
      console.log(usage());
      return;
    }
    validateCommandFlags(parsed);

    const jsonOutput = parsed.flags.pretty !== true;
    const commandContext = await buildRequest(parsed, cliPolicy);
    const { request, runConfig, surface, surfaceFromCli, benchRunOptions } = commandContext;
    errorContext.surface = surface === 'auto' ? null : surface;
    errorContext.request = request;
    const { saveDir, shouldSave, comparePath, manifestPath } = benchRunOptions;

    if (manifestPath) {
      const manifest = await loadManifest(String(manifestPath));
      const results = await runManifestSweep(
        manifest,
        {
          request,
          runConfig,
          surface,
          surfaceFromCli,
        },
        jsonOutput,
        cliPolicy
      );

      if (shouldSave) {
        for (const r of results) {
          if (r.response?.result) {
            const savedPath = await saveBenchResult(r.response.result, saveDir);
            if (!jsonOutput) console.error(`[save] ${r.label}: ${savedPath}`);
          }
        }
      }

      if (jsonOutput) {
        console.log(JSON.stringify(results.map((r) => r.response ?? r.error), null, 2));
        return;
      }

      printManifestSummary(results);
      for (const r of results) {
        if (r.response?.result) {
          console.log(`\n--- ${r.label} ---`);
          printMetricsSummary(r.response.result);
        }
      }
      return;
    }

    let response;
    if (surface === 'auto') {
      response = await runWithAutoSurface(request, runConfig, jsonOutput, cliPolicy);
    } else {
      response = await runCommandOnSurface(request, surface, runConfig, jsonOutput);
    }

    const isBench = response.result?.suite === 'bench';

    if (comparePath && isBench) {
      const baseline = await loadBaseline(String(comparePath), saveDir);
      if (baseline) {
        compareBenchResults(response.result, baseline);
      }
    }

    if (shouldSave && isBench) {
      const savedPath = await saveBenchResult(response.result, saveDir);
      if (!jsonOutput) {
        console.error(`[save] ${savedPath}`);
      }
    }

    if (jsonOutput) {
      const output = response?.result?.report !== undefined
        ? { ...response, result: { ...response.result, report: undefined } }
        : response;
      console.log(JSON.stringify(output, null, 2));
      return;
    }

    console.log(`[ok] ${toSummary(response.result)}`);
    printConvertContractSummary(response.result);
    printConvertReportSummary(response.result);
    printMetricsSummary(response.result);
  } catch (error) {
    if (jsonOutputRequested) {
      console.log(JSON.stringify(createCliToolingErrorEnvelope(error, errorContext), null, 2));
      process.exitCode = 1;
      return;
    }
    throw error;
  }
}

function isMainModule(metaUrl) {
  const entryPath = process.argv[1];
  if (!entryPath) {
    return false;
  }
  return path.resolve(fileURLToPath(metaUrl)) === path.resolve(entryPath);
}

if (isMainModule(import.meta.url)) {
  main().then(() => {
    process.exit(process.exitCode ?? 0);
  }).catch((error) => {
    console.error(`[error] ${error?.message || String(error)}`);
    process.exit(1);
  });
}
