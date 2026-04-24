#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { runNodeCommand } from '../tooling/node-command-runner.js';
import { runBrowserCommandInNode } from '../tooling/node-browser-command-runner.js';
import { writeProgramBundle } from '../tooling/program-bundle.js';
import {
  TOOLING_COMMANDS,
  normalizeToolingCommandRequest,
} from '../tooling/command-api.js';
import { createToolingErrorEnvelope } from '../tooling/command-envelope.js';
import {
  asStringOrNull,
  resolveBrowserModelUrl,
  resolveNodeModelUrl,
  resolveStaticRootDir,
  resolveRdrrRoot,
} from './cli-model-resolution.js';
import { isPlainObject } from '../utils/plain-object.js';
import {
  toSummary,
  formatNumber,
  formatMs,
  saveBenchResult,
  loadBaseline,
  compareBenchResults,
  printManifestSummary,
  printDeviceInfo,
  printConvertContractSummary,
  printConvertReportSummary,
  printMetricsSummary,
} from './cli-output.js';

export { resolveBrowserModelUrl, resolveNodeModelUrl } from './cli-model-resolution.js';

const NODE_WEBGPU_INCOMPLETE_MESSAGE = 'node command: WebGPU runtime is incomplete in Node';
const CLI_POLICY_PATH = fileURLToPath(new URL('./config/doppler-cli-policy.json', import.meta.url));
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

function usage() {
  return [
    'Usage:',
    '  doppler convert --config <path|url|json> [--surface auto|node]',
    '  doppler refresh-integrity --config <path|url|json> [--surface auto|node]',
    '  doppler debug --config <path|url|json> [--runtime-config <path|url|json>] [--surface auto|node|browser]',
    '  doppler bench --config <path|url|json> [--runtime-config <path|url|json>] [--surface auto|node|browser]',
    '  doppler verify --config <path|url|json> [--runtime-config <path|url|json>] [--surface auto|node|browser]',
    '  doppler lora --config <path|url|json> [--surface auto|node]',
    '  doppler distill --config <path|url|json> [--surface auto|node]',
    '  doppler program-bundle --config <path|json>',
    '  doppler program-bundle --manifest <path> --reference-report <path> --out <path> [--conversion-config <path>]',
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
    '  doppler refresh-integrity --config \'{"request":{"modelDir":"models/local/gemma-3-270m-it-q4k-ehf16-af32"}}\'',
    '  doppler verify --config \'{"request":{"workload":"inference","workloadType":"program-bundle","programBundlePath":"examples/program-bundles/gemma-3-270m-it-q4k-ehf16-af32.program-bundle.json"}}\'',
    '  doppler program-bundle --config \'{"manifestPath":"models/local/gemma-3-270m-it-q4k-ehf16-af32/manifest.json","referenceReportPath":"tests/fixtures/reports/gemma-3-270m-it-q4k-ehf16-af32/2026-03-18T13-33-38.973Z.json","outputPath":"examples/program-bundles/gemma-3-270m-it-q4k-ehf16-af32.program-bundle.json"}\'',
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

function validateProgramBundleFlags(parsed) {
  const allowedFlags = new Set([
    'config',
    'manifest',
    'model-dir',
    'reference-report',
    'conversion-config',
    'runtime-config',
    'out',
    'bundle-id',
    'created-at',
    'pretty',
    'json',
    'help',
    'h',
  ]);
  for (const key of Object.keys(parsed.flags || {})) {
    if (allowedFlags.has(key)) continue;
    throw new Error(`Unknown flag --${key} for "program-bundle".`);
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

async function buildProgramBundleOptions(parsed) {
  const repoRoot = path.resolve(fileURLToPath(new URL('../..', import.meta.url)));
  const configValue = asStringOrNull(parsed.flags.config);
  const config = configValue
    ? await readJsonObjectInput(configValue, '--config')
    : {};
  return {
    repoRoot,
    ...config,
    manifestPath: parsed.flags.manifest ?? config.manifestPath ?? null,
    modelDir: parsed.flags['model-dir'] ?? config.modelDir ?? null,
    referenceReportPath: parsed.flags['reference-report'] ?? config.referenceReportPath ?? null,
    conversionConfigPath: parsed.flags['conversion-config'] ?? config.conversionConfigPath ?? null,
    runtimeConfigPath: parsed.flags['runtime-config'] ?? config.runtimeConfigPath ?? null,
    outputPath: parsed.flags.out ?? config.outputPath ?? config.out ?? null,
    bundleId: parsed.flags['bundle-id'] ?? config.bundleId ?? null,
    createdAtUtc: parsed.flags['created-at'] ?? config.createdAtUtc ?? null,
  };
}

async function runProgramBundleCommand(parsed, jsonOutput) {
  const result = await writeProgramBundle(await buildProgramBundleOptions(parsed));
  const summary = {
    ok: true,
    outputPath: path.relative(process.cwd(), result.outputPath),
    modelId: result.bundle.modelId,
    bundleId: result.bundle.bundleId,
    executionGraphHash: result.bundle.sources.executionGraph.hash,
    artifactCount: result.bundle.artifacts.length,
    wgslModuleCount: result.bundle.wgslModules.length,
  };
  if (jsonOutput) {
    console.log(JSON.stringify(summary, null, 2));
    return;
  }
  console.log(
    `[ok] wrote ${summary.outputPath} ` +
    `(modelId=${summary.modelId}, artifacts=${summary.artifactCount}, wgsl=${summary.wgslModuleCount})`
  );
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
  if ((command === 'convert' || command === 'refresh-integrity') && normalized === 'browser') {
    throw new Error(`${command} is not supported on browser relay. Use --surface node or --surface auto.`);
  }
  if ((command === 'lora' || command === 'distill') && normalized === 'browser') {
    throw new Error(`${command} is not supported on browser relay. Use --surface node or --surface auto.`);
  }
  return normalized;
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
  if (request.command === 'convert' || request.command === 'refresh-integrity') {
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
    if (parsed.command === 'program-bundle') {
      validateProgramBundleFlags(parsed);
      await runProgramBundleCommand(parsed, parsed.flags.pretty !== true);
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
