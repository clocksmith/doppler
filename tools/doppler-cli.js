#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import { runNodeCommand } from '../src/tooling/node-command-runner.js';
import { runBrowserCommandInNode } from '../src/tooling/node-browser-command-runner.js';
import { TOOLING_COMMANDS } from '../src/tooling/command-api.js';

const NODE_WEBGPU_INCOMPLETE_MESSAGE = 'node command: WebGPU runtime is incomplete in Node';
const DEFAULT_BENCH_MODEL_ID = 'gemma-3-270m-it-wf16';
const DEFAULT_BENCH_SURFACE = 'browser';

function usage() {
  return [
    'Usage:',
    '  doppler convert <inputPath> --config <path.json> [--output-dir <path>] [--surface auto|node]',
    '  doppler debug --model-id <id> [--model-url <url>] [--runtime-preset <id>] [--runtime-config-url <url>] [--runtime-config-json <json>] [--surface auto|node|browser]',
    '  doppler bench [--model-id <id>] [--model-url <url>] [--runtime-preset <id>] [--runtime-config-url <url>] [--runtime-config-json <json>] [--surface auto|node|browser]',
    '  doppler test-model --suite <kernels|inference|diffusion|energy> [--model-id <id>] [--model-url <url>] [--runtime-preset <id>] [--runtime-config-url <url>] [--runtime-config-json <json>] [--surface auto|node|browser]',
    '',
    'Flags:',
    '  --surface <auto|node|browser>   Execution surface (default: auto)',
    '  --json                          Print machine-readable result JSON',
    '  --capture-output                Include captured output for supported suites',
    '  --keep-pipeline                 Keep loaded pipeline in result payload (node surface only)',
    '  --browser-channel <name>        Browser channel for Playwright launch (e.g. chrome)',
    '  --browser-executable <path>     Browser executable path for Playwright launch',
    '  --headed                        Run browser relay in headed mode',
    '  --headless <true|false>         Browser relay headless mode',
    '  --browser-headless <true|false>  Deprecated alias for --headless',
    '  --browser-port <port>           Static server port for browser relay (default: random)',
    '  --browser-timeout-ms <ms>       Browser command timeout (default: 180000)',
    '  --browser-arg <arg>            Extra launch arg (repeatable); WebGPU/Vulkan args are applied automatically',
    '  --browser-url-path <path>       Runner page path (default: /src/tooling/command-runner.html)',
    '  --browser-static-root <path>    Static server root directory (default: doppler root)',
    '  --browser-base-url <url>        Reuse an existing static server base URL',
    '  --browser-console               Stream browser console lines to stderr',
    '  --no-opfs-cache                 Disable OPFS caching (use HTTP shard loading every run)',
    '  --browser-user-data <path>      Persistent Chromium profile directory for OPFS cache',
    '',
    'Convert Flags:',
    '  --config <path>                 Converter config JSON (required for convert)',
    '  --output-dir <path>             Override output directory for convert',
    '  --converter-config <path>       Deprecated alias for --config (convert only)',
    '',
    'Bench Flags:',
    '  --save                          Save bench result JSON to disk',
    '  --save-dir <path>               Output directory for saved results (default: ./bench-results)',
    '  --compare <path|last>           Compare against a previous result file or "last"',
    '  --manifest <path>               Run a multi-model bench sweep from a manifest JSON',
    '  --cache-mode <cold|warm>        cold: wipe OPFS cache before run; warm: reuse (default: warm)',
    '',
    `Bench Defaults: --model-id ${DEFAULT_BENCH_MODEL_ID}, --surface ${DEFAULT_BENCH_SURFACE}, --browser-console (browser channel auto-selected)`,
  ].join('\n');
}

function applyCommandDefaults(parsed) {
  if (!parsed || parsed.command !== 'bench') {
    return parsed;
  }

  const flags = { ...parsed.flags };
  if (!flags['model-id'] && !flags['model-url']) {
    flags['model-id'] = DEFAULT_BENCH_MODEL_ID;
  }
  if (!flags.surface) {
    flags.surface = DEFAULT_BENCH_SURFACE;
  }
  flags['browser-console'] = true;

  return {
    ...parsed,
    flags,
  };
}

function parseArgs(argv) {
  const out = {
    command: null,
    positional: [],
    flags: {},
  };

  if (!argv.length) return out;
  out.command = argv[0] ?? null;

  for (let i = 1; i < argv.length; i += 1) {
    const token = argv[i];
    if (!token.startsWith('--')) {
      out.positional.push(token);
      continue;
    }

    const key = token.slice(2);
    if (
      key === 'json'
      || key === 'capture-output'
      || key === 'keep-pipeline'
      || key === 'headed'
      || key === 'help'
      || key === 'h'
      || key === 'browser-console'
      || key === 'no-opfs-cache'
      || key === 'save'
    ) {
      out.flags[key] = true;
      continue;
    }

    const value = argv[i + 1];
    if (value === undefined) {
      throw new Error(`Missing value for --${key}`);
    }

    if (key !== 'browser-arg' && value.startsWith('--')) {
      throw new Error(`Missing value for --${key}`);
    }

    if (key === 'browser-arg') {
      const previous = out.flags[key];
      if (Array.isArray(previous)) {
        previous.push(value);
      } else {
        out.flags[key] = [value];
      }
      i += 1;
      continue;
    }

    out.flags[key] = value;
    i += 1;
  }

  return out;
}

function parseRuntimeConfigJson(value) {
  if (!value) return null;
  try {
    const parsed = JSON.parse(value);
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      throw new Error('runtime config must be an object');
    }
    return parsed;
  } catch (error) {
    throw new Error(`Invalid --runtime-config-json: ${error.message}`);
  }
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

function resolveConvertConfigFlag(parsed) {
  const config = parsed.flags.config ?? null;
  const legacy = parsed.flags['converter-config'] ?? null;
  if (config && legacy) {
    throw new Error('convert accepts one config flag. Use --config only.');
  }
  if (config) {
    return { configPath: String(config), usedLegacyAlias: false };
  }
  if (legacy) {
    return { configPath: String(legacy), usedLegacyAlias: true };
  }
  return { configPath: null, usedLegacyAlias: false };
}

function parseBooleanFlag(value, label) {
  if (value === undefined || value === null || value === '') return null;
  if (typeof value === 'boolean') return value;
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (normalized === 'true') return true;
    if (normalized === 'false') return false;
  }
  throw new Error(`${label} must be true or false`);
}

function parseNumberFlag(value, label) {
  if (value === undefined || value === null || value === '') return null;
  const parsed = Number(value);
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

function resolveStaticRootDir(parsed) {
  const configured = parsed.flags['browser-static-root'];
  if (configured) {
    return path.resolve(String(configured));
  }
  return process.cwd();
}

async function resolveBrowserModelUrl(request, parsed) {
  if (request.modelUrl || !request.modelId) {
    return request;
  }

  const modelId = String(request.modelId);
  const encodedModelId = encodeURIComponent(modelId);

  if (parsed.flags['browser-base-url']) {
    return {
      ...request,
      modelUrl: `/models/${encodedModelId}`,
    };
  }

  const staticRootDir = resolveStaticRootDir(parsed);
  const curatedCandidate = {
    modelUrl: `/models/curated/${encodedModelId}`,
    manifestPath: path.join(staticRootDir, 'models', 'curated', modelId, 'manifest.json'),
  };
  const localCandidate = {
    modelUrl: `/models/local/${encodedModelId}`,
    manifestPath: path.join(staticRootDir, 'models', 'local', modelId, 'manifest.json'),
  };
  const legacyCandidate = {
    modelUrl: `/models/${encodedModelId}`,
    manifestPath: path.join(staticRootDir, 'models', modelId, 'manifest.json'),
  };
  const candidates = [
    curatedCandidate,
    localCandidate,
    legacyCandidate,
  ];
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
        return {
          ...request,
          modelUrl: candidate.modelUrl,
        };
      }
    } catch {
      return {
        ...request,
        modelUrl: candidate.modelUrl,
      };
    }
  }

  if (discoveredManifestCandidates.length > 0) {
    const firstCandidate = discoveredManifestCandidates[0];
    const paths = discoveredManifestCandidates
      .map((candidate) => candidate.modelUrl)
      .join(', ');
    throw new Error(
      `Model "${modelId}" was found, but no shard files (shard_*.bin) are present. ` +
      `Checked: ${paths}. Add shard files beside the manifest, or pass --model-url to a complete model directory.`
    );
  }

  return {
    ...request,
    modelUrl: `/models/${encodedModelId}`,
  };
}

function parseSurface(value, command) {
  const normalized = String(value || 'auto').trim().toLowerCase();
  if (normalized !== 'auto' && normalized !== 'node' && normalized !== 'browser') {
    throw new Error('--surface must be one of auto, node, browser');
  }
  if (command === 'convert' && normalized === 'browser') {
    throw new Error('convert is not supported on browser relay. Use --surface node or --surface auto.');
  }
  return normalized;
}

async function buildRequest(parsed, options = {}) {
  const command = parsed.command;
  if (!command || !TOOLING_COMMANDS.includes(command)) {
    throw new Error(`Unsupported command "${command || ''}"`);
  }
  const jsonOutput = options.jsonOutput === true;

  const common = {
    command,
    modelId: parsed.flags['model-id'] ?? null,
    modelUrl: parsed.flags['model-url'] ?? null,
    runtimePreset: parsed.flags['runtime-preset'] ?? null,
    runtimeConfigUrl: parsed.flags['runtime-config-url'] ?? null,
    runtimeConfig: parseRuntimeConfigJson(parsed.flags['runtime-config-json'] ?? null),
    captureOutput: parsed.flags['capture-output'] === true,
    keepPipeline: parsed.flags['keep-pipeline'] === true,
  };

  if (command === 'convert') {
    if (parsed.flags['model-id']) {
      throw new Error('convert does not accept --model-id. Set output.modelId in --config.');
    }

    const inputDir = parsed.positional[0] ?? null;
    if (!inputDir) {
      throw new Error('convert requires <inputPath>.');
    }
    if (parsed.positional.length > 1) {
      throw new Error('convert accepts only one positional argument: <inputPath>. Use --output-dir for output override.');
    }

    const outputDir = parsed.flags['output-dir'] ?? null;
    const { configPath, usedLegacyAlias } = resolveConvertConfigFlag(parsed);
    if (!configPath) {
      throw new Error('convert requires --config <path.json>.');
    }
    if (usedLegacyAlias && !jsonOutput) {
      console.error('[warn] --converter-config is deprecated; use --config.');
    }

    const converterConfig = await readJsonObjectFile(configPath, '--config');
    return {
      ...common,
      modelId: null,
      inputDir,
      outputDir,
      convertPayload: {
        converterConfig,
      },
    };
  }

  if (command === 'test-model') {
    return {
      ...common,
      suite: parsed.flags.suite ?? null,
    };
  }

  return common;
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

function buildBrowserRunOptions(parsed, jsonOutput) {
  const hasHeadlessFlag = Object.hasOwn(parsed.flags, 'headless');
  const hasBrowserHeadlessFlag = Object.hasOwn(parsed.flags, 'browser-headless');
  const hasHeadedFlag = Object.hasOwn(parsed.flags, 'headed');

  if (
    hasHeadedFlag
    && (hasHeadlessFlag || hasBrowserHeadlessFlag)
  ) {
    throw new Error('--headed is mutually exclusive with --headless / --browser-headless.');
  }

  let headless;
  if (hasHeadedFlag) {
    headless = false;
  } else {
    const rawHeadless = hasHeadlessFlag
      ? parsed.flags.headless
      : parsed.flags['browser-headless'];
    headless = parseBooleanFlag(rawHeadless, '--headless/--browser-headless');
    headless = headless === null ? true : headless;
  }
  const port = parseNumberFlag(parsed.flags['browser-port'], '--browser-port');
  const timeoutMs = parseNumberFlag(parsed.flags['browser-timeout-ms'], '--browser-timeout-ms');

  const options = {
    channel: parsed.flags['browser-channel'] ?? null,
    executablePath: parsed.flags['browser-executable'] ?? null,
    runnerPath: parsed.flags['browser-url-path'] ?? null,
    staticRootDir: parsed.flags['browser-static-root'] ?? null,
    baseUrl: parsed.flags['browser-base-url'] ?? null,
    browserArgs: parseBrowserArgs(parsed.flags['browser-arg']),
  };

  options.headless = headless;
  if (port !== null) {
    options.port = port;
  }
  if (timeoutMs !== null) {
    options.timeoutMs = timeoutMs;
  }

  if (parsed.flags['no-opfs-cache'] === true) {
    options.opfsCache = false;
  }
  if (parsed.flags['browser-user-data']) {
    options.userDataDir = String(parsed.flags['browser-user-data']);
  }
  if (parsed.flags['cache-mode'] === 'cold') {
    options.wipeCacheBeforeLaunch = true;
  }

  if (parsed.flags['browser-console'] === true && !jsonOutput) {
    options.onConsole = ({ type, text }) => {
      console.error(`[browser:${type}] ${text}`);
    };
  }

  return options;
}

function isNodeWebGPUFallbackCandidate(error) {
  const message = error?.message || String(error || '');
  return message.includes(NODE_WEBGPU_INCOMPLETE_MESSAGE);
}

async function runCommandOnSurface(request, surface, parsed, jsonOutput) {
  if (surface === 'node') {
    return runNodeCommand(request, buildNodeRunOptions(jsonOutput));
  }

  const browserRequest = await resolveBrowserModelUrl(request, parsed);

  if (!jsonOutput) {
    console.error('[progress] browser launching WebGPU harness...');
    if (browserRequest.modelUrl && browserRequest.modelUrl !== request.modelUrl) {
      console.error(`[progress] browser resolved modelUrl=${browserRequest.modelUrl}`);
    }
  }

  return runBrowserCommandInNode(browserRequest, buildBrowserRunOptions(parsed, jsonOutput));
}

async function runWithAutoSurface(request, parsed, jsonOutput) {
  if (request.command === 'convert') {
    return runCommandOnSurface(request, 'node', parsed, jsonOutput);
  }

  try {
    return await runCommandOnSurface(request, 'node', parsed, jsonOutput);
  } catch (error) {
    if (!isNodeWebGPUFallbackCandidate(error)) {
      throw error;
    }
    return runCommandOnSurface(request, 'browser', parsed, jsonOutput);
  }
}

function toSummary(result) {
  if (!result || typeof result !== 'object') {
    return 'ok';
  }

  if (result.manifest?.modelId) {
    return `converted ${result.manifest.modelId} (${result.tensorCount} tensors, ${result.shardCount} shards)`;
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

const DEFAULT_SAVE_DIR = './bench-results';

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

/**
 * Detect if a result is from a competitor harness (snake_case metric keys)
 * and normalize to Doppler's camelCase format for comparison.
 */
function normalizeBenchMetrics(result) {
  const m = result?.metrics;
  if (!m) return m;
  // Competitor detection: snake_case keys like decode_tokens_per_sec
  if ('decode_tokens_per_sec' in m) {
    return {
      medianDecodeTokensPerSec: m.decode_tokens_per_sec,
      medianPrefillTokensPerSec: m.prefill_tokens_per_sec,
      medianTtftMs: m.ttft_ms,
      medianTokensPerSec: m.tokens_per_sec,
      modelLoadMs: m.model_load_ms,
      decodeMsPerTokenP50: m.decode_ms_per_token_p50,
      decodeMsPerTokenP95: m.decode_ms_per_token_p95,
      decodeMsPerTokenP99: m.decode_ms_per_token_p99,
    };
  }
  return m;
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
    { label: 'tok/s (median)', cur: cm.medianTokensPerSec, base: bm.medianTokensPerSec, higherBetter: true },
    { label: 'decode tok/s', cur: cm.medianDecodeTokensPerSec, base: bm.medianDecodeTokensPerSec, higherBetter: true },
    { label: 'prefill tok/s', cur: cm.medianPrefillTokensPerSec, base: bm.medianPrefillTokensPerSec, higherBetter: true },
    { label: 'ttft (median)', cur: cm.medianTtftMs, base: bm.medianTtftMs, higherBetter: false },
    { label: 'prefill ms', cur: cm.medianPrefillMs, base: bm.medianPrefillMs, higherBetter: false },
    { label: 'decode ms', cur: cm.medianDecodeMs, base: bm.medianDecodeMs, higherBetter: false },
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

async function runManifestSweep(manifest, parsed, jsonOutput, surface) {
  const defaults = manifest.defaults || {};
  const results = [];

  for (let i = 0; i < manifest.runs.length; i++) {
    const run = manifest.runs[i];
    const label = run.label || run.modelId || `run-${i}`;
    if (!jsonOutput) {
      console.error(`[sweep] (${i + 1}/${manifest.runs.length}) ${label}`);
    }

    const mergedFlags = { ...parsed.flags };
    const modelId = run.modelId || defaults.modelId || parsed.flags['model-id'];
    if (modelId) mergedFlags['model-id'] = modelId;
    if (run.runtimePreset || defaults.runtimePreset) {
      mergedFlags['runtime-preset'] = run.runtimePreset || defaults.runtimePreset;
    }

    const mergedParsed = { ...parsed, flags: mergedFlags };
    const mergedWithDefaults = applyCommandDefaults(mergedParsed);
    const request = await buildRequest(mergedWithDefaults, { jsonOutput });
    if (!jsonOutput) mergedWithDefaults.flags['browser-console'] = true;

    try {
      const response = surface === 'auto'
        ? await runWithAutoSurface(request, mergedWithDefaults, jsonOutput)
        : await runCommandOnSurface(request, surface, mergedWithDefaults, jsonOutput);
      results.push({ label, response, error: null });
    } catch (error) {
      results.push({ label, response: null, error });
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
      `${formatNumber(m.medianTokensPerSec)} tok/s  ` +
      `decode=${formatNumber(m.medianDecodeTokensPerSec)}  ` +
      `prefill=${formatNumber(m.medianPrefillTokensPerSec)}  ` +
      `ttft=${formatMs(m.medianTtftMs)}`
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

function printMetricsSummary(result) {
  if (!result || typeof result !== 'object') return;
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
      `[metrics] ttft=${formatMs(metrics.ttftMs)} prefill=${formatMs(metrics.prefillMs)} ` +
      `decode=${formatMs(metrics.decodeMs)} total=${formatMs(metrics.totalMs)}`
    );
    console.log(
      `[metrics] tok/s total=${formatNumber(metrics.tokensPerSec)} ` +
      `prefill=${formatNumber(metrics.prefillTokensPerSec)} ` +
      `decode=${formatNumber(metrics.decodeTokensPerSec)}`
    );
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
      `[metrics] tok/s median=${formatNumber(metrics.medianTokensPerSec)} avg=${formatNumber(metrics.avgTokensPerSec)} ` +
      `prefill median=${formatNumber(metrics.medianPrefillTokensPerSec)} avg=${formatNumber(metrics.avgPrefillTokensPerSec)} ` +
      `decode median=${formatNumber(metrics.medianDecodeTokensPerSec)} avg=${formatNumber(metrics.avgDecodeTokensPerSec)}`
    );
    console.log(
      `[metrics] latency ttft median=${formatMs(metrics.medianTtftMs)} ` +
      `prefill median=${formatMs(metrics.medianPrefillMs)} decode median=${formatMs(metrics.medianDecodeMs)}`
    );
    printDeviceInfo(result);
    printGpuPhases(metrics);
    printMemoryReport(result);
  }
}

async function main() {
  const argv = process.argv.slice(2);
  if (!argv.length || argv[0] === '--help' || argv[0] === '-h') {
    console.log(usage());
    return;
  }

  const parsed = parseArgs(argv);
  const parsedWithDefaults = applyCommandDefaults(parsed);
  if (parsedWithDefaults.flags.help === true || parsedWithDefaults.flags.h === true) {
    console.log(usage());
    return;
  }

  const jsonOutput = parsedWithDefaults.flags.json === true;
  const surface = parseSurface(parsedWithDefaults.flags.surface, parsedWithDefaults.command);
  const saveDir = String(parsedWithDefaults.flags['save-dir'] || DEFAULT_SAVE_DIR);
  const shouldSave = parsedWithDefaults.flags.save === true;
  const comparePath = parsedWithDefaults.flags.compare ?? null;
  const manifestPath = parsedWithDefaults.flags.manifest ?? null;

  if (manifestPath) {
    const manifest = await loadManifest(String(manifestPath));
    const results = await runManifestSweep(manifest, parsedWithDefaults, jsonOutput, surface);

    if (shouldSave) {
      for (const r of results) {
        if (r.response?.result) {
          const savedPath = await saveBenchResult(r.response.result, saveDir);
          if (!jsonOutput) console.error(`[save] ${r.label}: ${savedPath}`);
        }
      }
    }

    if (jsonOutput) {
      console.log(JSON.stringify(results.map((r) => r.response ?? { error: r.error?.message }), null, 2));
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

  const request = await buildRequest(parsedWithDefaults, { jsonOutput });

  let response;
  if (surface === 'auto') {
    response = await runWithAutoSurface(request, parsedWithDefaults, jsonOutput);
  } else {
    response = await runCommandOnSurface(request, surface, parsedWithDefaults, jsonOutput);
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
    console.log(JSON.stringify(response, null, 2));
    return;
  }

  console.log(`[ok] ${toSummary(response.result)}`);
  printMetricsSummary(response.result);
}

main().catch((error) => {
  console.error(`[error] ${error?.message || String(error)}`);
  process.exit(1);
});
