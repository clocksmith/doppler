#!/usr/bin/env node

// Transformers.js WebGPU benchmark runner.
//
// Usage:
//   node benchmarks/runners/transformersjs-bench.js [options]
//
// Options:
//   --model <id>         HuggingFace model ID (default: onnx-community/gemma-3-1b-it-ONNX-GQA)
//   --prompt <text>      Input prompt
//   --max-tokens <n>     Max new tokens to generate (default: 128)
//   --temperature <n>    Sampling temperature (default: 0)
//   --top-k <n>          Sampling top-k (default: 32)
//   --top-p <n>          Sampling top-p (default: 1)
//   --warmup <n>         Warmup runs (default: 1)
//   --runs <n>           Timed runs (default: 3)
//   --seed <n>           Deterministic seed metadata (default: 0)
//   --workload <id>      Use a predefined workload from workloads.json
//   --cache-mode <mode>  cold|warm (default: warm)
//                        cold: wipe persistent profile before launch
//                        warm: reuse persistent profile
//   --browser-executable <path>  Browser executable path (defaults to Playwright default)
//   --load-mode <mode>   opfs|http|memory (default: http)
//                        warm+opfs performs an untimed cache-prime before timed runs
//   --timeout <ms>       Page timeout in ms (default: 600000)
//   --profile-ops <on|off>  Enable ONNX Runtime op profiling (default: on)
//   --profile-top <n>        Number of top ops in profiling summary (default: 20)
//   --tjs-version <3|4>  Transformers.js version (default: 4)
//   --dtype <fp16|q4|q4f16>  Transformers.js model dtype (default: fp16)
//   --browser-base-url <url>  Reuse an existing static base URL
//   --use-chat-template   Apply model chat template before generation
//   --save               Save result JSON to benchmarks/vendors/results/
//   --save-dir <dir>     Directory for saved results (default: ./benchmarks/vendors/results)
//   --timestamp <iso|ms> Result timestamp override (ISO-8601 or epoch milliseconds)
//   --json               Output only JSON (default: true)
//   --browser-console    Stream browser console to stderr

import fs from 'node:fs/promises';
import fsSync from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import os from 'node:os';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..', '..');
const BENCHMARKS_ROOT = path.join(REPO_ROOT, 'benchmarks');
const BENCHMARK_POLICY_PATH = path.join(BENCHMARKS_ROOT, 'vendors', 'benchmark-policy.json');

const DEFAULT_MODEL = 'onnx-community/gemma-3-1b-it-ONNX-GQA';
const DEFAULT_PROMPT = 'Summarize this input in one sentence.';
const DEFAULT_MAX_TOKENS = 128;
const DEFAULT_TEMPERATURE = 0;
const DEFAULT_TOP_K = 32;
const DEFAULT_TOP_P = 1;
const DEFAULT_WARMUP = 1;
const DEFAULT_RUNS = 3;
const DEFAULT_PROFILE_TOP_N = 20;
const DEFAULT_PROFILE_DIR = path.join(BENCHMARKS_ROOT, '.tjs-bench-profile');
const DEFAULT_SEED = 0;

const DEFAULT_BENCHMARK_POLICY = Object.freeze({
  timeoutsMs: Object.freeze({
    transformersjs: 600_000,
  }),
  browser: Object.freeze({
    webgpuArgs: Object.freeze([
      '--enable-unsafe-webgpu',
      '--enable-webgpu-developer-features',
      '--disable-dawn-features=disallow_unsafe_apis',
      '--ignore-gpu-blocklist',
    ]),
    stableArgs: Object.freeze([
      '--disable-breakpad',
      '--disable-gpu-sandbox',
      '--no-sandbox',
    ]),
    platformArgs: Object.freeze({
      darwin: Object.freeze(['--use-angle=metal']),
      linux: Object.freeze(['--use-angle=vulkan', '--enable-features=Vulkan', '--disable-vulkan-surface']),
      win32: Object.freeze([]),
    }),
  }),
});

function normalizeStringArray(value, label) {
  if (!Array.isArray(value)) {
    throw new Error(`${label} must be an array`);
  }
  return value.map((entry, index) => {
    if (typeof entry !== 'string' || entry.trim() === '') {
      throw new Error(`${label}[${index}] must be a non-empty string`);
    }
    return entry.trim();
  });
}

function normalizePositiveInteger(value, label) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || !Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${label} must be a positive integer`);
  }
  return parsed;
}

function normalizePlatformArgs(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  const normalized = {};
  for (const [platform, args] of Object.entries(value)) {
    normalized[platform] = Object.freeze(normalizeStringArray(args, `${label}.${platform}`));
  }
  return Object.freeze(normalized);
}

function loadBenchmarkPolicy() {
  let raw;
  try {
    raw = fsSync.readFileSync(BENCHMARK_POLICY_PATH, 'utf-8');
  } catch (error) {
    throw new Error(`Failed to read benchmark policy at ${BENCHMARK_POLICY_PATH}: ${error.message}`);
  }
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch (error) {
    throw new Error(`Invalid benchmark policy JSON at ${BENCHMARK_POLICY_PATH}: ${error.message}`);
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error(`benchmark policy at ${BENCHMARK_POLICY_PATH} must be an object`);
  }
  if (!Number.isInteger(parsed.schemaVersion) || parsed.schemaVersion !== 1) {
    throw new Error(`benchmark policy at ${BENCHMARK_POLICY_PATH} schemaVersion must be 1`);
  }

  const timeoutValue = parsed?.timeoutsMs?.transformersjs ?? DEFAULT_BENCHMARK_POLICY.timeoutsMs.transformersjs;
  const browserPolicy = parsed?.browser ?? DEFAULT_BENCHMARK_POLICY.browser;
  return Object.freeze({
    source: BENCHMARK_POLICY_PATH,
    schemaVersion: parsed.schemaVersion,
    updated: parsed.updated || null,
    timeoutMs: normalizePositiveInteger(timeoutValue, 'timeoutsMs.transformersjs'),
    webgpuArgs: Object.freeze(
      normalizeStringArray(browserPolicy.webgpuArgs ?? DEFAULT_BENCHMARK_POLICY.browser.webgpuArgs, 'browser.webgpuArgs')
    ),
    stableArgs: Object.freeze(
      normalizeStringArray(
        browserPolicy.stableArgs
          ?? browserPolicy.crashRecoveryArgs
          ?? DEFAULT_BENCHMARK_POLICY.browser.stableArgs,
        'browser.stableArgs'
      )
    ),
    platformArgs: normalizePlatformArgs(
      browserPolicy.platformArgs ?? DEFAULT_BENCHMARK_POLICY.browser.platformArgs,
      'browser.platformArgs'
    ),
  });
}

const BENCHMARK_POLICY = loadBenchmarkPolicy();
const DEFAULT_TIMEOUT = BENCHMARK_POLICY.timeoutMs;
const WEBGPU_ARGS = BENCHMARK_POLICY.webgpuArgs;
const STABLE_BROWSER_ARGS = BENCHMARK_POLICY.stableArgs;
const PLATFORM_ARGS = BENCHMARK_POLICY.platformArgs;

function uniqueArgs(values) {
  return [...new Set(values)];
}

function asNonEmptyString(value) {
  if (value == null) return null;
  const normalized = String(value).trim();
  return normalized === '' ? null : normalized;
}

function normalizeGpuBackend(value) {
  const raw = asNonEmptyString(value);
  if (!raw) return null;
  const normalized = raw.toLowerCase();
  if (normalized.includes('metal')) return 'metal';
  if (normalized.includes('vulkan')) return 'vulkan';
  if (normalized.includes('d3d12')) return 'd3d12';
  if (normalized.includes('d3d11')) return 'd3d11';
  if (normalized.includes('opengl') || normalized === 'gl') return 'opengl';
  if (normalized.includes('swiftshader')) return 'swiftshader';
  return normalized;
}

function readArgFlagValue(args, flagName) {
  if (!Array.isArray(args)) return null;
  for (let i = 0; i < args.length; i += 1) {
    const token = String(args[i] ?? '');
    if (token === flagName) {
      return asNonEmptyString(args[i + 1]);
    }
    if (token.startsWith(`${flagName}=`)) {
      return asNonEmptyString(token.slice(flagName.length + 1));
    }
  }
  return null;
}

function inferWebgpuBackendFromArgs(args, hostPlatform) {
  const useAngle = normalizeGpuBackend(readArgFlagValue(args, '--use-angle'));
  if (useAngle) return useAngle;
  const normalizedArgs = Array.isArray(args)
    ? args.map((value) => String(value ?? '').toLowerCase())
    : [];
  if (normalizedArgs.some((value) => value.includes('vulkan'))) return 'vulkan';
  if (normalizedArgs.some((value) => value.includes('metal'))) return 'metal';
  if (normalizedArgs.some((value) => value.includes('d3d12'))) return 'd3d12';
  if (normalizedArgs.some((value) => value.includes('d3d11'))) return 'd3d11';
  const platform = asNonEmptyString(hostPlatform);
  if (platform === 'darwin') return 'metal';
  if (platform === 'linux') return 'vulkan';
  if (platform === 'win32') return 'd3d12';
  return null;
}

function hasCrashRecoveryArgs(args = []) {
  const argSet = new Set(args);
  return STABLE_BROWSER_ARGS.every((value) => argSet.has(value));
}

function withCrashRecoveryArgs(args = []) {
  return uniqueArgs([...args, ...STABLE_BROWSER_ARGS]);
}

const PERSISTENT_LAUNCH_ERROR_HINTS = Object.freeze([
  'Target page, context or browser has been closed',
  'bootstrap_check_in',
  'Permission denied',
  'org.chromium.Chromium.MachPortRendezvousServer',
]);

function parseArgs(argv) {
  const flags = {};
  for (let i = 0; i < argv.length; i++) {
    const token = argv[i];
    if (!token.startsWith('--')) continue;
    const key = token.slice(2);
    if (key === 'json' || key === 'browser-console' || key === 'save' || key === 'use-chat-template') {
      flags[key] = true;
      continue;
    }
    if (key === 'browser-arg') {
      if (flags[key] == null) {
        flags[key] = [];
      } else if (!Array.isArray(flags[key])) {
        flags[key] = [flags[key]];
      }
      const value = argv[i + 1];
      if (value === undefined) {
        throw new Error(`Missing value for --${key}`);
      }
      flags[key].push(value);
      i += 1;
      continue;
    }
    const value = argv[i + 1];
    if (value === undefined) {
      throw new Error(`Missing value for --${key}`);
    }
    flags[key] = value;
    i++;
  }
  return flags;
}

function isRecoverablePersistentLaunchError(error) {
  const message = error?.message || String(error || '');
  return PERSISTENT_LAUNCH_ERROR_HINTS.some((hint) => message.includes(hint));
}

function parseBooleanFlag(value, fallback, label) {
  if (value == null || value === '') return fallback;
  const normalized = String(value).trim().toLowerCase();
  if (normalized === 'on' || normalized === 'true' || normalized === '1' || normalized === 'yes') return true;
  if (normalized === 'off' || normalized === 'false' || normalized === '0' || normalized === 'no') return false;
  throw new Error(`${label} must be one of: on, off, true, false, 1, 0`);
}

function parsePositiveInteger(value, fallback, label) {
  if (value == null || value === '') return fallback;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || !Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${label} must be a positive integer`);
  }
  return parsed;
}

function parseNonNegativeInt(value, flag, fallback) {
  if (value == null) return fallback;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || !Number.isInteger(parsed) || parsed < 0) {
    throw new Error(`${flag} must be a non-negative integer`);
  }
  return parsed;
}

function parseNonNegativeNumber(value, flag, fallback) {
  if (value == null) return fallback;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0) {
    throw new Error(`${flag} must be a non-negative number`);
  }
  return parsed;
}

function parseProbability(value, flag, fallback) {
  if (value == null) return fallback;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0 || parsed > 1) {
    throw new Error(`${flag} must be in the range (0, 1]`);
  }
  return parsed;
}

function parseLoadMode(value, flag, fallback = null) {
  if (value == null || value === '') return fallback;
  const normalized = String(value).trim().toLowerCase();
  if (normalized === 'opfs' || normalized === 'http' || normalized === 'memory') {
    return normalized;
  }
  throw new Error(`${flag} must be one of: opfs, http, memory`);
}

function parseTjsDtype(value, flag, fallback = 'fp16') {
  if (value == null || value === '') return fallback;
  const normalized = String(value).trim().toLowerCase();
  if (normalized === 'fp16' || normalized === 'q4' || normalized === 'q4f16') return normalized;
  throw new Error(`${flag} must be one of: fp16, q4, q4f16`);
}

async function loadWorkload(workloadId) {
  const workloadsPath = path.join(BENCHMARKS_ROOT, 'vendors', 'workloads.json');
  const raw = await fs.readFile(workloadsPath, 'utf-8');
  const { workloads } = JSON.parse(raw);
  const wl = workloads.find((w) => w.id === workloadId);
  if (!wl) {
    throw new Error(`Unknown workload "${workloadId}". Available: ${workloads.map((w) => w.id).join(', ')}`);
  }
  return wl;
}

async function wipeDir(dirPath) {
  try {
    await fs.rm(dirPath, { recursive: true, force: true });
  } catch { /* ignore if not exists */ }
}

function parseTimestampValue(rawValue, label) {
  if (rawValue == null || rawValue === '') return null;
  if (typeof rawValue !== 'string') {
    throw new Error(`${label} must be a string`);
  }
  const trimmed = rawValue.trim();
  if (trimmed === '') return null;
  const asMs = /^[-+]?\d+$/.test(trimmed) ? Number(trimmed) : NaN;
  const parsed = Number.isFinite(asMs) ? new Date(asMs) : new Date(trimmed);
  if (Number.isNaN(parsed.getTime())) {
    throw new Error(`${label} must be ISO-8601 or epoch milliseconds`);
  }
  return parsed.toISOString();
}

function compactTimestamp(timestamp = null) {
  const d = timestamp == null ? new Date() : new Date(timestamp);
  const pad = (n, w = 2) => String(n).padStart(w, '0');
  return `${d.getUTCFullYear()}${pad(d.getUTCMonth() + 1)}${pad(d.getUTCDate())}T${pad(d.getUTCHours())}${pad(d.getUTCMinutes())}${pad(d.getUTCSeconds())}`;
}

async function saveResult(result, saveDir, timestamp = null) {
  await fs.mkdir(saveDir, { recursive: true });
  const modelSlug = String(result?.modelId || 'unknown').replace(/[^a-zA-Z0-9_-]/g, '_');
  const ts = compactTimestamp(timestamp);
  const filename = `tjs_${modelSlug}_${ts}.json`;
  const filePath = path.join(saveDir, filename);
  const json = JSON.stringify(result, null, 2);
  await fs.writeFile(filePath, json, 'utf-8');
  await fs.writeFile(path.join(saveDir, 'tjs_latest.json'), json, 'utf-8');
  return filePath;
}

const DEFAULT_SERVER_PORT = 0;
const SERVER_HOSTS = Object.freeze(['127.0.0.1', 'localhost', '0.0.0.0']);

async function createStaticServer(root, preferredPort) {
  const listenPort = Number.isFinite(preferredPort) ? preferredPort : DEFAULT_SERVER_PORT;

  const mimeTypes = {
    '.html': 'text/html',
    '.js': 'text/javascript',
    '.json': 'application/json',
    '.wasm': 'application/wasm',
    '.onnx': 'application/octet-stream',
  };

  const serveRequest = async (req, res) => {
    const url = new URL(req.url, `http://${req.headers.host}`);
    const urlPath = decodeURIComponent(url.pathname);
    const safePath = urlPath.replace(/^\/+/, '') || '';
    const filePath = path.resolve(root, safePath);
    const rootPath = path.resolve(root);
    const normalizedRoot = rootPath.endsWith(path.sep) ? rootPath : `${rootPath}${path.sep}`;

    try {
      const isAllowed = filePath.startsWith(normalizedRoot);
      if (!isAllowed) {
        res.writeHead(403);
        res.end('Forbidden');
        return;
      }

      const stat = await fs.stat(filePath);
      if (!stat.isFile()) {
        res.writeHead(404);
        res.end('Not found');
        return;
      }
      const ext = path.extname(filePath);
      const contentType = mimeTypes[ext] || 'application/octet-stream';
      const data = await fs.readFile(filePath);
      res.writeHead(200, {
        'Content-Type': contentType,
        'Content-Length': data.byteLength,
        'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
        'Pragma': 'no-cache',
        'Expires': '0',
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Embedder-Policy': 'credentialless',
      });
      res.end(data);
    } catch {
      res.writeHead(404);
      res.end('Not found');
    }
  };

  const tryListen = (host) => new Promise((resolve, reject) => {
    const server = http.createServer(serveRequest);
    server.once('error', (error) => {
      reject(error);
    });
    if (host == null) {
      server.listen(listenPort, () => {
        const address = server.address();
        const resolvedPort = Number.isFinite(address?.port) ? address.port : listenPort;
        const resolvedHost = typeof address?.address === 'string'
          ? address.address
          : '127.0.0.1';
        const urlHost = resolvedHost === '::' || resolvedHost === '0.0.0.0'
          ? '127.0.0.1'
          : resolvedHost;
        resolve({
          server,
          port: resolvedPort,
          host: urlHost,
        });
      });
      return;
    }
    server.listen(listenPort, host, () => {
      const address = server.address();
      const resolvedPort = Number.isFinite(address?.port) ? address.port : listenPort;
      const resolvedHost = host && host.length > 0 ? host : '127.0.0.1';
      const urlHost = resolvedHost === '0.0.0.0' ? '127.0.0.1' : resolvedHost;
      resolve({
        server,
        port: resolvedPort,
        host: urlHost,
      });
    });
  });

  let lastError = null;
  const tryHosts = [...SERVER_HOSTS, null];
  for (const host of tryHosts) {
    try {
      const result = await tryListen(host);
      return {
        server: result.server,
        port: result.port,
        baseUrl: `http://${result.host}:${result.port}`,
      };
    } catch (error) {
      lastError = error;
      if (error?.code !== 'EACCES' && error?.code !== 'EADDRINUSE' && error?.code !== 'EPERM') {
        throw error;
      }
    }
  }

  if (lastError != null) {
    throw lastError;
  }

  throw new Error('unable to start static server: no hosts available');
}

function tryParseJson(text) {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function extractOrtProfilingEvents(messages) {
  const events = [];
  const buffered = [];
  let inArrayBlock = false;
  let parseErrorCount = 0;
  let rawLogCount = 0;

  const pushObject = (value) => {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return;
    if (!Object.prototype.hasOwnProperty.call(value, 'name') && !Object.prototype.hasOwnProperty.call(value, 'dur')) return;
    events.push(value);
  };

  const pushParsed = (value) => {
    if (Array.isArray(value)) {
      for (const item of value) {
        pushObject(item);
      }
      return;
    }
    pushObject(value);
  };

  const parseLine = (line) => {
    const trimmed = line.trim();
    if (!trimmed) return;
    const normalized = trimmed.endsWith(',') ? trimmed.slice(0, -1).trimEnd() : trimmed;
    const parsed = tryParseJson(normalized);
    if (parsed == null) {
      parseErrorCount++;
      return;
    }
    pushParsed(parsed);
  };

  const flushBuffered = () => {
    for (const line of buffered) {
      parseLine(line);
    }
    buffered.length = 0;
  };

  for (const message of messages) {
    if (message.type !== 'log') continue;
    const text = String(message.text ?? '').trim();
    if (!text) continue;
    rawLogCount++;

    if (inArrayBlock) {
      if (text === ']') {
        flushBuffered();
        inArrayBlock = false;
      } else {
        buffered.push(text);
      }
      continue;
    }

    if (text === '[') {
      inArrayBlock = true;
      buffered.length = 0;
      continue;
    }

    const parsed = tryParseJson(text);
    if (parsed != null) {
      pushParsed(parsed);
      continue;
    }

    if (text.startsWith('{')) {
      parseLine(text);
    }
  }

  if (inArrayBlock && buffered.length > 0) {
    flushBuffered();
  }

  return {
    events,
    parseErrorCount,
    rawLogCount,
  };
}

function summarizeOrtProfiling(events, topN) {
  const byName = new Map();
  const byCategory = new Map();
  let totalDurationMs = 0;
  let durationEventCount = 0;

  for (const event of events) {
    const durUs = Number(event.dur);
    if (!Number.isFinite(durUs) || durUs < 0) continue;

    const totalMs = durUs / 1000;
    const name = String(event.name || 'unknown');
    const category = String(event.cat || 'unknown');

    durationEventCount++;
    totalDurationMs += totalMs;

    const opEntry = byName.get(name) || { name, count: 0, totalMs: 0, maxMs: 0, categories: new Set() };
    opEntry.count++;
    opEntry.totalMs += totalMs;
    if (totalMs > opEntry.maxMs) opEntry.maxMs = totalMs;
    opEntry.categories.add(category);
    byName.set(name, opEntry);

    byCategory.set(category, (byCategory.get(category) || 0) + totalMs);
  }

  const topOperations = Array.from(byName.values())
    .sort((a, b) => b.totalMs - a.totalMs)
    .slice(0, topN)
    .map((entry) => ({
      name: entry.name,
      count: entry.count,
      totalMs: entry.totalMs,
      avgMs: entry.count > 0 ? entry.totalMs / entry.count : 0,
      maxMs: entry.maxMs,
      categories: Array.from(entry.categories).sort(),
    }));

  const categories = Array.from(byCategory.entries())
    .sort((a, b) => b[1] - a[1])
    .map(([name, totalMs]) => ({ name, totalMs }));

  return {
    eventCount: events.length,
    durationEventCount,
    totalDurationMs,
    topOperations,
    categories,
  };
}

function formatRecentBrowserMessages(messages, limit = 20) {
  if (!Array.isArray(messages) || messages.length === 0) return '';
  const start = Math.max(0, messages.length - limit);
  const tail = messages.slice(start);
  return tail.map((entry) => `[browser:${entry.type}] ${entry.text}`).join('\n');
}

function closeBaseServer(baseServer) {
  const server = baseServer?.server;
  if (server && typeof server.close === 'function') {
    server.close();
  }
}

async function main() {
  const flags = parseArgs(process.argv.slice(2));
  const showConsole = flags['browser-console'] === true;
  const cacheMode = flags['cache-mode'] || 'warm';
  const loadMode = parseLoadMode(flags['load-mode'], '--load-mode', 'http');
  const strictWarmOpfs = cacheMode === 'warm' && loadMode === 'opfs';
  const tjsVersion = flags['tjs-version'] || '4';
  const timestamp = parseTimestampValue(flags.timestamp, '--timestamp');
  const profileOps = parseBooleanFlag(flags['profile-ops'], true, '--profile-ops');
  const profileTopN = parsePositiveInteger(flags['profile-top'], DEFAULT_PROFILE_TOP_N, '--profile-top');
  const useChatTemplate = flags['use-chat-template'] === true;
  const tjsDtype = parseTjsDtype(flags.dtype, '--dtype', 'fp16');

  if (cacheMode !== 'cold' && cacheMode !== 'warm') {
    throw new Error('--cache-mode must be cold or warm');
  }
  if (tjsVersion !== '3' && tjsVersion !== '4') {
    throw new Error('--tjs-version must be 3 or 4');
  }

  let modelId = flags.model || DEFAULT_MODEL;
  let prompt = flags.prompt || DEFAULT_PROMPT;
  let maxNewTokens = parsePositiveInteger(flags['max-tokens'], DEFAULT_MAX_TOKENS, '--max-tokens');
  let temperature = parseNonNegativeNumber(flags.temperature, '--temperature', DEFAULT_TEMPERATURE);
  let topK = parsePositiveInteger(flags['top-k'], DEFAULT_TOP_K, '--top-k');
  let topP = parseProbability(flags['top-p'], '--top-p', DEFAULT_TOP_P);
  let warmupRuns = parseNonNegativeInt(flags.warmup, '--warmup', DEFAULT_WARMUP);
  let timedRuns = parsePositiveInteger(flags.runs, DEFAULT_RUNS, '--runs');
  const timeoutMs = parsePositiveInteger(flags.timeout, DEFAULT_TIMEOUT, '--timeout');
  const userDataDir = flags['user-data'] || DEFAULT_PROFILE_DIR;
  const serverPort = parseNonNegativeInt(flags['server-port'], '--server-port', DEFAULT_SERVER_PORT);
  const seed = parseNonNegativeInt(flags.seed, '--seed', DEFAULT_SEED);
  const localModelPath = flags['local-model-path'] || null;
  const browserBaseUrl = flags['browser-base-url'] || null;
  // Auto-read HF token from flags or the cached credential file (for gated models).
  let hfToken = flags['hf-token'] || null;
  if (!hfToken) {
    const tokenPath = path.join(process.env.HOME || process.env.USERPROFILE || '', '.cache', 'huggingface', 'token');
    try {
      hfToken = (await fs.readFile(tokenPath, 'utf-8')).trim() || null;
    } catch { /* no cached token */ }
  }

  if (flags.workload) {
    const wl = await loadWorkload(flags.workload);
    maxNewTokens = wl.decodeTokens;
    if (flags.temperature == null && Number.isFinite(wl?.sampling?.temperature)) {
      temperature = parseNonNegativeNumber(wl.sampling.temperature, '--workload.sampling.temperature', temperature);
    }
    if (flags['top-k'] == null && Number.isFinite(wl?.sampling?.topK)) {
      topK = Math.max(1, Math.floor(Number(wl.sampling.topK)));
    }
    if (flags['top-p'] == null && Number.isFinite(wl?.sampling?.topP)) {
      topP = parseProbability(wl.sampling.topP, '--workload.sampling.topP', topP);
    }
    if (!flags.prompt) {
      const words = [];
      for (let i = 0; i < wl.prefillTokens; i++) {
        words.push(`word${i}`);
      }
      prompt = words.join(' ');
    }
  }

  console.error(
    `[tjs-bench] tjs=v${tjsVersion} model=${modelId} maxTokens=${maxNewTokens} ` +
    `warmup=${warmupRuns} runs=${timedRuns} cache=${cacheMode} ` +
    `sampling=(temp=${temperature}, topK=${topK}, topP=${topP}) ` +
    `chatTemplate=${useChatTemplate ? 'on' : 'off'} dtype=${tjsDtype} ` +
    `profileOps=${profileOps ? 'on' : 'off'} timeout=${timeoutMs}ms`
  );
  if (strictWarmOpfs) {
    console.error('[tjs-bench] strict warm-opfs enabled: persistent profile required; timed run will execute offline.');
  }

  // Cold mode: wipe persistent profile to force full re-download + recompile
  if (cacheMode === 'cold') {
    console.error(`[tjs-bench] cold mode: wiping ${userDataDir}`);
    await wipeDir(userDataDir);
  }

  let playwright;
  try {
    playwright = await import('playwright');
  } catch {
    console.error('[tjs-bench] Error: playwright not installed. Run: npm install');
    process.exit(1);
  }

  const baseServer = browserBaseUrl
    ? null
    : await createStaticServer(REPO_ROOT, serverPort).catch((error) => {
      throw new Error(`failed to start static server (${error.message}). Pass --browser-base-url to reuse an existing server.`);
    });
  const baseUrl = browserBaseUrl || baseServer.baseUrl;
  console.error(`[tjs-bench] using baseUrl ${baseUrl}`);

  const platformArgs = PLATFORM_ARGS[process.platform] ?? [];
  const cliBrowserArgs = Array.isArray(flags['browser-arg']) ? flags['browser-arg'] : [];
  const allArgs = uniqueArgs([...WEBGPU_ARGS, ...platformArgs, ...cliBrowserArgs]);
  const browserExecutable = flags['browser-executable'] || null;
  const hostEnvironment = {
    platform: process.platform,
    arch: process.arch,
    nodeVersion: process.version,
    osRelease: typeof os.release === 'function' ? os.release() : null,
    cpuModel: (() => {
      const cpuInfo = typeof os.cpus === 'function' ? os.cpus() : null;
      if (!Array.isArray(cpuInfo) || cpuInfo.length === 0) return null;
      return asNonEmptyString(cpuInfo[0]?.model);
    })(),
  };
  const webgpuBackend = inferWebgpuBackendFromArgs(allArgs, hostEnvironment.platform);

  // Measure browser launch time as part of cold/warm UX
  const launchStart = performance.now();

  let context;
  let browser = null;
  try {
    await fs.mkdir(userDataDir, { recursive: true });
    const launchArgs = uniqueArgs([...allArgs]);
    const launchOptions = {
      headless: true,
      args: launchArgs,
    };
    if (browserExecutable) {
      launchOptions.executablePath = String(browserExecutable);
    }
    try {
      context = await playwright.chromium.launchPersistentContext(userDataDir, launchOptions);
    } catch (error) {
      if (!isRecoverablePersistentLaunchError(error)) {
        throw error;
      }
      const recoveryLaunchOptions = {
        ...launchOptions,
        args: hasCrashRecoveryArgs(launchOptions.args) ? launchOptions.args : withCrashRecoveryArgs(launchOptions.args),
      };
      if (strictWarmOpfs) {
        console.error('[tjs-bench] strict warm-opfs: persistent launch failed; retrying with crash-recovery args (no profile wipe).');
        try {
          context = await playwright.chromium.launchPersistentContext(userDataDir, recoveryLaunchOptions);
        } catch (retryError) {
          throw new Error(
            `strict warm-opfs requires persistent profile reuse; launch failed without wipe (${retryError.message})`
          );
        }
      } else {
        console.error('[tjs-bench] persistent launch failed; retrying with a clean profile.');
        await fs.rm(userDataDir, { recursive: true, force: true }).catch(() => {});
        try {
          context = await playwright.chromium.launchPersistentContext(userDataDir, launchOptions);
        } catch (retryError) {
          if (!isRecoverablePersistentLaunchError(retryError)) {
            throw retryError;
          }
          console.error('[tjs-bench] persistent launch still failing; retrying with crash-recovery args.');
          try {
            context = await playwright.chromium.launchPersistentContext(userDataDir, recoveryLaunchOptions);
          } catch (recoveryError) {
            if (!isRecoverablePersistentLaunchError(recoveryError)) {
              throw recoveryError;
            }
            console.error('[tjs-bench] persistent launch with recovery args failed; falling back to non-persistent browser.');
            browser = await playwright.chromium.launch(recoveryLaunchOptions);
            context = await browser.newContext();
          }
        }
      }
    }
  } catch (error) {
    console.error(`[tjs-bench] Failed to launch browser: ${error.message}`);
    if (baseServer) {
      closeBaseServer(baseServer);
    }
    process.exit(1);
  }

  const browserLaunchMs = performance.now() - launchStart;

  const page = await context.newPage();
  page.setDefaultTimeout(timeoutMs);
  page.setDefaultNavigationTimeout(timeoutMs);
  const browserMessages = [];

  page.on('console', (msg) => {
    const entry = { type: msg.type(), text: msg.text() };
    browserMessages.push(entry);
    if (showConsole) {
      console.error(`[browser:${entry.type}] ${entry.text}`);
    }
  });

  page.on('pageerror', (error) => {
    console.error(`[browser:error] ${error.message}`);
  });

  try {
    const navStart = performance.now();
    const runnerParams = new URLSearchParams({ v: tjsVersion });
    if (localModelPath) runnerParams.set('localModelPath', localModelPath);
    if (hfToken) runnerParams.set('hfToken', hfToken);
    const runnerUrl = new URL('/benchmarks/runners/transformersjs-runner.html', baseUrl);
    runnerUrl.search = runnerParams.toString();
    await page.goto(runnerUrl.toString(), { timeout: timeoutMs });
    await page.waitForFunction(() => window.__tfjsReady === true, { timeout: timeoutMs });
    const pageReadyMs = performance.now() - navStart;
    console.error(`[tjs-bench] runner page ready in ${pageReadyMs.toFixed(0)}ms`);

    const cachePrimeEnabled = cacheMode === 'warm' && loadMode === 'opfs';
    const cachePrime = {
      enabled: cachePrimeEnabled,
      primed: false,
      primeMs: 0,
    };

    if (cachePrimeEnabled) {
      const primeResult = await page.evaluate(async (primeConfig) => {
        if (typeof window.__primeBenchModel !== 'function') {
          throw new Error('__primeBenchModel is not available in runner page');
        }
        return window.__primeBenchModel(primeConfig);
      }, { modelId, dtype: tjsDtype });
      const reportedPrimeMs = Number(primeResult?.primeMs);
      cachePrime.primed = primeResult?.ok === true;
      cachePrime.primeMs = Number.isFinite(reportedPrimeMs) ? reportedPrimeMs : 0;
      if (!cachePrime.primed) {
        throw new Error(`warm-opfs cache prime failed for model "${modelId}"`);
      }
      console.error(`[tjs-bench] warm-opfs cache prime complete in ${cachePrime.primeMs.toFixed(0)}ms`);
      if (strictWarmOpfs) {
        await context.setOffline(true);
        console.error('[tjs-bench] strict warm-opfs: browser set offline for timed run.');
      }
    }

    console.error('[tjs-bench] starting benchmark...');

    const benchStart = performance.now();

    const result = await page.evaluate(
      async (config) => window.__runBench(config),
      {
        modelId,
        prompt,
        maxNewTokens,
        warmupRuns,
        timedRuns,
        cacheMode,
        loadMode,
        profileOps,
        profileTopN,
        seed,
        sampling: {
          temperature,
          topK,
          topP,
        },
        useChatTemplate,
        dtype: tjsDtype,
      },
    );

    const totalBenchMs = performance.now() - benchStart;
    const extracted = profileOps
      ? extractOrtProfilingEvents(browserMessages)
      : { events: [], parseErrorCount: 0, rawLogCount: 0 };
    const ortSummary = summarizeOrtProfiling(extracted.events, profileTopN);
    ortSummary.parseErrorCount = extracted.parseErrorCount;
    ortSummary.rawLogCount = extracted.rawLogCount;
    ortSummary.enabled = profileOps;
    ortSummary.topN = profileTopN;

    result.cacheMode = cacheMode;
    result.loadMode = loadMode;
    result.benchmarkPolicy = {
      source: BENCHMARK_POLICY.source,
      schemaVersion: BENCHMARK_POLICY.schemaVersion,
      updated: BENCHMARK_POLICY.updated,
    };
    result.determinism = {
      seed,
      decoding: {
        do_sample: temperature > 0,
        temperature,
        topK,
        topP,
      },
    };
    result.profiling = {
      ...(result.profiling && typeof result.profiling === 'object' ? result.profiling : {}),
      ortConsole: ortSummary,
    };
    result.metrics = {
      ...(result.metrics && typeof result.metrics === 'object' ? result.metrics : {}),
      ort_profiled_total_ms: ortSummary.totalDurationMs,
      ort_profiled_event_count: ortSummary.durationEventCount,
      ort_top_op_total_ms: ortSummary.topOperations[0]?.totalMs ?? null,
    };
    result.cachePrime = cachePrime;
    result.env = {
      ...(result.env && typeof result.env === 'object' ? result.env : {}),
      webgpuBackend,
      browserExecutable: asNonEmptyString(browserExecutable),
    };
    result.environment = {
      host: hostEnvironment,
      browser: {
        userAgent: asNonEmptyString(result?.env?.browserUserAgent),
        platform: asNonEmptyString(result?.env?.browserPlatform),
        language: asNonEmptyString(result?.env?.browserLanguage),
        vendor: asNonEmptyString(result?.env?.browserVendor),
        executable: asNonEmptyString(browserExecutable),
        channel: null,
      },
      gpu: {
        api: 'webgpu',
        backend: webgpuBackend,
        vendor: asNonEmptyString(result?.deviceInfo?.vendor),
        architecture: asNonEmptyString(result?.deviceInfo?.architecture),
        device: asNonEmptyString(result?.deviceInfo?.device),
        description: asNonEmptyString(result?.deviceInfo?.description),
        hasF16: typeof result?.deviceInfo?.hasF16 === 'boolean' ? result.deviceInfo.hasF16 : null,
        hasSubgroups: typeof result?.deviceInfo?.hasSubgroups === 'boolean' ? result.deviceInfo.hasSubgroups : null,
        hasTimestampQuery: typeof result?.deviceInfo?.hasTimestampQuery === 'boolean'
          ? result.deviceInfo.hasTimestampQuery
          : null,
      },
      runtime: {
        library: 'transformers.js',
        version: asNonEmptyString(result?.env?.version),
        surface: 'browser',
        device: asNonEmptyString(result?.env?.device),
        dtype: asNonEmptyString(result?.env?.dtype),
        requestedDtype: asNonEmptyString(result?.env?.requestedDtype),
        executionProviderMode: asNonEmptyString(result?.env?.executionProviderMode),
        cacheMode: asNonEmptyString(result?.cacheMode),
        loadMode: asNonEmptyString(result?.loadMode),
      },
    };

    console.log(JSON.stringify(result, null, 2));

    if (flags.save) {
      const saveDir = flags['save-dir'] || path.join(BENCHMARKS_ROOT, 'vendors', 'results');
      const savedPath = await saveResult(result, saveDir, timestamp);
      console.error(`[tjs-bench] saved to ${savedPath}`);
    }
  } catch (error) {
    console.error(`[tjs-bench] Benchmark failed: ${error.message}`);
    const recentLogs = formatRecentBrowserMessages(browserMessages, 20);
    if (recentLogs) {
      console.error('[tjs-bench] recent browser console:\n' + recentLogs);
    }
    process.exit(1);
  } finally {
    await page.close().catch(() => {});
    await context.close().catch(() => {});
    if (browser) {
      await browser.close().catch(() => {});
    }
    if (baseServer) {
      closeBaseServer(baseServer);
    }
  }
}

main().catch((error) => {
  console.error(`[tjs-bench] ${error.message}`);
  process.exit(1);
});
