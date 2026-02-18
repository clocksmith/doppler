#!/usr/bin/env node

/**
 * Transformers.js WebGPU benchmark runner.
 *
 * Launches headless Chromium via Playwright, loads a model through Transformers.js
 * with the WebGPU backend, runs timed inference, and outputs normalized JSON.
 *
 * Usage:
 *   node external/transformersjs-bench.mjs [options]
 *
 * Options:
 *   --model <id>         HuggingFace model ID (default: onnx-community/gemma-3-1b-it-ONNX-GQA)
 *   --prompt <text>      Input prompt
 *   --max-tokens <n>     Max new tokens to generate (default: 128)
 *   --warmup <n>         Warmup runs (default: 1)
 *   --runs <n>           Timed runs (default: 3)
 *   --workload <id>      Use a predefined workload from workloads.json
 *   --cache-mode <mode>  cold|warm (default: warm)
 *                        cold: wipe persistent profile before launch
 *                        warm: reuse persistent profile
 *   --timeout <ms>       Page timeout in ms (default: 600000)
 *   --profile-ops <on|off>  Enable ONNX Runtime op profiling (default: on)
 *   --profile-top <n>        Number of top ops in profiling summary (default: 20)
 *   --tjs-version <3|4>  Transformers.js version (default: 3)
 *   --save               Save result JSON to bench-results/
 *   --save-dir <dir>     Directory for saved results (default: ./bench-results)
 *   --json               Output only JSON (default: true)
 *   --browser-console    Stream browser console to stderr
 */

import fs from 'node:fs/promises';
import path from 'node:path';
import http from 'node:http';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DOPPLER_ROOT = path.resolve(__dirname, '..');

const DEFAULT_MODEL = 'onnx-community/gemma-3-1b-it-ONNX-GQA';
const DEFAULT_PROMPT = 'Summarize this input in one sentence.';
const DEFAULT_MAX_TOKENS = 128;
const DEFAULT_WARMUP = 1;
const DEFAULT_RUNS = 3;
const DEFAULT_TIMEOUT = 600_000;
const DEFAULT_PROFILE_TOP_N = 20;
const DEFAULT_PROFILE_DIR = path.join(DOPPLER_ROOT, '.tjs-bench-profile');

const WEBGPU_ARGS = [
  '--enable-unsafe-webgpu',
  '--enable-webgpu-developer-features',
  '--disable-dawn-features=disallow_unsafe_apis',
  '--ignore-gpu-blocklist',
];

const PLATFORM_ARGS = {
  darwin: ['--use-angle=metal'],
  linux: ['--use-angle=vulkan', '--enable-features=Vulkan', '--disable-vulkan-surface'],
  win32: [],
};

function parseArgs(argv) {
  const flags = {};
  for (let i = 0; i < argv.length; i++) {
    const token = argv[i];
    if (!token.startsWith('--')) continue;
    const key = token.slice(2);
    if (key === 'json' || key === 'browser-console' || key === 'save') {
      flags[key] = true;
      continue;
    }
    const value = argv[i + 1];
    if (value === undefined || value.startsWith('--')) {
      throw new Error(`Missing value for --${key}`);
    }
    flags[key] = value;
    i++;
  }
  return flags;
}

function parseToggle(value, flag, fallback) {
  if (value == null) return fallback;
  const normalized = String(value).trim().toLowerCase();
  if (normalized === 'on' || normalized === 'true' || normalized === '1' || normalized === 'yes') return true;
  if (normalized === 'off' || normalized === 'false' || normalized === '0' || normalized === 'no') return false;
  throw new Error(`${flag} must be one of: on, off, true, false, 1, 0`);
}

function parsePositiveInt(value, flag, fallback) {
  if (value == null) return fallback;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || !Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${flag} must be a positive integer`);
  }
  return parsed;
}

async function loadWorkload(workloadId) {
  const workloadsPath = path.join(DOPPLER_ROOT, 'benchmarks', 'competitors', 'workloads.json');
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

function compactTimestamp() {
  const d = new Date();
  const pad = (n, w = 2) => String(n).padStart(w, '0');
  return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}T${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
}

async function saveResult(result, saveDir) {
  await fs.mkdir(saveDir, { recursive: true });
  const modelSlug = String(result?.modelId || 'unknown').replace(/[^a-zA-Z0-9_-]/g, '_');
  const ts = compactTimestamp();
  const filename = `tjs_${modelSlug}_${ts}.json`;
  const filePath = path.join(saveDir, filename);
  const json = JSON.stringify(result, null, 2);
  await fs.writeFile(filePath, json, 'utf-8');
  await fs.writeFile(path.join(saveDir, 'tjs_latest.json'), json, 'utf-8');
  return filePath;
}

const DEFAULT_SERVER_PORT = 9999;

function createStaticServer(root, preferredPort) {
  const listenPort = preferredPort || DEFAULT_SERVER_PORT;
  return new Promise((resolve) => {
    const mimeTypes = {
      '.html': 'text/html',
      '.js': 'text/javascript',
      '.mjs': 'text/javascript',
      '.json': 'application/json',
      '.wasm': 'application/wasm',
      '.onnx': 'application/octet-stream',
    };

    const server = http.createServer(async (req, res) => {
      const url = new URL(req.url, `http://${req.headers.host}`);
      const filePath = path.join(root, decodeURIComponent(url.pathname));

      try {
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
          'Cross-Origin-Opener-Policy': 'same-origin',
          'Cross-Origin-Embedder-Policy': 'credentialless',
        });
        res.end(data);
      } catch {
        res.writeHead(404);
        res.end('Not found');
      }
    });

    server.listen(listenPort, '127.0.0.1', () => {
      const { port } = server.address();
      resolve({ server, port, baseUrl: `http://127.0.0.1:${port}` });
    });
  });
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

async function main() {
  const flags = parseArgs(process.argv.slice(2));
  const showConsole = flags['browser-console'] === true;
  const cacheMode = flags['cache-mode'] || 'warm';
  const tjsVersion = flags['tjs-version'] || '3';
  const profileOps = parseToggle(flags['profile-ops'], '--profile-ops', true);
  const profileTopN = parsePositiveInt(flags['profile-top'], '--profile-top', DEFAULT_PROFILE_TOP_N);

  if (cacheMode !== 'cold' && cacheMode !== 'warm') {
    throw new Error('--cache-mode must be cold or warm');
  }
  if (tjsVersion !== '3' && tjsVersion !== '4') {
    throw new Error('--tjs-version must be 3 or 4');
  }

  let modelId = flags.model || DEFAULT_MODEL;
  let prompt = flags.prompt || DEFAULT_PROMPT;
  let maxNewTokens = Number(flags['max-tokens'] || DEFAULT_MAX_TOKENS);
  let warmupRuns = Number(flags.warmup ?? DEFAULT_WARMUP);
  let timedRuns = Number(flags.runs ?? DEFAULT_RUNS);
  const timeoutMs = Number(flags.timeout ?? DEFAULT_TIMEOUT);
  const userDataDir = flags['user-data'] || DEFAULT_PROFILE_DIR;
  const serverPort = parsePositiveInt(flags['server-port'], '--server-port', DEFAULT_SERVER_PORT);
  const localModelPath = flags['local-model-path'] || null;

  if (flags.workload) {
    const wl = await loadWorkload(flags.workload);
    maxNewTokens = wl.decodeTokens;
    if (!flags.prompt) {
      const words = [];
      for (let i = 0; i < wl.prefillTokens; i++) {
        words.push(`word${i}`);
      }
      prompt = words.join(' ');
    }
  }

  console.error(`[tjs-bench] tjs=v${tjsVersion} model=${modelId} maxTokens=${maxNewTokens} warmup=${warmupRuns} runs=${timedRuns} cache=${cacheMode} profileOps=${profileOps ? 'on' : 'off'}`);

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

  // Serve from doppler root so both external/ and node_modules/ are accessible
  const { server, baseUrl } = await createStaticServer(DOPPLER_ROOT, serverPort);
  console.error(`[tjs-bench] static server at ${baseUrl}`);

  const platformArgs = PLATFORM_ARGS[process.platform] ?? [];
  const allArgs = [...new Set([...WEBGPU_ARGS, ...platformArgs])];

  // Measure browser launch time as part of cold/warm UX
  const launchStart = performance.now();

  let context;
  try {
    await fs.mkdir(userDataDir, { recursive: true });
    context = await playwright.chromium.launchPersistentContext(userDataDir, {
      headless: true,
      args: allArgs,
    });
  } catch (error) {
    console.error(`[tjs-bench] Failed to launch browser: ${error.message}`);
    server.close();
    process.exit(1);
  }

  const browserLaunchMs = performance.now() - launchStart;

  const page = await context.newPage();
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
    await page.goto(`${baseUrl}/external/transformersjs-runner.html?${runnerParams}`, { timeout: 30_000 });
    await page.waitForFunction(() => window.__tfjsReady === true, { timeout: 30_000 });
    const pageReadyMs = performance.now() - navStart;
    console.error(`[tjs-bench] runner page ready in ${pageReadyMs.toFixed(0)}ms, starting benchmark...`);

    const benchStart = performance.now();

    const result = await page.evaluate(
      async (config) => window.__runBench(config),
      { modelId, prompt, maxNewTokens, warmupRuns, timedRuns, profileOps, profileTopN },
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

    // Annotate with load class and UX timing
    result.loadClass = cacheMode;
    result.ux = {
      browserLaunchMs,
      pageReadyMs,
      modelLoadMs: result.modelLoadMs,
      firstResponseMs: result.modelLoadMs + (result.runs?.[0]?.totalMs ?? 0),
      totalBenchMs,
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

    console.log(JSON.stringify(result, null, 2));

    if (flags.save) {
      const saveDir = flags['save-dir'] || path.join(DOPPLER_ROOT, 'bench-results');
      const savedPath = await saveResult(result, saveDir);
      console.error(`[tjs-bench] saved to ${savedPath}`);
    }
  } catch (error) {
    console.error(`[tjs-bench] Benchmark failed: ${error.message}`);
    process.exit(1);
  } finally {
    await page.close().catch(() => {});
    await context.close().catch(() => {});
    server.close();
  }
}

main().catch((error) => {
  console.error(`[tjs-bench] ${error.message}`);
  process.exit(1);
});
