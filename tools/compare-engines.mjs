#!/usr/bin/env node

/**
 * Unified cross-engine benchmark comparison.
 *
 * Runs both Doppler and Transformers.js benchmarks and produces
 * a structured side-by-side comparison across three modes:
 *   - compute: model-load/decode/prefill(TTFT)/TTFT (warm-cache compare)
 *   - warm:    warm-start UX (model load from cache + first inference)
 *   - cold:    cold-start UX (full download/compile + first inference)
 *
 * Usage:
 *   node tools/compare-engines.mjs [options]
 *
 * Options:
 *   --model-id <id>        Doppler model ID (default: gemma-3-1b-it-wf16)
 *   --model-url <url>      Doppler model URL path (default: /models/local/<model-id>)
 *   --tjs-model <id>       TJS model ID (default: onnx-community/gemma-3-1b-it-ONNX-GQA)
 *   --tjs-version <3|4>    Transformers.js version (default: 3)
 *   --workload <id>        Shared workload id from benchmarks/competitors/workloads.json
 *   --prompt <text>        Prompt used for both engines (overrides workload-derived prompt)
 *   --prefill-tokens <n>   Synthetic prompt token target when --prompt is omitted
 *   --mode <mode>          compute|cold|warm|all (default: all)
 *   --max-tokens <n>       Max new tokens (default: 64)
 *   --temperature <n>      Sampling temperature (default from workload, fallback 0)
 *   --top-k <n>            Sampling top-k (default from workload, fallback 32)
 *   --top-p <n>            Sampling top-p (default from workload, fallback 1)
 *   --warmup <n>           Warmup runs per engine (default: 1)
 *   --runs <n>             Timed runs per engine (default: 3)
 *   --decode-profile <profile>  parity|throughput|custom (default: parity)
 *   --seed <n>             Deterministic seed metadata (default: 0)
 *   --doppler-kernel-path <id>  Doppler kernel path override (default: gemma3-f16-f32a for Gemma 3 1B; otherwise manifest default)
 *   --doppler-batch-size <n>     Doppler decode batch size (only with --decode-profile custom)
 *   --doppler-readback-interval <n>  Doppler decode readback interval (only with --decode-profile custom)
 *   --doppler-no-opfs-cache  Disable Doppler OPFS cache for browser runs
 *   --doppler-browser-user-data <path>  Doppler Chromium profile dir
 *   --doppler-browser-port <n>  Doppler browser relay static port (default: 0 = random)
 *   --tjs-profile-ops <on|off>    Transformers.js ORT op profiling (default: off)
 *   --tjs-timeout-ms <ms>         Transformers.js timeout (default: 600000)
 *   --tjs-server-port <n>         Transformers.js static server port (default: 0 = random)
 *   --tjs-browser-console         Stream browser console on TJS failures/retries
 *   --save                 Save results to bench-results/
 *   --save-dir <dir>       Directory for saved results (default: ./bench-results)
 *   --json                 JSON-only output
 */

import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const execFileAsync = promisify(execFile);
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DOPPLER_ROOT = path.resolve(__dirname, '..');
const WORKLOADS_PATH = path.join(DOPPLER_ROOT, 'benchmarks', 'competitors', 'workloads.json');

const DEFAULT_DOPPLER_MODEL = 'gemma-3-1b-it-wf16';
const DEFAULT_TJS_MODEL = 'onnx-community/gemma-3-1b-it-ONNX-GQA';
const FALLBACK_DEFAULT_WORKLOAD_ID = 'g3-p064-d064-t0-k32';
const DEFAULT_PROMPT = 'word0 word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 word12 word13 word14 word15 word16 word17 word18 word19 word20 word21 word22 word23 word24 word25 word26 word27 word28 word29 word30 word31';
const DEFAULT_MAX_TOKENS = 64;
const DEFAULT_WARMUP = 1;
const DEFAULT_RUNS = 3;
const DEFAULT_SEED = 0;
const DEFAULT_SHARED_SAMPLING = Object.freeze({
  temperature: 0,
  topK: 32,
  topP: 1,
});
const DEFAULT_DOPPLER_KERNEL_PATH = null;
const AUTO_DOPPLER_KERNEL_PATH_BY_MODEL = Object.freeze({
  'gemma-3-1b-it': 'gemma3-f16-fused-f32a-online',
  'gemma-3-1b-it-wf16': 'gemma3-f16-fused-f32a-online',
});
const DEFAULT_DECODE_PROFILE = 'parity';
const DEFAULT_TJS_PROFILE_OPS = false;
const DEFAULT_TJS_TIMEOUT_MS = 600_000;
const DEFAULT_TJS_SERVER_PORT = 0;
const DEFAULT_DOPPLER_BROWSER_PORT = 0;
const DECODE_PROFILE_PRESETS = Object.freeze({
  parity: Object.freeze({
    batchSize: 1,
    readbackInterval: 1,
    label: 'TJS-like per-token cadence',
  }),
  throughput: Object.freeze({
    batchSize: 4,
    readbackInterval: 4,
    label: 'Doppler throughput-tuned cadence',
  }),
});
const VALID_DECODE_PROFILES = Object.freeze([
  ...Object.keys(DECODE_PROFILE_PRESETS),
  'custom',
]);
const DEFAULT_DOPPLER_BATCH_SIZE = DECODE_PROFILE_PRESETS[DEFAULT_DECODE_PROFILE].batchSize;
const DEFAULT_DOPPLER_READBACK_INTERVAL = DECODE_PROFILE_PRESETS[DEFAULT_DECODE_PROFILE].readbackInterval;

function parsePositiveInt(value, fallback, label) {
  if (value == null || value === '') return fallback;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || !Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${label} must be a positive integer`);
  }
  return parsed;
}

function parseNonNegativeInt(value, fallback, label) {
  if (value == null || value === '') return fallback;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || !Number.isInteger(parsed) || parsed < 0) {
    throw new Error(`${label} must be a non-negative integer`);
  }
  return parsed;
}

function parseOnOff(value, fallback, label) {
  if (value == null || value === '') return fallback;
  const normalized = String(value).trim().toLowerCase();
  if (normalized === 'on' || normalized === 'true' || normalized === '1' || normalized === 'yes') return true;
  if (normalized === 'off' || normalized === 'false' || normalized === '0' || normalized === 'no') return false;
  throw new Error(`${label} must be one of: on, off, true, false, 1, 0`);
}

function parseNonNegativeNumber(value, fallback, label) {
  if (value == null || value === '') return fallback;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0) {
    throw new Error(`${label} must be a non-negative number`);
  }
  return parsed;
}

function parseTopP(value, fallback, label) {
  if (value == null || value === '') return fallback;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0 || parsed > 1) {
    throw new Error(`${label} must be in the range (0, 1]`);
  }
  return parsed;
}

function buildSyntheticPrompt(prefillTokens) {
  const words = [];
  for (let i = 0; i < prefillTokens; i++) {
    words.push(`word${i}`);
  }
  return words.join(' ');
}

async function loadWorkloadCatalog() {
  const raw = await fs.readFile(WORKLOADS_PATH, 'utf-8');
  const payload = JSON.parse(raw);
  const rows = Array.isArray(payload?.workloads) ? payload.workloads : [];
  const configuredDefault = payload?.defaults?.compareEngines;
  return {
    rows,
    defaultWorkloadId: typeof configuredDefault === 'string' && configuredDefault.trim() !== ''
      ? configuredDefault
      : null,
  };
}

function resolveWorkloadById(workloadCatalog, workloadId) {
  const rows = Array.isArray(workloadCatalog?.rows) ? workloadCatalog.rows : [];
  const selected = rows.find((row) => row.id === workloadId) || null;
  if (!selected) {
    throw new Error(`Unknown workload "${workloadId}". Available: ${rows.map((row) => row.id).join(', ')}`);
  }
  return selected;
}

function clipTail(text, maxChars = 3000) {
  if (text == null) return '';
  const str = String(text);
  if (str.length <= maxChars) return str;
  return str.slice(str.length - maxChars);
}

function toFailurePayload(library, error) {
  return {
    failed: true,
    env: { library },
    error: {
      message: String(error?.message || error),
      code: error?.code ?? null,
      signal: error?.signal ?? null,
      killed: error?.killed === true,
      stderrTail: clipTail(error?.stderr, 3000),
    },
  };
}

function parseDecodeProfile(value) {
  if (value == null || value === '') return DEFAULT_DECODE_PROFILE;
  const profile = String(value);
  if (!VALID_DECODE_PROFILES.includes(profile)) {
    throw new Error(
      `--decode-profile must be one of: ${VALID_DECODE_PROFILES.join(', ')}`
    );
  }
  return profile;
}

function resolveDopplerKernelPath(modelId, kernelPathOverride) {
  if (kernelPathOverride != null && kernelPathOverride !== '') {
    return {
      kernelPath: String(kernelPathOverride),
      source: 'cli',
    };
  }
  const normalizedModelId = String(modelId ?? '').trim().toLowerCase();
  const autoKernelPath = AUTO_DOPPLER_KERNEL_PATH_BY_MODEL[normalizedModelId] ?? null;
  if (autoKernelPath != null) {
    return {
      kernelPath: autoKernelPath,
      source: 'auto-model-default',
    };
  }
  return {
    kernelPath: DEFAULT_DOPPLER_KERNEL_PATH,
    source: 'manifest-default',
  };
}

function parseArgs(argv) {
  const flags = {};
  for (let i = 0; i < argv.length; i++) {
    const token = argv[i];
    if (!token.startsWith('--')) continue;
    const key = token.slice(2);
    if (
      key === 'json'
      || key === 'save'
      || key === 'doppler-no-opfs-cache'
      || key === 'tjs-browser-console'
    ) {
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

function buildSharedBenchmarkContract({
  prompt,
  maxTokens,
  warmupRuns,
  timedRuns,
  seed,
  sampling = DEFAULT_SHARED_SAMPLING,
  useChatTemplate = false,
}) {
  return {
    prompt: String(prompt ?? DEFAULT_PROMPT),
    maxTokens: parsePositiveInt(maxTokens, DEFAULT_MAX_TOKENS, '--max-tokens'),
    warmupRuns: parseNonNegativeInt(warmupRuns, DEFAULT_WARMUP, '--warmup'),
    timedRuns: parsePositiveInt(timedRuns, DEFAULT_RUNS, '--runs'),
    seed: parseNonNegativeInt(seed, DEFAULT_SEED, '--seed'),
    sampling: {
      temperature: Number(sampling.temperature),
      topK: Number(sampling.topK),
      topP: Number(sampling.topP),
    },
    useChatTemplate: useChatTemplate === true,
  };
}

function buildDopplerRuntimeConfig(sharedContract, engineOverlay) {
  const sampling = {
    temperature: sharedContract.sampling.temperature,
    topK: sharedContract.sampling.topK,
    topP: sharedContract.sampling.topP,
  };
  const runtimeConfig = {
    shared: {
      benchmark: {
        run: {
          customPrompt: sharedContract.prompt,
          maxNewTokens: sharedContract.maxTokens,
          warmupRuns: sharedContract.warmupRuns,
          timedRuns: sharedContract.timedRuns,
          useChatTemplate: sharedContract.useChatTemplate,
          sampling,
        },
      },
    },
    inference: {
      prompt: sharedContract.prompt,
      chatTemplate: {
        enabled: sharedContract.useChatTemplate,
      },
      batching: {
        maxTokens: sharedContract.maxTokens,
        batchSize: engineOverlay.batchSize,
        readbackInterval: engineOverlay.readbackInterval,
        stopCheckMode: 'per-token',
      },
      sampling,
    },
  };
  if (engineOverlay.kernelPath) {
    runtimeConfig.inference.kernelPath = engineOverlay.kernelPath;
  }
  return runtimeConfig;
}

async function runDoppler(modelId, modelUrl, sharedContract, cacheMode, options = {}) {
  const resolvedKernelPath = options.kernelPath ?? DEFAULT_DOPPLER_KERNEL_PATH;
  const resolvedBatchSize = parsePositiveInt(options.batchSize, DEFAULT_DOPPLER_BATCH_SIZE, '--doppler-batch-size');
  const resolvedReadbackInterval = parsePositiveInt(
    options.readbackInterval,
    DEFAULT_DOPPLER_READBACK_INTERVAL,
    '--doppler-readback-interval'
  );
  const resolvedBrowserPort = parseNonNegativeInt(
    options.browserPort,
    DEFAULT_DOPPLER_BROWSER_PORT,
    '--doppler-browser-port'
  );
  const runtimeConfig = buildDopplerRuntimeConfig(sharedContract, {
    kernelPath: resolvedKernelPath,
    batchSize: resolvedBatchSize,
    readbackInterval: resolvedReadbackInterval,
  });
  const baseArgs = [
    path.join(DOPPLER_ROOT, 'tools', 'doppler-cli.js'),
    'bench',
    '--model-id', modelId,
    '--model-url', modelUrl,
    '--json',
    '--cache-mode', cacheMode,
    '--browser-port', String(resolvedBrowserPort),
    '--runtime-config-json', JSON.stringify(runtimeConfig),
  ];
  if (options.noOpfsCache) {
    baseArgs.push('--no-opfs-cache');
  }
  if (options.browserUserData) {
    baseArgs.push('--browser-user-data', String(options.browserUserData));
  }

  console.error(`[compare] running Doppler (${cacheMode})...`);
  const runOnce = async (extraArgs = []) => {
    const args = [...baseArgs, ...extraArgs];
    const { stdout } = await execFileAsync('node', args, {
      cwd: DOPPLER_ROOT,
      timeout: 600_000,
      maxBuffer: 10 * 1024 * 1024,
    });
    const jsonMatch = stdout.match(/\{[\s\S]*\}/);
    if (!jsonMatch) throw new Error('No JSON in Doppler output');
    return JSON.parse(jsonMatch[0]);
  };

  try {
    return await runOnce();
  } catch (error) {
    const message = String(error?.message || '');
    const shouldRetryNoOpfs = !options.noOpfsCache && message.includes('Invalid manifest');
    if (shouldRetryNoOpfs) {
      console.error('[compare] Doppler failed with cached manifest mismatch; retrying with --no-opfs-cache...');
      try {
        return await runOnce(['--no-opfs-cache']);
      } catch (retryError) {
        console.error(`[compare] Doppler (${cacheMode}) retry failed: ${retryError.message}`);
        return toFailurePayload('doppler', retryError);
      }
    }
    console.error(`[compare] Doppler (${cacheMode}) failed: ${error.message}`);
    return toFailurePayload('doppler', error);
  }
}

async function runTjs(modelId, sharedContract, cacheMode, tjsVersion, localModelPath, options = {}) {
  const resolvedProfileOps = options.profileOps ?? DEFAULT_TJS_PROFILE_OPS;
  const resolvedTimeoutMs = parsePositiveInt(options.timeoutMs, DEFAULT_TJS_TIMEOUT_MS, '--tjs-timeout-ms');
  const resolvedServerPort = parseNonNegativeInt(options.serverPort, DEFAULT_TJS_SERVER_PORT, '--tjs-server-port');
  const args = [
    path.join(DOPPLER_ROOT, 'external', 'transformersjs-bench.mjs'),
    '--model', modelId,
    '--prompt', String(sharedContract.prompt),
    '--max-tokens', String(sharedContract.maxTokens),
    '--warmup', String(sharedContract.warmupRuns),
    '--runs', String(sharedContract.timedRuns),
    '--cache-mode', cacheMode,
    '--tjs-version', tjsVersion,
    '--profile-ops', resolvedProfileOps ? 'on' : 'off',
    '--timeout', String(resolvedTimeoutMs),
    '--server-port', String(resolvedServerPort),
    '--seed', String(sharedContract.seed),
    '--temperature', String(sharedContract.sampling.temperature),
    '--top-k', String(sharedContract.sampling.topK),
    '--top-p', String(sharedContract.sampling.topP),
  ];
  if (localModelPath) args.push('--local-model-path', localModelPath);
  if (options.browserConsole === true) args.push('--browser-console');

  console.error(`[compare] running TJS v${tjsVersion} (${cacheMode})...`);
  const runOnce = async (overrideArgs = []) => {
    const { stdout } = await execFileAsync('node', [...args, ...overrideArgs], {
      cwd: DOPPLER_ROOT,
      timeout: resolvedTimeoutMs,
      maxBuffer: 10 * 1024 * 1024,
    });
    const jsonMatch = stdout.match(/\{[\s\S]*\}/);
    if (!jsonMatch) throw new Error('No JSON in TJS output');
    return JSON.parse(jsonMatch[0]);
  };
  try {
    return await runOnce();
  } catch (error) {
    const message = String(error?.message || '');
    const pageClosed = /Target page, context or browser has been closed/i.test(message);
    if (pageClosed) {
      const retryArgs = resolvedProfileOps ? ['--profile-ops', 'off'] : [];
      console.error(
        resolvedProfileOps
          ? '[compare] TJS closed page/context during profiled run; retrying with --profile-ops off...'
          : '[compare] TJS closed page/context; retrying once...'
      );
      try {
        return await runOnce(retryArgs);
      } catch (retryError) {
        console.error(`[compare] TJS (${cacheMode}) retry failed: ${retryError.message}`);
        return toFailurePayload('transformers.js', retryError);
      }
    }
    console.error(`[compare] TJS (${cacheMode}) failed: ${error.message}`);
    return toFailurePayload('transformers.js', error);
  }
}

function getDopplerMetric(result, key) {
  // Doppler CLI outputs { ok, surface, request, result: { metrics: { ... } } }
  const m = result?.result?.metrics || result?.metrics;
  if (!m) return null;
  const prefillTokens = Number.isFinite(m.avgPrefillTokens)
    ? m.avgPrefillTokens
    : (Number.isFinite(m?.tokens?.prefill?.median) ? m.tokens.prefill.median : null);
  const prefillTokPerSecTtft = Number.isFinite(m.medianPrefillTokensPerSecTtft)
    ? m.medianPrefillTokensPerSecTtft
    : (Number.isFinite(prefillTokens) && Number.isFinite(m.medianTtftMs) && m.medianTtftMs > 0
      ? (prefillTokens / m.medianTtftMs) * 1000
      : null);
  const map = {
    decodeTokPerSec: m.medianDecodeTokensPerSec,
    prefillTokPerSec: prefillTokPerSecTtft,
    ttftMs: m.medianTtftMs,
    modelLoadMs: m.modelLoadMs,
  };
  return map[key] ?? null;
}

function getTjsMetric(result, key) {
  const m = result?.metrics;
  if (!m) return null;
  const map = {
    decodeTokPerSec: m.decode_tokens_per_sec,
    prefillTokPerSec: m.prefill_tokens_per_sec_ttft ?? m.prefill_tokens_per_sec,
    ttftMs: m.ttft_ms,
    modelLoadMs: m.model_load_ms,
  };
  return map[key] ?? null;
}

function formatVal(v, unit) {
  if (v == null || !Number.isFinite(v)) return '-';
  if (unit === 'ms') {
    if (v >= 1000) return `${(v / 1000).toFixed(1)}s`;
    return `${v.toFixed(0)}ms`;
  }
  if (unit === 'tok/s') return `${v.toFixed(1)}`;
  return String(v);
}

function formatDelta(dopplerVal, tjsVal, higherBetter) {
  if (dopplerVal == null || tjsVal == null || !Number.isFinite(dopplerVal) || !Number.isFinite(tjsVal)) return '-';
  if (tjsVal === 0 && dopplerVal === 0) return 'same';
  const ref = Math.min(Math.abs(dopplerVal), Math.abs(tjsVal));
  if (ref === 0) return '-';
  // Express as "X wins by Y%"
  const pct = Math.abs(dopplerVal - tjsVal) / ref * 100;
  const dopplerWins = higherBetter ? dopplerVal > tjsVal : dopplerVal < tjsVal;
  const winner = dopplerWins ? 'Doppler' : 'TJS';
  return `${pct.toFixed(0)}% ${winner}`;
}

function printRow(label, dopplerVal, tjsVal, unit, higherBetter) {
  const dStr = formatVal(dopplerVal, unit);
  const tStr = formatVal(tjsVal, unit);
  const delta = formatDelta(dopplerVal, tjsVal, higherBetter);
  console.log(`  ${label.padEnd(20)} ${dStr.padStart(14)} ${tStr.padStart(14)} ${delta.padStart(18)}`);
}

function printSection(title, dopplerResult, tjsResult, rows) {
  console.log(`\n=== ${title} ===`);
  console.log(`  ${''.padEnd(20)} ${'Doppler'.padStart(14)} ${'TJS'.padStart(14)} ${'delta'.padStart(18)}`);
  for (const row of rows) {
    const dVal = getDopplerMetric(dopplerResult, row.key);
    const tVal = getTjsMetric(tjsResult, row.key);
    printRow(row.label, dVal, tVal, row.unit, row.higherBetter);
  }
  // UX timing rows (from ux sub-object) if available
  if (dopplerResult?.ux || tjsResult?.ux) {
    const dUx = dopplerResult?.ux || {};
    const tUx = tjsResult?.ux || {};
    if (dUx.firstResponseMs != null || tUx.firstResponseMs != null) {
      printRow('first token (e2e)', dUx.firstResponseMs, tUx.firstResponseMs, 'ms', false);
    }
  }
}

function compactTimestamp() {
  const d = new Date();
  const pad = (n, w = 2) => String(n).padStart(w, '0');
  return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}T${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
}

async function main() {
  const flags = parseArgs(process.argv.slice(2));
  const workloadCatalog = await loadWorkloadCatalog();
  const defaultWorkloadId = workloadCatalog.defaultWorkloadId ?? FALLBACK_DEFAULT_WORKLOAD_ID;
  resolveWorkloadById(workloadCatalog, defaultWorkloadId);
  const useDefaultWorkload = flags.workload == null
    && flags.prompt == null
    && flags['prefill-tokens'] == null
    && flags['max-tokens'] == null;
  const workloadId = flags.workload ?? (useDefaultWorkload ? defaultWorkloadId : null);
  const workload = workloadId ? resolveWorkloadById(workloadCatalog, workloadId) : null;
  const prefillTokenTarget = parsePositiveInt(
    flags['prefill-tokens'],
    workload?.prefillTokens ?? null,
    '--prefill-tokens'
  );
  const workloadSampling = workload?.sampling || DEFAULT_SHARED_SAMPLING;
  const sampling = {
    temperature: parseNonNegativeNumber(flags.temperature, workloadSampling.temperature, '--temperature'),
    topK: parsePositiveInt(flags['top-k'], workloadSampling.topK, '--top-k'),
    topP: parseTopP(flags['top-p'], workloadSampling.topP, '--top-p'),
  };
  const promptInput = flags.prompt
    || (Number.isFinite(prefillTokenTarget) ? buildSyntheticPrompt(prefillTokenTarget) : DEFAULT_PROMPT);
  const maxTokensInput = flags['max-tokens'] ?? workload?.decodeTokens ?? DEFAULT_MAX_TOKENS;

  const dopplerModelId = flags['model-id'] || DEFAULT_DOPPLER_MODEL;
  const dopplerModelUrl = flags['model-url'] || `/models/local/${dopplerModelId}`;
  const tjsModelId = flags['tjs-model'] || DEFAULT_TJS_MODEL;
  const tjsLocalModelPath = flags['tjs-local-model-path'] || null;
  const tjsVersion = flags['tjs-version'] || '3';
  const mode = flags.mode || 'all';
  const sharedContract = buildSharedBenchmarkContract({
    prompt: promptInput,
    maxTokens: maxTokensInput,
    warmupRuns: flags.warmup,
    timedRuns: flags.runs,
    seed: flags.seed,
    sampling,
  });
  const prompt = sharedContract.prompt;
  const maxTokens = sharedContract.maxTokens;
  const warmupRuns = sharedContract.warmupRuns;
  const runs = sharedContract.timedRuns;
  const seed = sharedContract.seed;
  const decodeProfile = parseDecodeProfile(flags['decode-profile']);
  const tjsProfileOps = parseOnOff(flags['tjs-profile-ops'], DEFAULT_TJS_PROFILE_OPS, '--tjs-profile-ops');
  const tjsTimeoutMs = parsePositiveInt(flags['tjs-timeout-ms'], DEFAULT_TJS_TIMEOUT_MS, '--tjs-timeout-ms');
  const tjsServerPort = parseNonNegativeInt(flags['tjs-server-port'], DEFAULT_TJS_SERVER_PORT, '--tjs-server-port');
  const tjsBrowserConsole = flags['tjs-browser-console'] === true;
  const hasCustomDopplerBatchSize = flags['doppler-batch-size'] != null;
  const hasCustomDopplerReadbackInterval = flags['doppler-readback-interval'] != null;
  const hasCustomDopplerDecodeTuning = hasCustomDopplerBatchSize || hasCustomDopplerReadbackInterval;
  if (hasCustomDopplerDecodeTuning && decodeProfile !== 'custom') {
    throw new Error(
      'Use --decode-profile custom when setting --doppler-batch-size or --doppler-readback-interval.'
    );
  }
  const decodeProfilePreset = DECODE_PROFILE_PRESETS[decodeProfile] || DECODE_PROFILE_PRESETS[DEFAULT_DECODE_PROFILE];
  const dopplerKernelPathOverride = flags['doppler-kernel-path'] ?? DEFAULT_DOPPLER_KERNEL_PATH;
  const dopplerKernelResolution = resolveDopplerKernelPath(dopplerModelId, dopplerKernelPathOverride);
  const dopplerKernelPath = dopplerKernelResolution.kernelPath;
  const dopplerBatchSize = parsePositiveInt(
    flags['doppler-batch-size'],
    decodeProfilePreset.batchSize,
    '--doppler-batch-size'
  );
  const dopplerReadbackInterval = parsePositiveInt(
    flags['doppler-readback-interval'],
    decodeProfilePreset.readbackInterval,
    '--doppler-readback-interval'
  );
  const dopplerTokensPerReadback = dopplerBatchSize * dopplerReadbackInterval;
  const dopplerNoOpfsCache = flags['doppler-no-opfs-cache'] === true;
  const dopplerBrowserUserData = flags['doppler-browser-user-data'] || null;
  const dopplerBrowserPort = parseNonNegativeInt(
    flags['doppler-browser-port'],
    DEFAULT_DOPPLER_BROWSER_PORT,
    '--doppler-browser-port'
  );
  const jsonOutput = flags.json === true;
  const shouldSave = flags.save === true;
  const saveDir = flags['save-dir'] || path.join(DOPPLER_ROOT, 'bench-results');

  const validModes = ['compute', 'cold', 'warm', 'all'];
  if (!validModes.includes(mode)) {
    console.error(`Invalid --mode "${mode}". Must be one of: ${validModes.join(', ')}`);
    process.exit(1);
  }

  console.error(`[compare] Doppler model: ${dopplerModelId}`);
  console.error(`[compare] TJS model:     ${tjsModelId} (v${tjsVersion})`);
  console.error(
    `[compare] Doppler kernel path: ${dopplerKernelPath ?? 'manifest-default'} `
    + `(${dopplerKernelResolution.source})`
  );
  console.error(
    `[compare] mode: ${mode}, maxTokens: ${maxTokens}, warmupRuns: ${warmupRuns}, runs: ${runs}, `
    + `decodeProfile: ${decodeProfile}, workload=${workloadId ?? 'none'}, `
    + `sampling=(temp=${sharedContract.sampling.temperature}, topK=${sharedContract.sampling.topK}, topP=${sharedContract.sampling.topP}), `
    + `dopplerBatchSize: ${dopplerBatchSize}, `
    + `dopplerReadbackInterval: ${dopplerReadbackInterval}, dopplerTokensPerReadback: ${dopplerTokensPerReadback}`
  );

  const report = {
    timestamp: new Date().toISOString(),
    dopplerModelId,
    dopplerKernelPath: dopplerKernelPath ?? 'manifest-default',
    dopplerKernelPathSource: dopplerKernelResolution.source,
    decodeProfile,
    dopplerBatchSize,
    dopplerReadbackInterval,
    dopplerTokensPerReadback,
    tjsModelId,
    mode,
    prompt,
    maxTokens,
    warmupRuns,
    runs,
    seed,
    workload: {
      id: workloadId,
      prefillTokenTarget: prefillTokenTarget ?? null,
      decodeTokenTarget: maxTokens,
    },
    methodology: {
      prefillTokensPerSec: 'prompt_tokens / ttft_ms',
      deterministicDecoding: {
        seed,
        temperature: sharedContract.sampling.temperature,
        topK: sharedContract.sampling.topK,
        topP: sharedContract.sampling.topP,
      },
      promptParity: {
        dopplerChatTemplateEnabled: sharedContract.useChatTemplate,
        transformersChatTemplateEquivalent: 'raw-prompt',
      },
      cacheSemantics: {
        warm: 'Reuse engine-specific persistent cache state (Doppler OPFS/browser profile; TJS persistent browser profile).',
        cold: 'Wipe engine-specific persistent cache state before run (Doppler OPFS/profile wipe; TJS profile wipe).',
      },
      dopplerDecodeCadence: {
        batchSize: dopplerBatchSize,
        readbackInterval: dopplerReadbackInterval,
        tokensPerReadback: dopplerTokensPerReadback,
      },
      transformersjsDecodeCadence: {
        streamerCallbackGranularityTokens: 1,
        readbackControl: 'runtime-internal',
      },
    },
    sections: {},
  };

  // Compute measures warm-cache behavior and reports both parity/throughput Doppler cadence.
  const needCompute = mode === 'compute' || mode === 'all';
  const needWarm = mode === 'warm' || mode === 'all';
  const needCold = mode === 'cold' || mode === 'all';

  let dopplerComputeParity = null;
  let dopplerComputeThroughput = null;
  let tjsCompute = null;
  let dopplerWarm = null;
  let tjsWarm = null;
  let dopplerCold = null;
  let tjsCold = null;

  if (needCompute) {
    tjsCompute = await runTjs(
      tjsModelId,
      sharedContract,
      'warm',
      tjsVersion,
      tjsLocalModelPath,
      {
        profileOps: tjsProfileOps,
        timeoutMs: tjsTimeoutMs,
        serverPort: tjsServerPort,
        browserConsole: tjsBrowserConsole,
      }
    );
    dopplerComputeParity = await runDoppler(
      dopplerModelId,
      dopplerModelUrl,
      sharedContract,
      'warm',
      {
        kernelPath: dopplerKernelPath,
        batchSize: DECODE_PROFILE_PRESETS.parity.batchSize,
        readbackInterval: DECODE_PROFILE_PRESETS.parity.readbackInterval,
        noOpfsCache: dopplerNoOpfsCache,
        browserUserData: dopplerBrowserUserData,
        browserPort: dopplerBrowserPort,
      }
    );
    dopplerComputeThroughput = await runDoppler(
      dopplerModelId,
      dopplerModelUrl,
      sharedContract,
      'warm',
      {
        kernelPath: dopplerKernelPath,
        batchSize: DECODE_PROFILE_PRESETS.throughput.batchSize,
        readbackInterval: DECODE_PROFILE_PRESETS.throughput.readbackInterval,
        noOpfsCache: dopplerNoOpfsCache,
        browserUserData: dopplerBrowserUserData,
        browserPort: dopplerBrowserPort,
      }
    );
    report.sections.compute = {
      parity: { doppler: dopplerComputeParity, tjs: tjsCompute },
      throughput: { doppler: dopplerComputeThroughput, tjs: tjsCompute },
    };
  }

  if (needWarm) {
    dopplerWarm = await runDoppler(
      dopplerModelId,
      dopplerModelUrl,
      sharedContract,
      'warm',
      {
        kernelPath: dopplerKernelPath,
        batchSize: dopplerBatchSize,
        readbackInterval: dopplerReadbackInterval,
        noOpfsCache: dopplerNoOpfsCache,
        browserUserData: dopplerBrowserUserData,
        browserPort: dopplerBrowserPort,
      }
    );
    tjsWarm = await runTjs(
      tjsModelId,
      sharedContract,
      'warm',
      tjsVersion,
      tjsLocalModelPath,
      {
        profileOps: tjsProfileOps,
        timeoutMs: tjsTimeoutMs,
        serverPort: tjsServerPort,
        browserConsole: tjsBrowserConsole,
      }
    );
    report.sections.warm = { doppler: dopplerWarm, tjs: tjsWarm };
  }

  if (needCold) {
    dopplerCold = await runDoppler(
      dopplerModelId,
      dopplerModelUrl,
      sharedContract,
      'cold',
      {
        kernelPath: dopplerKernelPath,
        batchSize: dopplerBatchSize,
        readbackInterval: dopplerReadbackInterval,
        noOpfsCache: dopplerNoOpfsCache,
        browserUserData: dopplerBrowserUserData,
        browserPort: dopplerBrowserPort,
      }
    );
    tjsCold = await runTjs(
      tjsModelId,
      sharedContract,
      'cold',
      tjsVersion,
      tjsLocalModelPath,
      {
        profileOps: tjsProfileOps,
        timeoutMs: tjsTimeoutMs,
        serverPort: tjsServerPort,
        browserConsole: tjsBrowserConsole,
      }
    );
    report.sections.cold = { doppler: dopplerCold, tjs: tjsCold };
  }

  if (jsonOutput) {
    console.log(JSON.stringify(report, null, 2));
  } else {
    const decodeProfileLabel = decodeProfilePreset?.label || 'custom decode cadence';
    console.log(
      `[method] prefill=prompt_tokens/TTFT, decodeProfile=${decodeProfile} ` +
      `(${decodeProfileLabel}), Doppler tokens/readback=${dopplerTokensPerReadback}`
    );
    const computeRows = [
      { key: 'modelLoadMs', label: 'model load', unit: 'ms', higherBetter: false },
      { key: 'decodeTokPerSec', label: 'decode tok/s', unit: 'tok/s', higherBetter: true },
      { key: 'prefillTokPerSec', label: 'prompt tok/s (TTFT)', unit: 'tok/s', higherBetter: true },
      { key: 'ttftMs', label: 'TTFT', unit: 'ms', higherBetter: false },
    ];

    if (mode === 'compute' || mode === 'all') {
      printSection('COMPUTE (PARITY)', dopplerComputeParity, tjsCompute, computeRows);
      printSection('COMPUTE (THROUGHPUT)', dopplerComputeThroughput, tjsCompute, computeRows);
    }
    if (mode === 'warm' || mode === 'all') {
      printSection('WARM START', dopplerWarm, tjsWarm, computeRows);
    }
    if (mode === 'cold' || mode === 'all') {
      printSection('COLD START', dopplerCold, tjsCold, computeRows);
    }
    console.log('');
  }

  if (shouldSave) {
    await fs.mkdir(saveDir, { recursive: true });
    const ts = compactTimestamp();
    const filename = `compare_${ts}.json`;
    const filePath = path.join(saveDir, filename);
    await fs.writeFile(filePath, JSON.stringify(report, null, 2), 'utf-8');
    await fs.writeFile(path.join(saveDir, 'compare_latest.json'), JSON.stringify(report, null, 2), 'utf-8');
    console.error(`[compare] saved to ${filePath}`);
  }
}

main().catch((error) => {
  console.error(`[compare] ${error.message}`);
  process.exit(1);
});
