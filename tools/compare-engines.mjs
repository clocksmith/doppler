#!/usr/bin/env node

/**
 * Unified cross-engine benchmark comparison.
 *
 * Runs both Doppler and Transformers.js benchmarks and produces
 * a structured side-by-side comparison across three modes:
 *   - compute: decode/prefill/TTFT (raw engine performance)
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
 *   --prompt <text>        Prompt used for both engines (default: synthetic 64-word prompt)
 *   --mode <mode>          compute|cold|warm|all (default: all)
 *   --max-tokens <n>       Max new tokens (default: 64)
 *   --warmup <n>           Warmup runs per engine (default: 1)
 *   --runs <n>             Timed runs per engine (default: 3)
 *   --doppler-kernel-path <id>  Doppler kernel path override (default: gemma3-f16-f16a-online)
 *   --doppler-no-opfs-cache  Disable Doppler OPFS cache for browser runs
 *   --doppler-browser-user-data <path>  Doppler Chromium profile dir
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

const DEFAULT_DOPPLER_MODEL = 'gemma-3-1b-it-wf16';
const DEFAULT_TJS_MODEL = 'onnx-community/gemma-3-1b-it-ONNX-GQA';
const DEFAULT_PREFILL_WORDS = 64;
const DEFAULT_PROMPT = Array.from({ length: DEFAULT_PREFILL_WORDS }, (_, i) => `word${i}`).join(' ');
const DEFAULT_MAX_TOKENS = 64;
const DEFAULT_WARMUP = 1;
const DEFAULT_RUNS = 3;
const DEFAULT_DOPPLER_KERNEL_PATH = 'gemma3-f16-f16a-online';

function parseArgs(argv) {
  const flags = {};
  for (let i = 0; i < argv.length; i++) {
    const token = argv[i];
    if (!token.startsWith('--')) continue;
    const key = token.slice(2);
    if (key === 'json' || key === 'save' || key === 'doppler-no-opfs-cache') {
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

async function runDoppler(modelId, modelUrl, prompt, maxTokens, warmupRuns, runs, cacheMode, options = {}) {
  const resolvedPrompt = prompt || DEFAULT_PROMPT;
  const resolvedMaxTokens = maxTokens || DEFAULT_MAX_TOKENS;
  const resolvedWarmupRuns = warmupRuns || DEFAULT_WARMUP;
  const resolvedTimedRuns = runs || DEFAULT_RUNS;
  const resolvedKernelPath = options.kernelPath || DEFAULT_DOPPLER_KERNEL_PATH;
  const baseArgs = [
    path.join(DOPPLER_ROOT, 'tools', 'doppler-cli.js'),
    'bench',
    '--model-id', modelId,
    '--model-url', modelUrl,
    '--json',
    '--cache-mode', cacheMode,
    '--runtime-config-json', JSON.stringify({
      shared: {
        benchmark: {
          run: {
            customPrompt: resolvedPrompt,
            maxNewTokens: resolvedMaxTokens,
            warmupRuns: resolvedWarmupRuns,
            timedRuns: resolvedTimedRuns,
            sampling: {
              temperature: 0,
              topK: 1,
              topP: 1,
            },
          },
        },
      },
      inference: {
        kernelPath: resolvedKernelPath,
        prompt: resolvedPrompt,
        batching: {
          maxTokens: resolvedMaxTokens,
        },
        sampling: {
          temperature: 0,
          topK: 1,
          topP: 1,
        },
      },
    }),
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
        return null;
      }
    }
    console.error(`[compare] Doppler (${cacheMode}) failed: ${error.message}`);
    return null;
  }
}

async function runTjs(modelId, prompt, maxTokens, warmupRuns, runs, cacheMode, tjsVersion, localModelPath) {
  const args = [
    path.join(DOPPLER_ROOT, 'external', 'transformersjs-bench.mjs'),
    '--model', modelId,
    '--prompt', String(prompt || DEFAULT_PROMPT),
    '--max-tokens', String(maxTokens || DEFAULT_MAX_TOKENS),
    '--warmup', String(warmupRuns || DEFAULT_WARMUP),
    '--runs', String(runs || DEFAULT_RUNS),
    '--cache-mode', cacheMode,
    '--tjs-version', tjsVersion,
  ];
  if (localModelPath) args.push('--local-model-path', localModelPath);

  console.error(`[compare] running TJS v${tjsVersion} (${cacheMode})...`);
  try {
    const { stdout } = await execFileAsync('node', args, {
      cwd: DOPPLER_ROOT,
      timeout: 600_000,
      maxBuffer: 10 * 1024 * 1024,
    });
    const jsonMatch = stdout.match(/\{[\s\S]*\}/);
    if (!jsonMatch) throw new Error('No JSON in TJS output');
    return JSON.parse(jsonMatch[0]);
  } catch (error) {
    console.error(`[compare] TJS (${cacheMode}) failed: ${error.message}`);
    return null;
  }
}

function getDopplerMetric(result, key) {
  // Doppler CLI outputs { ok, surface, request, result: { metrics: { ... } } }
  const m = result?.result?.metrics || result?.metrics;
  if (!m) return null;
  const map = {
    decodeTokPerSec: m.medianDecodeTokensPerSec,
    prefillTokPerSec: m.medianPrefillTokensPerSec,
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
    prefillTokPerSec: m.prefill_tokens_per_sec,
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
  const dopplerModelId = flags['model-id'] || DEFAULT_DOPPLER_MODEL;
  const dopplerModelUrl = flags['model-url'] || `/models/local/${dopplerModelId}`;
  const tjsModelId = flags['tjs-model'] || DEFAULT_TJS_MODEL;
  const tjsLocalModelPath = flags['tjs-local-model-path'] || null;
  const tjsVersion = flags['tjs-version'] || '3';
  const mode = flags.mode || 'all';
  const prompt = flags.prompt || DEFAULT_PROMPT;
  const maxTokens = Number(flags['max-tokens'] || DEFAULT_MAX_TOKENS);
  const warmupRuns = Number(flags.warmup || DEFAULT_WARMUP);
  const runs = Number(flags.runs || DEFAULT_RUNS);
  const dopplerKernelPath = flags['doppler-kernel-path'] || DEFAULT_DOPPLER_KERNEL_PATH;
  const dopplerNoOpfsCache = flags['doppler-no-opfs-cache'] === true;
  const dopplerBrowserUserData = flags['doppler-browser-user-data'] || null;
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
  console.error(`[compare] mode: ${mode}, maxTokens: ${maxTokens}, warmupRuns: ${warmupRuns}, runs: ${runs}`);

  const report = {
    timestamp: new Date().toISOString(),
    dopplerModelId,
    dopplerKernelPath,
    tjsModelId,
    mode,
    prompt,
    maxTokens,
    warmupRuns,
    runs,
    sections: {},
  };

  // Warm runs serve double duty: compute metrics come from warm runs
  const needWarm = mode === 'warm' || mode === 'all' || mode === 'compute';
  const needCold = mode === 'cold' || mode === 'all';

  let dopplerWarm = null;
  let tjsWarm = null;
  let dopplerCold = null;
  let tjsCold = null;

  if (needWarm) {
    dopplerWarm = await runDoppler(
      dopplerModelId,
      dopplerModelUrl,
      prompt,
      maxTokens,
      warmupRuns,
      runs,
      'warm',
      {
        kernelPath: dopplerKernelPath,
        noOpfsCache: dopplerNoOpfsCache,
        browserUserData: dopplerBrowserUserData,
      }
    );
    tjsWarm = await runTjs(
      tjsModelId,
      prompt,
      maxTokens,
      warmupRuns,
      runs,
      'warm',
      tjsVersion,
      tjsLocalModelPath
    );
    report.sections.warm = { doppler: dopplerWarm, tjs: tjsWarm };
  }

  if (needCold) {
    dopplerCold = await runDoppler(
      dopplerModelId,
      dopplerModelUrl,
      prompt,
      maxTokens,
      warmupRuns,
      runs,
      'cold',
      {
        kernelPath: dopplerKernelPath,
        noOpfsCache: dopplerNoOpfsCache,
        browserUserData: dopplerBrowserUserData,
      }
    );
    tjsCold = await runTjs(
      tjsModelId,
      prompt,
      maxTokens,
      warmupRuns,
      runs,
      'cold',
      tjsVersion,
      tjsLocalModelPath
    );
    report.sections.cold = { doppler: dopplerCold, tjs: tjsCold };
  }

  if (jsonOutput) {
    console.log(JSON.stringify(report, null, 2));
  } else {
    const computeRows = [
      { key: 'decodeTokPerSec', label: 'decode tok/s', unit: 'tok/s', higherBetter: true },
      { key: 'prefillTokPerSec', label: 'prefill tok/s', unit: 'tok/s', higherBetter: true },
      { key: 'ttftMs', label: 'TTFT', unit: 'ms', higherBetter: false },
    ];
    const loadRows = [
      { key: 'modelLoadMs', label: 'model load', unit: 'ms', higherBetter: false },
      ...computeRows,
    ];

    if (mode === 'compute' || mode === 'all') {
      printSection('COMPUTE', dopplerWarm, tjsWarm, computeRows);
    }
    if (mode === 'warm' || mode === 'all') {
      printSection('WARM START', dopplerWarm, tjsWarm, loadRows);
    }
    if (mode === 'cold' || mode === 'all') {
      printSection('COLD START', dopplerCold, tjsCold, loadRows);
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
