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
 *   --tjs-model <id>       TJS model ID (default: onnx-community/gemma-3-1b-it-ONNX-GQA)
 *   --tjs-version <3|4>    Transformers.js version (default: 3)
 *   --mode <mode>          compute|cold|warm|all (default: all)
 *   --max-tokens <n>       Max new tokens (default: 128)
 *   --runs <n>             Timed runs per engine (default: 3)
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
const DEFAULT_MAX_TOKENS = 128;
const DEFAULT_RUNS = 3;

function parseArgs(argv) {
  const flags = {};
  for (let i = 0; i < argv.length; i++) {
    const token = argv[i];
    if (!token.startsWith('--')) continue;
    const key = token.slice(2);
    if (key === 'json' || key === 'save') {
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

async function runDoppler(modelId, maxTokens, runs, cacheMode) {
  const args = [
    path.join(DOPPLER_ROOT, 'tools', 'doppler-cli.js'),
    'bench',
    '--model-id', modelId,
    '--json',
    '--cache-mode', cacheMode,
  ];
  if (maxTokens) args.push('--runtime-config-json', JSON.stringify({ inference: { maxNewTokens: maxTokens } }));
  if (runs) args.push('--runtime-config-json', JSON.stringify({ bench: { timedRuns: runs } }));

  console.error(`[compare] running Doppler (${cacheMode})...`);
  try {
    const { stdout } = await execFileAsync('node', args, {
      cwd: DOPPLER_ROOT,
      timeout: 600_000,
      maxBuffer: 10 * 1024 * 1024,
    });
    // Extract JSON from stdout (skip any non-JSON lines)
    const jsonMatch = stdout.match(/\{[\s\S]*\}/);
    if (!jsonMatch) throw new Error('No JSON in Doppler output');
    return JSON.parse(jsonMatch[0]);
  } catch (error) {
    console.error(`[compare] Doppler (${cacheMode}) failed: ${error.message}`);
    return null;
  }
}

async function runTjs(modelId, maxTokens, runs, cacheMode, tjsVersion) {
  const args = [
    path.join(DOPPLER_ROOT, 'external', 'transformersjs-bench.mjs'),
    '--model', modelId,
    '--max-tokens', String(maxTokens || DEFAULT_MAX_TOKENS),
    '--runs', String(runs || DEFAULT_RUNS),
    '--cache-mode', cacheMode,
    '--tjs-version', tjsVersion,
  ];

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
  const m = result?.metrics;
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
  const tjsModelId = flags['tjs-model'] || DEFAULT_TJS_MODEL;
  const tjsVersion = flags['tjs-version'] || '3';
  const mode = flags.mode || 'all';
  const maxTokens = Number(flags['max-tokens'] || DEFAULT_MAX_TOKENS);
  const runs = Number(flags.runs || DEFAULT_RUNS);
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
  console.error(`[compare] mode: ${mode}, maxTokens: ${maxTokens}, runs: ${runs}`);

  const report = {
    timestamp: new Date().toISOString(),
    dopplerModelId,
    tjsModelId,
    mode,
    maxTokens,
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
    dopplerWarm = await runDoppler(dopplerModelId, maxTokens, runs, 'warm');
    tjsWarm = await runTjs(tjsModelId, maxTokens, runs, 'warm', tjsVersion);
    report.sections.warm = { doppler: dopplerWarm, tjs: tjsWarm };
  }

  if (needCold) {
    dopplerCold = await runDoppler(dopplerModelId, maxTokens, runs, 'cold');
    tjsCold = await runTjs(tjsModelId, maxTokens, runs, 'cold', tjsVersion);
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
