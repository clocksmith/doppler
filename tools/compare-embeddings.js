#!/usr/bin/env node

import { execFile } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { promisify } from 'node:util';
import { buildEntryRemoteBaseUrl } from '../src/tooling/hf-registry-utils.js';

const execFileAsync = promisify(execFile);
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.resolve(__dirname, '..');
const CONFIG_PATH = path.join(ROOT_DIR, 'benchmarks', 'vendors', 'embedding-compare.config.json');
const CATALOG_PATH = path.join(ROOT_DIR, 'models', 'catalog.json');
const SEMANTIC_FIXTURE_PATH = path.join(ROOT_DIR, 'src', 'inference', 'fixtures', 'embedding-semantic-fixtures.json');
const RESULTS_DIR = path.join(ROOT_DIR, 'benchmarks', 'vendors', 'results');
const DOPPLER_CLI_PATH = path.join(ROOT_DIR, 'src', 'cli', 'doppler-cli.js');
const TJS_BENCH_PATH = path.join(ROOT_DIR, 'benchmarks', 'runners', 'transformersjs-bench.js');
const DEFAULT_TIMEOUT_MS = 900_000;
const DOPPLER_SOURCES = Object.freeze(['local', 'quickstart-registry']);
const TJS_DTYPES = Object.freeze(['fp32', 'fp16', 'q8', 'q4', 'q4f16']);
const TJS_FORMATS = Object.freeze(['onnx', 'safetensors']);

function usage() {
  return [
    'Usage: node tools/compare-embeddings.js [options]',
    '',
    'Options:',
    '  --model-id <id>              Doppler model ID (default: first embedding profile)',
    '  --all                       Run every embedding profile in embedding-compare.config.json',
    '  --prompt <text>             Shared embedding prompt',
    '  --warmup <n>                Warmup runs per engine',
    '  --runs <n>                  Timed runs per engine',
    '  --doppler-source <source>   local|quickstart-registry',
    '  --doppler-surface <surface> auto|node|browser',
    '  --tjs-model <id>            Transformers.js model ID override',
    '  --tjs-dtype <dtype>         fp32|fp16|q8|q4|q4f16',
    '  --tjs-format <format>       onnx|safetensors',
    '  --cache-mode <mode>         warm|cold',
    '  --load-mode <mode>          http|opfs|memory',
    '  --timeout-ms <ms>           Per-engine command timeout',
    '  --save                      Save compare JSON under benchmarks/vendors/results/',
    '  --save-dir <dir>            Save directory override',
    '  --timestamp <iso|ms>        Deterministic timestamp override',
    '  --browser-console           Stream Transformers.js browser logs',
    '  --json                      Print JSON only',
    '  --help, -h                  Show this help text',
  ].join('\n');
}

function parseArgs(argv) {
  const flags = {};
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === '-h') {
      flags.h = true;
      continue;
    }
    if (!token.startsWith('--')) continue;
    const key = token.slice(2);
    if (key === 'help' || key === 'h' || key === 'json' || key === 'save' || key === 'all' || key === 'browser-console') {
      flags[key] = true;
      continue;
    }
    const value = argv[i + 1];
    if (value === undefined) {
      throw new Error(`Missing value for --${key}`);
    }
    flags[key] = value;
    i += 1;
  }
  return flags;
}

function parsePositiveInteger(value, fallback, label) {
  if (value == null || value === '') return fallback;
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${label} must be a positive integer`);
  }
  return parsed;
}

function parseChoice(value, allowed, fallback, label) {
  if (value == null || value === '') return fallback;
  const normalized = String(value).trim();
  if (allowed.includes(normalized)) return normalized;
  const lower = normalized.toLowerCase();
  if (allowed.includes(lower)) return lower;
  throw new Error(`${label} must be one of: ${allowed.join(', ')}`);
}

function parseTimestampValue(rawValue, label) {
  if (rawValue == null || rawValue === '') return new Date().toISOString();
  const trimmed = String(rawValue).trim();
  const asMs = /^[-+]?\d+$/.test(trimmed) ? Number(trimmed) : NaN;
  const parsed = Number.isFinite(asMs) ? new Date(asMs) : new Date(trimmed);
  if (Number.isNaN(parsed.getTime())) {
    throw new Error(`${label} must be ISO-8601 or epoch milliseconds`);
  }
  return parsed.toISOString();
}

function compactTimestamp(timestamp) {
  const d = new Date(timestamp);
  const pad = (n, w = 2) => String(n).padStart(w, '0');
  return `${d.getUTCFullYear()}${pad(d.getUTCMonth() + 1)}${pad(d.getUTCDate())}T${pad(d.getUTCHours())}${pad(d.getUTCMinutes())}${pad(d.getUTCSeconds())}`;
}

function hashText(text) {
  return crypto.createHash('sha256').update(text).digest('hex');
}

function redactSecrets(text) {
  return String(text ?? '')
    .replace(/hfToken=hf_[A-Za-z0-9._-]+/g, 'hfToken=<redacted>')
    .replace(/\bhf_[A-Za-z0-9._-]+\b/g, '<HF_TOKEN_REDACTED>')
    .replace(/(authorization:\s*bearer\s+)[^\s"'`]+/gi, '$1<redacted>');
}

function parseJsonBlock(stdout, label) {
  const text = String(stdout ?? '').trim();
  if (!text) {
    throw new Error(`${label} produced no output`);
  }
  try {
    return JSON.parse(text);
  } catch {
    const start = text.indexOf('{');
    if (start >= 0) {
      try {
        return JSON.parse(text.slice(start));
      } catch { /* fall through */ }
    }
  }
  throw new Error(`${label} did not produce strict JSON. Output tail:\n${redactSecrets(text.slice(-2000))}`);
}

async function readJsonWithHash(filePath) {
  const raw = await fs.readFile(filePath, 'utf-8');
  return {
    payload: JSON.parse(raw),
    source: filePath,
    sourceSha256: hashText(raw),
  };
}

function catalogEntryById(catalog, modelId) {
  const rows = Array.isArray(catalog?.models) ? catalog.models : [];
  return rows.find((row) => row?.modelId === modelId) ?? null;
}

function profileByModelId(config, modelId) {
  const rows = Array.isArray(config?.modelProfiles) ? config.modelProfiles : [];
  return rows.find((row) => row?.dopplerModelId === modelId) ?? null;
}

function optionalString(value) {
  return typeof value === 'string' && value.trim() !== '' ? value.trim() : null;
}

function validateProfile(profile, modelId) {
  const label = `${CONFIG_PATH} profile "${modelId}"`;
  if (profile?.compareLane !== 'performance_comparable' && profile?.compareLane !== 'capability_only') {
    throw new Error(`${label} compareLane must be performance_comparable or capability_only`);
  }
  if (typeof profile.releaseClaimable !== 'boolean') {
    throw new Error(`${label} releaseClaimable must be explicit true or false`);
  }
  if (profile.releaseClaimable === true && profile.compareLane !== 'performance_comparable') {
    throw new Error(`${label} releaseClaimable=true requires compareLane=performance_comparable`);
  }
}

function resolveTransformersjsBenchmark({ flags, profile, catalogEntry, modelId }) {
  const label = `${CONFIG_PATH} profile "${modelId}"`;
  const catalogTjs = catalogEntry?.vendorBenchmark?.transformersjs ?? null;
  const catalogRepoId = optionalString(catalogTjs?.repoId);
  const catalogDtype = optionalString(catalogTjs?.dtype);
  const profileRepoId = optionalString(profile.defaultTjsModelId);
  const profileDtype = optionalString(profile.defaultTjsDtype);
  if (catalogRepoId && profileRepoId && catalogRepoId !== profileRepoId) {
    throw new Error(`${label} defaultTjsModelId must match models/catalog.json vendorBenchmark.transformersjs.repoId`);
  }
  if (catalogDtype && profileDtype && catalogDtype !== profileDtype) {
    throw new Error(`${label} defaultTjsDtype must match models/catalog.json vendorBenchmark.transformersjs.dtype`);
  }
  const modelIdOverride = optionalString(flags['tjs-model']);
  const dtypeOverride = optionalString(flags['tjs-dtype']);
  const modelIdValue = modelIdOverride ?? catalogRepoId ?? profileRepoId;
  const dtypeValue = dtypeOverride ?? catalogDtype ?? profileDtype;
  if (!modelIdValue) {
    throw new Error(`${label} needs models/catalog.json vendorBenchmark.transformersjs.repoId or --tjs-model`);
  }
  if (!dtypeValue) {
    throw new Error(`${label} needs models/catalog.json vendorBenchmark.transformersjs.dtype or --tjs-dtype`);
  }
  return {
    modelId: modelIdValue,
    dtype: dtypeValue,
  };
}

function defaultModelId(config) {
  const first = Array.isArray(config?.modelProfiles) ? config.modelProfiles[0] : null;
  const modelId = typeof first?.dopplerModelId === 'string' ? first.dopplerModelId.trim() : '';
  if (!modelId) {
    throw new Error('embedding-compare.config.json must include at least one model profile.');
  }
  return modelId;
}

function resolveDopplerModelUrl(modelId, source, catalogEntry) {
  if (source === 'local') {
    return {
      modelUrl: pathToFileURL(path.join(ROOT_DIR, 'models', 'local', modelId)).href,
      source,
      locator: `models/local/${modelId}`,
    };
  }
  if (source === 'quickstart-registry') {
    const modelUrl = buildEntryRemoteBaseUrl(catalogEntry);
    if (!modelUrl) {
      throw new Error(`Catalog entry "${modelId}" does not have complete hosted HF coordinates.`);
    }
    return {
      modelUrl,
      source,
      locator: modelUrl,
    };
  }
  throw new Error(`Unsupported Doppler model source: ${source}`);
}

async function runNodeJson(commandArgs, label, timeoutMs) {
  const { stdout, stderr } = await execFileAsync(process.execPath, commandArgs, {
    cwd: ROOT_DIR,
    timeout: timeoutMs,
    maxBuffer: 1024 * 1024 * 128,
    env: process.env,
  }).catch((error) => {
    const stderrTail = redactSecrets(error?.stderr ?? '').slice(-4000);
    const stdoutTail = redactSecrets(error?.stdout ?? '').slice(-4000);
    throw new Error(`${label} failed: ${error.message}\nstdout:\n${stdoutTail}\nstderr:\n${stderrTail}`);
  });
  if (stderr && !String(stderr).includes('[tjs-bench]')) {
    // Keep stderr available in failure paths without making successful JSON noisy.
  }
  return parseJsonBlock(stdout, label);
}

function buildDopplerConfig({
  modelId,
  modelUrl,
  runtimeProfile,
  prompt,
  warmupRuns,
  timedRuns,
  loadMode,
  surface,
  timestamp,
}) {
  const request = {
    workload: 'embedding',
    modelId,
    modelUrl,
    runtimeConfig: {
      shared: {
        benchmark: {
          run: {
            customPrompt: prompt,
            warmupRuns,
            timedRuns,
            loadMode,
          },
        },
      },
      inference: {
        prompt,
      },
    },
  };
  if (runtimeProfile !== null) {
    request.runtimeProfile = runtimeProfile;
  }
  return {
    request: {
      ...request,
    },
    run: {
      surface,
      timestamp,
    },
  };
}

function unwrapToolingResult(payload) {
  if (payload?.ok === true && payload?.result && typeof payload.result === 'object') {
    return payload.result;
  }
  return payload;
}

function finiteNumber(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function extractDopplerSpeed(result) {
  const metrics = result?.metrics ?? {};
  return {
    embeddingDim: finiteNumber(metrics.embeddingDim),
    medianEmbeddingMs: finiteNumber(metrics.medianEmbeddingMs),
    avgEmbeddingMs: finiteNumber(metrics.avgEmbeddingMs),
    avgEmbeddingsPerSec: finiteNumber(metrics.avgEmbeddingsPerSec),
    p95EmbeddingMs: finiteNumber(metrics.p95EmbeddingMs),
    modelLoadMs: finiteNumber(metrics.modelLoadMs ?? result?.timing?.modelLoadMs),
    nonFiniteValues: finiteNumber(metrics.nonFiniteValues),
    avgEmbeddingL2Norm: finiteNumber(metrics.avgEmbeddingL2Norm),
  };
}

function extractDopplerCorrectness(result) {
  const metrics = result?.metrics ?? {};
  return {
    semanticPassed: metrics.semanticPassed === true,
    semanticRetrievalTop1Acc: finiteNumber(metrics.semanticRetrievalTop1Acc),
    semanticPairAcc: finiteNumber(metrics.semanticPairAcc),
    finiteRatio: finiteNumber(metrics.finiteRatio),
    nonFiniteValues: finiteNumber(metrics.nonFiniteValues),
    embeddingDim: finiteNumber(metrics.embeddingDim),
    embeddingL2Norm: finiteNumber(metrics.embeddingL2Norm),
  };
}

function extractTjsSpeed(result) {
  const metrics = result?.metrics ?? {};
  return {
    embeddingDim: finiteNumber(metrics.embeddingDim),
    medianEmbeddingMs: finiteNumber(metrics.medianEmbeddingMs),
    avgEmbeddingMs: finiteNumber(metrics.avgEmbeddingMs),
    avgEmbeddingsPerSec: finiteNumber(metrics.avgEmbeddingsPerSec),
    p95EmbeddingMs: finiteNumber(metrics.p95EmbeddingMs),
    modelLoadMs: finiteNumber(metrics.modelLoadMs ?? result?.timing?.modelLoadMs),
    nonFiniteValues: finiteNumber(metrics.nonFiniteValues),
    avgEmbeddingL2Norm: finiteNumber(metrics.avgEmbeddingL2Norm),
  };
}

function extractTjsCorrectness(result) {
  const semantic = result?.semantic ?? null;
  return {
    semanticPassed: semantic?.passed === true,
    semanticRetrievalTop1Acc: finiteNumber(semantic?.retrievalTop1Acc),
    semanticPairAcc: finiteNumber(semantic?.pairAcc),
    finiteRatio: result?.metrics?.nonFiniteValues === 0 ? 1 : null,
    nonFiniteValues: finiteNumber(result?.metrics?.nonFiniteValues),
    embeddingDim: finiteNumber(result?.metrics?.embeddingDim),
    embeddingL2Norm: finiteNumber(result?.metrics?.avgEmbeddingL2Norm),
  };
}

function ratio(a, b) {
  if (!Number.isFinite(a) || !Number.isFinite(b) || b === 0) return null;
  return Number((a / b).toFixed(4));
}

function buildSummary(dopplerBench, dopplerVerify, tjsBench) {
  const dopplerSpeed = extractDopplerSpeed(dopplerBench);
  const dopplerCorrectness = extractDopplerCorrectness(dopplerVerify);
  const tjsSpeed = extractTjsSpeed(tjsBench);
  const tjsCorrectness = extractTjsCorrectness(tjsBench);
  const correctnessOk = dopplerCorrectness.semanticPassed === true
    && tjsCorrectness.semanticPassed === true
    && dopplerSpeed.nonFiniteValues === 0
    && tjsSpeed.nonFiniteValues === 0;
  return {
    correctnessOk,
    doppler: {
      speed: dopplerSpeed,
      correctness: dopplerCorrectness,
    },
    transformersjs: {
      speed: tjsSpeed,
      correctness: tjsCorrectness,
    },
    ratios: {
      medianEmbeddingMsDopplerOverTjs: ratio(dopplerSpeed.medianEmbeddingMs, tjsSpeed.medianEmbeddingMs),
      avgEmbeddingsPerSecDopplerOverTjs: ratio(dopplerSpeed.avgEmbeddingsPerSec, tjsSpeed.avgEmbeddingsPerSec),
      modelLoadMsDopplerOverTjs: ratio(dopplerSpeed.modelLoadMs, tjsSpeed.modelLoadMs),
    },
  };
}

async function saveCompareResult(result, saveDir, timestamp) {
  await fs.mkdir(saveDir, { recursive: true });
  const modelSlug = String(result.model?.dopplerModelId ?? 'embedding').replace(/[^a-zA-Z0-9_-]/g, '_');
  const filename = `embedding_compare_${modelSlug}_${compactTimestamp(timestamp)}.json`;
  const filePath = path.join(saveDir, filename);
  const json = JSON.stringify(result, null, 2);
  await fs.writeFile(filePath, json, 'utf-8');
  await fs.writeFile(path.join(saveDir, 'embedding_compare_latest.json'), json, 'utf-8');
  return filePath;
}

async function runOne({ modelId, flags, configBundle, catalogBundle, timestamp }) {
  const config = configBundle.payload;
  const catalog = catalogBundle.payload;
  const profile = profileByModelId(config, modelId);
  if (!profile) {
    throw new Error(`No embedding compare profile for "${modelId}" in ${CONFIG_PATH}`);
  }
  validateProfile(profile, modelId);
  const catalogEntry = catalogEntryById(catalog, modelId);
  if (!catalogEntry) {
    throw new Error(`No model catalog entry for "${modelId}"`);
  }
  const defaults = config.defaults ?? {};
  const prompt = flags.prompt ?? defaults.prompt;
  const warmupRuns = parsePositiveInteger(flags.warmup, defaults.warmupRuns, '--warmup');
  const timedRuns = parsePositiveInteger(flags.runs, defaults.timedRuns, '--runs');
  const cacheMode = parseChoice(flags['cache-mode'], ['warm', 'cold'], defaults.cacheMode, '--cache-mode');
  const loadMode = parseChoice(flags['load-mode'], ['http', 'opfs', 'memory'], defaults.loadMode, '--load-mode');
  const dopplerSource = parseChoice(
    flags['doppler-source'],
    DOPPLER_SOURCES,
    profile.defaultDopplerSource,
    '--doppler-source'
  );
  const dopplerSurface = parseChoice(
    flags['doppler-surface'],
    ['auto', 'node', 'browser'],
    defaults.dopplerSurface,
    '--doppler-surface'
  );
  const tjsBenchmark = resolveTransformersjsBenchmark({ flags, profile, catalogEntry, modelId });
  const tjsModelId = tjsBenchmark.modelId;
  const tjsDtype = parseChoice(tjsBenchmark.dtype, TJS_DTYPES, null, '--tjs-dtype');
  const tjsFormat = parseChoice(flags['tjs-format'], TJS_FORMATS, profile.defaultTjsFormat, '--tjs-format');
  const timeoutMs = parsePositiveInteger(flags['timeout-ms'], DEFAULT_TIMEOUT_MS, '--timeout-ms');
  const dopplerLocator = resolveDopplerModelUrl(modelId, dopplerSource, catalogEntry);
  const dopplerRuntimeProfile = Object.prototype.hasOwnProperty.call(profile, 'dopplerRuntimeProfile')
    ? profile.dopplerRuntimeProfile
    : defaults.dopplerRuntimeProfile;
  const dopplerVerifyRuntimeProfile = Object.prototype.hasOwnProperty.call(profile, 'dopplerVerifyRuntimeProfile')
    ? profile.dopplerVerifyRuntimeProfile
    : defaults.dopplerVerifyRuntimeProfile;

  const dopplerBenchConfig = buildDopplerConfig({
    modelId,
    modelUrl: dopplerLocator.modelUrl,
    runtimeProfile: dopplerRuntimeProfile,
    prompt,
    warmupRuns,
    timedRuns,
    loadMode,
    surface: dopplerSurface,
    timestamp,
  });
  const dopplerVerifyConfig = buildDopplerConfig({
    modelId,
    modelUrl: dopplerLocator.modelUrl,
    runtimeProfile: dopplerVerifyRuntimeProfile,
    prompt,
    warmupRuns: 0,
    timedRuns: 1,
    loadMode,
    surface: dopplerSurface,
    timestamp,
  });

  const dopplerBenchArgs = [
    DOPPLER_CLI_PATH,
    'bench',
    '--config',
    JSON.stringify(dopplerBenchConfig),
    '--json',
  ];
  const dopplerVerifyArgs = [
    DOPPLER_CLI_PATH,
    'verify',
    '--config',
    JSON.stringify(dopplerVerifyConfig),
    '--json',
  ];
  const tjsArgs = [
    TJS_BENCH_PATH,
    '--task',
    'embedding',
    '--model',
    tjsModelId,
    '--prompt',
    prompt,
    '--warmup',
    String(warmupRuns),
    '--runs',
    String(timedRuns),
    '--cache-mode',
    cacheMode,
    '--load-mode',
    loadMode,
    '--dtype',
    tjsDtype,
    '--format',
    tjsFormat,
    '--embedding-mode',
    profile.embeddingMode,
    '--embedding-pooling',
    profile.embeddingPooling,
    '--embedding-normalize',
    profile.embeddingNormalize === false ? 'off' : 'on',
    '--embedding-style',
    profile.embeddingStyle,
    '--semantic-fixture',
    SEMANTIC_FIXTURE_PATH,
    '--profile-ops',
    'off',
    '--timeout',
    String(timeoutMs),
    '--timestamp',
    timestamp,
    '--json',
  ];
  if (flags['browser-console']) {
    tjsArgs.push('--browser-console');
  }

  const dopplerBench = unwrapToolingResult(await runNodeJson(dopplerBenchArgs, `${modelId} Doppler embedding bench`, timeoutMs));
  const dopplerVerify = unwrapToolingResult(await runNodeJson(dopplerVerifyArgs, `${modelId} Doppler embedding verify`, timeoutMs));
  const tjsBench = await runNodeJson(tjsArgs, `${modelId} Transformers.js embedding bench`, timeoutMs);
  const summary = buildSummary(dopplerBench, dopplerVerify, tjsBench);

  const result = {
    schemaVersion: 1,
    kind: 'embedding-engine-compare',
    timestamp,
    model: {
      dopplerModelId: modelId,
      sourceCheckpointId: catalogEntry.sourceCheckpointId ?? null,
      dopplerSource,
      dopplerLocator: dopplerLocator.locator,
      transformersjsModelId: tjsModelId,
      transformersjsDtype: tjsDtype,
      transformersjsFormat: tjsFormat,
    },
    workload: {
      prompt,
      warmupRuns,
      timedRuns,
      cacheMode,
      loadMode,
      semanticFixture: path.relative(ROOT_DIR, SEMANTIC_FIXTURE_PATH).split(path.sep).join('/'),
      embeddingMode: profile.embeddingMode,
      embeddingPooling: profile.embeddingPooling,
      embeddingNormalize: profile.embeddingNormalize !== false,
      embeddingStyle: profile.embeddingStyle,
    },
    compareLane: {
      lane: profile.compareLane,
      reason: profile.compareLaneReason,
      correctnessOk: summary.correctnessOk,
      localComparable: summary.correctnessOk === true && profile.compareLane === 'performance_comparable',
      releaseClaimable: profile.releaseClaimable,
      claimable: summary.correctnessOk === true
        && profile.compareLane === 'performance_comparable'
        && profile.releaseClaimable === true,
    },
    sources: {
      config: {
        source: path.relative(ROOT_DIR, configBundle.source).split(path.sep).join('/'),
        sourceSha256: configBundle.sourceSha256,
      },
      catalog: {
        source: path.relative(ROOT_DIR, catalogBundle.source).split(path.sep).join('/'),
        sourceSha256: catalogBundle.sourceSha256,
      },
    },
    commands: {
      dopplerBench: [process.execPath, ...dopplerBenchArgs],
      dopplerVerify: [process.execPath, ...dopplerVerifyArgs],
      transformersjsBench: [process.execPath, ...tjsArgs],
    },
    summary,
    raw: {
      dopplerBench,
      dopplerVerify,
      transformersjsBench: tjsBench,
    },
  };

  if (flags.save) {
    const saveDir = flags['save-dir']
      ? path.resolve(process.cwd(), flags['save-dir'])
      : RESULTS_DIR;
    result.savedPath = await saveCompareResult(result, saveDir, timestamp);
  }

  return result;
}

async function main() {
  const flags = parseArgs(process.argv.slice(2));
  if (flags.help || flags.h) {
    console.log(usage());
    return;
  }
  const timestamp = parseTimestampValue(flags.timestamp, '--timestamp');
  const [configBundle, catalogBundle] = await Promise.all([
    readJsonWithHash(CONFIG_PATH),
    readJsonWithHash(CATALOG_PATH),
  ]);
  const modelIds = flags.all
    ? configBundle.payload.modelProfiles.map((profile) => profile.dopplerModelId)
    : [flags['model-id'] ?? defaultModelId(configBundle.payload)];
  const results = [];
  for (const modelId of modelIds) {
    results.push(await runOne({ modelId, flags, configBundle, catalogBundle, timestamp }));
  }
  const output = flags.all
    ? {
        schemaVersion: 1,
        kind: 'embedding-engine-compare-batch',
        timestamp,
        results,
      }
    : results[0];
  console.log(JSON.stringify(output, null, 2));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => {
    console.error(`[compare-embeddings] ${redactSecrets(error.message)}`);
    process.exit(1);
  });
}

export {
  buildSummary,
  parseArgs,
  main,
};
