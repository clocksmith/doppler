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
const CONFIG_PATH = path.join(ROOT_DIR, 'benchmarks', 'vendors', 'rerank-compare.config.json');
const CATALOG_PATH = path.join(ROOT_DIR, 'models', 'catalog.json');
const SEMANTIC_FIXTURE_PATH = path.join(ROOT_DIR, 'src', 'inference', 'fixtures', 'rerank-semantic-fixtures.json');
const RESULTS_DIR = path.join(ROOT_DIR, 'benchmarks', 'vendors', 'results');
const DOPPLER_CLI_PATH = path.join(ROOT_DIR, 'src', 'cli', 'doppler-cli.js');
const HF_BENCH_PATH = path.join(ROOT_DIR, 'benchmarks', 'runners', 'hf-transformers-reranker-bench.py');
const DEFAULT_TIMEOUT_MS = 900_000;
const DEFAULT_HF_DEVICE = 'cpu';
const DEFAULT_HF_DTYPE = 'auto';
const DEFAULT_HF_BATCH_SIZE = 1;
const DOPPLER_SOURCES = Object.freeze(['local', 'quickstart-registry']);
const HF_DEVICES = Object.freeze(['cpu', 'cuda', 'auto']);
const HF_DTYPES = Object.freeze(['auto', 'float32', 'float16', 'bfloat16']);

function usage() {
  return [
    'Usage: node tools/compare-rerankers-hf.js [options]',
    '',
    'Options:',
    '  --model-id <id>              Doppler model ID (default: first rerank profile)',
    '  --query <text>              Shared rerank query',
    '  --document <text>           Shared rerank document, repeatable',
    '  --document-count <n>        Expand default documents to n deterministic docs',
    '  --warmup <n>                Warmup runs per engine (default from rerank compare config)',
    '  --runs <n>                  Timed runs per engine (default from rerank compare config)',
    '  --doppler-source <source>   local|quickstart-registry',
    '  --doppler-surface <surface> auto|node|browser',
    '  --hf-model <id>             HF Transformers model ID/path (default: catalog sourceCheckpointId)',
    '  --hf-device <device>        cpu|cuda|auto (default: cpu)',
    '  --hf-dtype <dtype>          auto|float32|float16|bfloat16',
    '  --hf-batch-size <n>         HF scoring batch size',
    '  --hf-allow-download         Allow HF runner to download missing files',
    '  --cache-mode <mode>         warm|cold',
    '  --load-mode <mode>          http|opfs|memory',
    '  --timeout-ms <ms>           Per-engine command timeout',
    '  --save                      Save compare JSON under benchmarks/vendors/results/',
    '  --save-dir <dir>            Save directory override',
    '  --timestamp <iso|ms>        Deterministic timestamp override',
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
    if (key === 'help' || key === 'h' || key === 'json' || key === 'save' || key === 'hf-allow-download') {
      flags[key] = true;
      continue;
    }
    if (key === 'document') {
      const value = argv[i + 1];
      if (value === undefined) throw new Error(`Missing value for --${key}`);
      if (!Array.isArray(flags.document)) flags.document = [];
      flags.document.push(value);
      i += 1;
      continue;
    }
    const value = argv[i + 1];
    if (value === undefined) throw new Error(`Missing value for --${key}`);
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

function parseNonNegativeInteger(value, fallback, label) {
  if (value == null || value === '') return fallback;
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 0) {
    throw new Error(`${label} must be a non-negative integer`);
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
  if (!text) throw new Error(`${label} produced no output`);
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

function optionalString(value) {
  return typeof value === 'string' && value.trim() !== '' ? value.trim() : null;
}

function catalogEntryById(catalog, modelId) {
  const rows = Array.isArray(catalog?.models) ? catalog.models : [];
  return rows.find((row) => row?.modelId === modelId) ?? null;
}

function profileByModelId(config, modelId) {
  const rows = Array.isArray(config?.modelProfiles) ? config.modelProfiles : [];
  return rows.find((row) => row?.dopplerModelId === modelId) ?? null;
}

function defaultModelId(config) {
  const first = Array.isArray(config?.modelProfiles) ? config.modelProfiles[0] : null;
  const modelId = optionalString(first?.dopplerModelId);
  if (!modelId) throw new Error('rerank-compare.config.json must include at least one model profile.');
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

function normalizeStringList(value, label) {
  const source = Array.isArray(value) ? value : (value == null ? [] : [value]);
  const out = source.map((entry) => String(entry ?? '').trim()).filter(Boolean);
  if (out.length === 0) {
    throw new Error(`${label} must include at least one non-empty string`);
  }
  return out;
}

function expandDocuments(documents, count) {
  if (!Number.isInteger(count)) {
    return documents;
  }
  if (count <= documents.length) return documents.slice(0, count);
  const expanded = [...documents];
  const topics = [
    'database migrations',
    'CSS layout',
    'password managers',
    'weather forecasts',
    'shipping logistics',
    'financial ledgers',
    'audio compression',
    'test orchestration',
    'compiler warnings',
    'recipe indexing',
  ];
  for (let i = expanded.length; i < count; i += 1) {
    const topic = topics[i % topics.length];
    expanded.push(
      `Distractor passage ${i}: this document discusses ${topic}, operational details, and unrelated implementation notes. ` +
      'It does not identify the browser API that exposes GPU hardware for compute workloads.'
    );
  }
  return expanded;
}

async function loadRerankScoringConfig(modelId) {
  const manifestPath = path.join(ROOT_DIR, 'models', 'local', modelId, 'manifest.json');
  const manifest = await readJsonWithHash(manifestPath);
  const config = manifest.payload?.inference?.rerank;
  if (!config || typeof config !== 'object' || Array.isArray(config)) {
    throw new Error(`${path.relative(ROOT_DIR, manifestPath)} must define inference.rerank`);
  }
  return {
    config,
    source: path.relative(ROOT_DIR, manifest.source).split(path.sep).join('/'),
    sourceSha256: manifest.sourceSha256,
  };
}

function buildDopplerConfig({
  modelId,
  modelUrl,
  runtimeProfile,
  query,
  documents,
  warmupRuns,
  timedRuns,
  cacheMode,
  loadMode,
  surface,
  timestamp,
}) {
  const request = {
    workload: 'rerank',
    modelId,
    modelUrl,
    cacheMode,
    loadMode,
    runtimeConfig: {
      shared: {
        benchmark: {
          run: {
            warmupRuns,
            timedRuns,
            loadMode,
          },
        },
      },
      inference: {
        rerank: {
          query,
          documents,
        },
      },
    },
  };
  if (runtimeProfile !== null) {
    request.runtimeProfile = runtimeProfile;
  }
  return {
    request,
    run: {
      surface,
      timestamp,
      ...(loadMode === 'http'
        ? {
            browser: {
              opfsCache: false,
            },
          }
        : {}),
    },
  };
}

async function runJson(command, commandArgs, label, timeoutMs) {
  const { stdout } = await execFileAsync(command, commandArgs, {
    cwd: ROOT_DIR,
    timeout: timeoutMs,
    maxBuffer: 1024 * 1024 * 128,
    env: process.env,
  }).catch((error) => {
    const stderrTail = redactSecrets(error?.stderr ?? '').slice(-4000);
    const stdoutTail = redactSecrets(error?.stdout ?? '').slice(-4000);
    throw new Error(`${label} failed: ${error.message}\nstdout:\n${stdoutTail}\nstderr:\n${stderrTail}`);
  });
  return parseJsonBlock(stdout, label);
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

function extractRerankSpeed(result) {
  const metrics = result?.metrics ?? {};
  return {
    documentCount: finiteNumber(metrics.documentCount),
    medianRerankMs: finiteNumber(metrics.medianRerankMs),
    avgRerankMs: finiteNumber(metrics.avgRerankMs),
    avgReranksPerSec: finiteNumber(metrics.avgReranksPerSec),
    p95RerankMs: finiteNumber(metrics.p95RerankMs),
    modelLoadMs: finiteNumber(metrics.modelLoadMs ?? result?.timing?.modelLoadMs),
    topDocumentIndex: finiteNumber(metrics.topDocumentIndex),
    nonFiniteScores: finiteNumber(metrics.nonFiniteScores),
  };
}

function extractRerankCorrectness(result) {
  const metrics = result?.metrics ?? {};
  return {
    semanticPassed: metrics.semanticPassed === true,
    semanticPairAcc: finiteNumber(metrics.semanticPairAcc),
    topDocumentIndex: finiteNumber(metrics.topDocumentIndex),
    nonFiniteScores: finiteNumber(metrics.nonFiniteScores),
    validRuns: finiteNumber(metrics.validRuns),
    invalidRuns: finiteNumber(metrics.invalidRuns),
  };
}

function ratio(a, b) {
  if (!Number.isFinite(a) || !Number.isFinite(b) || b === 0) return null;
  return Number((a / b).toFixed(4));
}

function buildSummary(dopplerBench, dopplerVerify, hfBench, expectedTopDocumentIndex) {
  const dopplerSpeed = extractRerankSpeed(dopplerBench);
  const dopplerCorrectness = extractRerankCorrectness(dopplerVerify);
  const hfSpeed = extractRerankSpeed(hfBench);
  const hfCorrectness = extractRerankCorrectness(hfBench);
  const expected = Number.isInteger(expectedTopDocumentIndex) ? expectedTopDocumentIndex : 0;
  const correctnessOk = dopplerCorrectness.semanticPassed === true
    && hfCorrectness.semanticPassed === true
    && dopplerSpeed.nonFiniteScores === 0
    && hfSpeed.nonFiniteScores === 0
    && dopplerCorrectness.topDocumentIndex === expected
    && hfCorrectness.topDocumentIndex === expected;
  return {
    correctnessOk,
    expectedTopDocumentIndex: expected,
    doppler: {
      speed: dopplerSpeed,
      correctness: dopplerCorrectness,
    },
    hfPytorch: {
      speed: hfSpeed,
      correctness: hfCorrectness,
    },
    ratios: {
      medianRerankMsDopplerOverHfPytorch: ratio(dopplerSpeed.medianRerankMs, hfSpeed.medianRerankMs),
      avgReranksPerSecDopplerOverHfPytorch: ratio(dopplerSpeed.avgReranksPerSec, hfSpeed.avgReranksPerSec),
      modelLoadMsDopplerOverHfPytorch: ratio(dopplerSpeed.modelLoadMs, hfSpeed.modelLoadMs),
    },
  };
}

function buildFairnessAudit({ summary }) {
  const invalidReasons = [];
  const correctnessOk = summary.correctnessOk === true;
  if (!correctnessOk) invalidReasons.push('correctness-failed');
  const claimGrade = invalidReasons.length === 0;
  return {
    schemaVersion: 1,
    claimGrade,
    releaseClaimable: false,
    localComparable: claimGrade,
    correctnessOk,
    primarySection: 'rerank',
    invalidReason: invalidReasons[0] ?? null,
    invalidReasons,
    sourceFairness: {
      schemaVersion: 1,
      competitor: 'hf-pytorch',
      releaseConfigured: false,
      blocksReleaseClaim: true,
      reason: 'HF/PyTorch local baseline is a non-WebGPU diagnostic lane, not a release claim lane.',
    },
  };
}

async function saveCompareResult(result, saveDir, timestamp) {
  await fs.mkdir(saveDir, { recursive: true });
  const modelSlug = String(result.model?.dopplerModelId ?? 'rerank').replace(/[^a-zA-Z0-9_-]/g, '_');
  const filename = `rerank_hf_pytorch_compare_${modelSlug}_${compactTimestamp(timestamp)}.json`;
  const filePath = path.join(saveDir, filename);
  const json = JSON.stringify(result, null, 2);
  await fs.writeFile(filePath, json, 'utf-8');
  await fs.writeFile(path.join(saveDir, 'rerank_hf_pytorch_compare_latest.json'), json, 'utf-8');
  return filePath;
}

async function runOne({ flags, configBundle, catalogBundle, timestamp }) {
  const config = configBundle.payload;
  const catalog = catalogBundle.payload;
  const modelId = optionalString(flags['model-id']) ?? defaultModelId(config);
  const profile = profileByModelId(config, modelId);
  if (!profile) throw new Error(`No rerank compare profile for "${modelId}" in ${CONFIG_PATH}`);
  const catalogEntry = catalogEntryById(catalog, modelId);
  if (!catalogEntry) throw new Error(`No model catalog entry for "${modelId}"`);
  const defaults = config.defaults ?? {};
  const query = optionalString(flags.query) ?? optionalString(profile.query) ?? optionalString(defaults.query);
  if (!query) throw new Error(`${modelId}: rerank compare requires a query`);
  const baseDocuments = normalizeStringList(
    flags.document ?? profile.documents ?? defaults.documents,
    `${modelId}: rerank documents`
  );
  const documentCount = parsePositiveInteger(flags['document-count'], null, '--document-count');
  const documents = expandDocuments(baseDocuments, documentCount);
  const expectedTopDocumentIndex = Number.isInteger(profile.expectedTopDocumentIndex)
    ? profile.expectedTopDocumentIndex
    : defaults.expectedTopDocumentIndex;
  const warmupRuns = parseNonNegativeInteger(flags.warmup, profile.warmupRuns ?? defaults.warmupRuns ?? 1, '--warmup');
  const timedRuns = parsePositiveInteger(flags.runs, profile.timedRuns ?? defaults.timedRuns ?? 3, '--runs');
  const cacheMode = parseChoice(flags['cache-mode'], ['warm', 'cold'], profile.cacheMode ?? defaults.cacheMode ?? 'warm', '--cache-mode');
  const loadMode = parseChoice(flags['load-mode'], ['http', 'opfs', 'memory'], profile.loadMode ?? defaults.loadMode ?? 'http', '--load-mode');
  const dopplerSource = parseChoice(flags['doppler-source'], DOPPLER_SOURCES, profile.defaultDopplerSource, '--doppler-source');
  const dopplerSurface = parseChoice(
    flags['doppler-surface'],
    ['auto', 'node', 'browser'],
    profile.defaultDopplerSurface ?? defaults.dopplerSurface ?? 'browser',
    '--doppler-surface'
  );
  if (dopplerSurface === 'browser' && loadMode === 'memory') {
    throw new Error('--load-mode memory is supported by the Doppler Node surface, not browser');
  }
  const hfModelId = optionalString(flags['hf-model']) ?? optionalString(catalogEntry.sourceCheckpointId);
  if (!hfModelId) throw new Error(`${modelId}: --hf-model or catalog sourceCheckpointId is required`);
  const hfDevice = parseChoice(flags['hf-device'], HF_DEVICES, DEFAULT_HF_DEVICE, '--hf-device');
  const hfDtype = parseChoice(flags['hf-dtype'], HF_DTYPES, DEFAULT_HF_DTYPE, '--hf-dtype');
  const hfBatchSize = parsePositiveInteger(flags['hf-batch-size'], DEFAULT_HF_BATCH_SIZE, '--hf-batch-size');
  const timeoutMs = parsePositiveInteger(flags['timeout-ms'], DEFAULT_TIMEOUT_MS, '--timeout-ms');
  const dopplerLocator = resolveDopplerModelUrl(modelId, dopplerSource, catalogEntry);
  const dopplerRuntimeProfile = Object.prototype.hasOwnProperty.call(profile, 'dopplerRuntimeProfile')
    ? profile.dopplerRuntimeProfile
    : defaults.dopplerRuntimeProfile;
  const dopplerVerifyRuntimeProfile = Object.prototype.hasOwnProperty.call(profile, 'dopplerVerifyRuntimeProfile')
    ? profile.dopplerVerifyRuntimeProfile
    : defaults.dopplerVerifyRuntimeProfile;
  const scoring = await loadRerankScoringConfig(modelId);

  const dopplerBenchConfig = buildDopplerConfig({
    modelId,
    modelUrl: dopplerLocator.modelUrl,
    runtimeProfile: dopplerRuntimeProfile,
    query,
    documents,
    warmupRuns,
    timedRuns,
    cacheMode,
    loadMode,
    surface: dopplerSurface,
    timestamp,
  });
  const dopplerVerifyConfig = buildDopplerConfig({
    modelId,
    modelUrl: dopplerLocator.modelUrl,
    runtimeProfile: dopplerVerifyRuntimeProfile,
    query,
    documents,
    warmupRuns: 0,
    timedRuns: 1,
    cacheMode,
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
  const hfArgs = [
    HF_BENCH_PATH,
    '--model',
    hfModelId,
    '--query',
    query,
    '--documents-json',
    JSON.stringify(documents),
    '--rerank-config',
    JSON.stringify(scoring.config),
    '--semantic-fixture',
    SEMANTIC_FIXTURE_PATH,
    '--warmup',
    String(warmupRuns),
    '--runs',
    String(timedRuns),
    '--batch-size',
    String(hfBatchSize),
    '--device',
    hfDevice,
    '--dtype',
    hfDtype,
    '--timestamp',
    timestamp,
    '--json',
  ];
  if (flags['hf-allow-download']) hfArgs.push('--allow-download');

  const python = process.env.PYTHON || 'python3';
  const dopplerBench = unwrapToolingResult(await runJson(process.execPath, dopplerBenchArgs, `${modelId} Doppler rerank bench`, timeoutMs));
  const dopplerVerify = unwrapToolingResult(await runJson(process.execPath, dopplerVerifyArgs, `${modelId} Doppler rerank verify`, timeoutMs));
  const hfBench = await runJson(python, hfArgs, `${modelId} HF PyTorch rerank bench`, timeoutMs);
  const summary = buildSummary(dopplerBench, dopplerVerify, hfBench, expectedTopDocumentIndex);
  const fairness = buildFairnessAudit({ summary });

  const result = {
    schemaVersion: 1,
    kind: 'rerank-hf-pytorch-local-compare',
    timestamp,
    model: {
      dopplerModelId: modelId,
      sourceCheckpointId: catalogEntry.sourceCheckpointId ?? null,
      dopplerSource,
      dopplerLocator: dopplerLocator.locator,
      hfModelId,
      hfDevice,
      hfDtype,
      hfBatchSize,
    },
    workload: {
      query,
      documents,
      documentCount: documents.length,
      expectedTopDocumentIndex,
      warmupRuns,
      timedRuns,
      cacheMode,
      loadMode,
      semanticFixture: path.relative(ROOT_DIR, SEMANTIC_FIXTURE_PATH).split(path.sep).join('/'),
      scoringConfig: scoring.config,
    },
    compareLane: {
      lane: 'local_non_webgpu_baseline',
      reason: 'Local Doppler reranker path compared against a local HF Transformers PyTorch baseline with identical Qwen yes/no-logit scoring prompts.',
      claimGrade: fairness.claimGrade,
      correctnessOk: fairness.correctnessOk,
      localComparable: fairness.localComparable,
      releaseClaimable: false,
      claimable: false,
      fairnessInvalidReason: fairness.invalidReason,
      fairnessInvalidReasons: fairness.invalidReasons,
    },
    fairness,
    sources: {
      config: {
        source: path.relative(ROOT_DIR, configBundle.source).split(path.sep).join('/'),
        sourceSha256: configBundle.sourceSha256,
      },
      catalog: {
        source: path.relative(ROOT_DIR, catalogBundle.source).split(path.sep).join('/'),
        sourceSha256: catalogBundle.sourceSha256,
      },
      scoring,
    },
    commands: {
      dopplerBench: [process.execPath, ...dopplerBenchArgs],
      dopplerVerify: [process.execPath, ...dopplerVerifyArgs],
      hfPytorchBench: [python, ...hfArgs],
    },
    summary,
    raw: {
      dopplerBench,
      dopplerVerify,
      hfPytorchBench: hfBench,
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
  const result = await runOne({ flags, configBundle, catalogBundle, timestamp });
  console.log(JSON.stringify(result, null, 2));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => {
    console.error(`[compare-rerankers-hf] ${redactSecrets(error.message)}`);
    process.exit(1);
  });
}

export {
  buildFairnessAudit,
  buildSummary,
  parseArgs,
  main,
};
