#!/usr/bin/env node

import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { promises as fs } from 'node:fs';

import { runBrowserCommandInNode } from '../src/tooling/node-browser-command-runner.js';
import { parseManifest } from '../src/formats/rdrr/index.js';
import { getSourceRuntimeMetadata } from '../src/tooling/source-runtime-bundle.js';
import {
  buildEntryRemoteBaseUrl,
  findCatalogEntry,
  loadJsonFile,
} from './hf-registry-utils.js';

const DEFAULT_MODEL_ID = 'gemma-3-270m-it-q4k-ehf16-af32';
const DEFAULT_CATALOG_FILE = path.join(process.cwd(), 'models', 'catalog.json');
const DEFAULT_GENERATION_PROMPT = Object.freeze({
  messages: Object.freeze([
    Object.freeze({
      role: 'user',
      content: 'Answer with one number only: What is 2 + 2?',
    }),
  ]),
});
const DEFAULT_EMBEDDING_PROMPT = 'Search query: local-first AI with WebGPU inference.';
const DEFAULT_EXPECTED_FIRST_TOKEN = '4';
const DEFAULT_EXPECT_MODE = 'generation';
const DEFAULT_MAX_TOKENS = 8;
const DEFAULT_TIMEOUT_MS = 300_000;
const DEFAULT_BROWSER_ARGS = Object.freeze([
  '--use-angle=swiftshader',
  '--disable-vulkan-surface',
]);
const OPTIONAL_AUX_FILES = Object.freeze([
  'config.json',
  'generation_config.json',
  'tokenizer_config.json',
  'special_tokens_map.json',
]);

function usage() {
  console.error(
    'Usage: node tools/ci-browser-opfs-registry-smoke.mjs '
    + '[--model-id <id>] [--catalog-file <path>] [--cache-root <dir>] [--profile-dir <dir>] '
    + '[--channel <name>] [--timeout-ms <ms>] [--prompt <json>] [--expect-mode <generation|embedding>] '
    + '[--runtime-preset <id>] '
    + '[--expected-first-token <token>] [--expected-embedding-dim <n>] [--expected-semantic-style <id>] '
    + '[--kernel-path <id>] [--activation-dtype <f16|f32>] [--kv-dtype <f16|f32>] '
    + '[--output-dtype <f16|f32>] '
    + '[--keep-opfs-profile] [--json]'
  );
}

function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function parsePositiveInt(value, label, defaultValue) {
  if (value == null || value === '') return defaultValue;
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || !Number.isInteger(numeric) || numeric <= 0) {
    throw new Error(`${label} must be a positive integer.`);
  }
  return numeric;
}

function normalizeExpectMode(value) {
  const normalized = normalizeText(value).toLowerCase();
  if (!normalized) return DEFAULT_EXPECT_MODE;
  if (normalized !== 'generation' && normalized !== 'embedding') {
    throw new Error('--expect-mode must be "generation" or "embedding".');
  }
  return normalized;
}

function parseArgs(argv) {
  const out = {
    modelId: DEFAULT_MODEL_ID,
    catalogFile: DEFAULT_CATALOG_FILE,
    cacheRoot: path.join(os.homedir(), '.cache', 'doppler', 'ci-rdrr'),
    profileDir: path.join(os.homedir(), '.cache', 'doppler', 'ci-opfs', DEFAULT_MODEL_ID),
    channel: 'chromium',
    timeoutMs: DEFAULT_TIMEOUT_MS,
    prompt: null,
    promptProvided: false,
    expectMode: DEFAULT_EXPECT_MODE,
    runtimePreset: null,
    expectedFirstToken: DEFAULT_EXPECTED_FIRST_TOKEN,
    expectedEmbeddingDim: null,
    expectedSemanticStyle: null,
    kernelPath: null,
    activationDtype: null,
    kvDtype: null,
    outputDtype: null,
    keepOpfsProfile: false,
    json: false,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const readValue = () => {
      const value = argv[i + 1];
      if (value == null || String(value).startsWith('--')) {
        throw new Error(`Missing value for ${arg}`);
      }
      i += 1;
      return String(value);
    };

    if (arg === '--model-id') {
      out.modelId = normalizeText(readValue()) || DEFAULT_MODEL_ID;
      continue;
    }
    if (arg === '--catalog-file') {
      out.catalogFile = path.resolve(readValue());
      continue;
    }
    if (arg === '--cache-root') {
      out.cacheRoot = path.resolve(readValue());
      continue;
    }
    if (arg === '--profile-dir') {
      out.profileDir = path.resolve(readValue());
      continue;
    }
    if (arg === '--channel') {
      out.channel = normalizeText(readValue()) || 'chromium';
      continue;
    }
    if (arg === '--timeout-ms') {
      out.timeoutMs = parsePositiveInt(readValue(), '--timeout-ms', DEFAULT_TIMEOUT_MS);
      continue;
    }
    if (arg === '--prompt') {
      out.prompt = JSON.parse(readValue());
      out.promptProvided = true;
      continue;
    }
    if (arg === '--expect-mode') {
      out.expectMode = normalizeExpectMode(readValue());
      continue;
    }
    if (arg === '--runtime-preset') {
      out.runtimePreset = normalizeText(readValue()) || null;
      continue;
    }
    if (arg === '--expected-first-token') {
      out.expectedFirstToken = normalizeText(readValue()).toLowerCase();
      continue;
    }
    if (arg === '--expected-embedding-dim') {
      out.expectedEmbeddingDim = parsePositiveInt(readValue(), '--expected-embedding-dim', null);
      continue;
    }
    if (arg === '--expected-semantic-style') {
      out.expectedSemanticStyle = normalizeText(readValue()) || null;
      continue;
    }
    if (arg === '--kernel-path') {
      out.kernelPath = normalizeText(readValue()) || null;
      continue;
    }
    if (arg === '--activation-dtype') {
      out.activationDtype = normalizeText(readValue()) || null;
      continue;
    }
    if (arg === '--kv-dtype') {
      out.kvDtype = normalizeText(readValue()) || null;
      continue;
    }
    if (arg === '--output-dtype') {
      out.outputDtype = normalizeText(readValue()) || null;
      continue;
    }
    if (arg === '--keep-opfs-profile') {
      out.keepOpfsProfile = true;
      continue;
    }
    if (arg === '--json') {
      out.json = true;
      continue;
    }
    if (arg === '--help' || arg === '-h') {
      usage();
      process.exit(0);
    }
    throw new Error(`Unknown flag: ${arg}`);
  }

  return out;
}

function normalizePrompt(prompt) {
  if (typeof prompt === 'string' && prompt.trim()) {
    return prompt;
  }
  if (prompt && typeof prompt === 'object' && !Array.isArray(prompt)) {
    return prompt;
  }
  throw new Error('Prompt must be a non-empty string or a structured prompt object.');
}

function normalizeOptionalDtype(value, label) {
  if (value == null) return null;
  const normalized = String(value).trim().toLowerCase();
  if (!normalized) return null;
  if (normalized !== 'f16' && normalized !== 'f32') {
    throw new Error(`${label} must be "f16" or "f32".`);
  }
  return normalized;
}

function normalizeFirstToken(output) {
  const normalized = String(output ?? '')
    .trim()
    .toLowerCase()
    .replace(/^[^a-z0-9-]+/u, '')
    .replace(/\s+/g, ' ');
  const firstToken = normalized.split(' ')[0] ?? '';
  return firstToken.replace(/^[^a-z0-9-]+|[^a-z0-9-]+$/gu, '');
}

function assertGenerationSmokeResult(label, response, expectedFirstToken) {
  if (!response?.ok || !response.result) {
    throw new Error(`${label}: browser smoke did not return a successful result envelope.`);
  }

  const output = String(response.result.output ?? '');
  const firstToken = normalizeFirstToken(output);
  if (!firstToken) {
    throw new Error(`${label}: generated output is empty.`);
  }
  if (firstToken !== expectedFirstToken) {
    throw new Error(
      `${label}: expected first token "${expectedFirstToken}" but received "${firstToken}". `
      + `Output: ${JSON.stringify(output)}`
    );
  }

  const metrics = response.result.metrics ?? {};
  if (!Number.isFinite(metrics.modelLoadMs) || metrics.modelLoadMs < 0) {
    throw new Error(`${label}: modelLoadMs must be finite.`);
  }
  if (!Number.isFinite(metrics.firstTokenMs) || metrics.firstTokenMs <= 0) {
    throw new Error(`${label}: firstTokenMs must be > 0.`);
  }
  if (!Number.isFinite(metrics.tokensGenerated) || metrics.tokensGenerated <= 0) {
    throw new Error(`${label}: tokensGenerated must be > 0.`);
  }
}

function assertEmbeddingSmokeResult(label, response, expectedEmbeddingDim, expectedSemanticStyle) {
  if (!response?.ok || !response.result) {
    throw new Error(`${label}: browser smoke did not return a successful result envelope.`);
  }

  const output = response.result.output ?? {};
  const metrics = response.result.metrics ?? {};
  if (output.mode !== 'embedding') {
    throw new Error(`${label}: expected embedding output mode but received ${JSON.stringify(output.mode)}.`);
  }

  const embeddingDim = Number(output.embeddingDim ?? metrics.embeddingDim);
  if (!Number.isFinite(embeddingDim) || embeddingDim <= 0) {
    throw new Error(`${label}: embeddingDim must be > 0.`);
  }
  if (Number.isFinite(expectedEmbeddingDim) && embeddingDim !== expectedEmbeddingDim) {
    throw new Error(`${label}: expected embeddingDim ${expectedEmbeddingDim} but received ${embeddingDim}.`);
  }

  const nonFiniteValues = Number(output.nonFiniteValues ?? metrics.nonFiniteValues ?? NaN);
  if (!Number.isFinite(nonFiniteValues) || nonFiniteValues !== 0) {
    throw new Error(`${label}: embedding contains non-finite values (${nonFiniteValues}).`);
  }

  const finiteRatio = Number(output.finiteRatio ?? metrics.finiteRatio ?? NaN);
  if (!Number.isFinite(finiteRatio) || finiteRatio < 0.999999) {
    throw new Error(`${label}: finiteRatio must be ~1.0; received ${finiteRatio}.`);
  }

  const semantic = output.semantic ?? {};
  const semanticPassed = semantic.passed ?? metrics.semanticPassed;
  if (semanticPassed !== true) {
    throw new Error(`${label}: embedding semantic checks did not pass.`);
  }

  const semanticStyle = normalizeText(semantic.style ?? metrics.semanticStyle);
  if (expectedSemanticStyle && semanticStyle !== expectedSemanticStyle) {
    throw new Error(
      `${label}: expected semantic style "${expectedSemanticStyle}" but received "${semanticStyle}".`
    );
  }

  if (!Number.isFinite(metrics.modelLoadMs) || metrics.modelLoadMs < 0) {
    throw new Error(`${label}: modelLoadMs must be finite.`);
  }
  if (!Number.isFinite(metrics.embeddingMs) || metrics.embeddingMs <= 0) {
    throw new Error(`${label}: embeddingMs must be > 0.`);
  }
  if (!Number.isFinite(metrics.semanticDurationMs) || metrics.semanticDurationMs <= 0) {
    throw new Error(`${label}: semanticDurationMs must be > 0.`);
  }
  if (!Number.isFinite(metrics.semanticRetrievalTotal) || metrics.semanticRetrievalTotal <= 0) {
    throw new Error(`${label}: semanticRetrievalTotal must be > 0.`);
  }
  if (!Number.isFinite(metrics.semanticPairTotal) || metrics.semanticPairTotal <= 0) {
    throw new Error(`${label}: semanticPairTotal must be > 0.`);
  }
  if (!Array.isArray(output.preview) || output.preview.length === 0) {
    throw new Error(`${label}: embedding preview is missing.`);
  }
}

function assertSmokeResult(label, response, options = {}) {
  const expectMode = normalizeExpectMode(options.expectMode);
  if (expectMode === 'embedding') {
    assertEmbeddingSmokeResult(
      label,
      response,
      options.expectedEmbeddingDim,
      options.expectedSemanticStyle
    );
    return;
  }
  assertGenerationSmokeResult(label, response, options.expectedFirstToken);
}

function collectTokenizerPaths(tokenizer) {
  if (!tokenizer || typeof tokenizer !== 'object') {
    return [];
  }
  const keys = [
    'file',
    'sentencepieceModel',
    'tokenizerFile',
    'vocabFile',
    'mergesFile',
    'configFile',
    'specialTokensFile',
    'spieceFile',
    'modelFile',
  ];
  const paths = [];
  for (const key of keys) {
    const value = normalizeText(tokenizer[key]);
    if (value) {
      paths.push(value);
    }
  }
  return [...new Set(paths)];
}

async function pathExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function fetchWithRetry(url, options = {}) {
  const attempts = Number.isFinite(options.attempts) ? Math.max(1, options.attempts) : 3;
  const timeoutMs = Number.isFinite(options.timeoutMs) ? options.timeoutMs : 120_000;
  let lastError = null;

  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      const response = await fetch(url, {
        signal: AbortSignal.timeout(timeoutMs),
        headers: { Connection: 'close' },
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      return response;
    } catch (error) {
      lastError = error;
      if (attempt < attempts) {
        await new Promise((resolve) => setTimeout(resolve, attempt * 1000));
      }
    }
  }

  throw new Error(`Failed to fetch ${url}: ${lastError?.message || lastError}`);
}

async function writeFetchedFile(url, targetPath, options = {}) {
  await fs.mkdir(path.dirname(targetPath), { recursive: true });

  if (options.skipIfExists === true && await pathExists(targetPath)) {
    if (!Number.isFinite(options.expectedBytes)) {
      return;
    }
    const stat = await fs.stat(targetPath);
    if (stat.size === options.expectedBytes) {
      return;
    }
  }

  const response = await fetchWithRetry(url);
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (Number.isFinite(options.expectedBytes) && bytes.byteLength !== options.expectedBytes) {
    throw new Error(
      `Downloaded size mismatch for ${url}: expected ${options.expectedBytes}, got ${bytes.byteLength}.`
    );
  }
  await fs.writeFile(targetPath, bytes);
}

async function ensureModelCache(modelId, catalogFile, cacheRoot) {
  const catalog = await loadJsonFile(catalogFile, catalogFile);
  const entry = findCatalogEntry(catalog, modelId);
  if (!entry) {
    throw new Error(`Model "${modelId}" was not found in ${catalogFile}.`);
  }

  const remoteBaseUrl = buildEntryRemoteBaseUrl(entry);
  if (!remoteBaseUrl) {
    throw new Error(`Model "${modelId}" does not define a pinned Hugging Face source.`);
  }

  const revision = normalizeText(entry?.hf?.revision);
  if (!revision) {
    throw new Error(`Model "${modelId}" is missing hf.revision in ${catalogFile}.`);
  }

  const rdrrRoot = path.join(cacheRoot, modelId, revision);
  const modelDir = path.join(rdrrRoot, modelId);
  await fs.mkdir(modelDir, { recursive: true });

  const manifestUrl = `${remoteBaseUrl}/manifest.json`;
  const manifestText = await (await fetchWithRetry(manifestUrl)).text();
  const manifest = parseManifest(manifestText);
  const sourceRuntime = getSourceRuntimeMetadata(manifest);
  await fs.writeFile(path.join(modelDir, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`, 'utf8');

  const requiredPaths = [];
  if (typeof manifest.tensorsFile === 'string' && manifest.tensorsFile.trim()) {
    requiredPaths.push({ relativePath: manifest.tensorsFile.trim(), expectedBytes: null });
  }
  for (const shard of Array.isArray(manifest.shards) ? manifest.shards : []) {
    const relativePath = normalizeText(shard?.filename);
    if (!relativePath) continue;
    requiredPaths.push({
      relativePath,
      expectedBytes: Number.isFinite(shard?.size) ? Number(shard.size) : null,
    });
  }
  if (sourceRuntime) {
    for (const entry of Array.isArray(sourceRuntime.sourceFiles) ? sourceRuntime.sourceFiles : []) {
      const relativePath = normalizeText(entry?.path);
      if (!relativePath) continue;
      requiredPaths.push({
        relativePath,
        expectedBytes: Number.isFinite(entry?.size) ? Number(entry.size) : null,
      });
    }
    for (const entry of Array.isArray(sourceRuntime.auxiliaryFiles) ? sourceRuntime.auxiliaryFiles : []) {
      const relativePath = normalizeText(entry?.path);
      if (!relativePath) continue;
      requiredPaths.push({
        relativePath,
        expectedBytes: Number.isFinite(entry?.size) ? Number(entry.size) : null,
      });
    }
  }
  for (const tokenizerPath of collectTokenizerPaths(manifest.tokenizer)) {
    requiredPaths.push({ relativePath: tokenizerPath, expectedBytes: null });
  }

  for (const relativePath of OPTIONAL_AUX_FILES) {
    const localPath = path.join(modelDir, relativePath);
    if (await pathExists(localPath)) {
      continue;
    }
    const url = `${remoteBaseUrl}/${relativePath}`;
    try {
      await writeFetchedFile(url, localPath, { skipIfExists: false });
    } catch (error) {
      if (!String(error?.message || error).includes('HTTP 404')) {
        throw error;
      }
    }
  }

  for (const item of requiredPaths) {
    await writeFetchedFile(
      `${remoteBaseUrl}/${item.relativePath}`,
      path.join(modelDir, item.relativePath),
      {
        skipIfExists: true,
        expectedBytes: item.expectedBytes,
      }
    );
  }

  const metadata = {
    modelId,
    revision,
    remoteBaseUrl,
    cachedAt: new Date().toISOString(),
  };
  await fs.writeFile(path.join(rdrrRoot, 'source.json'), `${JSON.stringify(metadata, null, 2)}\n`, 'utf8');

  return {
    modelId,
    revision,
    rdrrRoot,
    modelDir,
    modelUrl: `/models/external/${encodeURIComponent(modelId)}`,
    remoteBaseUrl,
  };
}

function createSmokeProfiles(expectMode) {
  if (normalizeExpectMode(expectMode) === 'embedding') {
    return [
      {
        label: 'embedding-opfs',
        sampling: null,
      },
    ];
  }
  return [
    {
      label: 'greedy-opfs',
      sampling: {
        temperature: 0,
        topP: 1,
        topK: 1,
        repetitionPenalty: 1,
        greedyThreshold: 1,
      },
    },
    {
      label: 'topk40-opfs',
      sampling: {
        temperature: 0,
        topP: 1,
        topK: 40,
        repetitionPenalty: 1,
        greedyThreshold: 0,
      },
    },
  ];
}

async function runSmokeRequest({
  label,
  modelId,
  modelUrl,
  prompt,
  timeoutMs,
  profileDir,
  rdrrRoot,
  channel,
  loadMode,
  wipeCacheBeforeLaunch,
  sampling,
  kernelPath,
  activationDtype,
  kvDtype,
  outputDtype,
  expectMode,
  runtimePreset,
}) {
  const explicitInferenceOverride = {};
  if (activationDtype) {
    explicitInferenceOverride.compute = { activationDtype };
  }
  if (kvDtype) {
    explicitInferenceOverride.kvcache = { kvDtype };
  }
  if (outputDtype) {
    explicitInferenceOverride.session = {
      compute: {
        defaults: {
          outputDtype,
        },
      },
    };
  }
  if (kernelPath) {
    explicitInferenceOverride.kernelPath = kernelPath;
    explicitInferenceOverride.kernelPathPolicy = {
      mode: 'locked',
      sourceScope: ['config'],
      onIncompatible: 'remap',
    };
  }

  const normalizedExpectMode = normalizeExpectMode(expectMode);
  const inferenceConfig = {
    prompt,
    batching: {
      maxTokens: normalizedExpectMode === 'embedding' ? 1 : DEFAULT_MAX_TOKENS,
    },
    ...explicitInferenceOverride,
  };
  if (normalizedExpectMode === 'embedding') {
    inferenceConfig.generation = {
      embeddingMode: 'mean',
    };
  } else if (sampling) {
    inferenceConfig.sampling = sampling;
  }

  const response = await runBrowserCommandInNode({
    command: 'verify',
    suite: 'inference',
    modelId,
    modelUrl,
    loadMode,
    captureOutput: true,
    runtimePreset,
    runtimeConfig: {
      inference: inferenceConfig,
    },
  }, {
    channel,
    headless: true,
    timeoutMs,
    opfsCache: true,
    userDataDir: profileDir,
    wipeCacheBeforeLaunch,
    browserArgs: [...DEFAULT_BROWSER_ARGS],
    staticMounts: [
      {
        urlPrefix: '/models/external',
        rootDir: rdrrRoot,
      },
    ],
  });

  return {
    label,
    loadMode,
    output: response?.result?.output ?? null,
    timing: response?.result?.timing ?? null,
    metrics: response?.result?.metrics ?? null,
    response,
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const expectMode = normalizeExpectMode(args.expectMode);
  const prompt = normalizePrompt(
    args.promptProvided
      ? args.prompt
      : (expectMode === 'embedding' ? DEFAULT_EMBEDDING_PROMPT : DEFAULT_GENERATION_PROMPT)
  );
  const activationDtype = normalizeOptionalDtype(args.activationDtype, '--activation-dtype');
  const kvDtype = normalizeOptionalDtype(args.kvDtype, '--kv-dtype');
  const outputDtype = normalizeOptionalDtype(args.outputDtype, '--output-dtype');

  if (!args.keepOpfsProfile) {
    await fs.rm(args.profileDir, { recursive: true, force: true }).catch(() => {});
  }

  const modelCache = await ensureModelCache(args.modelId, args.catalogFile, args.cacheRoot);

  const primeRun = await runSmokeRequest({
    label: 'prime-http',
    modelId: modelCache.modelId,
    modelUrl: modelCache.modelUrl,
    prompt,
    timeoutMs: args.timeoutMs,
    profileDir: args.profileDir,
    rdrrRoot: modelCache.rdrrRoot,
    channel: args.channel,
    loadMode: 'http',
    wipeCacheBeforeLaunch: !args.keepOpfsProfile,
    sampling: {
      temperature: 0,
      topP: 1,
      topK: 1,
      repetitionPenalty: 1,
      greedyThreshold: 1,
    },
    kernelPath: args.kernelPath,
    activationDtype,
    kvDtype,
    outputDtype,
    expectMode,
    runtimePreset: args.runtimePreset,
  });
  assertSmokeResult(primeRun.label, primeRun.response, {
    expectMode,
    expectedFirstToken: args.expectedFirstToken,
    expectedEmbeddingDim: args.expectedEmbeddingDim,
    expectedSemanticStyle: args.expectedSemanticStyle,
  });

  const checks = [];
  for (const profile of createSmokeProfiles(expectMode)) {
    const run = await runSmokeRequest({
      label: profile.label,
      modelId: modelCache.modelId,
      modelUrl: modelCache.modelUrl,
      prompt,
      timeoutMs: args.timeoutMs,
      profileDir: args.profileDir,
      rdrrRoot: modelCache.rdrrRoot,
      channel: args.channel,
      loadMode: 'opfs',
      wipeCacheBeforeLaunch: false,
      sampling: profile.sampling,
      kernelPath: args.kernelPath,
      activationDtype,
      kvDtype,
      outputDtype,
      expectMode,
      runtimePreset: args.runtimePreset,
    });
    assertSmokeResult(run.label, run.response, {
      expectMode,
      expectedFirstToken: args.expectedFirstToken,
      expectedEmbeddingDim: args.expectedEmbeddingDim,
      expectedSemanticStyle: args.expectedSemanticStyle,
    });
    if (run.response?.result?.loadMode !== 'opfs' && run.response?.result?.timing?.loadMode !== 'opfs') {
      throw new Error(`${run.label}: browser smoke did not report loadMode=opfs.`);
    }
    checks.push(run);
  }

  const summary = {
    ok: true,
    modelId: modelCache.modelId,
    revision: modelCache.revision,
    remoteBaseUrl: modelCache.remoteBaseUrl,
    rdrrRoot: modelCache.rdrrRoot,
    profileDir: args.profileDir,
    expectMode,
    prompt,
    expectedFirstToken: expectMode === 'generation' ? args.expectedFirstToken : null,
    expectedEmbeddingDim: expectMode === 'embedding' ? args.expectedEmbeddingDim : null,
    expectedSemanticStyle: expectMode === 'embedding' ? args.expectedSemanticStyle : null,
    runtimePreset: args.runtimePreset,
    kernelPath: args.kernelPath,
    activationDtype,
    kvDtype,
    outputDtype,
    prime: {
      label: primeRun.label,
      output: primeRun.output,
      timing: primeRun.timing,
    },
    checks: checks.map((run) => ({
      label: run.label,
      loadMode: run.loadMode,
      output: run.output,
      timing: run.timing,
    })),
  };

  if (args.json) {
    console.log(JSON.stringify(summary, null, 2));
    return;
  }

  console.log(
    `[opfs-smoke] mode=${summary.expectMode} model=${summary.modelId} revision=${summary.revision} `
    + `prime=${JSON.stringify(summary.prime.output)}`
  );
  for (const run of summary.checks) {
    console.log(`[opfs-smoke] ${run.label} output=${JSON.stringify(run.output)}`);
  }
}

main().catch((error) => {
  console.error(`[opfs-smoke] ${error?.message || error}`);
  process.exit(1);
});
