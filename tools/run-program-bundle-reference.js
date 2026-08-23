#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { runNodeCommand } from '../src/tooling/node-command-runner.js';
import { runBrowserCommandInNode } from '../src/tooling/node-browser-command-runner.js';
import { writeProgramBundle } from '../src/tooling/program-bundle.js';
import { destroyDevice } from '../src/gpu/device.js';
import { releaseNodeWebGPU } from '../src/tooling/node-webgpu.js';
import { sha256Hex } from '../src/utils/sha256.js';

const DEFAULT_PROMPT = 'The color of the sky is';
const DEFAULT_MAX_TOKENS = 8;

function usage() {
  return [
    'Usage:',
    '  node tools/run-program-bundle-reference.js --manifest <manifest.json> --out <bundle.json> [options]',
    '',
    'Options:',
    '  --model-dir <dir>              Model directory; defaults to manifest parent.',
    '  --model-id <id>                Model id; defaults to manifest.modelId.',
    '  --model-url <url|path>         Replay model URL; defaults to file://<model-dir>.',
    '  --conversion-config <path>     Conversion config artifact to include.',
    '  --runtime-config <path|json>   Runtime config input for the verify run.',
    '  --expected-transcript <path>  Pinned upstream transcript required for exact token parity.',
    '  --surface <node|browser>       Reference surface; default browser.',
    '  --prompt <text>                Prompt for the bounded proof run.',
    '  --max-tokens <n>               Max generated tokens; default 8.',
    '  --report-out <path>            Where to write the captured report.',
    '  --created-at <iso>             Bundle timestamp override.',
    '  --bundle-id <id>               Bundle id override.',
    '  --tsir-fixture-dir <dir>       Capture handoff and TSIR boundary activations',
    '                                 (pre_layer_input/post_rmsnorm/post_qkv/post_attn/post_ffn)',
    '                                 and write them as .npy under <dir>/layer_<N>/<probe>.npy.',
    '                                 Used to produce Doe frozen Doppler reference fixtures.',
    '  --tsir-fixture-layers <list>   Comma-separated layer indices to capture (default: all).',
    '  --tsir-fixture-generation-step <n>',
    '                                 Also capture the declared generated-token step (1-based).',
  ].join('\n');
}

function readFlag(argv, index) {
  const value = argv[index + 1];
  if (value === undefined || value.startsWith('--')) {
    throw new Error(`Missing value for ${argv[index]}.`);
  }
  return value;
}

export function parseArgs(argv) {
  const args = {
    manifestPath: null,
    modelDir: null,
    modelId: null,
    modelUrl: null,
    conversionConfigPath: null,
    runtimeConfig: null,
    expectedTranscriptPath: null,
    surface: 'browser',
    prompt: DEFAULT_PROMPT,
    maxTokens: DEFAULT_MAX_TOKENS,
    referenceReportPath: null,
    outputPath: null,
    createdAtUtc: null,
    bundleId: null,
    tsirFixtureDir: null,
    tsirFixtureLayers: null,
    tsirFixtureGenerationStep: null,
    help: false,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--help' || arg === '-h') {
      args.help = true;
      continue;
    }
    if (arg === '--manifest') {
      args.manifestPath = readFlag(argv, index);
      index += 1;
      continue;
    }
    if (arg === '--model-dir') {
      args.modelDir = readFlag(argv, index);
      index += 1;
      continue;
    }
    if (arg === '--model-id') {
      args.modelId = readFlag(argv, index);
      index += 1;
      continue;
    }
    if (arg === '--model-url') {
      args.modelUrl = readFlag(argv, index);
      index += 1;
      continue;
    }
    if (arg === '--conversion-config') {
      args.conversionConfigPath = readFlag(argv, index);
      index += 1;
      continue;
    }
    if (arg === '--runtime-config') {
      args.runtimeConfig = readFlag(argv, index);
      index += 1;
      continue;
    }
    if (arg === '--expected-transcript') {
      args.expectedTranscriptPath = readFlag(argv, index);
      index += 1;
      continue;
    }
    if (arg === '--surface') {
      args.surface = readFlag(argv, index);
      index += 1;
      continue;
    }
    if (arg === '--prompt') {
      args.prompt = readFlag(argv, index);
      index += 1;
      continue;
    }
    if (arg === '--max-tokens') {
      const value = Number(readFlag(argv, index));
      if (!Number.isInteger(value) || value < 1) {
        throw new Error('--max-tokens must be a positive integer.');
      }
      args.maxTokens = value;
      index += 1;
      continue;
    }
    if (arg === '--report-out') {
      args.referenceReportPath = readFlag(argv, index);
      index += 1;
      continue;
    }
    if (arg === '--out') {
      args.outputPath = readFlag(argv, index);
      index += 1;
      continue;
    }
    if (arg === '--created-at') {
      args.createdAtUtc = readFlag(argv, index);
      index += 1;
      continue;
    }
    if (arg === '--bundle-id') {
      args.bundleId = readFlag(argv, index);
      index += 1;
      continue;
    }
    if (arg === '--tsir-fixture-dir') {
      args.tsirFixtureDir = readFlag(argv, index);
      index += 1;
      continue;
    }
    if (arg === '--tsir-fixture-layers') {
      args.tsirFixtureLayers = readFlag(argv, index);
      index += 1;
      continue;
    }
    if (arg === '--tsir-fixture-generation-step') {
      const value = Number(readFlag(argv, index));
      if (!Number.isInteger(value) || value < 1) {
        throw new Error('--tsir-fixture-generation-step must be a positive integer.');
      }
      args.tsirFixtureGenerationStep = value;
      index += 1;
      continue;
    }
    throw new Error(`Unknown argument: ${arg}`);
  }

  if (args.surface !== 'node' && args.surface !== 'browser') {
    throw new Error('--surface must be "node" or "browser".');
  }
  if (args.surface === 'browser' && args.tsirFixtureDir) {
    throw new Error('--tsir-fixture-dir requires --surface node because fixture capture writes local .npy files.');
  }
  if (args.tsirFixtureGenerationStep != null && !args.tsirFixtureDir) {
    throw new Error('--tsir-fixture-generation-step requires --tsir-fixture-dir.');
  }
  return args;
}

async function readJsonFile(filePath, label) {
  const raw = await fs.readFile(path.resolve(filePath), 'utf8');
  try {
    return JSON.parse(raw);
  } catch (error) {
    throw new Error(`${label} must contain valid JSON: ${error.message}`);
  }
}

async function readJsonEvidenceFile(filePath, label) {
  const resolved = path.resolve(filePath);
  const raw = await fs.readFile(resolved, 'utf8');
  try {
    return {
      path: resolved,
      hash: `sha256:${sha256Hex(raw)}`,
      json: JSON.parse(raw),
    };
  } catch (error) {
    throw new Error(`${label} must contain valid JSON: ${error.message}`);
  }
}

function toRepoRelativeUrlPath(repoRoot, modelDir) {
  const relative = path.relative(path.resolve(repoRoot), path.resolve(modelDir));
  if (!relative || relative.startsWith('..') || path.isAbsolute(relative)) {
    return null;
  }
  return `/${relative.split(path.sep).map((part) => encodeURIComponent(part)).join('/')}`;
}

export function normalizeModelUrl(value, modelDir, options = {}) {
  const raw = typeof value === 'string' ? value.trim() : '';
  if (!raw) {
    if (options.surface === 'browser') {
      const repoRelativePath = toRepoRelativeUrlPath(options.repoRoot ?? process.cwd(), modelDir);
      if (repoRelativePath) {
        return repoRelativePath;
      }
    }
    return pathToFileURL(path.resolve(modelDir)).href;
  }
  if (/^[a-z][a-z0-9+.-]*:\/\//u.test(raw)) {
    return raw;
  }
  return pathToFileURL(path.resolve(raw)).href;
}

function timestampLabel(value = new Date()) {
  return value.toISOString().replace(/[:]/g, '-');
}

async function resolveOptions(args) {
  if (!args.manifestPath) {
    throw new Error('--manifest is required.');
  }
  if (!args.outputPath) {
    throw new Error('--out is required.');
  }

  const repoRoot = process.cwd();
  const manifestPath = path.resolve(args.manifestPath);
  const manifest = await readJsonFile(manifestPath, 'manifest');
  const modelId = args.modelId || manifest.modelId;
  if (typeof modelId !== 'string' || !modelId.trim()) {
    throw new Error('--model-id is required when manifest.modelId is missing.');
  }
  const modelDir = path.resolve(args.modelDir || path.dirname(manifestPath));
  const referenceReportPath = args.referenceReportPath
    ? path.resolve(args.referenceReportPath)
    : path.resolve(
      repoRoot,
      'reports',
      'program-bundles',
      modelId,
      `${timestampLabel()}.reference.json`
    );
  const expectedTranscriptFile = args.expectedTranscriptPath
    ? await readJsonEvidenceFile(args.expectedTranscriptPath, 'expected transcript')
    : null;

  return {
    repoRoot,
    manifestPath,
    manifest,
    modelDir,
    modelId,
    modelUrl: normalizeModelUrl(args.modelUrl, modelDir, {
      repoRoot,
      surface: args.surface,
    }),
    localArtifactModelDir: args.modelUrl
      ? null
      : modelDir,
    conversionConfigPath: args.conversionConfigPath ? path.resolve(args.conversionConfigPath) : null,
    runtimeConfig: args.runtimeConfig,
    runtimeConfigPath: resolveRuntimeConfigArtifactPath(args.runtimeConfig),
    expectedTranscript: expectedTranscriptFile
      ? {
        path: path.relative(repoRoot, expectedTranscriptFile.path).split(path.sep).join('/'),
        hash: expectedTranscriptFile.hash,
        json: expectedTranscriptFile.json,
      }
      : null,
    surface: args.surface,
    prompt: args.prompt,
    maxTokens: args.maxTokens,
    referenceReportPath,
    outputPath: path.resolve(args.outputPath),
    createdAtUtc: args.createdAtUtc,
    bundleId: args.bundleId,
    tsirFixtureDir: args.tsirFixtureDir ? path.resolve(args.tsirFixtureDir) : null,
    tsirFixtureLayers: args.tsirFixtureLayers
      ? args.tsirFixtureLayers.split(',').map((s) => Number.parseInt(s.trim(), 10)).filter((n) => Number.isInteger(n))
      : null,
    tsirFixtureGenerationStep: args.tsirFixtureGenerationStep,
  };
}

export function resolveRuntimeConfigArtifactPath(input) {
  if (input == null) return null;
  const raw = String(input).trim();
  if (!raw || raw.startsWith('{')) return null;
  return path.resolve(raw);
}

async function normalizeRuntimeConfigInput(input) {
  if (input == null || input === '') return {};
  const raw = String(input).trim();
  if (!raw) return {};
  if (raw.startsWith('{')) {
    return { runtimeConfig: JSON.parse(raw) };
  }
  return { runtimeConfigUrl: pathToFileURL(path.resolve(raw)).href };
}

function isPlainObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function mergePlainObjects(base, patch) {
  const output = { ...(isPlainObject(base) ? base : {}) };
  for (const [key, value] of Object.entries(patch)) {
    if (isPlainObject(value) && isPlainObject(output[key])) {
      output[key] = mergePlainObjects(output[key], value);
    } else {
      output[key] = value;
    }
  }
  return output;
}

function withReferenceTranscriptRuntimeConfig(runtimeInput, options = {}) {
  const harnessPatch = {
    referenceTranscript: {
      enabled: true,
      captureLogits: true,
      captureKvBytes: true,
    },
  };
  if (options.tsirFixtureDir) {
    harnessPatch.tsirFixture = {
      dir: options.tsirFixtureDir,
      layerFilter: options.tsirFixtureLayers ?? null,
      prefillOnly: options.tsirFixtureGenerationStep == null,
      generationStep: options.tsirFixtureGenerationStep ?? null,
    };
  }
  const proofRuntimeConfig = {
    shared: {
      harness: harnessPatch,
    },
  };
  return {
    ...runtimeInput,
    runtimeConfig: mergePlainObjects(runtimeInput.runtimeConfig ?? {}, proofRuntimeConfig),
  };
}

function localModelDirFromUrl(modelUrl) {
  if (typeof modelUrl !== 'string' || !modelUrl.startsWith('file://')) {
    return null;
  }
  return fileURLToPath(modelUrl);
}

function resolveLocalArtifactRoot(modelDir, artifactRoot) {
  const root = typeof artifactRoot === 'string' ? artifactRoot.trim() : '';
  if (!root) return null;
  if (/^file:\/\//i.test(root)) {
    return fileURLToPath(root);
  }
  if (/^[a-z][a-z0-9+.-]*:\/\//iu.test(root)) {
    return null;
  }
  return path.resolve(modelDir, root);
}

async function resolveLocalStorageArtifact(options, modelDir) {
  const weightsRef = options.manifest?.weightsRef;
  if (!weightsRef) {
    return {
      manifest: options.manifest,
      modelDir,
    };
  }
  const storageModelDir = resolveLocalArtifactRoot(modelDir, weightsRef.artifactRoot);
  if (!storageModelDir) {
    return {
      manifest: options.manifest,
      modelDir,
    };
  }
  const storageManifestPath = path.join(storageModelDir, 'manifest.json');
  const storageManifest = await readJsonFile(
    storageManifestPath,
    `weightsRef target manifest at ${storageManifestPath}`,
  );
  return {
    manifest: storageManifest,
    modelDir: storageModelDir,
  };
}

export async function assertLocalModelArtifactsReadable(options) {
  const modelDir = options.localArtifactModelDir || localModelDirFromUrl(options.modelUrl);
  if (!modelDir) return;
  const storageArtifact = await resolveLocalStorageArtifact(options, modelDir);
  const artifactDir = storageArtifact.modelDir;
  const manifest = storageArtifact.manifest;
  const missing = [];
  const tokenizerFile = manifest?.tokenizer?.file;
  if (typeof tokenizerFile === 'string' && tokenizerFile.trim()) {
    const tokenizerPath = path.resolve(artifactDir, tokenizerFile);
    try {
      await fs.access(tokenizerPath);
    } catch {
      missing.push(path.relative(process.cwd(), tokenizerPath));
    }
  }
  const shards = Array.isArray(manifest?.shards) ? manifest.shards : [];
  for (const shard of shards) {
    const filename = typeof shard?.filename === 'string'
      ? shard.filename
      : (typeof shard?.path === 'string' ? shard.path : null);
    if (!filename) continue;
    const shardPath = path.resolve(artifactDir, filename);
    try {
      await fs.access(shardPath);
    } catch {
      missing.push(path.relative(process.cwd(), shardPath));
      if (missing.length >= 5) break;
    }
  }
  if (missing.length > 0) {
    throw new Error(
      `program bundle reference: local model artifacts are incomplete under ${artifactDir}. ` +
      `Missing: ${missing.join(', ')}${missing.length >= 5 ? ', ...' : ''}. ` +
      'Pass --model-url to a complete hosted/local artifact or restore the shard files before running the proof lane.'
    );
  }
}

async function runReferenceVerify(options) {
  await assertLocalModelArtifactsReadable(options);
  const runtimeInput = withReferenceTranscriptRuntimeConfig(
    await normalizeRuntimeConfigInput(options.runtimeConfig),
    {
      tsirFixtureDir: options.tsirFixtureDir,
      tsirFixtureLayers: options.tsirFixtureLayers,
      tsirFixtureGenerationStep: options.tsirFixtureGenerationStep,
    },
  );
  const request = {
    command: 'verify',
    workload: 'inference',
    modelId: options.modelId,
    modelUrl: options.modelUrl,
    loadMode: options.modelUrl.startsWith('file://') ? 'http' : null,
    inferenceInput: {
      prompt: options.prompt,
      maxTokens: options.maxTokens,
    },
    ...runtimeInput,
  };

  if (options.surface === 'node') {
    return runOwnedNodeReferenceCommand(request);
  }

  return runBrowserCommandInNode(request, {
    opfsCache: false,
    timeoutMs: 600000,
    staticRootDir: options.repoRoot,
  });
}

export async function runOwnedNodeReferenceCommand(request, lifecycle = {}) {
  const runCommand = lifecycle.runCommand ?? runNodeCommand;
  const destroy = lifecycle.destroyDevice ?? destroyDevice;
  const release = lifecycle.releaseNodeWebGPU ?? releaseNodeWebGPU;
  try {
    return await runCommand(request);
  } finally {
    try {
      destroy();
    } finally {
      await release();
    }
  }
}

function compareTokenIds(expected, observed) {
  const expectedIds = Array.isArray(expected) ? expected : [];
  const observedIds = Array.isArray(observed) ? observed : [];
  const compared = Math.max(expectedIds.length, observedIds.length);
  let firstMismatchIndex = null;
  for (let index = 0; index < compared; index += 1) {
    if (expectedIds[index] !== observedIds[index]) {
      firstMismatchIndex = index;
      break;
    }
  }
  return {
    passed: firstMismatchIndex === null,
    expectedCount: expectedIds.length,
    observedCount: observedIds.length,
    firstMismatchIndex,
    expectedTokenId: firstMismatchIndex === null ? null : (expectedIds[firstMismatchIndex] ?? null),
    observedTokenId: firstMismatchIndex === null ? null : (observedIds[firstMismatchIndex] ?? null),
  };
}

export function buildSourceParity(report, expectedTranscript) {
  const expected = expectedTranscript.json;
  if (!/^sha256:[0-9a-f]{64}$/.test(expectedTranscript.hash || '')) {
    throw new Error('expected transcript evidence requires a SHA-256 hash.');
  }
  for (const [label, value] of [
    ['model', expected.model],
    ['revision', expected.revision],
    ['execution.sampling', expected.execution?.sampling],
  ]) {
    if (typeof value !== 'string' || !value.trim()) {
      throw new Error(`expected transcript must declare ${label}.`);
    }
  }
  if (!Array.isArray(expected.promptTokenIds) || expected.promptTokenIds.length < 1) {
    throw new Error('expected transcript must contain non-empty promptTokenIds.');
  }
  const prompt = compareTokenIds(
    expected.promptTokenIds,
    report.metrics?.referenceTranscript?.prompt?.ids
  );
  const generation = compareTokenIds(
    expected.generatedTokenIds,
    report.metrics?.referenceTranscript?.tokens?.ids
  );
  const expectedCount = expected.generatedTokens ?? expected.generation?.maxNewTokens;
  if (!Number.isInteger(expectedCount) || expectedCount < 1) {
    throw new Error('expected transcript must declare generatedTokens or generation.maxNewTokens.');
  }
  if (!Array.isArray(expected.generatedTokenIds) || expected.generatedTokenIds.length !== expectedCount) {
    throw new Error('expected transcript generatedTokenIds length does not match its declared generated token count.');
  }
  return {
    schema: 'doppler.source-token-parity/v1',
    status: prompt.passed && generation.passed ? 'passed' : 'failed',
    expectedTranscriptPath: expectedTranscript.path,
    expectedTranscriptHash: expectedTranscript.hash,
    sourceModel: expected.model ?? null,
    sourceRevision: expected.revision ?? null,
    sampling: expected.execution?.sampling ?? null,
    prompt,
    generation,
  };
}

async function writeReferenceReport(response, reportPath, expectedTranscript = null) {
  const report = response?.result?.report;
  if (!report || typeof report !== 'object' || Array.isArray(report)) {
    throw new Error(
      'program bundle reference: verify response did not include result.report. ' +
      'Use a command runner that returns the full report object.'
    );
  }
  const sourceParity = expectedTranscript ? buildSourceParity(report, expectedTranscript) : null;
  if (sourceParity) {
    const mismatch = sourceParity.prompt.firstMismatchIndex === null
      ? `generation index ${sourceParity.generation.firstMismatchIndex}`
      : `prompt index ${sourceParity.prompt.firstMismatchIndex}`;
    report.metrics.sourceParity = sourceParity;
    report.results = [
      ...(Array.isArray(report.results) ? report.results : []),
      {
        name: 'source-token-parity',
        passed: sourceParity.status === 'passed',
        duration: 0,
        ...(sourceParity.status === 'passed'
          ? {}
          : { error: `First source-token mismatch at ${mismatch}.` }),
      },
    ];
  }
  await fs.mkdir(path.dirname(reportPath), { recursive: true });
  await fs.writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
  return { report, sourceParity };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    console.log(usage());
    return;
  }
  const options = await resolveOptions(args);
  const response = await runReferenceVerify(options);
  const { report, sourceParity } = await writeReferenceReport(
    response,
    options.referenceReportPath,
    options.expectedTranscript
  );
  if (sourceParity?.status === 'failed') {
    throw new Error(
      `source-token parity failed: prompt mismatch ${sourceParity.prompt.firstMismatchIndex}, `
      + `generation mismatch ${sourceParity.generation.firstMismatchIndex}.`
    );
  }
  const result = await writeProgramBundle({
    repoRoot: options.repoRoot,
    manifestPath: options.manifestPath,
    modelDir: options.modelDir,
    referenceReportPath: options.referenceReportPath,
    conversionConfigPath: options.conversionConfigPath,
    runtimeConfigPath: options.runtimeConfigPath,
    outputPath: options.outputPath,
    createdAtUtc: options.createdAtUtc,
    bundleId: options.bundleId,
  });

  console.log(JSON.stringify({
    ok: true,
    surface: options.surface,
    modelId: result.bundle.modelId,
    reportPath: path.relative(options.repoRoot, options.referenceReportPath),
    outputPath: path.relative(options.repoRoot, result.outputPath),
    bundleId: result.bundle.bundleId,
    executionGraphHash: result.bundle.sources.executionGraph.hash,
    tokensGenerated: report.metrics?.tokensGenerated ?? null,
    stopReason: report.metrics?.stopReason ?? null,
  }, null, 2));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => {
    console.error(`[program-bundle:reference] ${error.message}`);
    process.exit(1);
  });
}
