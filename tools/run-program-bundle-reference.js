#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { runNodeCommand } from '../src/tooling/node-command-runner.js';
import { runBrowserCommandInNode } from '../src/tooling/node-browser-command-runner.js';
import { writeProgramBundle } from '../src/tooling/program-bundle.js';

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
    '  --surface <node|browser>       Reference surface; default browser.',
    '  --prompt <text>                Prompt for the bounded proof run.',
    '  --max-tokens <n>               Max generated tokens; default 8.',
    '  --report-out <path>            Where to write the captured report.',
    '  --created-at <iso>             Bundle timestamp override.',
    '  --bundle-id <id>               Bundle id override.',
  ].join('\n');
}

function readFlag(argv, index) {
  const value = argv[index + 1];
  if (value === undefined || value.startsWith('--')) {
    throw new Error(`Missing value for ${argv[index]}.`);
  }
  return value;
}

function parseArgs(argv) {
  const args = {
    manifestPath: null,
    modelDir: null,
    modelId: null,
    modelUrl: null,
    conversionConfigPath: null,
    runtimeConfig: null,
    surface: 'browser',
    prompt: DEFAULT_PROMPT,
    maxTokens: DEFAULT_MAX_TOKENS,
    referenceReportPath: null,
    outputPath: null,
    createdAtUtc: null,
    bundleId: null,
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
    throw new Error(`Unknown argument: ${arg}`);
  }

  if (args.surface !== 'node' && args.surface !== 'browser') {
    throw new Error('--surface must be "node" or "browser".');
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

function normalizeModelUrl(value, modelDir) {
  const raw = typeof value === 'string' ? value.trim() : '';
  if (!raw) {
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

  return {
    repoRoot,
    manifestPath,
    manifest,
    modelDir,
    modelId,
    modelUrl: normalizeModelUrl(args.modelUrl, modelDir),
    conversionConfigPath: args.conversionConfigPath ? path.resolve(args.conversionConfigPath) : null,
    runtimeConfig: args.runtimeConfig,
    surface: args.surface,
    prompt: args.prompt,
    maxTokens: args.maxTokens,
    referenceReportPath,
    outputPath: path.resolve(args.outputPath),
    createdAtUtc: args.createdAtUtc,
    bundleId: args.bundleId,
  };
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

function withReferenceTranscriptRuntimeConfig(runtimeInput) {
  const proofRuntimeConfig = {
    shared: {
      harness: {
        referenceTranscript: {
          enabled: true,
          captureLogits: true,
          captureKvBytes: true,
        },
      },
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

async function assertLocalModelArtifactsReadable(options) {
  const modelDir = localModelDirFromUrl(options.modelUrl);
  if (!modelDir) return;
  const missing = [];
  const tokenizerFile = options.manifest?.tokenizer?.file;
  if (typeof tokenizerFile === 'string' && tokenizerFile.trim()) {
    const tokenizerPath = path.resolve(modelDir, tokenizerFile);
    try {
      await fs.access(tokenizerPath);
    } catch {
      missing.push(path.relative(process.cwd(), tokenizerPath));
    }
  }
  const shards = Array.isArray(options.manifest?.shards) ? options.manifest.shards : [];
  for (const shard of shards) {
    const filename = typeof shard?.filename === 'string'
      ? shard.filename
      : (typeof shard?.path === 'string' ? shard.path : null);
    if (!filename) continue;
    const shardPath = path.resolve(modelDir, filename);
    try {
      await fs.access(shardPath);
    } catch {
      missing.push(path.relative(process.cwd(), shardPath));
      if (missing.length >= 5) break;
    }
  }
  if (missing.length > 0) {
    throw new Error(
      `program bundle reference: local model artifacts are incomplete under ${modelDir}. ` +
      `Missing: ${missing.join(', ')}${missing.length >= 5 ? ', ...' : ''}. ` +
      'Pass --model-url to a complete hosted/local artifact or restore the shard files before running the proof lane.'
    );
  }
}

async function runReferenceVerify(options) {
  await assertLocalModelArtifactsReadable(options);
  const runtimeInput = withReferenceTranscriptRuntimeConfig(
    await normalizeRuntimeConfigInput(options.runtimeConfig)
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
    return runNodeCommand(request);
  }

  return runBrowserCommandInNode(request, {
    opfsCache: false,
    timeoutMs: 600000,
    staticRootDir: options.repoRoot,
  });
}

async function writeReferenceReport(response, reportPath) {
  const report = response?.result?.report;
  if (!report || typeof report !== 'object' || Array.isArray(report)) {
    throw new Error(
      'program bundle reference: verify response did not include result.report. ' +
      'Use a command runner that returns the full report object.'
    );
  }
  await fs.mkdir(path.dirname(reportPath), { recursive: true });
  await fs.writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
  return report;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    console.log(usage());
    return;
  }
  const options = await resolveOptions(args);
  const response = await runReferenceVerify(options);
  const report = await writeReferenceReport(response, options.referenceReportPath);
  const result = await writeProgramBundle({
    repoRoot: options.repoRoot,
    manifestPath: options.manifestPath,
    modelDir: options.modelDir,
    referenceReportPath: options.referenceReportPath,
    conversionConfigPath: options.conversionConfigPath,
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

main().catch((error) => {
  console.error(`[program-bundle:reference] ${error.message}`);
  process.exit(1);
});
