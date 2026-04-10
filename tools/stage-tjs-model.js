#!/usr/bin/env node

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_OUTPUT_ROOT = path.join(os.homedir(), '.cache', 'doppler', 'tjs-models');
const DEFAULT_PRESET = 'text-generation';
const DEFAULT_DTYPE = 'fp16';
const MODEL_CATALOG_PATH = path.join(__dirname, '..', 'models', 'catalog.json');
const TJS_DTYPES = Object.freeze(['fp16', 'q4', 'q4f16']);

function printHelp() {
  console.log([
    'Usage: node tools/stage-tjs-model.js --model-id <repo/id> [options]',
    '',
    'Options:',
      '  --model-id <repo/id>   Hugging Face model repo id to stage locally',
      `  --output-root <path>   Local models root (default: ${DEFAULT_OUTPUT_ROOT})`,
      `  --preset <id>          Snapshot preset: ${DEFAULT_PRESET}|full`,
      `  --dtype <id>           ONNX dtype selector for text-generation preset (default: catalog vendor benchmark dtype or ${DEFAULT_DTYPE})`,
      '  --force-download       Redownload matched files even when metadata exists',
      '  --revision <ref>       Optional Hugging Face revision',
      '  --help, -h             Show this help',
  ].join('\n'));
}

function parseArgs(argv) {
  const parsed = {
    modelId: null,
    outputRoot: DEFAULT_OUTPUT_ROOT,
    preset: DEFAULT_PRESET,
    dtype: null,
    forceDownload: false,
    revision: null,
    help: false,
  };

  for (let index = 2; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--help' || arg === '-h') {
      parsed.help = true;
      continue;
    }
    if (arg === '--model-id') {
      parsed.modelId = argv[index + 1] || null;
      index += 1;
      continue;
    }
    if (arg === '--output-root') {
      parsed.outputRoot = argv[index + 1] || DEFAULT_OUTPUT_ROOT;
      index += 1;
      continue;
    }
    if (arg === '--revision') {
      parsed.revision = argv[index + 1] || null;
      index += 1;
      continue;
    }
    if (arg === '--preset') {
      parsed.preset = argv[index + 1] || DEFAULT_PRESET;
      index += 1;
      continue;
    }
    if (arg === '--dtype') {
      parsed.dtype = argv[index + 1] || null;
      index += 1;
      continue;
    }
    if (arg === '--force-download') {
      parsed.forceDownload = true;
      continue;
    }
    throw new Error(`Unknown argument "${arg}".`);
  }

  if (!parsed.help && (!parsed.modelId || parsed.modelId.trim() === '')) {
    throw new Error('--model-id is required.');
  }

  return parsed;
}

function resolveDownloadCli() {
  const candidates = ['hf', 'huggingface-cli'];
  for (const candidate of candidates) {
    const probe = spawnSync(candidate, ['--help'], {
      stdio: 'ignore',
      encoding: 'utf8',
    });
    if (!probe.error && probe.status === 0) {
      return candidate;
    }
  }
  throw new Error('Could not find `hf` or `huggingface-cli` on PATH.');
}

function normalizePreset(value) {
  const normalized = String(value || DEFAULT_PRESET).trim().toLowerCase();
  if (normalized === 'text-generation' || normalized === 'full') {
    return normalized;
  }
  throw new Error(`Unsupported --preset "${value}". Use text-generation or full.`);
}

function normalizeDtype(value, label = '--dtype') {
  const normalized = String(value || '').trim().toLowerCase();
  if (!normalized) {
    throw new Error(`${label} must be a non-empty string.`);
  }
  if (!TJS_DTYPES.includes(normalized)) {
    throw new Error(`${label} must be one of: ${TJS_DTYPES.join(', ')}`);
  }
  return normalized;
}

function buildDownloadArgs(cli, modelId, outputDir, revision, forceDownload, includePatterns = []) {
  const args = cli === 'hf'
    ? ['download', modelId, '--repo-type', 'model', '--local-dir', outputDir]
    : ['download', modelId, '--repo-type', 'model', '--local-dir', outputDir];
  if (revision && revision.trim() !== '') {
    args.push('--revision', revision.trim());
  }
  if (forceDownload === true) {
    args.push('--force-download');
  }
  if (Array.isArray(includePatterns) && includePatterns.length > 0) {
    args.push('--include', ...includePatterns);
  }
  return args;
}

function buildTextGenerationIncludePatterns(dtype) {
  const normalizedDtype = normalizeDtype(dtype);
  return [
    'config.json',
    'generation_config.json',
    'tokenizer.json',
    'tokenizer_config.json',
    'tokenizer.model',
    'special_tokens_map.json',
    'added_tokens.json',
    'chat_template.jinja',
    'preprocessor_config.json',
    `onnx/decoder*${normalizedDtype}*`,
    `onnx/embed*${normalizedDtype}*`,
    `onnx/model*${normalizedDtype}*`,
    'onnx/decoder_model_merged.onnx',
    'onnx/embed_tokens.onnx',
    'onnx/model.onnx',
  ];
}

function buildTextGenerationRequiredSnapshotFiles(dtype) {
  const normalizedDtype = normalizeDtype(dtype);
  return {
    requiredFiles: [
      'config.json',
      'tokenizer_config.json',
      `onnx/embed_tokens_${normalizedDtype}.onnx`,
      `onnx/decoder_model_merged_${normalizedDtype}.onnx`,
    ],
    requiredOneOf: [
      ['tokenizer.json', 'tokenizer.model'],
    ],
    requiredDataPrefixes: [
      `onnx/embed_tokens_${normalizedDtype}.onnx_data`,
      `onnx/decoder_model_merged_${normalizedDtype}.onnx_data`,
    ],
  };
}

function loadCatalogTransformersjsRepoDtypes(catalogPath = MODEL_CATALOG_PATH) {
  if (!fs.existsSync(catalogPath)) {
    return new Map();
  }
  const payload = JSON.parse(fs.readFileSync(catalogPath, 'utf8'));
  const rows = Array.isArray(payload?.models) ? payload.models : [];
  const repoDtypeMap = new Map();
  for (const row of rows) {
    const repoId = typeof row?.vendorBenchmark?.transformersjs?.repoId === 'string'
      ? row.vendorBenchmark.transformersjs.repoId.trim()
      : '';
    const dtype = typeof row?.vendorBenchmark?.transformersjs?.dtype === 'string'
      ? row.vendorBenchmark.transformersjs.dtype.trim().toLowerCase()
      : '';
    if (!repoId || !dtype) {
      continue;
    }
    if (!TJS_DTYPES.includes(dtype)) {
      throw new Error(
        `models/catalog.json vendorBenchmark.transformersjs.dtype for ${row?.modelId || repoId} `
        + `must be one of: ${TJS_DTYPES.join(', ')}`
      );
    }
    const existing = repoDtypeMap.get(repoId.toLowerCase()) || null;
    if (existing && existing !== dtype) {
      throw new Error(
        `models/catalog.json has conflicting vendorBenchmark.transformersjs dtype values for `
        + `"${repoId}": ${existing} vs ${dtype}`
      );
    }
    repoDtypeMap.set(repoId.toLowerCase(), dtype);
  }
  return repoDtypeMap;
}

function resolveCatalogDefaultDtype(modelId, repoDtypeMap) {
  const normalizedModelId = typeof modelId === 'string' ? modelId.trim().toLowerCase() : '';
  if (!normalizedModelId) {
    return null;
  }
  return repoDtypeMap.get(normalizedModelId) || null;
}

function resolveRequestedDtype(explicitDtype, modelId, repoDtypeMap) {
  if (explicitDtype != null) {
    return normalizeDtype(explicitDtype);
  }
  return resolveCatalogDefaultDtype(modelId, repoDtypeMap) || DEFAULT_DTYPE;
}

function hasMatchingDataShard(outputDir, prefix) {
  const absolutePrefix = path.join(outputDir, prefix);
  if (fs.existsSync(absolutePrefix)) {
    return true;
  }
  const directory = path.dirname(absolutePrefix);
  const basenamePrefix = path.basename(absolutePrefix);
  if (!fs.existsSync(directory)) {
    return false;
  }
  return fs.readdirSync(directory).some((name) => name.startsWith(basenamePrefix));
}

function clearPartialDownloads(outputDir) {
  const downloadRoot = path.join(outputDir, '.cache', 'huggingface', 'download');
  if (!fs.existsSync(downloadRoot)) {
    return;
  }
  const queue = [downloadRoot];
  while (queue.length > 0) {
    const currentDir = queue.pop();
    const entries = fs.readdirSync(currentDir, { withFileTypes: true });
    for (const entry of entries) {
      const entryPath = path.join(currentDir, entry.name);
      if (entry.isDirectory()) {
        queue.push(entryPath);
        continue;
      }
      if (entry.name.endsWith('.lock') || entry.name.endsWith('.incomplete')) {
        fs.rmSync(entryPath, { force: true });
      }
    }
  }
}

function validateStagedSnapshot(outputDir, preset, dtype) {
  if (preset !== 'text-generation') {
    return;
  }
  const required = buildTextGenerationRequiredSnapshotFiles(dtype);
  const missing = [];
  for (const relativePath of required.requiredFiles) {
    if (!fs.existsSync(path.join(outputDir, relativePath))) {
      missing.push(relativePath);
    }
  }
  for (const options of required.requiredOneOf) {
    const hasAny = options.some((relativePath) => fs.existsSync(path.join(outputDir, relativePath)));
    if (!hasAny) {
      missing.push(options.join(' | '));
    }
  }
  for (const prefix of required.requiredDataPrefixes) {
    if (!hasMatchingDataShard(outputDir, prefix)) {
      missing.push(`${prefix}*`);
    }
  }
  if (missing.length > 0) {
    throw new Error(
      `Staged snapshot is incomplete for preset=${preset}, dtype=${dtype}. `
      + `Missing: ${missing.join(', ')}. `
      + 'Rerun with --force-download to rebuild the local Hugging Face snapshot.'
    );
  }
}

function main(argv = process.argv) {
  const args = parseArgs(argv);
  if (args.help) {
    printHelp();
    return;
  }

  const modelId = args.modelId.trim();
  const outputRoot = path.resolve(args.outputRoot);
  const outputDir = path.join(outputRoot, modelId);
  const preset = normalizePreset(args.preset);
  const repoDtypeMap = loadCatalogTransformersjsRepoDtypes();
  const dtype = resolveRequestedDtype(args.dtype, modelId, repoDtypeMap);
  const cli = resolveDownloadCli();

  fs.mkdirSync(outputDir, { recursive: true });
  clearPartialDownloads(outputDir);

  const downloadArgs = buildDownloadArgs(
    cli,
    modelId,
    outputDir,
    args.revision,
    args.forceDownload === true,
    preset === 'text-generation' ? buildTextGenerationIncludePatterns(dtype) : []
  );

  const result = spawnSync(cli, downloadArgs, {
    stdio: 'inherit',
    encoding: 'utf8',
  });
  if (result.error) {
    throw result.error;
  }
  if (result.status !== 0) {
    throw new Error(`${cli} download exited with status ${result.status}.`);
  }
  validateStagedSnapshot(outputDir, preset, dtype);

  console.log(JSON.stringify({
    ok: true,
    modelId,
    outputRoot,
    outputDir,
    preset,
    dtype,
    forceDownload: args.forceDownload === true,
    revision: args.revision,
    localModelPathFlagValue: outputRoot,
  }, null, 2));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    main();
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  }
}

export {
  buildTextGenerationIncludePatterns,
  buildTextGenerationRequiredSnapshotFiles,
  loadCatalogTransformersjsRepoDtypes,
  normalizeDtype,
  normalizePreset,
  parseArgs,
  resolveCatalogDefaultDtype,
  resolveRequestedDtype,
  validateStagedSnapshot,
};
