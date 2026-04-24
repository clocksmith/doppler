#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import { parseManifest, parseTensorMap } from '../src/formats/rdrr/parsing.js';
import { createConverterConfig } from '../src/config/schema/index.js';
import { resolveConversionPlan } from '../src/converter/conversion-plan.js';
import { inferBundledTokenizerBehaviorFlags } from '../src/inference/tokenizers/behavior-flags.js';
import { buildManifestIntegrityFromModelDir } from '../src/tooling/rdrr-integrity-refresh.js';

function fail(message) {
  console.error(`[refresh-manifest] ${message}`);
  process.exit(1);
}

function parseArgs(argv) {
  const args = {
    modelDir: null,
    conversionConfigPath: null,
    manifestPath: null,
    modelId: null,
    blockSize: null,
    loweringEntryPaths: [],
    skipShardCheck: false,
    dryRun: false,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--config' || arg === '--conversion-config') {
      args.conversionConfigPath = argv[i + 1] ?? null;
      i += 1;
      continue;
    }
    if (arg === '--manifest' || arg === '--manifest-path') {
      args.manifestPath = argv[i + 1] ?? null;
      i += 1;
      continue;
    }
    if (arg === '--model-id') {
      args.modelId = argv[i + 1] ?? null;
      i += 1;
      continue;
    }
    if (arg === '--block-size') {
      args.blockSize = argv[i + 1] ?? null;
      i += 1;
      continue;
    }
    if (arg === '--lowering-entry') {
      args.loweringEntryPaths.push(argv[i + 1] ?? null);
      i += 1;
      continue;
    }
    if (arg === '--skip-shard-check') {
      args.skipShardCheck = true;
      continue;
    }
    if (arg === '--dry-run') {
      args.dryRun = true;
      continue;
    }
    if (arg.startsWith('-')) {
      fail(`Unknown flag: ${arg}`);
    }
    if (!args.modelDir) {
      args.modelDir = arg;
      continue;
    }
    fail(`Unexpected positional argument: ${arg}`);
  }

  if (!args.modelDir || !args.conversionConfigPath) {
    fail('Usage: node tools/refresh-converted-manifest.js <model-dir> --config <conversion-config.json> [--manifest <manifest.json>] [--model-id <id>] [--block-size <bytes>] [--lowering-entry <entry.json>]... [--skip-shard-check] [--dry-run]');
  }

  if (typeof args.conversionConfigPath !== 'string' || !args.conversionConfigPath.trim()) {
    fail('Missing --config path.');
  }

  return args;
}

function resolveOptionalBlockSize(value) {
  if (value == null) return null;
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    fail('--block-size must be a positive integer.');
  }
  return parsed;
}

async function readJson(filePath, label) {
  let raw;
  try {
    raw = await fs.readFile(filePath, 'utf8');
  } catch (error) {
    fail(`Failed to read ${label}: ${error.message}`);
  }
  try {
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      fail(`Invalid ${label}: expected a JSON object.`);
    }
    return parsed;
  } catch (error) {
    fail(`Invalid JSON in ${label}: ${error.message}`);
  }
}

function toSafeString(value) {
  if (typeof value !== 'string') return '';
  const trimmed = value.trim();
  return trimmed || '';
}

function normalizeQuantizationTag(value) {
  const raw = toSafeString(value).toUpperCase();
  if (!raw) return 'f16';
  if (raw === 'Q4_K_M') return 'q4k';
  if (raw === 'Q4_K') return 'q4k';
  return raw.toLowerCase();
}

function resolveArchitectureHint(architecture) {
  if (!architecture) return '';
  if (typeof architecture === 'string') return architecture;
  return (
    toSafeString(architecture.id)
    || toSafeString(architecture.name)
    || toSafeString(architecture.type)
    || ''
  );
}

function resolveHeadDim(architecture) {
  const headDim = Number(architecture?.headDim ?? architecture?.head_dim);
  return Number.isFinite(headDim) && headDim > 0 ? headDim : null;
}

function extractSourceQuantization(manifest) {
  const explicitWeights = toSafeString(manifest?.quantizationInfo?.weights);
  if (explicitWeights) return explicitWeights;
  const explicitQuant = toSafeString(manifest?.quantization);
  if (explicitQuant) return explicitQuant;
  return '';
}

async function loadTensorEntries(manifest, modelDir) {
  if (manifest.tensors && !Array.isArray(manifest.tensors) && typeof manifest.tensors === 'object') {
    return Object.entries(manifest.tensors).map(([name, tensor]) => ({
      name,
      dtype: tensor?.dtype ?? null,
      shape: tensor?.shape ?? null,
      role: tensor?.role ?? null,
      layout: tensor?.layout ?? null,
    }));
  }

  if (typeof manifest.tensorsFile === 'string' && manifest.tensorsFile.trim()) {
    const tensorsPath = path.join(modelDir, manifest.tensorsFile);
    let tensorsRaw;
    try {
      tensorsRaw = await fs.readFile(tensorsPath, 'utf8');
    } catch (error) {
      fail(`Failed to read tensorsFile (${manifest.tensorsFile}): ${error.message}`);
    }
    const tensorsJson = parseTensorMap(tensorsRaw);
    return Object.entries(tensorsJson).map(([name, tensor]) => ({
      name,
      dtype: tensor?.dtype ?? null,
      shape: tensor?.shape ?? null,
      role: tensor?.role ?? null,
      layout: tensor?.layout ?? null,
    }));
  }

  return [];
}

async function verifyShards(modelDir, manifest) {
  const shards = Array.isArray(manifest?.shards) ? manifest.shards : [];
  if (shards.length === 0) {
    return;
  }

  const missing = [];
  for (const shard of shards) {
    const shardName = shard?.filename;
    if (!toSafeString(shardName)) {
      fail('Manifest contains invalid shard entry (missing filename).');
    }
    try {
      await fs.access(path.join(modelDir, shardName));
    } catch {
      missing.push(shardName);
    }
  }

  if (missing.length > 0) {
    fail(`Missing shard files referenced by manifest: ${missing.join(', ')}`);
  }
}

function mergeMetadata(manifest, conversionConfigPath) {
  const metadata = manifest.metadata && typeof manifest.metadata === 'object'
    ? { ...manifest.metadata }
    : {};

  metadata.manifestRefresh = {
    ...(typeof metadata.manifestRefresh === 'object' && metadata.manifestRefresh !== null
      ? metadata.manifestRefresh
      : {}),
    at: new Date().toISOString(),
    config: path.basename(conversionConfigPath),
  };

  return metadata;
}

function resolveTokenizerBehaviorField(tokenizerConfig, ...keys) {
  if (!tokenizerConfig || typeof tokenizerConfig !== 'object' || Array.isArray(tokenizerConfig)) {
    return null;
  }
  for (const key of keys) {
    if (tokenizerConfig[key] != null) {
      return tokenizerConfig[key];
    }
  }
  return null;
}

export async function resolveBundledTokenizerRefreshPatch(modelDir, manifest) {
  const tokenizer = manifest?.tokenizer;
  if (!tokenizer || tokenizer.type !== 'bundled') {
    return null;
  }

  const tokenizerFile = toSafeString(tokenizer.file) || 'tokenizer.json';
  const tokenizerPath = path.join(modelDir, tokenizerFile);
  let tokenizerJsonRaw;
  try {
    tokenizerJsonRaw = await fs.readFile(tokenizerPath, 'utf8');
  } catch (error) {
    throw new Error(`Failed to read bundled tokenizer (${tokenizerFile}): ${error.message}`);
  }

  let tokenizerJson;
  try {
    tokenizerJson = JSON.parse(tokenizerJsonRaw);
  } catch (error) {
    throw new Error(`Invalid JSON in bundled tokenizer (${tokenizerFile}): ${error.message}`);
  }

  const patch = {};
  const addBosToken = resolveTokenizerBehaviorField(tokenizerJson, 'add_bos_token', 'addBosToken');
  const addEosToken = resolveTokenizerBehaviorField(tokenizerJson, 'add_eos_token', 'addEosToken');
  const inferredFlags = inferBundledTokenizerBehaviorFlags(tokenizerJson);
  if (addBosToken != null) {
    patch.addBosToken = addBosToken;
  } else if (inferredFlags.addBosToken != null) {
    patch.addBosToken = inferredFlags.addBosToken;
  }
  if (addEosToken != null) {
    patch.addEosToken = addEosToken;
  } else if (inferredFlags.addEosToken != null) {
    patch.addEosToken = inferredFlags.addEosToken;
  }
  return Object.keys(patch).length > 0 ? patch : null;
}

function buildRefreshRawConfig(manifest) {
  const baseConfig = (manifest?.config && typeof manifest.config === 'object')
    ? { ...manifest.config }
    : {};
  const manifestModelType = toSafeString(manifest?.modelType);

  if (manifestModelType && !toSafeString(baseConfig.model_type)) {
    baseConfig.model_type = manifestModelType;
  }

  if (Array.isArray(baseConfig.layer_types) && baseConfig.layer_types.length > 0) {
    return baseConfig;
  }

  const manifestLayerTypes = manifest?.inference?.layerPattern?.layerTypes;
  if (Array.isArray(manifestLayerTypes) && manifestLayerTypes.length > 0) {
    return {
      ...baseConfig,
      layer_types: [...manifestLayerTypes],
    };
  }

  return baseConfig;
}

function assertRefreshManifestContract(manifest, rawConfig) {
  const sourceQuantization = extractSourceQuantization(manifest);
  if (!sourceQuantization) {
    fail(
      'Manifest refresh requires explicit quantization metadata. ' +
      'Set manifest.quantizationInfo.weights or manifest.quantization before refresh.'
    );
  }
  if (manifest.modelType !== 'diffusion') {
    const modelType = toSafeString(rawConfig?.model_type) || toSafeString(rawConfig?.text_config?.model_type);
    if (!modelType) {
      fail(
        'Manifest refresh requires explicit config.model_type for non-diffusion models. ' +
        'The refresh tool will not infer model_type from removed family metadata.'
      );
    }
  }
  return sourceQuantization;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));

  const modelDir = path.resolve(args.modelDir);
  const conversionConfigPath = path.resolve(args.conversionConfigPath);
  const manifestPath = path.resolve(args.manifestPath || path.join(modelDir, 'manifest.json'));

  const manifest = await readJson(manifestPath, 'manifest.json');
  const rawConversionConfig = await readJson(conversionConfigPath, 'conversion config');
  const converterConfig = createConverterConfig(rawConversionConfig);

  if (!manifest.modelType || typeof manifest.modelType !== 'string') {
    fail('Manifest is missing modelType.');
  }

  const tensorEntries = await loadTensorEntries(manifest, modelDir);
  const architecture = manifest.architecture && typeof manifest.architecture === 'object'
    ? manifest.architecture
    : null;
  const refreshRawConfig = buildRefreshRawConfig(manifest);
  const sourceQuantization = assertRefreshManifestContract(manifest, refreshRawConfig);

  const plan = resolveConversionPlan({
    rawConfig: refreshRawConfig,
    tensors: tensorEntries,
    converterConfig,
    sourceQuantization: normalizeQuantizationTag(sourceQuantization),
    modelKind: manifest.modelType === 'diffusion' ? 'diffusion' : 'transformer',
    architectureHint: resolveArchitectureHint(manifest.architecture),
    architectureConfig: architecture,
    headDim: resolveHeadDim(architecture),
  });

  if (!args.skipShardCheck) {
    await verifyShards(modelDir, manifest);
  }

  const nextModelId = toSafeString(args.modelId) || toSafeString(manifest.modelId);
  if (!nextModelId) {
    fail('Manifest is missing modelId and --model-id was not provided.');
  }

  const refreshed = {
    ...manifest,
    modelId: nextModelId,
    modelType: plan.modelType || manifest.modelType,
    quantization: plan.manifestQuantization || manifest.quantization,
    quantizationInfo: plan.quantizationInfo || manifest.quantizationInfo || null,
    inference: plan.manifestInference || manifest.inference,
    metadata: mergeMetadata(manifest, conversionConfigPath),
  };
  const tokenizerPatch = await resolveBundledTokenizerRefreshPatch(modelDir, manifest);
  if (tokenizerPatch) {
    refreshed.tokenizer = {
      ...(manifest.tokenizer ?? {}),
      ...tokenizerPatch,
    };
  }
  const builtIntegrity = await buildManifestIntegrityFromModelDir(refreshed, {
    modelDir,
    blockSize: resolveOptionalBlockSize(args.blockSize),
    loweringEntryPaths: args.loweringEntryPaths,
  });
  refreshed.integrityExtensions = builtIntegrity.integrityExtensions;

  // Validate and normalize before writing to avoid writing partial/invalid files.
  const validated = parseManifest(JSON.stringify(refreshed));

  if (args.dryRun) {
    console.log('[refresh-manifest] dry-run successful');
    console.log(`  modelId: ${validated.modelId}`);
    console.log(`  quantization: ${validated.quantization}`);
    console.log(`  integrityExtensionsHash: ${builtIntegrity.integrityExtensionsHash}`);
    return;
  }

  await fs.writeFile(manifestPath, JSON.stringify(validated, null, 2), 'utf8');
  console.log(`[refresh-manifest] wrote ${manifestPath}`);
  console.log(`[refresh-manifest] modelId=${validated.modelId} quantization=${validated.quantization}`);
}

export {
  buildRefreshRawConfig,
  extractSourceQuantization,
};

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    fail(error?.message || String(error));
  });
}
