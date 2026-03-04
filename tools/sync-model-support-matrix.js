#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import {
  PRESET_DETECTION_ORDER,
  getPreset,
  listPresets,
  resolvePreset,
} from '../src/config/loader.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..');
const DEFAULT_OUTPUT_PATH = path.join(REPO_ROOT, 'docs/model-support-matrix.md');
const CONVERSION_CONFIG_DIR = path.join(REPO_ROOT, 'tools/configs/conversion');
const CATALOG_PATH = path.join(REPO_ROOT, 'models/catalog.json');
const RUNTIME_BLOCKED_MODEL_TYPES = new Set(['mamba', 'rwkv']);

function parseArgs(argv) {
  const args = {
    check: false,
    outputPath: DEFAULT_OUTPUT_PATH,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const entry = argv[i];
    if (entry === '--check') {
      args.check = true;
      continue;
    }
    if (entry === '--output') {
      const candidate = String(argv[i + 1] || '').trim();
      if (!candidate) {
        throw new Error('Missing value for --output');
      }
      args.outputPath = path.resolve(REPO_ROOT, candidate);
      i += 1;
      continue;
    }
    throw new Error(`Unknown argument: ${entry}`);
  }
  return args;
}

function normalizeText(value) {
  return typeof value === 'string' ? value.trim().toLowerCase() : '';
}

function normalizeList(values) {
  const out = [];
  const seen = new Set();
  for (const entry of values) {
    const normalized = normalizeText(entry);
    if (!normalized || seen.has(normalized)) continue;
    seen.add(normalized);
    out.push(normalized);
  }
  return out;
}

function relativePath(filePath) {
  return path.relative(REPO_ROOT, filePath).replace(/\\/g, '/');
}

async function readJson(filePath) {
  const raw = await fs.readFile(filePath, 'utf8');
  return JSON.parse(raw);
}

async function collectJsonFiles(rootDir) {
  const entries = await fs.readdir(rootDir, { withFileTypes: true });
  const files = [];
  const ordered = entries.sort((left, right) => left.name.localeCompare(right.name));
  for (const entry of ordered) {
    const fullPath = path.join(rootDir, entry.name);
    if (entry.isDirectory()) {
      const nested = await collectJsonFiles(fullPath);
      files.push(...nested);
      continue;
    }
    if (entry.isFile() && entry.name.endsWith('.json')) {
      files.push(fullPath);
    }
  }
  return files;
}

function buildPresetOrder(presetIds) {
  const presetSet = new Set(presetIds);
  const order = [];
  for (const presetId of PRESET_DETECTION_ORDER) {
    if (!presetSet.has(presetId)) continue;
    order.push(presetId);
  }
  const remaining = presetIds
    .filter((presetId) => !order.includes(presetId))
    .sort((left, right) => left.localeCompare(right));
  order.push(...remaining);
  return order;
}

function inferPresetFromCatalogModel(model, presetOrder, presetSet) {
  const explicitPreset = normalizeText(model?.preset);
  if (explicitPreset && presetSet.has(explicitPreset)) {
    return explicitPreset;
  }

  const tokens = normalizeList([
    model?.modelId,
    ...(Array.isArray(model?.aliases) ? model.aliases : []),
    model?.label,
    model?.description,
  ]);
  if (tokens.length === 0) {
    return null;
  }

  for (const presetId of presetOrder) {
    const detection = getPreset(presetId)?.detection;
    if (!detection || typeof detection !== 'object') continue;
    const patterns = normalizeList([
      ...(Array.isArray(detection.architecturePatterns) ? detection.architecturePatterns : []),
      ...(Array.isArray(detection.modelTypePatterns) ? detection.modelTypePatterns : []),
    ]);
    if (patterns.length === 0) continue;
    for (const pattern of patterns) {
      if (tokens.some((token) => token.includes(pattern))) {
        return presetId;
      }
    }
  }
  return null;
}

function resolveRuntimeModelType(presetId) {
  try {
    const preset = resolvePreset(presetId);
    return normalizeText(preset?.modelType) || 'unknown';
  } catch {
    return 'unknown';
  }
}

function resolveRuntimeStatus(modelType) {
  return RUNTIME_BLOCKED_MODEL_TYPES.has(modelType) ? 'blocked' : 'active';
}

function resolveRowStatus(row) {
  if (row.conversionCount === 0) return 'missing-conversion';
  if (row.runtimeStatus === 'blocked') return 'blocked-runtime';
  if (row.catalogCount > 0) return 'ready';
  return 'conversion-ready';
}

function summarizeList(values, maxItems = 3) {
  if (values.length === 0) return '0';
  if (values.length <= maxItems) {
    return `${values.length} (${values.join(', ')})`;
  }
  const visible = values.slice(0, maxItems).join(', ');
  return `${values.length} (${visible}, +${values.length - maxItems} more)`;
}

function renderMatrix(rows, metadata) {
  const lines = [];
  lines.push('# Model Support Matrix');
  lines.push('');
  lines.push('Auto-generated from preset registry (`src/config/loader.js`), conversion configs (`tools/configs/conversion/**`), and catalog (`models/catalog.json`).');
  lines.push('Run `npm run support:matrix:sync` after adding/changing presets, conversion configs, or catalog entries.');
  lines.push('');
  lines.push(`Updated at: ${metadata.generatedAt}`);
  lines.push('');
  lines.push('| Preset | Runtime modelType | Runtime | Conversion configs | Catalog models | Status | Notes |');
  lines.push('| --- | --- | --- | --- | --- | --- | --- |');
  for (const row of rows) {
    const notes = [];
    if (row.runtimeStatus === 'blocked') {
      notes.push('fail-closed runtime path');
    }
    if (row.catalogCount === 0) {
      notes.push('not in local catalog');
    }
    const noteText = notes.length > 0 ? notes.join('; ') : '-';
    lines.push(
      `| ${row.presetId} | ${row.runtimeModelType} | ${row.runtimeStatus} | ` +
      `${summarizeList(row.conversionFiles)} | ${summarizeList(row.catalogModels)} | ${row.status} | ${noteText} |`
    );
  }
  lines.push('');
  lines.push('## Summary');
  lines.push('');
  lines.push(`- Presets tracked: ${metadata.presetCount}`);
  lines.push(`- Presets with conversion configs: ${metadata.presetsWithConversion}`);
  lines.push(`- Presets present in catalog: ${metadata.presetsInCatalog}`);
  lines.push(`- Ready presets (active runtime + conversion + catalog): ${metadata.readyCount}`);
  lines.push(`- Blocked runtime presets: ${metadata.blockedCount}`);
  lines.push(`- Catalog entries: ${metadata.catalogCount}`);
  lines.push('');
  return `${lines.join('\n')}\n`;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const presetIds = listPresets();
  const presetOrder = buildPresetOrder(presetIds);
  const presetSet = new Set(presetIds);

  const conversionFiles = await collectJsonFiles(CONVERSION_CONFIG_DIR);
  const conversionByPreset = new Map(presetIds.map((presetId) => [presetId, []]));
  for (const filePath of conversionFiles) {
    const payload = await readJson(filePath);
    const presetId = normalizeText(payload?.presets?.model);
    if (!presetId || !presetSet.has(presetId)) continue;
    conversionByPreset.get(presetId).push(relativePath(filePath));
  }
  for (const presetId of presetIds) {
    conversionByPreset.get(presetId).sort((left, right) => left.localeCompare(right));
  }

  const catalogPayload = await readJson(CATALOG_PATH);
  const catalogModels = Array.isArray(catalogPayload?.models) ? catalogPayload.models : [];
  const catalogByPreset = new Map(presetIds.map((presetId) => [presetId, []]));
  const unmappedCatalogModels = [];
  for (const model of catalogModels) {
    const inferredPreset = inferPresetFromCatalogModel(model, presetOrder, presetSet);
    if (!inferredPreset) {
      const modelId = typeof model?.modelId === 'string' && model.modelId.trim()
        ? model.modelId.trim()
        : 'unknown-model';
      unmappedCatalogModels.push(modelId);
      continue;
    }
    const modelId = typeof model?.modelId === 'string' && model.modelId.trim()
      ? model.modelId.trim()
      : 'unknown-model';
    catalogByPreset.get(inferredPreset).push(modelId);
  }
  for (const presetId of presetIds) {
    catalogByPreset.get(presetId).sort((left, right) => left.localeCompare(right));
  }

  const rows = presetOrder.map((presetId) => {
    const runtimeModelType = resolveRuntimeModelType(presetId);
    const runtimeStatus = resolveRuntimeStatus(runtimeModelType);
    const conversionFilesForPreset = conversionByPreset.get(presetId) || [];
    const catalogModelsForPreset = catalogByPreset.get(presetId) || [];
    const row = {
      presetId,
      runtimeModelType,
      runtimeStatus,
      conversionFiles: conversionFilesForPreset,
      conversionCount: conversionFilesForPreset.length,
      catalogModels: catalogModelsForPreset,
      catalogCount: catalogModelsForPreset.length,
    };
    return {
      ...row,
      status: resolveRowStatus(row),
    };
  });

  const missingConversionPresets = rows
    .filter((row) => row.conversionCount === 0)
    .map((row) => row.presetId);
  if (missingConversionPresets.length > 0) {
    throw new Error(
      `Missing conversion config coverage for presets: ${missingConversionPresets.join(', ')}`
    );
  }

  if (unmappedCatalogModels.length > 0) {
    throw new Error(
      `Catalog entries missing preset mapping: ${unmappedCatalogModels.join(', ')}`
    );
  }

  const metadata = {
    generatedAt: typeof catalogPayload?.updatedAt === 'string' && catalogPayload.updatedAt.trim()
      ? catalogPayload.updatedAt.trim()
      : 'unknown',
    presetCount: rows.length,
    presetsWithConversion: rows.filter((row) => row.conversionCount > 0).length,
    presetsInCatalog: rows.filter((row) => row.catalogCount > 0).length,
    readyCount: rows.filter((row) => row.status === 'ready').length,
    blockedCount: rows.filter((row) => row.runtimeStatus === 'blocked').length,
    catalogCount: catalogModels.length,
  };
  const nextContent = renderMatrix(rows, metadata);

  if (args.check) {
    let currentContent;
    try {
      currentContent = await fs.readFile(args.outputPath, 'utf8');
    } catch (error) {
      if (error && error.code === 'ENOENT') {
        throw new Error(`Missing ${relativePath(args.outputPath)}. Run npm run support:matrix:sync`);
      }
      throw error;
    }
    if (currentContent !== nextContent) {
      throw new Error(
        `Model support matrix is out of date at ${relativePath(args.outputPath)}. ` +
        'Run npm run support:matrix:sync'
      );
    }
    console.log(`[support-matrix] up to date (${rows.length} presets)`);
    return;
  }

  await fs.mkdir(path.dirname(args.outputPath), { recursive: true });
  await fs.writeFile(args.outputPath, nextContent, 'utf8');
  console.log(`[support-matrix] wrote ${relativePath(args.outputPath)} (${rows.length} presets)`);
}

main().catch((error) => {
  console.error(`[support-matrix] ${error.message}`);
  process.exit(1);
});
