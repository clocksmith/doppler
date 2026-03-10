#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';
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
const QUICKSTART_REGISTRY_PATH = path.join(REPO_ROOT, 'src', 'client', 'doppler-registry.json');
const RUNTIME_BLOCKED_MODEL_TYPES = new Set(['mamba', 'rwkv']);

export function parseArgs(argv) {
  const args = {
    check: false,
    outputPath: DEFAULT_OUTPUT_PATH,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const entry = argv[i];
    const nextValue = () => {
      const candidate = argv[i + 1];
      if (candidate == null || String(candidate).startsWith('--')) {
        throw new Error(`Missing value for ${entry}`);
      }
      i += 1;
      return String(candidate).trim();
    };
    if (entry === '--check') {
      args.check = true;
      continue;
    }
    if (entry === '--output') {
      const candidate = nextValue();
      if (!candidate) {
        throw new Error('Missing value for --output');
      }
      args.outputPath = path.resolve(REPO_ROOT, candidate);
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

export function validateCatalogMatrixInputs(payload) {
  const errors = [];
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    return ['catalog payload must be a JSON object'];
  }
  if (!Array.isArray(payload.models)) {
    return ['catalog payload must include a models array'];
  }
  const updatedAt = normalizeText(payload.updatedAt);
  if (!updatedAt) {
    errors.push('catalog updatedAt must be a non-empty string');
  }

  const seenModelIds = new Set();
  for (const model of payload.models) {
    const modelId = normalizeText(model?.modelId);
    if (!modelId) {
      errors.push('catalog entries must include modelId');
      continue;
    }
    if (seenModelIds.has(modelId)) {
      errors.push(`duplicate catalog modelId: ${modelId}`);
    }
    seenModelIds.add(modelId);

    const lifecycle = model?.lifecycle && typeof model.lifecycle === 'object' ? model.lifecycle : {};
    const availability = lifecycle?.availability && typeof lifecycle.availability === 'object'
      ? lifecycle.availability
      : {};
    const status = lifecycle?.status && typeof lifecycle.status === 'object' ? lifecycle.status : {};
    const demo = normalizeText(status.demo);
    const baseUrl = normalizeText(model?.baseUrl);
    const hf = model?.hf && typeof model.hf === 'object' ? model.hf : {};

    if (availability.hf === true) {
      if (!normalizeText(hf.repoId)) {
        errors.push(`${modelId}: lifecycle.availability.hf=true requires hf.repoId`);
      }
      if (!normalizeText(hf.revision)) {
        errors.push(`${modelId}: lifecycle.availability.hf=true requires hf.revision`);
      }
      if (!normalizeText(hf.path)) {
        errors.push(`${modelId}: lifecycle.availability.hf=true requires hf.path`);
      }
    }

    if (availability.curated === true && !(baseUrl.startsWith('./local/') || baseUrl.startsWith('local/'))) {
      errors.push(`${modelId}: lifecycle.availability.curated=true requires a repo-local baseUrl`);
    }
    if (
      availability.local === true
      && !(baseUrl.startsWith('./local/')
        || baseUrl.startsWith('local/'))
    ) {
      errors.push(`${modelId}: lifecycle.availability.local=true requires a repo-local baseUrl`);
    }

    if (demo === 'curated' && !(baseUrl.startsWith('./local/') || baseUrl.startsWith('local/'))) {
      errors.push(`${modelId}: lifecycle.status.demo=curated requires a repo-local baseUrl`);
    }
    if (demo === 'local' && !(baseUrl.startsWith('./local/') || baseUrl.startsWith('local/'))) {
      errors.push(`${modelId}: lifecycle.status.demo=local requires a local baseUrl`);
    }
  }

  return errors;
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
  if (!explicitPreset) {
    return null;
  }
  if (!presetSet.has(explicitPreset)) {
    const modelId = normalizeText(model?.modelId) || 'unknown-model';
    throw new Error(
      `Catalog model "${modelId}" has preset "${explicitPreset}" which is not in the preset registry. ` +
      `Valid presets: ${[...presetSet].sort().join(', ')}`
    );
  }
  return explicitPreset;
}

function resolveRuntimeModelType(presetId) {
  const preset = resolvePreset(presetId);
  const modelType = normalizeText(preset?.modelType);
  if (!modelType) {
    throw new Error(`Preset "${presetId}" resolved without a modelType field`);
  }
  return modelType;
}

function resolveRuntimeStatus(modelType) {
  return RUNTIME_BLOCKED_MODEL_TYPES.has(modelType) ? 'blocked' : 'active';
}

function normalizeModelId(value) {
  return typeof value === 'string' && value.trim() ? value.trim() : 'unknown-model';
}

function summarizeModes(model) {
  const modes = Array.isArray(model?.modes)
    ? model.modes.filter((entry) => typeof entry === 'string' && entry.trim())
    : [];
  return modes.length > 0 ? modes.join(', ') : 'run';
}

function createEmptyLifecycleAggregate() {
  return {
    hosted: false,
    demo: 'none',
    tested: 'unknown',
    testedAt: null,
  };
}

function normalizeTestedState(value) {
  const normalized = normalizeText(value);
  if (normalized === 'verified' || normalized === 'pass' || normalized === 'passed') return 'verified';
  if (normalized === 'failed' || normalized === 'fail') return 'failed';
  return 'unknown';
}

function resolveCatalogLifecycle(model) {
  const lifecycle = model?.lifecycle && typeof model.lifecycle === 'object' ? model.lifecycle : {};
  const availability = lifecycle?.availability && typeof lifecycle.availability === 'object'
    ? lifecycle.availability
    : {};
  const status = lifecycle?.status && typeof lifecycle.status === 'object' ? lifecycle.status : {};
  const tested = lifecycle?.tested && typeof lifecycle.tested === 'object' ? lifecycle.tested : {};

  const baseUrl = typeof model?.baseUrl === 'string' ? model.baseUrl.trim() : '';
  const fallbackDemo = baseUrl.startsWith('./local/') || baseUrl.startsWith('local/')
    ? 'local'
    : 'none';
  const demo = normalizeText(status.demo) || fallbackDemo;

  const hosted = typeof availability.hf === 'boolean'
    ? availability.hf
    : (model?.hf && typeof model.hf === 'object');

  const testedState = normalizeTestedState(tested.result || status.tested);
  const testedAt = typeof tested.lastVerifiedAt === 'string' && tested.lastVerifiedAt.trim()
    ? tested.lastVerifiedAt.trim()
    : null;

  return {
    hosted,
    demo,
    tested: testedState,
    testedAt,
  };
}

function mergeLifecycleAggregate(left, right) {
  const tested = left.tested === 'failed' || right.tested === 'failed'
    ? 'failed'
    : (left.tested === 'verified' || right.tested === 'verified' ? 'verified' : 'unknown');
  const testedAt = [left.testedAt, right.testedAt]
    .filter((value) => typeof value === 'string' && value.length > 0)
    .sort((a, b) => b.localeCompare(a))[0] || null;
  const demo = left.demo === 'curated' || right.demo === 'curated'
    ? 'curated'
    : (left.demo === 'local' || right.demo === 'local' ? 'local' : 'none');
  return {
    hosted: left.hosted || right.hosted,
    demo,
    tested,
    testedAt,
  };
}

export function resolveRowStatus(row) {
  if (row.conversionCount === 0) return 'missing-conversion';
  if (row.runtimeStatus === 'blocked') return 'blocked-runtime';
  if (row.catalogCount > 0) {
    if (row.lifecycleTested === 'verified') return 'verified';
    if (row.lifecycleTested === 'failed') return 'verification-failed';
    return 'verification-pending';
  }
  return 'conversion-ready';
}

function compareCatalogModels(left, right) {
  const leftOrder = Number.isFinite(left?.sortOrder) ? left.sortOrder : Number.POSITIVE_INFINITY;
  const rightOrder = Number.isFinite(right?.sortOrder) ? right.sortOrder : Number.POSITIVE_INFINITY;
  if (leftOrder !== rightOrder) {
    return leftOrder - rightOrder;
  }
  return normalizeModelId(left?.modelId).localeCompare(normalizeModelId(right?.modelId));
}

function buildCatalogModelStatusEntry(model) {
  const lifecycle = resolveCatalogLifecycle(model);
  const status = model?.lifecycle && typeof model.lifecycle === 'object' && model.lifecycle.status && typeof model.lifecycle.status === 'object'
    ? model.lifecycle.status
    : {};
  const tested = model?.lifecycle && typeof model.lifecycle === 'object' && model.lifecycle.tested && typeof model.lifecycle.tested === 'object'
    ? model.lifecycle.tested
    : {};
  return {
    modelId: normalizeModelId(model?.modelId),
    preset: normalizeText(model?.preset) || 'unknown',
    modes: summarizeModes(model),
    runtimeStatus: normalizeText(status.runtime) || 'unknown',
    tested: lifecycle.tested,
    testedAt: lifecycle.testedAt,
    surface: typeof tested.surface === 'string' && tested.surface.trim() ? tested.surface.trim() : null,
    notes: typeof tested.notes === 'string' && tested.notes.trim() ? tested.notes.trim() : null,
  };
}

function buildPresetCoverageEntry(row) {
  const notes = [];
  if (row.runtimeStatus === 'blocked') {
    notes.push('runtime path is fail-closed');
  } else {
    notes.push('conversion configs exist, but there is no cataloged model entry yet');
  }
  return {
    entry: row.presetId,
    type: 'preset family',
    status: row.status,
    notes: notes.join('; '),
  };
}

export function buildCurrentInferenceStatusBuckets({ catalogModels, quickStartModelIds, rows }) {
  const verified = [];
  const loadsButUnverified = [];
  const knownFailing = [];
  const everythingElseCatalog = [];
  const sortedCatalogModels = Array.isArray(catalogModels)
    ? catalogModels.slice().sort(compareCatalogModels)
    : [];
  const catalogModelIds = new Set();
  for (const model of sortedCatalogModels) {
    const entry = buildCatalogModelStatusEntry(model);
    catalogModelIds.add(entry.modelId);
    if (entry.tested === 'verified') {
      verified.push(entry);
      continue;
    }
    if (entry.tested === 'failed') {
      knownFailing.push(entry);
      continue;
    }
    if (entry.runtimeStatus === 'active') {
      loadsButUnverified.push(entry);
      continue;
    }
    everythingElseCatalog.push({
      entry: entry.modelId,
      type: 'catalog model',
      status: entry.runtimeStatus || 'unknown',
      notes: entry.notes || 'Cataloged model without a verified or failing inference lifecycle result.',
    });
  }

  const quickstartOnly = Array.isArray(quickStartModelIds)
    ? quickStartModelIds
      .filter((modelId) => typeof modelId === 'string' && modelId.trim() && !catalogModelIds.has(modelId.trim()))
      .sort((left, right) => left.localeCompare(right))
      .map((modelId) => ({
        modelId: modelId.trim(),
        source: 'quickstart registry',
        notes: 'Downloadable through the quickstart path, but not yet represented in models/catalog.json.',
      }))
    : [];

  const everythingElsePresets = Array.isArray(rows)
    ? rows
      .filter((row) => row.catalogCount === 0)
      .sort((left, right) => left.presetId.localeCompare(right.presetId))
      .map((row) => buildPresetCoverageEntry(row))
    : [];

  return {
    verified,
    loadsButUnverified,
    knownFailing,
    quickstartOnly,
    everythingElse: [...everythingElseCatalog, ...everythingElsePresets],
  };
}

function summarizeList(values, maxItems = 3) {
  if (values.length === 0) return '0';
  if (values.length <= maxItems) {
    return `${values.length} (${values.join(', ')})`;
  }
  const visible = values.slice(0, maxItems).join(', ');
  return `${values.length} (${visible}, +${values.length - maxItems} more)`;
}

function formatCell(value) {
  if (value === null || value === undefined || value === '') return '-';
  return String(value).replace(/\|/g, '\\|');
}

function pushTable(lines, headers, rows) {
  lines.push(`| ${headers.join(' | ')} |`);
  lines.push(`| ${headers.map(() => '---').join(' | ')} |`);
  for (const row of rows) {
    lines.push(`| ${row.map((value) => formatCell(value)).join(' | ')} |`);
  }
  lines.push('');
}

function renderCurrentInferenceStatus(lines, buckets) {
  lines.push('## Current Inference Status');
  lines.push('');
  lines.push('This section answers "which models work now?" from `models/catalog.json` lifecycle metadata plus the quickstart registry.');
  lines.push('');

  lines.push('### 1. Verified');
  lines.push('');
  if (buckets.verified.length === 0) {
    lines.push('None.');
    lines.push('');
  } else {
    pushTable(lines,
      ['Model ID', 'Preset', 'Modes', 'Last verified', 'Surface', 'Notes'],
      buckets.verified.map((entry) => [
        entry.modelId,
        entry.preset,
        entry.modes,
        entry.testedAt || null,
        entry.surface || null,
        entry.notes || null,
      ]));
  }

  lines.push('### 2. Loads But Unverified');
  lines.push('');
  if (buckets.loadsButUnverified.length === 0) {
    lines.push('None right now.');
    lines.push('');
  } else {
    pushTable(lines,
      ['Model ID', 'Preset', 'Modes', 'Runtime', 'Notes'],
      buckets.loadsButUnverified.map((entry) => [
        entry.modelId,
        entry.preset,
        entry.modes,
        entry.runtimeStatus,
        entry.notes || 'Cataloged model without a passing or failing verification result yet.',
      ]));
  }

  lines.push('### 3. Known Failing');
  lines.push('');
  if (buckets.knownFailing.length === 0) {
    lines.push('None right now.');
    lines.push('');
  } else {
    pushTable(lines,
      ['Model ID', 'Preset', 'Modes', 'Last checked', 'Surface', 'Notes'],
      buckets.knownFailing.map((entry) => [
        entry.modelId,
        entry.preset,
        entry.modes,
        entry.testedAt || null,
        entry.surface || null,
        entry.notes || null,
      ]));
  }

  lines.push('### 4. Quickstart-Supported Only');
  lines.push('');
  if (buckets.quickstartOnly.length === 0) {
    lines.push('None right now.');
    lines.push('');
  } else {
    pushTable(lines,
      ['Model ID', 'Source', 'Notes'],
      buckets.quickstartOnly.map((entry) => [
        entry.modelId,
        entry.source,
        entry.notes,
      ]));
  }

  lines.push('### 5. Everything Else');
  lines.push('');
  if (buckets.everythingElse.length === 0) {
    lines.push('None.');
    lines.push('');
  } else {
    pushTable(lines,
      ['Entry', 'Type', 'Status', 'Notes'],
      buckets.everythingElse.map((entry) => [
        entry.entry,
        entry.type,
        entry.status,
        entry.notes,
      ]));
  }
}

function renderMatrix(rows, metadata, buckets) {
  const lines = [];
  lines.push('# Model Support Matrix');
  lines.push('');
  lines.push('Auto-generated from preset registry (`src/config/loader.js`), conversion configs (`tools/configs/conversion/**`), and catalog (`models/catalog.json`).');
  lines.push('`models/catalog.json` lifecycle metadata is the canonical source for hosted/demo/tested status.');
  lines.push('Run `npm run support:matrix:sync` after adding/changing presets, conversion configs, or catalog entries.');
  lines.push('');
  lines.push(`Updated at: ${metadata.generatedAt}`);
  lines.push('');
  renderCurrentInferenceStatus(lines, buckets);
  lines.push('## Preset Coverage Matrix');
  lines.push('');
  lines.push('| Preset | Runtime modelType | Runtime | Conversion configs | Catalog models | Hosted (HF) | Demo | Tested | Status | Notes |');
  lines.push('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |');
  for (const row of rows) {
    const notes = [];
    if (row.runtimeStatus === 'blocked') {
      notes.push('fail-closed runtime path');
    }
    if (row.catalogCount === 0) {
      notes.push('not in local catalog');
    }
    if (row.lifecycleTested === 'unknown') {
      notes.push('not verified in catalog lifecycle');
    }
    const noteText = notes.length > 0 ? notes.join('; ') : '-';
    const testedLabel = row.lifecycleTested === 'verified' && row.lifecycleTestedAt
      ? `verified (${row.lifecycleTestedAt})`
      : row.lifecycleTested;
    lines.push(
      `| ${row.presetId} | ${row.runtimeModelType} | ${row.runtimeStatus} | ` +
      `${summarizeList(row.conversionFiles)} | ${summarizeList(row.catalogModels)} | ` +
      `${row.lifecycleHosted ? 'yes' : 'no'} | ${row.lifecycleDemo} | ${testedLabel} | ${row.status} | ${noteText} |`
    );
  }
  lines.push('');
  lines.push('## Summary');
  lines.push('');
  lines.push(`- Presets tracked: ${metadata.presetCount}`);
  lines.push(`- Presets with conversion configs: ${metadata.presetsWithConversion}`);
  lines.push(`- Presets present in catalog: ${metadata.presetsInCatalog}`);
  lines.push(`- Verified presets (active runtime + conversion + catalog + passing verification): ${metadata.verifiedReadyCount}`);
  lines.push(`- Cataloged presets pending verification: ${metadata.verificationPendingCount}`);
  lines.push(`- Presets with HF-hosted catalog entries: ${metadata.hostedCount}`);
  lines.push(`- Presets with verified catalog lifecycle: ${metadata.verifiedCount}`);
  lines.push(`- Presets with failed catalog verification: ${metadata.failedVerificationCount}`);
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
  const catalogInputErrors = validateCatalogMatrixInputs(catalogPayload);
  if (catalogInputErrors.length > 0) {
    throw new Error(`Catalog lifecycle metadata is invalid:\n${catalogInputErrors.join('\n')}`);
  }
  const catalogModels = Array.isArray(catalogPayload?.models) ? catalogPayload.models : [];
  const catalogByPreset = new Map(presetIds.map((presetId) => [presetId, []]));
  const lifecycleByPreset = new Map(
    presetIds.map((presetId) => [presetId, createEmptyLifecycleAggregate()])
  );
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
    lifecycleByPreset.set(
      inferredPreset,
      mergeLifecycleAggregate(
        lifecycleByPreset.get(inferredPreset) || createEmptyLifecycleAggregate(),
        resolveCatalogLifecycle(model)
      )
    );
  }
  for (const presetId of presetIds) {
    catalogByPreset.get(presetId).sort((left, right) => left.localeCompare(right));
  }

  const rows = presetOrder.map((presetId) => {
    const runtimeModelType = resolveRuntimeModelType(presetId);
    const runtimeStatus = resolveRuntimeStatus(runtimeModelType);
    const conversionFilesForPreset = conversionByPreset.get(presetId) || [];
    const catalogModelsForPreset = catalogByPreset.get(presetId) || [];
    const lifecycleForPreset = lifecycleByPreset.get(presetId) || createEmptyLifecycleAggregate();
    const row = {
      presetId,
      runtimeModelType,
      runtimeStatus,
      conversionFiles: conversionFilesForPreset,
      conversionCount: conversionFilesForPreset.length,
      catalogModels: catalogModelsForPreset,
      catalogCount: catalogModelsForPreset.length,
      lifecycleHosted: lifecycleForPreset.hosted === true,
      lifecycleDemo: lifecycleForPreset.demo,
      lifecycleTested: lifecycleForPreset.tested,
      lifecycleTestedAt: lifecycleForPreset.testedAt,
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

  const quickstartRegistry = await readJson(QUICKSTART_REGISTRY_PATH);
  const quickStartModelIds = Array.isArray(quickstartRegistry?.models)
    ? quickstartRegistry.models
      .map((entry) => normalizeText(entry?.modelId) ? String(entry.modelId).trim() : null)
      .filter((entry) => typeof entry === 'string' && entry.length > 0)
    : [];

  const buckets = buildCurrentInferenceStatusBuckets({
    catalogModels,
    quickStartModelIds,
    rows,
  });

  const metadata = {
    generatedAt: typeof catalogPayload?.updatedAt === 'string' && catalogPayload.updatedAt.trim()
      ? catalogPayload.updatedAt.trim()
      : 'unknown',
    presetCount: rows.length,
    presetsWithConversion: rows.filter((row) => row.conversionCount > 0).length,
    presetsInCatalog: rows.filter((row) => row.catalogCount > 0).length,
    verifiedReadyCount: rows.filter((row) => row.status === 'verified').length,
    verificationPendingCount: rows.filter((row) => row.status === 'verification-pending').length,
    hostedCount: rows.filter((row) => row.lifecycleHosted).length,
    verifiedCount: rows.filter((row) => row.lifecycleTested === 'verified').length,
    failedVerificationCount: rows.filter((row) => row.lifecycleTested === 'failed').length,
    blockedCount: rows.filter((row) => row.runtimeStatus === 'blocked').length,
    catalogCount: catalogModels.length,
  };
  const nextContent = renderMatrix(rows, metadata, buckets);

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

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(`[support-matrix] ${error.message}`);
    process.exit(1);
  });
}
