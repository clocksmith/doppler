#!/usr/bin/env node

import fs from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';

import {
  DEFAULT_EXTERNAL_MODELS_ROOT,
  DEFAULT_EXTERNAL_RDRR_INDEX_PATH,
  DEFAULT_EXTERNAL_SUPPORT_REGISTRY_PATH,
  ensureCatalogPayload,
  getEntryHfSpec,
  loadJsonFile,
  normalizeRepoPath,
  normalizeText,
} from './hf-registry-utils.js';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_CATALOG_FILE = path.join(REPO_ROOT, 'models', 'catalog.json');
const DEFAULT_VOLUME_ROOT = DEFAULT_EXTERNAL_MODELS_ROOT;
const DEFAULT_RDRR_INDEX = DEFAULT_EXTERNAL_RDRR_INDEX_PATH;
const DEFAULT_JSON_OUTPUT = DEFAULT_EXTERNAL_SUPPORT_REGISTRY_PATH;
const DEFAULT_MD_OUTPUT = path.join(DEFAULT_VOLUME_ROOT, 'DOPPLER_SUPPORT_REGISTRY.md');

function parseArgs(argv) {
  const out = {
    check: false,
    volumeRoot: path.resolve(DEFAULT_VOLUME_ROOT),
    rdrrIndex: path.resolve(DEFAULT_RDRR_INDEX),
    catalogFile: path.resolve(DEFAULT_CATALOG_FILE),
    sourceSupportFile: '',
    jsonOutput: path.resolve(DEFAULT_JSON_OUTPUT),
    mdOutput: path.resolve(DEFAULT_MD_OUTPUT),
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const nextValue = () => {
      const value = String(argv[i + 1] || '').trim();
      if (!value) {
        throw new Error(`Missing value for ${arg}`);
      }
      i += 1;
      return value;
    };

    if (arg === '--check') {
      out.check = true;
      continue;
    }
    if (arg === '--volume-root') {
      out.volumeRoot = path.resolve(nextValue());
      continue;
    }
    if (arg === '--rdrr-index') {
      out.rdrrIndex = path.resolve(nextValue());
      continue;
    }
    if (arg === '--catalog-file') {
      out.catalogFile = path.resolve(nextValue());
      continue;
    }
    if (arg === '--source-support-file') {
      out.sourceSupportFile = path.resolve(nextValue());
      continue;
    }
    if (arg === '--json-output') {
      out.jsonOutput = path.resolve(nextValue());
      continue;
    }
    if (arg === '--md-output') {
      out.mdOutput = path.resolve(nextValue());
      continue;
    }
    throw new Error(`Unknown argument: ${arg}`);
  }

  return out;
}

function resolveSupportSourceFile(args) {
  if (normalizeText(args.sourceSupportFile)) {
    return args.sourceSupportFile;
  }
  if (existsSync(args.jsonOutput)) {
    return args.jsonOutput;
  }
  return args.catalogFile;
}

function ensureExternalIndexPayload(payload, label = 'external RDRR index') {
  if (!payload || typeof payload !== 'object' || !Array.isArray(payload.sourceModels)) {
    throw new Error(`${label} payload must be an object with a sourceModels array.`);
  }
  return payload;
}

function toPosix(filePath) {
  return String(filePath || '').replace(/\\/g, '/');
}

function escapeCell(value) {
  return String(value ?? '').replace(/\|/g, '\\|').replace(/\n/g, ' ');
}

function flattenRdrrVariants(payload) {
  const out = [];
  for (const source of payload.sourceModels) {
    const sourceModel = normalizeText(source?.sourceModel);
    const sourceFormats = Array.isArray(source?.sourceFormats)
      ? source.sourceFormats.map((value) => normalizeText(value)).filter(Boolean)
      : [];
    const variants = Array.isArray(source?.variants) ? source.variants : [];
    for (const variant of variants) {
      out.push({
        sourceModel,
        sourceFormats,
        rdrrModelId: normalizeText(variant?.rdrrModelId),
        variant: normalizeText(variant?.variant),
        quantization: normalizeText(variant?.quantization),
        convertedAt: normalizeText(variant?.convertedAt) || null,
        totalSizeBytes: Number.isFinite(Number(variant?.totalSizeBytes)) ? Number(variant.totalSizeBytes) : 0,
        shardCount: Number.isFinite(Number(variant?.shardCount)) ? Number(variant.shardCount) : 0,
        sourceFormat: normalizeText(variant?.sourceFormat) || sourceFormats[0] || null,
        sourceRevision: normalizeText(variant?.sourceRevision) || null,
        pathRelativeToVolume: toPosix(variant?.pathRelativeToVolume || ''),
        pathRelativeToRdrrRoot: toPosix(variant?.pathRelativeToRdrrRoot || ''),
        manifestPath: toPosix(variant?.manifestPath || ''),
        hasOrigin: variant?.hasOrigin === true,
      });
    }
  }
  return out.filter((variant) => variant.rdrrModelId);
}

function collectEntryTokens(entry) {
  const tokens = new Set();
  const addToken = (value) => {
    const normalized = normalizeText(value);
    if (normalized) {
      tokens.add(normalized);
    }
  };

  addToken(entry?.modelId);
  const aliases = Array.isArray(entry?.aliases) ? entry.aliases : [];
  for (const alias of aliases) {
    addToken(alias);
  }

  const hfPath = normalizeRepoPath(getEntryHfSpec(entry).path);
  if (hfPath) {
    addToken(path.posix.basename(hfPath));
  }

  return [...tokens];
}

function resolveMatchedVariant(entry, variantsById) {
  const matches = [];
  const seen = new Set();
  for (const token of collectEntryTokens(entry)) {
    const variant = variantsById.get(token);
    if (!variant || seen.has(variant.rdrrModelId)) {
      continue;
    }
    seen.add(variant.rdrrModelId);
    matches.push(variant);
  }

  if (matches.length > 1) {
    throw new Error(
      `${normalizeText(entry?.modelId) || 'unknown-model'}: ambiguous external RDRR match ` +
      `(${matches.map((variant) => variant.rdrrModelId).join(', ')})`
    );
  }

  return matches[0] || null;
}

function buildExternalBlock(entry, variant) {
  const hfSpec = getEntryHfSpec(entry);
  const hostedPathBasename = hfSpec.path ? path.posix.basename(normalizeRepoPath(hfSpec.path)) : '';
  if (!variant) {
    return null;
  }
  return {
    rdrrModelId: variant.rdrrModelId,
    manifestModelId: normalizeText(variant?.manifestModelId) || null,
    manifestModelIdMatchesCatalogModelId: normalizeText(variant?.manifestModelId)
      ? normalizeText(variant.manifestModelId) === normalizeText(entry?.modelId)
      : null,
    sourceModel: variant.sourceModel,
    sourceFormats: variant.sourceFormats,
    sourceFormat: variant.sourceFormat,
    sourceRevision: variant.sourceRevision,
    variant: variant.variant,
    quantization: variant.quantization,
    convertedAt: variant.convertedAt,
    totalSizeBytes: variant.totalSizeBytes,
    shardCount: variant.shardCount,
    manifestPath: variant.manifestPath,
    pathRelativeToVolume: variant.pathRelativeToVolume,
    pathRelativeToRdrrRoot: variant.pathRelativeToRdrrRoot,
    hasOrigin: variant.hasOrigin,
    hostedPathBasename: hostedPathBasename || null,
    hostedPathMatchesRdrr: Boolean(hostedPathBasename && hostedPathBasename === variant.rdrrModelId),
  };
}

async function loadVariantManifestIdentity(variant, volumeRoot) {
  const relPath = normalizeText(variant?.pathRelativeToVolume);
  const manifestPath = relPath
    ? path.join(volumeRoot, relPath, 'manifest.json')
    : normalizeText(variant?.manifestPath);
  if (!manifestPath) {
    return {
      manifestModelId: null,
    };
  }
  try {
    const manifest = JSON.parse(await fs.readFile(manifestPath, 'utf8'));
    return {
      manifestModelId: normalizeText(manifest?.modelId) || null,
    };
  } catch (error) {
    throw new Error(
      `${normalizeText(variant?.rdrrModelId) || 'unknown-rdrr-model'}: failed to read manifest identity ` +
      `from ${manifestPath} (${error.message})`
    );
  }
}

async function buildCatalogBackedEntries(catalog, variantsById, volumeRoot) {
  const models = Array.isArray(catalog?.models) ? catalog.models : [];
  const entries = [];
  const matchedVariantIds = new Set();
  const errors = [];

  for (const model of models) {
    const entry = structuredClone(model);
    const format = normalizeText(entry?.artifact?.format);
    let variant = null;
    if (format === 'rdrr') {
      try {
        variant = resolveMatchedVariant(entry, variantsById);
      } catch (error) {
        errors.push(error.message);
        continue;
      }
    }
    if (format === 'rdrr' && !variant) {
      errors.push(
        `${normalizeText(entry?.modelId) || 'unknown-model'}: missing external RDRR variant match in the canonical external inventory`
      );
      continue;
    }
    if (variant) {
      const manifestIdentity = await loadVariantManifestIdentity(variant, volumeRoot);
      matchedVariantIds.add(variant.rdrrModelId);
      entry.external = buildExternalBlock(entry, {
        ...variant,
        ...manifestIdentity,
      });
      if (!Number.isFinite(Number(entry.sizeBytes)) || Number(entry.sizeBytes) <= 0) {
        entry.sizeBytes = variant.totalSizeBytes;
      }
    } else {
      entry.external = null;
    }
    entries.push(entry);
  }

  return { entries, matchedVariantIds, errors };
}

function buildUncatalogedVariants(variants, matchedVariantIds) {
  return variants
    .filter((variant) => !matchedVariantIds.has(variant.rdrrModelId))
    .sort((left, right) => left.rdrrModelId.localeCompare(right.rdrrModelId));
}

function buildMarkdown(payload) {
  const lines = [];
  lines.push('# Doppler External Support Registry');
  lines.push('');
  lines.push(`Generated: ${payload.generatedAt}`);
  lines.push(`Volume root: \`${payload.volumeRoot}\``);
  lines.push(`Source support registry: \`${payload.supportSource}\``);
  lines.push(`Source RDRR index: \`${payload.rdrrIndexSource}\``);
  lines.push('');
  lines.push(`- Catalog-backed models: ${payload.summary.catalogModelCount}`);
  lines.push(`- Catalog-backed external RDRR variants: ${payload.summary.catalogBackedRdrrCount}`);
  lines.push(`- Verified models: ${payload.summary.verifiedCount}`);
  lines.push(`- HF-approved models: ${payload.summary.hfApprovedCount}`);
  lines.push(`- HF path mismatches vs external RDRR: ${payload.summary.hfPathMismatchCount}`);
  lines.push(`- Manifest modelId mismatches vs catalog modelId: ${payload.summary.manifestModelIdMismatchCount}`);
  lines.push(`- Uncataloged external RDRR variants: ${payload.summary.uncatalogedRdrrCount}`);
  lines.push('');
  lines.push('## Catalog-backed Models');
  lines.push('');
  lines.push('| Model ID | Tested | HF | External RDRR | Manifest Model ID Match | HF Path Match | Source Model | Manifest |');
  lines.push('| --- | --- | --- | --- | --- | --- | --- | --- |');
  for (const model of payload.models) {
    const lifecycle = model?.lifecycle && typeof model.lifecycle === 'object' ? model.lifecycle : {};
    const status = lifecycle.status && typeof lifecycle.status === 'object' ? lifecycle.status : {};
    const availability = lifecycle.availability && typeof lifecycle.availability === 'object' ? lifecycle.availability : {};
    const external = model?.external && typeof model.external === 'object' ? model.external : null;
    lines.push(
      `| ${escapeCell(model.modelId)} | ${escapeCell(status.tested || 'unknown')} | ${availability.hf === true ? 'yes' : 'no'} | ${escapeCell(external?.rdrrModelId || '—')} | ${external ? (external.manifestModelIdMatchesCatalogModelId === true ? 'yes' : 'no') : '—'} | ${external ? (external.hostedPathMatchesRdrr ? 'yes' : 'no') : '—'} | ${escapeCell(external?.sourceModel || '—')} | ${escapeCell(external?.pathRelativeToVolume || '—')} |`
    );
  }
  lines.push('');
  lines.push('## Uncataloged External RDRR Variants');
  lines.push('');
  lines.push('| External RDRR | Source Model | Variant | Quantization | Converted | Path |');
  lines.push('| --- | --- | --- | --- | --- | --- |');
  for (const variant of payload.uncatalogedRdrrVariants) {
    lines.push(
      `| ${escapeCell(variant.rdrrModelId)} | ${escapeCell(variant.sourceModel)} | ${escapeCell(variant.variant || 'unknown')} | ${escapeCell(variant.quantization || 'unknown')} | ${escapeCell(variant.convertedAt || 'unknown')} | ${escapeCell(variant.pathRelativeToVolume)} |`
    );
  }
  lines.push('');
  return `${lines.join('\n')}\n`;
}

export async function buildExternalSupportRegistry(args, generatedAt = new Date().toISOString()) {
  const supportSource = resolveSupportSourceFile(args);
  const sourceSupportRegistry = ensureCatalogPayload(
    await loadJsonFile(supportSource, supportSource),
    supportSource
  );
  const rdrrIndex = ensureExternalIndexPayload(
    JSON.parse(await fs.readFile(args.rdrrIndex, 'utf8')),
    args.rdrrIndex
  );
  const flattenedVariants = flattenRdrrVariants(rdrrIndex);
  const variantsById = new Map(flattenedVariants.map((variant) => [variant.rdrrModelId, variant]));
  const { entries, matchedVariantIds, errors } = await buildCatalogBackedEntries(sourceSupportRegistry, variantsById, args.volumeRoot);
  if (errors.length > 0) {
    throw new Error(errors.join('\n'));
  }
  const uncatalogedRdrrVariants = buildUncatalogedVariants(flattenedVariants, matchedVariantIds);

  const verifiedCount = entries.filter((entry) => normalizeText(entry?.lifecycle?.status?.tested) === 'verified').length;
  const hfApprovedCount = entries.filter((entry) => entry?.lifecycle?.availability?.hf === true).length;
  const hfPathMismatchCount = entries.filter(
    (entry) => entry?.external && entry.external.hostedPathMatchesRdrr === false
  ).length;
  const manifestModelIdMismatchCount = entries.filter(
    (entry) => entry?.external && entry.external.manifestModelIdMatchesCatalogModelId === false
  ).length;

  const payload = {
    version: Number.isFinite(Number(sourceSupportRegistry.version)) ? Number(sourceSupportRegistry.version) : 1,
    lifecycleSchemaVersion: Number.isFinite(Number(sourceSupportRegistry.lifecycleSchemaVersion))
      ? Number(sourceSupportRegistry.lifecycleSchemaVersion)
      : 1,
    updatedAt: normalizeText(sourceSupportRegistry.updatedAt) || generatedAt.slice(0, 10),
    generatedAt,
    volumeRoot: toPosix(args.volumeRoot),
    supportSource: toPosix(supportSource),
    rdrrIndexSource: toPosix(args.rdrrIndex),
    summary: {
      catalogModelCount: entries.length,
      catalogBackedRdrrCount: matchedVariantIds.size,
      verifiedCount,
      hfApprovedCount,
      hfPathMismatchCount,
      manifestModelIdMismatchCount,
      uncatalogedRdrrCount: uncatalogedRdrrVariants.length,
    },
    models: entries,
    uncatalogedRdrrVariants,
  };

  return {
    payload,
    json: `${JSON.stringify(payload, null, 2)}\n`,
    md: buildMarkdown(payload),
  };
}

export async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  const outputs = await buildExternalSupportRegistry(args);

  if (args.check) {
    const [currentJson, currentMd] = await Promise.all([
      fs.readFile(args.jsonOutput, 'utf8'),
      fs.readFile(args.mdOutput, 'utf8'),
    ]);
    if (currentJson !== outputs.json || currentMd !== outputs.md) {
      throw new Error(
        'External support registry is out of date. Run: node tools/sync-external-support-registry.js'
      );
    }
    console.log(
      `[external-support-registry] up to date (${outputs.payload.summary.catalogModelCount} catalog-backed models)`
    );
    return;
  }

  await fs.writeFile(args.jsonOutput, outputs.json, 'utf8');
  await fs.writeFile(args.mdOutput, outputs.md, 'utf8');
  console.log(
    `[external-support-registry] wrote ${outputs.payload.summary.catalogModelCount} catalog-backed models to ${toPosix(args.jsonOutput)}`
  );
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(`[external-support-registry] ${error.message}`);
    process.exit(1);
  });
}
