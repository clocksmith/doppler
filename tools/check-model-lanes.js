#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_CATALOG_PATH = path.join(REPO_ROOT, 'models', 'catalog.json');

function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function isPlainObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function parseArgs(argv) {
  const args = {
    catalogPath: DEFAULT_CATALOG_PATH,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    const nextValue = () => {
      const candidate = argv[i + 1];
      if (candidate == null || String(candidate).startsWith('--')) {
        throw new Error(`Missing value for ${token}`);
      }
      i += 1;
      return path.resolve(REPO_ROOT, String(candidate).trim());
    };
    if (token === '--catalog') {
      args.catalogPath = nextValue();
      continue;
    }
    throw new Error(`Unknown argument: ${token}`);
  }
  return args;
}

async function readJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, 'utf8'));
}

function hasCompleteHfPointer(entry) {
  const hf = isPlainObject(entry?.hf) ? entry.hf : null;
  return Boolean(
    normalizeText(hf?.repoId)
    && normalizeText(hf?.revision)
    && normalizeText(hf?.path)
  );
}

function hasAnyHfPointer(entry) {
  const hf = isPlainObject(entry?.hf) ? entry.hf : null;
  return Boolean(
    normalizeText(hf?.repoId)
    || normalizeText(hf?.revision)
    || normalizeText(hf?.path)
  );
}

function isPrimaryLane(entry) {
  return entry?.artifactCompleteness === 'complete' && entry?.weightsRefAllowed === false;
}

function isWeightsRefLane(entry) {
  return entry?.artifactCompleteness === 'weights-ref' && entry?.weightsRefAllowed === true;
}

function validateCatalog(catalog) {
  const errors = [];
  if (!isPlainObject(catalog) || !Array.isArray(catalog.models)) {
    return ['models/catalog.json must be an object with a models array'];
  }

  const byModelId = new Map();
  const primaryByWeightPackId = new Map();
  for (const entry of catalog.models) {
    const modelId = normalizeText(entry?.modelId);
    if (!modelId) {
      errors.push('catalog entry is missing modelId');
      continue;
    }
    if (byModelId.has(modelId)) {
      errors.push(`${modelId}: duplicate modelId`);
    }
    byModelId.set(modelId, entry);
    if (isPrimaryLane(entry) && normalizeText(entry.weightPackId)) {
      primaryByWeightPackId.set(entry.weightPackId, entry);
    }
  }

  for (const entry of catalog.models) {
    const modelId = normalizeText(entry?.modelId) || 'unknown-model';
    const availability = isPlainObject(entry?.lifecycle?.availability)
      ? entry.lifecycle.availability
      : {};
    const hfAvailable = availability.hf === true;

    if (hfAvailable && !hasCompleteHfPointer(entry)) {
      errors.push(`${modelId}: lifecycle.availability.hf=true requires hf.repoId, hf.revision, and hf.path`);
    }
    if (!hfAvailable && hasAnyHfPointer(entry)) {
      errors.push(`${modelId}: hf metadata must be absent unless lifecycle.availability.hf=true`);
    }

    if (entry?.runtimePromotionState === 'manifest-owned') {
      for (const field of ['sourceCheckpointId', 'weightPackId', 'manifestVariantId', 'artifactCompleteness']) {
        if (!normalizeText(entry?.[field])) {
          errors.push(`${modelId}: runtimePromotionState=manifest-owned requires ${field}`);
        }
      }
    }

    if (isWeightsRefLane(entry)) {
      const primary = primaryByWeightPackId.get(entry.weightPackId);
      if (!primary) {
        errors.push(`${modelId}: weights-ref lane has no complete primary for weightPackId=${entry.weightPackId}`);
      } else if (normalizeText(primary.sourceCheckpointId) !== normalizeText(entry.sourceCheckpointId)) {
        errors.push(`${modelId}: weights-ref sourceCheckpointId must match primary ${primary.modelId}`);
      }
    }

    const preferredId = normalizeText(entry?.demoPreferredVariantId);
    if (preferredId) {
      const preferred = byModelId.get(preferredId);
      if (!preferred) {
        errors.push(`${modelId}: demoPreferredVariantId points at missing model ${preferredId}`);
      } else {
        if (!isPrimaryLane(entry)) {
          errors.push(`${modelId}: demoPreferredVariantId is only valid on a complete primary lane`);
        }
        if (!isWeightsRefLane(preferred)) {
          errors.push(`${modelId}: demoPreferredVariantId target ${preferredId} must be artifactCompleteness=weights-ref and weightsRefAllowed=true`);
        }
        if (normalizeText(preferred.weightPackId) !== normalizeText(entry.weightPackId)) {
          errors.push(`${modelId}: demoPreferredVariantId target ${preferredId} must share weightPackId`);
        }
        if (preferred?.runtimePromotionState !== 'manifest-owned') {
          errors.push(`${modelId}: demoPreferredVariantId target ${preferredId} must be runtimePromotionState=manifest-owned`);
        }
      }
    }
  }

  return errors;
}

export async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  const catalog = await readJson(args.catalogPath);
  const errors = validateCatalog(catalog);
  if (errors.length > 0) {
    for (const error of errors) {
      console.error(`model-lanes: ${error}`);
    }
    process.exitCode = 1;
    return;
  }
  console.log('model-lanes: catalog lane metadata ok');
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error?.stack || error?.message || String(error));
    process.exitCode = 1;
  });
}
