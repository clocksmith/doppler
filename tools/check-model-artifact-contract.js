#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_POLICY_PATH = path.join(REPO_ROOT, 'tools', 'policies', 'model-artifact-contract.json');
const ID_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function isPlainObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function resolveRepoPath(value, label) {
  const normalized = normalizeText(value);
  if (
    !normalized
    || path.isAbsolute(normalized)
    || normalized.includes('\\')
    || normalized.split('/').includes('..')
  ) {
    throw new Error(`${label} must be a repo-relative path`);
  }
  return path.join(REPO_ROOT, normalized);
}

async function readJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, 'utf8'));
}

function valueAtPath(object, fieldPath) {
  const parts = normalizeText(fieldPath).split('.').filter(Boolean);
  let current = object;
  for (const part of parts) {
    if (!isPlainObject(current) || !Object.prototype.hasOwnProperty.call(current, part)) {
      return undefined;
    }
    current = current[part];
  }
  return current;
}

function stableJsonValue(value) {
  if (Array.isArray(value)) {
    return value.map((entry) => stableJsonValue(entry));
  }
  if (!isPlainObject(value)) {
    return value;
  }
  return Object.fromEntries(
    Object.keys(value)
      .sort()
      .map((key) => [key, stableJsonValue(value[key])])
  );
}

function sameJson(left, right) {
  return JSON.stringify(stableJsonValue(left)) === JSON.stringify(stableJsonValue(right));
}

function validateStringArray(value, label, errors) {
  if (!Array.isArray(value) || value.length === 0) {
    errors.push(`${label} must be a non-empty array`);
    return [];
  }
  const seen = new Set();
  const out = [];
  for (const item of value) {
    const normalized = normalizeText(item);
    if (!normalized) {
      errors.push(`${label} entries must be non-empty strings`);
      continue;
    }
    if (seen.has(normalized)) {
      errors.push(`${label} contains duplicate entry ${normalized}`);
      continue;
    }
    seen.add(normalized);
    out.push(normalized);
  }
  return out;
}

function validatePolicy(policy) {
  const errors = [];
  if (!isPlainObject(policy)) {
    return ['model artifact policy must be an object'];
  }
  if (policy.schemaVersion !== 1) {
    errors.push('model artifact policy schemaVersion must be 1');
  }
  if (policy.source !== 'doppler') {
    errors.push('model artifact policy source must be "doppler"');
  }
  if (!isPlainObject(policy.registrySelection)) {
    errors.push('model artifact policy registrySelection must be an object');
  }
  validateStringArray(policy.mirroredFields, 'model artifact policy mirroredFields', errors);
  return errors;
}

function validateCatalog(catalog, errors) {
  if (!isPlainObject(catalog)) {
    errors.push('catalog must be an object');
    return new Map();
  }
  if (!Array.isArray(catalog.models)) {
    errors.push('catalog.models must be an array');
    return new Map();
  }

  const modelById = new Map();
  const aliasOwner = new Map();
  for (const model of catalog.models) {
    const modelId = normalizeText(model?.modelId);
    if (!modelId) {
      errors.push('catalog model is missing modelId');
      continue;
    }
    if (!ID_PATTERN.test(modelId)) {
      errors.push(`${modelId}: modelId must be lowercase kebab-case`);
    }
    if (modelById.has(modelId)) {
      errors.push(`${modelId}: duplicate catalog modelId`);
    }
    modelById.set(modelId, model);

    if (normalizeText(model?.artifact?.format) !== 'rdrr') {
      errors.push(`${modelId}: artifact.format must be rdrr`);
    }
    for (const field of ['sourceCheckpointId', 'weightPackId', 'manifestVariantId', 'artifactCompleteness', 'runtimePromotionState']) {
      if (!normalizeText(model?.[field])) {
        errors.push(`${modelId}: ${field} is required`);
      }
    }
    if (typeof model?.weightsRefAllowed !== 'boolean') {
      errors.push(`${modelId}: weightsRefAllowed must be boolean`);
    }
    const aliases = validateStringArray(model?.aliases, `${modelId}: aliases`, errors);
    validateStringArray(model?.modes, `${modelId}: modes`, errors);
    for (const alias of aliases) {
      const owner = aliasOwner.get(alias);
      if (owner && owner !== modelId) {
        errors.push(`${modelId}: alias ${alias} is already owned by ${owner}`);
      }
      aliasOwner.set(alias, modelId);
    }
  }
  for (const [alias, owner] of aliasOwner) {
    if (modelById.has(alias) && alias !== owner) {
      errors.push(`${owner}: alias ${alias} shadows catalog modelId ${alias}`);
    }
  }
  return modelById;
}

function validateRegistry(registry, errors) {
  if (!isPlainObject(registry)) {
    errors.push('quickstart registry must be an object');
    return [];
  }
  if (normalizeText(registry.source) !== 'models/catalog.json') {
    errors.push('quickstart registry source must be models/catalog.json');
  }
  if (!Array.isArray(registry.models)) {
    errors.push('quickstart registry models must be an array');
    return [];
  }
  const seen = new Set();
  for (const model of registry.models) {
    const modelId = normalizeText(model?.modelId);
    if (!modelId) {
      errors.push('quickstart registry model is missing modelId');
      continue;
    }
    if (seen.has(modelId)) {
      errors.push(`${modelId}: duplicate quickstart registry modelId`);
    }
    seen.add(modelId);
  }
  return registry.models;
}

function modelMatchesSelection(model, selection) {
  return model?.quickstart === selection.quickstart
    && normalizeText(model?.artifact?.format) === selection.artifactFormat
    && normalizeText(model?.artifactCompleteness) === selection.artifactCompleteness
    && normalizeText(model?.runtimePromotionState) === selection.runtimePromotionState
    && model?.weightsRefAllowed === selection.weightsRefAllowed
    && model?.lifecycle?.availability?.hf === selection.hfAvailability
    && normalizeText(model?.lifecycle?.status?.runtime) === selection.runtimeStatus
    && normalizeText(model?.lifecycle?.status?.tested) === selection.testedStatus
    && normalizeText(model?.lifecycle?.tested?.result) === selection.testedResult
    && model?.lifecycle?.tested?.contracts?.executionContractOk === selection.executionContractOk
    && isPlainObject(model?.hf);
}

function validateRegistryAgainstCatalog(policy, modelById, registryModels, errors) {
  const selection = policy.registrySelection;
  const mirroredFields = policy.mirroredFields;
  const expectedRegistryIds = new Set();
  for (const [modelId, model] of modelById) {
    if (modelMatchesSelection(model, selection)) {
      expectedRegistryIds.add(modelId);
    }
  }

  const actualRegistryIds = new Set();
  for (const registryModel of registryModels) {
    const modelId = normalizeText(registryModel?.modelId);
    if (!modelId) continue;
    actualRegistryIds.add(modelId);
    const catalogModel = modelById.get(modelId);
    if (!catalogModel) {
      errors.push(`${modelId}: quickstart registry model is not declared in catalog`);
      continue;
    }
    if (!modelMatchesSelection(catalogModel, selection)) {
      errors.push(`${modelId}: quickstart registry model does not satisfy registry selection policy`);
    }
    for (const field of mirroredFields) {
      const registryValue = valueAtPath(registryModel, field);
      const catalogValue = valueAtPath(catalogModel, field);
      if (!sameJson(registryValue, catalogValue)) {
        errors.push(`${modelId}: registry field ${field} does not match catalog`);
      }
    }
  }

  for (const modelId of expectedRegistryIds) {
    if (!actualRegistryIds.has(modelId)) {
      errors.push(`${modelId}: catalog model satisfies registry selection but is missing from quickstart registry`);
    }
  }
}

export async function buildModelArtifactContractReport(options = {}) {
  const policyPath = options.policyPath || DEFAULT_POLICY_PATH;
  const policy = await readJson(policyPath);
  const errors = validatePolicy(policy);
  if (errors.length > 0) {
    return {
      ok: false,
      policyPath: path.relative(REPO_ROOT, policyPath),
      errors,
      catalogModels: 0,
      registryModels: 0,
    };
  }

  const catalogPath = resolveRepoPath(policy.catalogPath, 'catalogPath');
  const registryPath = resolveRepoPath(policy.quickstartRegistryPath, 'quickstartRegistryPath');
  const [catalog, registry] = await Promise.all([
    readJson(catalogPath),
    readJson(registryPath),
  ]);
  const modelById = validateCatalog(catalog, errors);
  const registryModels = validateRegistry(registry, errors);
  validateRegistryAgainstCatalog(policy, modelById, registryModels, errors);

  return {
    ok: errors.length === 0,
    policyPath: path.relative(REPO_ROOT, policyPath),
    catalogPath: policy.catalogPath,
    quickstartRegistryPath: policy.quickstartRegistryPath,
    errors,
    catalogModels: modelById.size,
    registryModels: registryModels.length,
  };
}

export async function main(argv = process.argv.slice(2)) {
  const json = argv.includes('--json');
  const unsupported = argv.filter((token) => token !== '--json');
  if (unsupported.length > 0) {
    throw new Error(`Unknown argument: ${unsupported[0]}`);
  }
  const report = await buildModelArtifactContractReport();
  if (json) {
    console.log(JSON.stringify(report, null, 2));
  } else if (report.ok) {
    console.log(`model-artifact-contract: registry ok (${report.registryModels}/${report.catalogModels} catalog models exposed)`);
  } else {
    for (const error of report.errors) {
      console.error(`model-artifact-contract: ${error}`);
    }
  }
  if (!report.ok) {
    process.exitCode = 1;
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error?.stack || error?.message || String(error));
    process.exitCode = 1;
  });
}
