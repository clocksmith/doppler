import fs from 'node:fs';

const TAXONOMY_URL = new URL('../../models/model-type-taxonomy.json', import.meta.url);

function readCanonicalTaxonomy() {
  return JSON.parse(fs.readFileSync(TAXONOMY_URL, 'utf8'));
}

export const MODEL_TYPE_TAXONOMY = Object.freeze(readCanonicalTaxonomy());

function isObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function compareText(left, right) {
  return left.localeCompare(right);
}

function validateVocabulary(values, field) {
  const errors = [];
  if (!Array.isArray(values) || values.length === 0) {
    return [`taxonomy ${field} must be a non-empty array`];
  }
  const seen = new Set();
  for (const value of values) {
    const normalized = normalizeText(value);
    if (!normalized) {
      errors.push(`taxonomy ${field} entries must be non-empty strings`);
      continue;
    }
    if (seen.has(normalized)) {
      errors.push(`taxonomy ${field} contains duplicate value "${normalized}"`);
    }
    seen.add(normalized);
  }
  return errors;
}

function validateClassificationList(values, field, vocabulary) {
  const errors = [];
  if (!Array.isArray(values) || values.length === 0) {
    return [`classification.${field} must be a non-empty array`];
  }
  const seen = new Set();
  for (const value of values) {
    const normalized = normalizeText(value);
    if (!normalized) {
      errors.push(`classification.${field} entries must be non-empty strings`);
      continue;
    }
    if (!vocabulary.has(normalized)) {
      errors.push(`classification.${field} contains unknown value "${normalized}"`);
    }
    if (seen.has(normalized)) {
      errors.push(`classification.${field} contains duplicate value "${normalized}"`);
    }
    seen.add(normalized);
  }
  return errors;
}

function matchesCluster(classification, cluster) {
  const match = cluster.match;
  if (classification.domain !== match.domain) return false;
  if (!classification.tasks.includes(match.task)) return false;
  if (match.architectureRole && classification.architectureRole !== match.architectureRole) return false;
  if (Array.isArray(match.inputsExact)) {
    const expected = [...match.inputsExact].sort(compareText);
    const actual = [...classification.inputs].sort(compareText);
    if (expected.length !== actual.length) return false;
    if (expected.some((value, index) => value !== actual[index])) return false;
  }
  if (Array.isArray(match.inputsAny) && !match.inputsAny.some((value) => classification.inputs.includes(value))) {
    return false;
  }
  return true;
}

export function validateModelTypeTaxonomy(taxonomy = MODEL_TYPE_TAXONOMY) {
  const errors = [];
  if (!isObject(taxonomy)) {
    return ['model type taxonomy must be an object'];
  }
  if (taxonomy.schema !== 'doppler.model-type-taxonomy.v1') {
    errors.push('taxonomy schema must be "doppler.model-type-taxonomy.v1"');
  }
  if (taxonomy.version !== 1) {
    errors.push('taxonomy version must be 1');
  }
  for (const field of ['domains', 'tasks', 'architectureRoles', 'inputs', 'outputs']) {
    errors.push(...validateVocabulary(taxonomy[field], field));
  }
  if (!Array.isArray(taxonomy.clusters) || taxonomy.clusters.length === 0) {
    errors.push('taxonomy clusters must be a non-empty array');
    return errors;
  }

  const vocabularies = {
    domain: new Set(taxonomy.domains || []),
    task: new Set(taxonomy.tasks || []),
    architectureRole: new Set(taxonomy.architectureRoles || []),
    input: new Set(taxonomy.inputs || []),
  };
  const clusterIds = new Set();
  for (const cluster of taxonomy.clusters) {
    const clusterId = normalizeText(cluster?.id);
    if (!clusterId) {
      errors.push('taxonomy cluster id must be a non-empty string');
      continue;
    }
    if (clusterIds.has(clusterId)) {
      errors.push(`taxonomy contains duplicate cluster id "${clusterId}"`);
    }
    clusterIds.add(clusterId);
    if (!normalizeText(cluster?.label)) errors.push(`${clusterId}: label is required`);
    if (!normalizeText(cluster?.description)) errors.push(`${clusterId}: description is required`);
    if (!isObject(cluster?.match)) {
      errors.push(`${clusterId}: match must be an object`);
      continue;
    }
    for (const field of ['domain', 'task']) {
      const value = normalizeText(cluster.match[field]);
      if (!vocabularies[field].has(value)) {
        errors.push(`${clusterId}: match.${field} contains unknown value "${value}"`);
      }
    }
    const architectureRole = normalizeText(cluster.match.architectureRole);
    if (architectureRole && !vocabularies.architectureRole.has(architectureRole)) {
      errors.push(`${clusterId}: match.architectureRole contains unknown value "${architectureRole}"`);
    }
    for (const field of ['inputsExact', 'inputsAny']) {
      if (cluster.match[field] === undefined) continue;
      errors.push(...validateClassificationList(cluster.match[field], field, vocabularies.input)
        .map((error) => `${clusterId}: ${error.replace('classification.', 'match.')}`));
    }
  }
  return errors;
}

export function validateModelClassification(classification, taxonomy = MODEL_TYPE_TAXONOMY) {
  const errors = [];
  if (!isObject(classification)) {
    return ['classification must be an object'];
  }
  const domains = new Set(taxonomy.domains || []);
  const architectureRoles = new Set(taxonomy.architectureRoles || []);
  const domain = normalizeText(classification.domain);
  const architectureRole = normalizeText(classification.architectureRole);
  if (!domains.has(domain)) {
    errors.push(`classification.domain contains unknown value "${domain}"`);
  }
  if (!architectureRoles.has(architectureRole)) {
    errors.push(`classification.architectureRole contains unknown value "${architectureRole}"`);
  }
  errors.push(...validateClassificationList(classification.tasks, 'tasks', new Set(taxonomy.tasks || [])));
  errors.push(...validateClassificationList(classification.inputs, 'inputs', new Set(taxonomy.inputs || [])));
  errors.push(...validateClassificationList(classification.outputs, 'outputs', new Set(taxonomy.outputs || [])));
  return errors;
}

export function resolveModelTypeCluster(classification, taxonomy = MODEL_TYPE_TAXONOMY) {
  const classificationErrors = validateModelClassification(classification, taxonomy);
  if (classificationErrors.length > 0) {
    throw new Error(classificationErrors.join('; '));
  }
  const matches = taxonomy.clusters.filter((cluster) => matchesCluster(classification, cluster));
  if (matches.length !== 1) {
    const ids = matches.map((cluster) => cluster.id).join(', ') || 'none';
    throw new Error(`classification must resolve to exactly one model type cluster; matched ${ids}`);
  }
  return matches[0];
}

export function validateCatalogClassifications(catalog, taxonomy = MODEL_TYPE_TAXONOMY) {
  const errors = validateModelTypeTaxonomy(taxonomy);
  if (errors.length > 0) return errors;
  if (!isObject(catalog) || !Array.isArray(catalog.models)) {
    return ['catalog must be an object with a models array'];
  }
  for (const model of catalog.models) {
    const modelId = normalizeText(model?.modelId) || 'unknown-model';
    const classificationErrors = validateModelClassification(model?.classification, taxonomy);
    errors.push(...classificationErrors.map((error) => `${modelId}: ${error}`));
    if (classificationErrors.length > 0) continue;
    try {
      resolveModelTypeCluster(model.classification, taxonomy);
    } catch (error) {
      errors.push(`${modelId}: ${error.message}`);
    }
  }
  return errors;
}

export function buildModelTypeClusters(models, taxonomy = MODEL_TYPE_TAXONOMY) {
  const groups = new Map(taxonomy.clusters.map((cluster) => [cluster.id, {
    ...cluster,
    models: [],
  }]));
  for (const model of models || []) {
    const cluster = resolveModelTypeCluster(model.classification, taxonomy);
    groups.get(cluster.id).models.push(model);
  }
  return taxonomy.clusters.map((cluster) => {
    const group = groups.get(cluster.id);
    group.models.sort((left, right) => normalizeText(left?.modelId).localeCompare(normalizeText(right?.modelId)));
    return group;
  });
}
