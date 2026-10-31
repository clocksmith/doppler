import { isPlainObject } from '../../formats/plain-object.js';
import fs from 'node:fs/promises';
import { existsSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';

export function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function detectDefaultExternalModelsRoot() {
  const envRoot = normalizeText(process.env.DOPPLER_EXTERNAL_MODELS_ROOT);
  if (envRoot) {
    return envRoot;
  }
  const linuxMediaRoot = path.join('/media', os.userInfo().username, 'models');
  for (const candidate of ['/Volumes/models', '/Volumes/models2', linuxMediaRoot]) {
    if (existsSync(candidate)) {
      return candidate;
    }
  }
  return linuxMediaRoot;
}

export const DEFAULT_EXTERNAL_MODELS_ROOT = detectDefaultExternalModelsRoot();
export { isPlainObject };


export function ensureCatalogPayload(payload, label = 'catalog') {
  if (!isPlainObject(payload) || !Array.isArray(payload.models)) {
    throw new Error(`${label} payload must be an object with a models array.`);
  }
  return payload;
}

export async function loadJsonFile(filePath, label = filePath) {
  const raw = await fs.readFile(filePath, 'utf8');
  const payload = JSON.parse(raw);
  return ensureCatalogPayload(payload, label);
}

export async function writeJsonFile(filePath, payload) {
  await fs.writeFile(filePath, `${JSON.stringify(payload, null, 2)}\n`, 'utf8');
}

export function collectDuplicateModelIds(models) {
  const seen = new Set();
  const duplicates = new Set();
  for (const model of models || []) {
    const modelId = normalizeText(model?.modelId);
    if (!modelId) continue;
    if (seen.has(modelId)) {
      duplicates.add(modelId);
      continue;
    }
    seen.add(modelId);
  }
  return [...duplicates].sort((a, b) => a.localeCompare(b));
}

export function findCatalogEntry(payload, modelId) {
  const models = Array.isArray(payload?.models) ? payload.models : [];
  const needle = normalizeText(modelId);
  return models.find((entry) => normalizeText(entry?.modelId) === needle) || null;
}
