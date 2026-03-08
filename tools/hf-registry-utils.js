#!/usr/bin/env node

import fs from 'node:fs/promises';

export const DEFAULT_HF_REPO_ID = 'Clocksmith/rdrr';
export const DEFAULT_HF_REGISTRY_PATH = 'registry/catalog.json';
export const DEFAULT_HF_REGISTRY_URL = `https://huggingface.co/${DEFAULT_HF_REPO_ID}/resolve/main/${DEFAULT_HF_REGISTRY_PATH}`;

export function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

export function normalizeToken(value) {
  return normalizeText(value).toLowerCase();
}

export function normalizeRepoPath(value) {
  return normalizeText(value).replace(/^\/+/, '');
}

export function isPlainObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

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

export function buildHfResolveUrl(repoId, revision, repoPath) {
  const normalizedRepoId = normalizeText(repoId);
  const normalizedRevision = normalizeText(revision);
  const normalizedRepoPath = normalizeRepoPath(repoPath);
  if (!normalizedRepoId || !normalizedRevision || !normalizedRepoPath) {
    return '';
  }
  return `https://huggingface.co/${normalizedRepoId}/resolve/${encodeURIComponent(normalizedRevision)}/${normalizedRepoPath}`;
}

export function getEntryHfSpec(entry) {
  const hf = isPlainObject(entry?.hf) ? entry.hf : {};
  const repoId = normalizeText(hf.repoId);
  const revision = normalizeText(hf.revision);
  const path = normalizeRepoPath(hf.path);
  return {
    repoId,
    revision,
    path,
    complete: Boolean(repoId && revision && path),
  };
}

export function buildEntryRemoteBaseUrl(entry) {
  const hfSpec = getEntryHfSpec(entry);
  if (hfSpec.complete) {
    return buildHfResolveUrl(hfSpec.repoId, hfSpec.revision, hfSpec.path).replace(/\/+$/, '');
  }
  const baseUrl = normalizeText(entry?.baseUrl);
  if (!baseUrl) return '';
  try {
    return new URL(baseUrl).toString().replace(/\/+$/, '');
  } catch {
    return '';
  }
}

export function resolveDemoRegistryEntryBaseUrl(entry, catalogSourceUrl) {
  const hfSpec = getEntryHfSpec(entry);
  if (hfSpec.complete) {
    return buildHfResolveUrl(hfSpec.repoId, hfSpec.revision, hfSpec.path).replace(/\/+$/, '');
  }
  const baseUrl = normalizeText(entry?.baseUrl);
  if (!baseUrl) return '';
  try {
    return new URL(baseUrl, catalogSourceUrl).toString().replace(/\/+$/, '');
  } catch {
    return '';
  }
}

export function shouldDemoSurfaceRemoteRegistryEntry(entry, catalogSourceUrl) {
  return Boolean(resolveDemoRegistryEntryBaseUrl(entry, catalogSourceUrl));
}

export function buildManifestUrl(baseUrl) {
  const normalizedBaseUrl = normalizeText(baseUrl).replace(/\/+$/, '');
  if (!normalizedBaseUrl) return '';
  return `${normalizedBaseUrl}/manifest.json`;
}

export function buildShardUrl(baseUrl, shard) {
  const normalizedBaseUrl = normalizeText(baseUrl).replace(/\/+$/, '');
  const filename = normalizeText(shard?.filename);
  if (!normalizedBaseUrl || !filename) return '';
  return `${normalizedBaseUrl}/${filename}`;
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

export function buildPublishedRegistryEntry(localEntry, revision) {
  const modelId = normalizeText(localEntry?.modelId);
  if (!modelId) {
    throw new Error('Published registry entry requires a non-empty modelId.');
  }
  const next = structuredClone(localEntry);
  const hf = isPlainObject(next.hf) ? next.hf : {};
  const repoId = normalizeText(hf.repoId);
  const repoPath = normalizeRepoPath(hf.path);
  if (!repoId) {
    throw new Error(
      `Published registry entry for "${modelId}" requires explicit hf.repoId.`
    );
  }
  if (!repoPath) {
    throw new Error(
      `Published registry entry for "${modelId}" requires explicit hf.path.`
    );
  }
  const lifecycle = isPlainObject(next.lifecycle) ? next.lifecycle : {};
  const availability = isPlainObject(lifecycle.availability) ? lifecycle.availability : {};
  next.hf = {
    ...hf,
    repoId,
    revision: normalizeText(revision),
    path: repoPath,
  };
  next.lifecycle = {
    ...lifecycle,
    availability: {
      ...availability,
      hf: true,
    },
  };
  return next;
}

export function extractCommitShaFromUrl(value) {
  const raw = normalizeText(value);
  if (!raw) return '';
  const directMatch = raw.match(/\b([a-f0-9]{40})\b/i);
  return directMatch ? directMatch[1].toLowerCase() : '';
}

export function validateLocalHfEntryShape(entry) {
  const errors = [];
  const modelId = normalizeText(entry?.modelId) || 'unknown-model';
  const hfSpec = getEntryHfSpec(entry);
  if (!hfSpec.repoId) {
    errors.push(`${modelId}: hf.repoId is required when lifecycle.availability.hf=true`);
  }
  if (!hfSpec.revision) {
    errors.push(`${modelId}: hf.revision is required when lifecycle.availability.hf=true`);
  }
  if (!hfSpec.path) {
    errors.push(`${modelId}: hf.path is required when lifecycle.availability.hf=true`);
  }
  return errors;
}

export async function probeUrl(url, options = {}) {
  const timeoutMs = Number.isFinite(options.timeoutMs) ? options.timeoutMs : 30000;
  const methods = Array.isArray(options.methods) && options.methods.length > 0
    ? options.methods
    : ['HEAD', 'GET'];
  let lastError = null;

  for (const method of methods) {
    try {
      const headers = method === 'GET'
        ? { Connection: 'close', Range: 'bytes=0-0' }
        : { Connection: 'close' };
      const response = await fetch(url, {
        method,
        headers,
        redirect: 'follow',
        signal: AbortSignal.timeout(timeoutMs),
      });
      if (response.ok || response.status === 206) {
        return {
          ok: true,
          status: response.status,
          url: response.url,
          method,
        };
      }
      lastError = new Error(`HTTP ${response.status}: ${url}`);
    } catch (error) {
      lastError = error;
    }
  }

  return {
    ok: false,
    status: null,
    url,
    method: methods[methods.length - 1] || 'HEAD',
    error: lastError,
  };
}

export async function fetchJson(url, options = {}) {
  const timeoutMs = Number.isFinite(options.timeoutMs) ? options.timeoutMs : 30000;
  const response = await fetch(url, {
    headers: {
      Connection: 'close',
    },
    redirect: 'follow',
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${url}`);
  }
  return response.json();
}
