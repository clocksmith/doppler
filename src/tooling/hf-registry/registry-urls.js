import { isPlainObject, normalizeText } from './catalog-io.js';

const DEFAULT_HF_REPO_ID = 'clocksmith/rdrr';
export const DEFAULT_HF_REGISTRY_PATH = 'registry/catalog.json';
export const DEFAULT_HF_REGISTRY_URL = `https://huggingface.co/${DEFAULT_HF_REPO_ID}/resolve/main/${DEFAULT_HF_REGISTRY_PATH}`;

function normalizeRepoPath(value) {
  return normalizeText(value).replace(/^\/+/, '');
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

export function extractCommitShaFromUrl(value) {
  const raw = normalizeText(value);
  if (!raw) return '';
  const directMatch = raw.match(/\b([a-f0-9]{40})\b/i);
  return directMatch ? directMatch[1].toLowerCase() : '';
}
