import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { buildHfResolveBaseUrl } from '../utils/hf-resolve-url.js';
import { DEFAULT_EXTERNAL_MODELS_ROOT } from '../tooling/hf-registry-utils.js';

const DEFAULT_EXTERNAL_RDRR_ROOT = path.join(DEFAULT_EXTERNAL_MODELS_ROOT, 'rdrr');
const PACKAGE_ROOT = fileURLToPath(new URL('../..', import.meta.url));
const CATALOG_PATH = path.join(PACKAGE_ROOT, 'models', 'catalog.json');

export function asStringOrNull(value) {
  if (value === undefined || value === null) return null;
  if (typeof value === 'string') {
    const trimmed = value.trim();
    if (!trimmed.length) return null;
    return trimmed;
  }
  return String(value);
}

async function pathExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

export function resolveStaticRootDir(browserOptions = {}) {
  const configured = asStringOrNull(browserOptions.staticRootDir);
  if (configured) {
    return path.resolve(String(configured));
  }
  return PACKAGE_ROOT;
}

export function resolveRdrrRoot(options = {}) {
  return path.resolve(asStringOrNull(options.rdrrRoot) || DEFAULT_EXTERNAL_RDRR_ROOT);
}

async function findResolvableModelCandidate(candidates) {
  const discoveredManifestCandidates = [];

  for (const candidate of candidates) {
    if (!await pathExists(candidate.manifestPath)) {
      continue;
    }
    discoveredManifestCandidates.push(candidate);

    const modelDir = path.dirname(candidate.manifestPath);
    try {
      const files = await fs.readdir(modelDir, { withFileTypes: true });
      const hasShards = files.some((entry) =>
        entry.isFile() && /^shard_\d+\.bin$/u.test(entry.name)
      );
      if (hasShards) {
        return { candidate, discoveredManifestCandidates };
      }
    } catch {
      return { candidate, discoveredManifestCandidates };
    }
  }

  return { candidate: null, discoveredManifestCandidates };
}

async function resolveExternalModelDirectory(rdrrRoot, modelId) {
  const directModelDir = path.join(rdrrRoot, modelId);
  const directManifestPath = path.join(directModelDir, 'manifest.json');
  if (await pathExists(directManifestPath)) {
    return {
      modelDir: directModelDir,
      manifestPath: directManifestPath,
    };
  }

  let entries = [];
  try {
    entries = await fs.readdir(rdrrRoot, { withFileTypes: true });
  } catch {
    return null;
  }

  const matches = [];
  for (const entry of entries) {
    if (!entry.isDirectory()) {
      continue;
    }
    const manifestPath = path.join(rdrrRoot, entry.name, 'manifest.json');
    if (!await pathExists(manifestPath)) {
      continue;
    }
    let manifest = null;
    try {
      manifest = JSON.parse(await fs.readFile(manifestPath, 'utf8'));
    } catch {
      continue;
    }
    if (manifest?.modelId !== modelId) {
      continue;
    }
    matches.push({
      modelDir: path.join(rdrrRoot, entry.name),
      manifestPath,
    });
  }

  if (matches.length > 1) {
    const matchPaths = matches.map((match) => match.modelDir).join(', ');
    throw new Error(
      `Model "${modelId}" matched multiple external directories. ` +
      `Disambiguate by setting request.modelUrl in --config. Matches: ${matchPaths}`
    );
  }

  return matches[0] || null;
}

async function resolveCatalogEntry(modelId) {
  let catalog;
  try {
    catalog = JSON.parse(await fs.readFile(CATALOG_PATH, 'utf8'));
  } catch {
    return null;
  }
  if (!Array.isArray(catalog?.models)) {
    return null;
  }
  return catalog.models.find((entry) => (
    entry.modelId === modelId
    || (Array.isArray(entry.aliases) && entry.aliases.includes(modelId))
  )) || null;
}

function buildCatalogModelUrl(entry) {
  if (!entry?.hf?.repoId || !entry?.hf?.path) {
    return null;
  }
  return buildHfResolveBaseUrl(entry.hf);
}

export async function resolveBrowserModelUrl(request, browserOptions = {}) {
  if (request.modelUrl || !request.modelId) {
    return request;
  }

  const modelId = String(request.modelId);
  const encodedModelId = encodeURIComponent(modelId);

  if (asStringOrNull(browserOptions.baseUrl)) {
    return {
      ...request,
      modelUrl: `/models/${encodedModelId}`,
    };
  }

  const staticRootDir = resolveStaticRootDir(browserOptions);
  const externalModel = await resolveExternalModelDirectory(resolveRdrrRoot(browserOptions), modelId);
  const candidates = [
    {
    modelUrl: `/models/local/${encodedModelId}`,
    manifestPath: path.join(staticRootDir, 'models', 'local', modelId, 'manifest.json'),
    },
    {
    modelUrl: `/models/${encodedModelId}`,
    manifestPath: path.join(staticRootDir, 'models', modelId, 'manifest.json'),
    },
    {
      modelUrl: `/models/external/${encodeURIComponent(path.basename(externalModel?.modelDir || modelId))}`,
      manifestPath: externalModel?.manifestPath || path.join(resolveRdrrRoot(browserOptions), modelId, 'manifest.json'),
    },
  ];

  const { candidate, discoveredManifestCandidates } = await findResolvableModelCandidate(candidates);
  if (candidate) {
    return {
      ...request,
      modelUrl: candidate.modelUrl,
    };
  }

  if (discoveredManifestCandidates.length > 0) {
    const paths = discoveredManifestCandidates
      .map((candidate) => candidate.modelUrl)
      .join(', ');
    throw new Error(
      `Model "${modelId}" was found, but no shard files (shard_*.bin) are present. ` +
      `Checked: ${paths}. Add shard files beside the manifest, or set request.modelUrl in --config to a complete model directory.`
    );
  }

  const catalogEntry = await resolveCatalogEntry(modelId);
  if (catalogEntry) {
    const hfUrl = buildCatalogModelUrl(catalogEntry);
    if (hfUrl) {
      return {
        ...request,
        modelUrl: hfUrl,
      };
    }
  }

  return {
    ...request,
    modelUrl: `/models/${encodedModelId}`,
  };
}

export async function resolveNodeModelUrl(request, options = {}) {
  if (request.modelUrl || !request.modelId) {
    return request;
  }

  const modelId = String(request.modelId);
  const staticRootDir = resolveStaticRootDir(options);
  const rdrrRoot = resolveRdrrRoot(options);
  const externalModel = await resolveExternalModelDirectory(rdrrRoot, modelId);
  const localCandidates = [
    {
      modelDir: path.join(staticRootDir, 'models', 'local', modelId),
      manifestPath: path.join(staticRootDir, 'models', 'local', modelId, 'manifest.json'),
    },
    {
      modelDir: path.join(staticRootDir, 'models', modelId),
      manifestPath: path.join(staticRootDir, 'models', modelId, 'manifest.json'),
    },
  ];
  const candidates = [
    ...localCandidates,
    {
      modelDir: externalModel?.modelDir || path.join(rdrrRoot, modelId),
      manifestPath: externalModel?.manifestPath || path.join(rdrrRoot, modelId, 'manifest.json'),
    },
  ];
  const { candidate, discoveredManifestCandidates } =
    await findResolvableModelCandidate(candidates);

  if (candidate) {
    return {
      ...request,
      modelUrl: pathToFileURL(candidate.modelDir).href.replace(/\/$/, ''),
    };
  }

  if (discoveredManifestCandidates.length > 0) {
    const paths = discoveredManifestCandidates
      .map((candidate) => candidate.modelDir)
      .join(', ');
    throw new Error(
      `Model "${modelId}" was found, but no shard files (shard_*.bin) are present. ` +
      `Checked: ${paths}. Add shard files beside the manifest, or set request.modelUrl to a complete model directory.`
    );
  }

  const catalogEntry = await resolveCatalogEntry(modelId);
  if (catalogEntry) {
    const hfUrl = buildCatalogModelUrl(catalogEntry);
    if (hfUrl) {
      return {
        ...request,
        modelUrl: hfUrl,
      };
    }
    throw new Error(
      `Model "${modelId}" found in catalog as "${catalogEntry.modelId}" but has no HF source configured.`
    );
  }

  throw new Error(
    `Model "${modelId}" not found. Searched local: ${rdrrRoot}. Not in catalog. `
    + 'Set request.modelUrl to a file:// or https:// URL, '
    + 'or set DOPPLER_EXTERNAL_MODELS_ROOT to the parent of your rdrr/ folder.'
  );
}
