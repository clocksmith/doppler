import { getCdnBasePath } from '../storage/download-types.js';
import { loadJson } from '../utils/load-json.js';

let registryPromise = null;

function normalizeEntry(entry) {
  if (!entry || typeof entry !== 'object') {
    return null;
  }
  const modelId = typeof entry.modelId === 'string' ? entry.modelId.trim() : '';
  if (!modelId) {
    return null;
  }
  return {
    modelId,
    aliases: Array.isArray(entry.aliases)
      ? entry.aliases.filter((alias) => typeof alias === 'string' && alias.trim().length > 0)
      : [],
    modes: Array.isArray(entry.modes)
      ? entry.modes.filter((mode) => typeof mode === 'string' && mode.trim().length > 0)
      : [],
    hf: entry.hf && typeof entry.hf === 'object'
      ? {
        repoId: typeof entry.hf.repoId === 'string' ? entry.hf.repoId.trim() : '',
        revision: typeof entry.hf.revision === 'string' && entry.hf.revision.trim().length > 0
          ? entry.hf.revision.trim()
          : null,
        path: typeof entry.hf.path === 'string' ? entry.hf.path.trim() : '',
      }
      : null,
  };
}

async function loadRegistry() {
  if (!registryPromise) {
    registryPromise = loadJson(
      './doppler-registry.json',
      import.meta.url,
      'Failed to load Doppler quickstart registry'
    ).then((raw) => {
      const entries = Array.isArray(raw?.models)
        ? raw.models.map(normalizeEntry).filter(Boolean)
        : [];
      return { models: entries };
    });
  }
  return registryPromise;
}

export async function listQuickstartModels() {
  const registry = await loadRegistry();
  return registry.models.map((entry) => ({
    modelId: entry.modelId,
    aliases: [...entry.aliases],
    modes: [...entry.modes],
  }));
}

export async function resolveQuickstartModel(model) {
  const requested = typeof model === 'string' ? model.trim() : '';
  if (!requested) {
    throw new Error('Quickstart model id is required.');
  }

  const registry = await loadRegistry();
  const resolved = registry.models.find((entry) => (
    entry.modelId === requested || entry.aliases.includes(requested)
  ));
  if (resolved) {
    return resolved;
  }

  const available = registry.models.map((entry) => entry.modelId).join(', ');
  throw new Error(`Unknown quickstart model "${requested}". Available: ${available}`);
}

export function buildQuickstartModelBaseUrl(entry, options = {}) {
  if (!entry?.hf?.repoId || !entry?.hf?.path) {
    throw new Error(`Quickstart model "${entry?.modelId ?? 'unknown'}" does not have a hosted Hugging Face source.`);
  }
  const cdnBasePath = typeof options.cdnBasePath === 'string' && options.cdnBasePath.length > 0
    ? options.cdnBasePath
    : (getCdnBasePath() || 'https://huggingface.co');
  const revision = entry.hf.revision || 'main';
  const base = cdnBasePath.replace(/\/$/, '');
  const path = entry.hf.path.replace(/^\/+/, '');
  return `${base}/${entry.hf.repoId}/resolve/${revision}/${path}`;
}
