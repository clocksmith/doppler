import { log } from '../../src/debug/index.js';
import { openModelStore, loadManifestFromStore } from '../../src/storage/shard-manager.js';
import { state } from '../state.js';

export function normalizeModelType(value) {
  if (typeof value !== 'string') return null;
  const normalized = value.trim().toLowerCase();
  return normalized || null;
}

export function isCompatibleModelType(modelType, mode) {
  const normalized = normalizeModelType(modelType);
  if (mode === 'diffusion') {
    return normalized === 'diffusion';
  }
  if (mode === 'energy') {
    return normalized === 'energy';
  }
  if (mode === 'run') {
    return normalized !== 'diffusion' && normalized !== 'energy';
  }
  return true;
}

export function isModeModelSelectable(mode) {
  return mode === 'run' || mode === 'diffusion' || mode === 'energy' || mode === 'kernels';
}

export function getModeModelLabel(mode) {
  if (mode === 'diffusion') return 'diffusion';
  if (mode === 'energy') return 'energy';
  if (mode === 'run') return 'text';
  if (mode === 'kernels' || mode === 'diagnostics' || mode === 'models') return 'models';
  return 'local';
}

export async function getModelTypeForId(modelId) {
  if (!modelId) return null;
  const cached = state.modelTypeCache[modelId];
  if (cached) return cached;
  try {
    await openModelStore(modelId);
    const manifestText = await loadManifestFromStore();
    if (!manifestText) return null;
    const manifest = JSON.parse(manifestText);
    const modelType = normalizeModelType(manifest?.modelType) || 'transformer';
    state.modelTypeCache[modelId] = modelType;
    return modelType;
  } catch (error) {
    log.warn('DopplerDemo', `Failed to read manifest for ${modelId}: ${error.message}`);
    return null;
  }
}
