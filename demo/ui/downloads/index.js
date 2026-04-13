import {
  downloadModel,
  pauseDownload,
  resumeDownload,
  listDownloads,
  cancelDownload,
} from '../../../src/storage/downloader.js';
import { listStorageInventory, deleteStorageEntry } from '../../../src/storage/inventory.js';
import { state } from '../state.js';

const downloadCallbacks = {
  onModelRegistered: null,
  onModelsUpdated: null,
  onProgress: null,
  onStateChange: null,
};

function emitStateChange(update) {
  if (typeof downloadCallbacks.onStateChange === 'function') {
    downloadCallbacks.onStateChange(update);
  }
}

function emitProgress(update) {
  if (typeof downloadCallbacks.onProgress === 'function') {
    downloadCallbacks.onProgress(update);
  }
}

function requireActiveDownloadId() {
  const modelId = typeof state.activeDownloadId === 'string' ? state.activeDownloadId.trim() : '';
  if (!modelId) {
    throw new Error('No active download selected.');
  }
  return modelId;
}

async function cleanupPartialImport(modelId) {
  const normalizedModelId = String(modelId ?? '').trim();
  if (!normalizedModelId) {
    return;
  }
  const inventory = await listStorageInventory();
  const entry = Array.isArray(inventory?.entries)
    ? inventory.entries.find((candidate) => candidate?.modelId === normalizedModelId)
    : null;
  if (!entry) {
    return;
  }
  await deleteStorageEntry(entry);
}

export function configureDownloadCallbacks(callbacks = {}) {
  downloadCallbacks.onModelRegistered = typeof callbacks.onModelRegistered === 'function'
    ? callbacks.onModelRegistered
    : null;
  downloadCallbacks.onModelsUpdated = typeof callbacks.onModelsUpdated === 'function'
    ? callbacks.onModelsUpdated
    : null;
  downloadCallbacks.onProgress = typeof callbacks.onProgress === 'function'
    ? callbacks.onProgress
    : null;
  downloadCallbacks.onStateChange = typeof callbacks.onStateChange === 'function'
    ? callbacks.onStateChange
    : null;
}

export async function refreshDownloads() {
  const downloads = await listDownloads();
  if (!Array.isArray(downloads) || downloads.length === 0) {
    emitStateChange({ modelId: state.activeDownloadId ?? null, active: false });
    return [];
  }
  const active = downloads.find((entry) => entry?.status === 'downloading') || downloads[0];
  const modelId = typeof active?.modelId === 'string' ? active.modelId : null;
  state.activeDownloadId = modelId;
  emitStateChange({ modelId, active: true });
  emitProgress(active);
  return downloads;
}

export async function startDownloadFromBaseUrl(baseUrl, modelIdOverride = '') {
  const normalizedBaseUrl = String(baseUrl ?? '').trim();
  const normalizedModelId = String(modelIdOverride ?? '').trim();
  if (!normalizedBaseUrl) {
    throw new Error('startDownloadFromBaseUrl requires a baseUrl.');
  }

  const working = {
    modelId: normalizedModelId || null,
  };

  if (working.modelId) {
    state.activeDownloadId = working.modelId;
  }
  emitStateChange({ modelId: working.modelId, active: true });

  try {
    const imported = await downloadModel(
      normalizedBaseUrl,
      (progress) => {
        const progressModelId = typeof progress?.modelId === 'string' && progress.modelId.trim()
          ? progress.modelId.trim()
          : working.modelId || state.activeDownloadId || null;
        if (progressModelId) {
          working.modelId = progressModelId;
          state.activeDownloadId = progressModelId;
        }
        emitProgress({
          ...progress,
          modelId: progressModelId,
        });
      },
      working.modelId ? { modelId: working.modelId } : {}
    );

    if (imported && working.modelId && typeof downloadCallbacks.onModelRegistered === 'function') {
      await downloadCallbacks.onModelRegistered(working.modelId);
    }
    if (imported && typeof downloadCallbacks.onModelsUpdated === 'function') {
      await downloadCallbacks.onModelsUpdated();
    }
    return imported;
  } catch (error) {
    await cleanupPartialImport(working.modelId);
    throw error;
  } finally {
    emitStateChange({ modelId: working.modelId || state.activeDownloadId || null, active: false });
  }
}

export async function startDownload() {
  throw new Error('Manual download start is not wired in this demo surface. Use quick model actions or startDownloadFromBaseUrl().');
}

export function pauseActiveDownload() {
  return pauseDownload(requireActiveDownloadId());
}

export async function resumeActiveDownload() {
  const modelId = requireActiveDownloadId();
  emitStateChange({ modelId, active: true });
  try {
    return await resumeDownload(modelId, (progress) => {
      emitProgress({
        ...progress,
        modelId,
      });
    });
  } finally {
    emitStateChange({ modelId, active: false });
  }
}

export async function cancelActiveDownload() {
  const modelId = requireActiveDownloadId();
  try {
    return await cancelDownload(modelId);
  } finally {
    emitStateChange({ modelId, active: false });
  }
}
