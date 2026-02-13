import {
  log,
  getRuntimeConfig,
  listRegisteredModels,
  removeRegisteredModel,
  formatBytes,
  getQuotaInfo,
  listStorageInventory,
  deleteStorageEntry,
  exportModelToDirectory,
} from '@doppler/core';
import { state } from '../state.js';
import { $, setText } from '../dom.js';
import { showErrorModal } from '../ui.js';

function getRegistryModelId(entry) {
  if (!entry) return '';
  return entry.modelId || entry.id || '';
}

function getRegistryTags(entry) {
  if (!entry) return [];
  const tags = [];
  const quant = entry.quantization ? String(entry.quantization).toLowerCase() : '';
  if (quant) {
    if (quant === 'f16' || quant === 'bf16' || quant === 'f32') {
      tags.push(`weights:${quant}`);
    } else if (
      quant.startsWith('w') &&
      (quant.endsWith('f16') || quant.endsWith('bf16') || quant.endsWith('f32'))
    ) {
      tags.push(`weights:${quant.slice(1)}`);
    } else {
      tags.push(`quant:${quant}`);
    }
  }
  if (entry.hashAlgorithm) {
    tags.push(String(entry.hashAlgorithm));
  }
  return tags;
}

function setStorageInspectorStatus(message) {
  const status = $('storage-inspector-status');
  if (!status) return;
  status.textContent = message;
}

function sanitizeDirectoryName(name) {
  // Keep consistent with modelId normalization (avoid nested dirs / invalid chars).
  return String(name).replace(/[^a-zA-Z0-9_-]/g, '_');
}

export async function updateStorageInfo() {
  const storageUsed = $('storage-used');
  if (!storageUsed) return;
  try {
    const info = await getQuotaInfo();
    state.storageUsageBytes = info.usage || 0;
    state.storageQuotaBytes = info.quota || 0;
    if (!info.quota) {
      storageUsed.textContent = 'Storage unavailable';
      return;
    }
    const used = formatBytes(info.usage || 0);
    const total = formatBytes(info.quota || 0);
    storageUsed.textContent = `${used} / ${total}`;
    state.lastStorageRefresh = Date.now();
  } catch (error) {
    storageUsed.textContent = `Storage error: ${error.message}`;
  }
}

async function handleDeleteStorageEntry(entry, callbacks) {
  if (!entry?.modelId) return;
  const sizeLabel = Number.isFinite(entry.totalBytes) ? formatBytes(entry.totalBytes) : 'unknown size';
  const backendLabel = entry.backend ? entry.backend.toUpperCase() : 'storage';
  const confirmed = window.confirm(`Delete ${entry.modelId} (${sizeLabel}) from ${backendLabel}?`);
  if (!confirmed) return;

  // Allow deleting the currently-loaded model by unloading first.
  if (entry.modelId === state.activeModelId && state.activePipeline) {
    if (typeof callbacks?.onUnloadActiveModel === 'function') {
      try {
        setStorageInspectorStatus(`Unloading ${entry.modelId}...`);
        await callbacks.onUnloadActiveModel(entry.modelId);
      } catch (error) {
        window.alert(`Could not unload ${entry.modelId}: ${error.message}`);
        return;
      }
    } else {
      window.alert('Unload the active model before deleting its storage.');
      return;
    }
  }

  try {
    await deleteStorageEntry(entry);
  } catch (error) {
    log.warn('DopplerDemo', `Failed to delete ${entry.modelId}: ${error.message}`);
  }

  try {
    await removeRegisteredModel(entry.modelId);
  } catch (error) {
    log.warn('DopplerDemo', `Failed to remove registry entry: ${error.message}`);
  }

  delete state.modelTypeCache[entry.modelId];
  if (entry.modelId === state.activeModelId) {
    state.activeModelId = null;
    state.activePipelineModelId = null;
  }

  if (callbacks?.onModelsUpdated) {
    await callbacks.onModelsUpdated();
    return;
  }
  await refreshStorageInspector(callbacks);
}

async function handleExportStorageEntry(entry) {
  if (!entry?.modelId) return;
  if (entry.missingStorage) {
    showErrorModal('Export unavailable: model is registry-only (no local storage).');
    return;
  }
  if (typeof window === 'undefined' || typeof window.showDirectoryPicker !== 'function') {
    showErrorModal('Export requires the File System Access API (showDirectoryPicker). Use a Chromium-based browser.');
    return;
  }

  let destRoot;
  try {
    destRoot = await window.showDirectoryPicker({ mode: 'readwrite' });
  } catch (error) {
    if (String(error?.name || '').includes('Abort')) return;
    showErrorModal(`Export canceled: ${error.message}`);
    return;
  }

  const dest = await destRoot.getDirectoryHandle(sanitizeDirectoryName(entry.modelId), { create: true });
  const startedAt = performance.now();
  setStorageInspectorStatus(`Exporting ${entry.modelId}...`);

  try {
    await exportModelToDirectory(entry.modelId, dest, {
      onProgress: (p) => {
        if (!p) return;
        if (p.stage === 'file_start') {
          setStorageInspectorStatus(`Exporting ${entry.modelId}: ${p.filename} (${(p.index ?? 0) + 1}/${p.total ?? 0})`);
        } else if (p.stage === 'done') {
          // handled below
        }
      },
    });
    const elapsed = ((performance.now() - startedAt) / 1000).toFixed(2);
    setStorageInspectorStatus(`Exported ${entry.modelId} in ${elapsed}s`);
  } catch (error) {
    log.warn('DopplerDemo', `Export failed: ${error.message}`);
    setStorageInspectorStatus('Ready');
    showErrorModal(`Export failed: ${error.message}`);
  }
}

export async function refreshStorageInspector(callbacks = {}) {
  const listEl = $('storage-inspector-list');
  const backendEl = $('storage-inspector-backend');
  const summaryEl = $('storage-inspector-summary');
  const systemSection = $('storage-inspector-system-section');
  const systemList = $('storage-inspector-system');
  if (!listEl || !systemSection || !systemList) return;
  if (state.storageInspectorScanning) return;

  state.storageInspectorScanning = true;
  listEl.innerHTML = '';
  systemList.innerHTML = '';
  systemSection.hidden = true;
  setStorageInspectorStatus('Scanning storage...');

  const runtime = getRuntimeConfig();

  let registryEntries = [];
  const registryIds = new Set();
  const registryById = new Map();
  try {
    registryEntries = await listRegisteredModels();
    for (const entry of registryEntries) {
      const id = getRegistryModelId(entry);
      if (!id) continue;
      registryIds.add(id);
      registryById.set(id, entry);
    }
  } catch (error) {
    log.warn('DopplerDemo', `Registry unavailable: ${error.message}`);
  }

  try {
    const inventory = await listStorageInventory();
    const storageEntries = inventory.entries.map((entry) => ({
      ...entry,
      registered: registryIds.has(entry.modelId),
      registryEntry: registryById.get(entry.modelId) || null,
    }));
    state.quickModelStorageIds = [...new Set(storageEntries
      .map((entry) => entry.modelId)
      .filter((modelId) => typeof modelId === 'string' && modelId.length > 0)
    )];
    if (typeof callbacks?.onStorageInventoryRefreshed === 'function') {
      callbacks.onStorageInventoryRefreshed(state.quickModelStorageIds.slice());
    }
    const storageIds = new Set(storageEntries.map((entry) => entry.modelId));
    const registryOnlyEntries = [];
    for (const [modelId, registryEntry] of registryById.entries()) {
      if (storageIds.has(modelId)) continue;
      const totalBytes = Number.isFinite(registryEntry?.totalSize)
        ? registryEntry.totalSize
        : Number.NaN;
      registryOnlyEntries.push({
        modelId,
        backend: registryEntry?.backend || 'unknown',
        root: '',
        totalBytes,
        fileCount: 0,
        shardCount: 0,
        hasManifest: false,
        registered: true,
        missingStorage: true,
        registryEntry,
      });
    }
    const entries = [...storageEntries, ...registryOnlyEntries];
    const systemEntries = inventory.systemEntries;
    const opfsRoots = inventory.opfsRoots;
    const backendParts = [];

    if (inventory.backendAvailability.opfs) {
      backendParts.push(
        opfsRoots.length ? `OPFS: ${opfsRoots.join(', ')}` : 'OPFS: empty'
      );
    } else {
      backendParts.push('OPFS: unavailable');
    }

    const idbName = runtime?.loading?.storage?.backend?.indexeddb?.dbName || 'indexeddb';
    if (inventory.backendAvailability.indexeddb) {
      backendParts.push(`IDB: ${idbName}`);
    } else {
      backendParts.push('IDB: unavailable');
    }
    if (backendEl) {
      backendEl.textContent = backendParts.join(' • ');
    }

    entries.sort((a, b) => {
      const aActive = a.modelId === state.activeModelId ? 1 : 0;
      const bActive = b.modelId === state.activeModelId ? 1 : 0;
      if (aActive !== bActive) return bActive - aActive;
      const aMissing = a.missingStorage ? 1 : 0;
      const bMissing = b.missingStorage ? 1 : 0;
      if (aMissing !== bMissing) return aMissing - bMissing;
      const aSize = Number.isFinite(a.totalBytes) ? a.totalBytes : 0;
      const bSize = Number.isFinite(b.totalBytes) ? b.totalBytes : 0;
      return bSize - aSize;
    });

    const totals = new Map();
    const counts = new Map();
    for (const entry of storageEntries) {
      const backend = entry.backend || 'unknown';
      counts.set(backend, (counts.get(backend) || 0) + 1);
      if (Number.isFinite(entry.totalBytes)) {
        totals.set(backend, (totals.get(backend) || 0) + entry.totalBytes);
      }
    }

    const systemBytes = systemEntries.reduce((sum, entry) => (
      sum + (Number.isFinite(entry.totalBytes) ? entry.totalBytes : 0)
    ), 0);
    const summaryParts = [];
    if (entries.length) {
      summaryParts.push(`Models: ${entries.length}`);
    }
    if (registryOnlyEntries.length) {
      summaryParts.push(`Registry-only: ${registryOnlyEntries.length}`);
    }
    const orderedBackends = ['opfs', 'indexeddb', 'memory', 'unknown'];
    for (const backend of orderedBackends) {
      const count = counts.get(backend);
      if (!count) continue;
      const bytes = totals.get(backend);
      const label = backend === 'indexeddb' ? 'IDB' : backend.toUpperCase();
      if (Number.isFinite(bytes) && bytes > 0) {
        summaryParts.push(`${label}: ${formatBytes(bytes)} (${count})`);
      } else {
        summaryParts.push(`${label}: ${count}`);
      }
    }
    if (systemEntries.length) {
      summaryParts.push(`System: ${formatBytes(systemBytes)}`);
    }
    if (summaryEl) {
      summaryEl.textContent = summaryParts.length
        ? summaryParts.join(' • ')
        : 'No models found';
    }

    if (!entries.length) {
      listEl.innerHTML = '<div class="type-caption">No models found.</div>';
      setStorageInspectorStatus('Ready');
    } else {
      for (const entry of entries) {
        const row = document.createElement('div');
        row.className = 'storage-entry';
        if (entry.modelId === state.activeModelId) {
          row.classList.add('is-active');
        }
        if (entry.missingStorage) {
          row.classList.add('is-missing');
        }

        const main = document.createElement('div');
        main.className = 'storage-entry-main';

        const title = document.createElement('div');
        title.className = 'storage-entry-title';

        const name = document.createElement('span');
        name.className = 'type-caption';
        name.textContent = entry.modelId;
        title.appendChild(name);

        const backendTag = document.createElement('span');
        backendTag.className = 'storage-tag';
        backendTag.textContent = entry.backend === 'indexeddb' ? 'idb' : entry.backend;
        title.appendChild(backendTag);

        const tag = document.createElement('span');
        tag.className = `storage-tag${entry.registered ? '' : ' orphan'}`;
        tag.textContent = entry.registered ? 'registered' : 'orphan';
        title.appendChild(tag);

        if (entry.modelId === state.activeModelId) {
          const activeTag = document.createElement('span');
          activeTag.className = 'storage-tag active';
          activeTag.textContent = 'active';
          title.appendChild(activeTag);
        }

        const registryTags = getRegistryTags(entry.registryEntry);
        for (const registryTag of registryTags) {
          const registryTagEl = document.createElement('span');
          registryTagEl.className = 'storage-tag';
          registryTagEl.textContent = registryTag;
          title.appendChild(registryTagEl);
        }

        if (entry.missingStorage) {
          const missingStorage = document.createElement('span');
          missingStorage.className = 'storage-tag missing';
          missingStorage.textContent = 'registry only';
          title.appendChild(missingStorage);
        }

        if (!entry.hasManifest) {
          const missing = document.createElement('span');
          missing.className = 'storage-tag missing';
          missing.textContent = 'no manifest';
          title.appendChild(missing);
        }

        main.appendChild(title);

        const shardLabel = entry.missingStorage
          ? 'registry only'
          : entry.shardCount
            ? `${entry.shardCount} shards`
            : `${entry.fileCount} files`;
        const detail = document.createElement('span');
        detail.className = 'type-caption';
        const rootLabel = entry.root ? ` • root: ${entry.root}` : '';
        const sizeBytes = Number.isFinite(entry.totalBytes)
          ? entry.totalBytes
          : Number.isFinite(entry.registryEntry?.totalSize)
            ? entry.registryEntry.totalSize
            : Number.NaN;
        const sizeLabel = Number.isFinite(sizeBytes)
          ? formatBytes(sizeBytes)
          : 'unknown size';
        detail.textContent = entry.missingStorage
          ? `${sizeLabel} • ${shardLabel}`
          : `${sizeLabel} • ${shardLabel}${rootLabel}`;
        main.appendChild(detail);

        row.appendChild(main);

        if (!entry.missingStorage) {
          const actions = document.createElement('div');
          actions.className = 'storage-entry-actions';

          const tryBtn = document.createElement('button');
          tryBtn.className = 'btn btn-small btn-primary';
          tryBtn.type = 'button';
          tryBtn.textContent = 'Try It';
          tryBtn.addEventListener('click', async (event) => {
            event.stopPropagation();
            if (callbacks?.onTryModel) {
              await callbacks.onTryModel(entry.modelId);
              return;
            }
            if (callbacks?.onSelectModel) {
              callbacks.onSelectModel(entry.modelId);
            }
          });
          actions.appendChild(tryBtn);

          const exportBtn = document.createElement('button');
          exportBtn.className = 'btn btn-small';
          exportBtn.type = 'button';
          exportBtn.textContent = 'Export';
          exportBtn.addEventListener('click', async (event) => {
            event.stopPropagation();
            await handleExportStorageEntry(entry);
          });
          actions.appendChild(exportBtn);

          const deleteBtn = document.createElement('button');
          deleteBtn.className = 'btn btn-small';
          deleteBtn.type = 'button';
          deleteBtn.textContent = 'Delete';
          if (entry.modelId === state.activeModelId && state.activePipeline) {
            deleteBtn.title = 'Will unload active model, then delete.';
          }
          deleteBtn.addEventListener('click', async (event) => {
            event.stopPropagation();
            try {
              await handleDeleteStorageEntry(entry, callbacks);
            } catch (error) {
              log.warn('DopplerDemo', `Delete failed: ${error.message}`);
              showErrorModal(`Delete failed: ${error.message}`);
            }
          });
          actions.appendChild(deleteBtn);
          row.appendChild(actions);
          if (callbacks?.onSelectModel) {
            row.addEventListener('click', () => callbacks.onSelectModel(entry.modelId));
          }
        }
        listEl.appendChild(row);
      }
    }

    if (systemEntries.length) {
      systemSection.hidden = false;
      for (const entry of systemEntries) {
        const row = document.createElement('div');
        row.className = 'storage-entry';

        const main = document.createElement('div');
        main.className = 'storage-entry-main';

        const title = document.createElement('div');
        title.className = 'storage-entry-title';

        const name = document.createElement('span');
        name.className = 'type-caption';
        name.textContent = entry.label;
        title.appendChild(name);

        const tag = document.createElement('span');
        tag.className = 'storage-tag system';
        tag.textContent = 'system';
        title.appendChild(tag);

        main.appendChild(title);

        const detail = document.createElement('span');
        detail.className = 'type-caption';
        const systemRoot = entry.root ? ` • root: ${entry.root}` : '';
        const bytesLabel = Number.isFinite(entry.totalBytes) ? formatBytes(entry.totalBytes) : '--';
        detail.textContent = `${bytesLabel} • ${entry.fileCount} files${systemRoot}`;
        main.appendChild(detail);

        row.appendChild(main);
        systemList.appendChild(row);
      }
    }

    setStorageInspectorStatus('Ready');
  } catch (error) {
    state.quickModelStorageIds = [];
    if (typeof callbacks?.onStorageInventoryRefreshed === 'function') {
      callbacks.onStorageInventoryRefreshed([]);
    }
    if (summaryEl) {
      summaryEl.textContent = '--';
    }
    setStorageInspectorStatus(`Storage scan failed: ${error.message}`);
  } finally {
    state.storageInspectorScanning = false;
    state.storageInspectorLastScan = Date.now();
  }
}
