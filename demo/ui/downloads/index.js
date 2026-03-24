import {
  deleteStorageEntry,
  listStorageInventory,
  openModelStore,
  parseManifest,
  saveManifest,
  saveTensorsToStore,
  saveTokenizer,
  saveTokenizerModel,
  saveAuxFile,
  writeShard,
} from 'doppler-gpu';

const AUX_IMPORT_FILENAMES = Object.freeze([
  'config.json',
  'generation_config.json',
  'tokenizer_config.json',
  'special_tokens_map.json',
]);

function sumManifestShardBytes(shards) {
  if (!Array.isArray(shards)) return 0;
  return shards.reduce((acc, shard) => {
    const size = Number(shard?.size);
    if (!Number.isFinite(size) || size <= 0) return acc;
    return acc + Math.floor(size);
  }, 0);
}

function clampPercent(value) {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, value));
}

function getTextByteLength(text) {
  return new TextEncoder().encode(String(text || '')).byteLength;
}

let callbacks = Object.freeze({
  onModelRegistered: null,
  onModelsUpdated: null,
  onProgress: null,
  onStateChange: null,
});

let activeDownload = null;
const downloadLog = [];
const MODEL_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{1,127}$/;
const HF_COMMIT_REVISION_PATTERN = /^[a-f0-9]{7,64}$/i;

function toErrorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}

function getElement(id) {
  return typeof document === 'undefined' ? null : document.getElementById(id);
}

function setElementHidden(id, hidden) {
  const element = getElement(id);
  if (!element) return;
  element.hidden = hidden;
}

function setElementText(id, value) {
  const element = getElement(id);
  if (!element) return;
  element.textContent = value;
}

function setProgressPercent(percent) {
  const bar = getElement('download-progress');
  if (!(bar instanceof HTMLElement)) return;
  const clamped = Number.isFinite(percent) ? Math.max(0, Math.min(100, percent)) : 0;
  bar.style.width = `${clamped}%`;
}

function appendLog(message) {
  if (!message) return;
  const timestamp = new Date().toLocaleTimeString();
  downloadLog.unshift(`[${timestamp}] ${message}`);
  if (downloadLog.length > 8) {
    downloadLog.length = 8;
  }
  const list = getElement('download-list');
  if (!(list instanceof HTMLElement)) return;
  list.textContent = downloadLog.join('\n');
}

function renderDownloadStatus({ message, percent, visible }) {
  setElementHidden('download-status', !visible);
  if (typeof percent === 'number') {
    setProgressPercent(percent);
  }
  if (typeof message === 'string') {
    setElementText('download-message', message);
  }
}

function emitProgress(update) {
  if (typeof callbacks.onProgress === 'function') {
    callbacks.onProgress(update);
  }
}

function emitStateChange(update) {
  if (typeof callbacks.onStateChange === 'function') {
    callbacks.onStateChange(update);
  }
}

async function cleanupPartialImport(modelId) {
  if (!modelId) return;
  const inventory = await listStorageInventory();
  const matches = inventory.entries.filter((entry) => entry.modelId === modelId);
  await Promise.all(matches.map((entry) => deleteStorageEntry(entry)));
}

function normalizeBaseUrl(baseUrl) {
  const raw = typeof baseUrl === 'string' ? baseUrl.trim() : '';
  if (!raw) return '';
  try {
    const url = new URL(raw);
    return url.toString().replace(/\/+$/, '');
  } catch {
    return '';
  }
}

function normalizeModelIdInput(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function isValidModelId(value) {
  return MODEL_ID_PATTERN.test(normalizeModelIdInput(value));
}

function assertValidModelId(value, sourceLabel) {
  const modelId = normalizeModelIdInput(value);
  if (!modelId) {
    throw new Error(`${sourceLabel} is required.`);
  }
  if (!isValidModelId(modelId)) {
    throw new Error(
      `${sourceLabel} must match ${MODEL_ID_PATTERN.source} (2-128 chars, alnum, dot, underscore, hyphen).`
    );
  }
  return modelId;
}

function toFileUrl(baseUrl, relativePath) {
  const base = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`;
  return new URL(relativePath, base).toString();
}

function isHuggingFaceHost(hostname) {
  if (typeof hostname !== 'string' || !hostname) return false;
  const lowered = hostname.toLowerCase();
  return lowered === 'huggingface.co' || lowered.endsWith('.huggingface.co');
}

function normalizePathname(pathname) {
  return typeof pathname === 'string' ? pathname.replace(/\/+/g, '/') : '';
}

function extractHfResolveRevision(url) {
  try {
    const parsed = new URL(url);
    if (!isHuggingFaceHost(parsed.hostname)) return null;
    const parts = normalizePathname(parsed.pathname).split('/').filter(Boolean);
    const resolveIndex = parts.indexOf('resolve');
    if (resolveIndex < 0 || resolveIndex + 1 >= parts.length) return null;
    return decodeURIComponent(parts[resolveIndex + 1]);
  } catch {
    return null;
  }
}

function resolveFetchCacheMode(url) {
  const revision = extractHfResolveRevision(url);
  return revision && HF_COMMIT_REVISION_PATTERN.test(revision) ? 'force-cache' : 'default';
}

async function fetchText(url, signal) {
  const response = await fetch(url, { cache: resolveFetchCacheMode(url), signal });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} while fetching ${url}`);
  }
  return response.text();
}

async function fetchBytes(url, signal) {
  const response = await fetch(url, { cache: resolveFetchCacheMode(url), signal });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} while fetching ${url}`);
  }
  return new Uint8Array(await response.arrayBuffer());
}

function markStep(progressState, status) {
  progressState.completed += 1;
  const percent = progressState.total > 0
    ? (progressState.completed / progressState.total) * 100
    : 100;
  const update = {
    modelId: progressState.modelId,
    percent: clampPercent(percent),
    downloadedBytes: 0,
    totalBytes: 0,
    totalShards: progressState.totalShards || 0,
    completedShards: progressState.completedShards || 0,
    currentShard: progressState.currentShard || null,
    status,
  };
  renderDownloadStatus({ message: status, percent: clampPercent(percent), visible: true });
  emitProgress(update);
}

function markStepWithBytes(progressState, status, options = {}) {
  const additionalBytes = Number.isFinite(options.downloadedBytes) ? Math.max(0, options.downloadedBytes) : 0;
  if (Number.isFinite(options.currentShard)) {
    progressState.currentShard = options.currentShard;
  }
  if (options.incrementCompletedShards) {
    progressState.completedShards = (Number.isFinite(progressState.completedShards) ? progressState.completedShards : 0) + 1;
  }
  if (additionalBytes > 0) {
    progressState.downloadedBytes = (Number.isFinite(progressState.downloadedBytes) ? progressState.downloadedBytes : 0) + additionalBytes;
  }

  const percent = progressState.totalBytes > 0
    ? (progressState.downloadedBytes / progressState.totalBytes) * 100
    : (progressState.total > 0 ? (progressState.completed / progressState.total) * 100 : 0);

  const update = {
    modelId: progressState.modelId,
    percent: clampPercent(percent),
    downloadedBytes: progressState.downloadedBytes,
    totalBytes: progressState.totalBytes || 0,
    totalShards: progressState.totalShards || 0,
    completedShards: progressState.completedShards || 0,
    currentShard: Number.isFinite(progressState.currentShard) ? progressState.currentShard : null,
    status,
  };
  renderDownloadStatus({ message: status, percent, visible: true });
  emitProgress(update);
}

async function persistOptionalFiles(baseUrl, manifest, signal, progressState) {
  if (manifest.tensorsFile) {
    const tensorsText = await fetchText(toFileUrl(baseUrl, manifest.tensorsFile), signal);
    await saveTensorsToStore(tensorsText);
    markStepWithBytes(progressState, `Saved ${manifest.tensorsFile}`, { downloadedBytes: getTextByteLength(tensorsText) });
  }

  const tokenizerFile = manifest?.tokenizer?.file;
  if (typeof tokenizerFile === 'string' && tokenizerFile.length > 0) {
    const tokenizerPath = tokenizerFile.trim();
    if (tokenizerPath.endsWith('.model')) {
      const tokenizerBytes = await fetchBytes(toFileUrl(baseUrl, tokenizerPath), signal);
      await saveTokenizerModel(tokenizerBytes.buffer);
      markStepWithBytes(progressState, `Saved ${tokenizerPath}`, { downloadedBytes: tokenizerBytes.byteLength });
    } else {
      const tokenizerText = await fetchText(toFileUrl(baseUrl, tokenizerPath), signal);
      await saveTokenizer(tokenizerText);
      markStepWithBytes(progressState, `Saved ${tokenizerPath}`, { downloadedBytes: getTextByteLength(tokenizerText) });
    }
  }

  for (const filename of AUX_IMPORT_FILENAMES) {
    try {
      const bytes = await fetchBytes(toFileUrl(baseUrl, filename), signal);
      await saveAuxFile(filename, bytes.buffer);
      markStepWithBytes(progressState, `Saved ${filename}`, { downloadedBytes: bytes.byteLength });
    } catch (error) {
      const message = toErrorMessage(error);
      if (!message.includes('HTTP 404')) {
        throw error;
      }
    }
  }
}

async function persistShards(baseUrl, manifest, signal, progressState) {
  const shards = Array.isArray(manifest.shards) ? manifest.shards : [];
  for (let i = 0; i < shards.length; i += 1) {
    const shard = shards[i];
    const filename = shard?.filename;
    if (typeof filename !== 'string' || filename.length === 0) {
      throw new Error(`Shard entry ${i} is missing filename.`);
    }
    const shardBytes = await fetchBytes(toFileUrl(baseUrl, filename), signal);
    if (Number.isFinite(shard?.size) && shardBytes.byteLength !== shard.size) {
      throw new Error(`Shard size mismatch for ${filename}: expected ${shard.size}, got ${shardBytes.byteLength}`);
    }
    await writeShard(Number(shard.index), shardBytes, { verify: true });
    const shardNumber = Number.isFinite(shard.index) ? shard.index + 1 : i + 1;
    markStepWithBytes(progressState, `Imported shard ${shardNumber}/${shards.length}`, {
      downloadedBytes: shardBytes.byteLength,
      currentShard: shardNumber,
      incrementCompletedShards: true,
    });
  }
}

function setActiveDownload(record) {
  activeDownload = record;
  emitStateChange({
    active: !!record,
    modelId: record?.modelId || '',
  });
}

export function configureDownloadCallbacks(nextCallbacks = {}) {
  callbacks = Object.freeze({
    onModelRegistered: typeof nextCallbacks.onModelRegistered === 'function' ? nextCallbacks.onModelRegistered : null,
    onModelsUpdated: typeof nextCallbacks.onModelsUpdated === 'function' ? nextCallbacks.onModelsUpdated : null,
    onProgress: typeof nextCallbacks.onProgress === 'function' ? nextCallbacks.onProgress : null,
    onStateChange: typeof nextCallbacks.onStateChange === 'function' ? nextCallbacks.onStateChange : null,
  });
}

export async function refreshDownloads() {
  if (activeDownload?.modelId) {
    renderDownloadStatus({
      message: `Importing ${activeDownload.modelId}...`,
      percent: 5,
      visible: true,
    });
    appendLog(`Active: importing ${activeDownload.modelId}`);
    return;
  }
  renderDownloadStatus({
    message: 'Idle',
    percent: 0,
    visible: false,
  });
}

export async function startDownload() {
  const baseUrlInput = getElement('download-base-url');
  const modelIdInput = getElement('download-model-id');
  const baseUrl = baseUrlInput instanceof HTMLInputElement ? baseUrlInput.value : '';
  const modelIdOverride = modelIdInput instanceof HTMLInputElement ? modelIdInput.value : '';

  try {
    return await startDownloadFromBaseUrl(baseUrl, modelIdOverride);
  } catch (error) {
    const message = toErrorMessage(error);
    renderDownloadStatus({ message: `Import failed: ${message}`, percent: 0, visible: true });
    appendLog(`Import failed: ${message}`);
    return false;
  }
}

export async function startDownloadFromBaseUrl(baseUrl, modelIdOverride = '') {
  if (activeDownload) {
    throw new Error('A model import is already in progress.');
  }

  const normalizedBaseUrl = normalizeBaseUrl(baseUrl);
  if (!normalizedBaseUrl) {
    throw new Error('Enter a valid RDRR base URL.');
  }

  const abortController = new AbortController();
  const working = {
    abortController,
    modelId: '',
    baseUrl: normalizedBaseUrl,
  };
  setActiveDownload(working);

  try {
    renderDownloadStatus({
      message: 'Fetching manifest...',
      percent: 2,
      visible: true,
    });
    const manifestText = await fetchText(toFileUrl(normalizedBaseUrl, 'manifest.json'), abortController.signal);
    const parsed = parseManifest(manifestText);
    const manifest = typeof structuredClone === 'function'
      ? structuredClone(parsed)
      : JSON.parse(JSON.stringify(parsed));

    const modelId = modelIdOverride
      ? assertValidModelId(modelIdOverride, 'Model ID override')
      : assertValidModelId(manifest?.modelId, 'RDRR manifest modelId');
    manifest.modelId = modelId;
    working.modelId = modelId;
    emitStateChange({ active: true, modelId });

    const shards = Array.isArray(manifest.shards) ? manifest.shards : [];
    const tokenizerStep = manifest?.tokenizer?.file ? 1 : 0;
    const tensorsStep = manifest?.tensorsFile ? 1 : 0;
    const totalSteps = Math.max(1, shards.length + tokenizerStep + tensorsStep + AUX_IMPORT_FILENAMES.length + 2);
    const shardByteTotal = sumManifestShardBytes(shards);
    const manifestTotalBytes = Number.isFinite(manifest.totalSize) && manifest.totalSize > 0
      ? Math.floor(manifest.totalSize)
      : shardByteTotal;
    const progressState = {
      total: totalSteps,
      completed: 0,
      modelId,
      totalBytes: manifestTotalBytes,
      downloadedBytes: 0,
      totalShards: shards.length,
      completedShards: 0,
      currentShard: null,
    };

    await openModelStore(modelId);
    await saveManifest(JSON.stringify(manifest, null, 2));
    markStep(progressState, `Saved manifest for ${modelId}`);

    await persistOptionalFiles(normalizedBaseUrl, manifest, abortController.signal, progressState);
    await persistShards(normalizedBaseUrl, manifest, abortController.signal, progressState);

    if (typeof callbacks.onModelRegistered === 'function') {
      await callbacks.onModelRegistered(modelId);
    }
    if (typeof callbacks.onModelsUpdated === 'function') {
      await callbacks.onModelsUpdated();
    }

    renderDownloadStatus({
      message: `Import complete: ${modelId}`,
      percent: 100,
      visible: true,
    });
    emitProgress({
      modelId,
      percent: 100,
      downloadedBytes: progressState.downloadedBytes,
      totalBytes: progressState.totalBytes,
      totalShards: progressState.totalShards,
      completedShards: progressState.completedShards,
      currentShard: progressState.currentShard,
      status: 'complete',
    });
    appendLog(`Imported ${modelId}`);
    return true;
  } catch (error) {
    if (working.modelId) {
      try {
        await cleanupPartialImport(working.modelId);
      } catch (cleanupError) {
        appendLog(`Partial import cleanup failed: ${toErrorMessage(cleanupError)}`);
      }
    }
    if (abortController.signal.aborted) {
      renderDownloadStatus({
        message: 'Import canceled.',
        percent: 0,
        visible: true,
      });
      appendLog('Import canceled');
      return false;
    }
    const message = toErrorMessage(error);
    renderDownloadStatus({
      message: `Import failed: ${message}`,
      percent: 0,
      visible: true,
    });
    appendLog(`Import failed: ${message}`);
    throw error;
  } finally {
    setActiveDownload(null);
  }
}

export function pauseActiveDownload() {
  renderDownloadStatus({
    message: 'Pause is not supported for URL import. Use Cancel to stop.',
    percent: 0,
    visible: true,
  });
}

export function resumeActiveDownload() {
  renderDownloadStatus({
    message: 'Resume is not supported for URL import. Restart the import.',
    percent: 0,
    visible: true,
  });
}

export function cancelActiveDownload() {
  if (!activeDownload?.abortController) return;
  activeDownload.abortController.abort();
}
