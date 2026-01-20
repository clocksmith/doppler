
import { getRuntimeConfig } from '../config/runtime.js';
import { isIndexedDBAvailable, isOPFSAvailable } from './quota.js';
import { createIdbStore } from './backends/idb-store.js';
import { getStorageBackendType } from './shard-manager.js';

const REGISTRY_FILENAME = 'models.json';
const REGISTRY_MODEL_ID = 'registry:models';

function normalizeRegistry(registry) {
  if (!registry || typeof registry !== 'object' || Array.isArray(registry)) {
    return { models: [] };
  }
  const models = Array.isArray(registry.models) ? registry.models : [];
  return { models };
}

async function readRegistryFromOpfs() {
  const runtime = getRuntimeConfig();
  const root = await navigator.storage.getDirectory();
  const opfsRoot = await root.getDirectoryHandle(runtime.loading.opfsPath.opfsRootDir, { create: true });
  const fileHandle = await opfsRoot.getFileHandle(REGISTRY_FILENAME, { create: true });
  const file = await fileHandle.getFile();
  if (file.size === 0) {
    return { models: [] };
  }
  const text = await file.text();
  return normalizeRegistry(JSON.parse(text));
}

async function writeRegistryToOpfs(registry) {
  const runtime = getRuntimeConfig();
  const root = await navigator.storage.getDirectory();
  const opfsRoot = await root.getDirectoryHandle(runtime.loading.opfsPath.opfsRootDir, { create: true });
  const fileHandle = await opfsRoot.getFileHandle(REGISTRY_FILENAME, { create: true });
  const writable = await fileHandle.createWritable();
  await writable.write(JSON.stringify(registry, null, 2));
  await writable.close();
  return { backend: 'opfs', path: REGISTRY_FILENAME };
}

async function readRegistryFromIdb() {
  const runtime = getRuntimeConfig();
  const store = createIdbStore(runtime.loading.storage.backend.indexeddb);
  await store.openModel(REGISTRY_MODEL_ID, { create: true });
  try {
    const text = await store.readFile(REGISTRY_FILENAME);
    await store.cleanup();
    if (!text) return { models: [] };
    return normalizeRegistry(JSON.parse(text));
  } catch {
    await store.cleanup();
    return { models: [] };
  }
}

async function writeRegistryToIdb(registry) {
  const runtime = getRuntimeConfig();
  const store = createIdbStore(runtime.loading.storage.backend.indexeddb);
  await store.openModel(REGISTRY_MODEL_ID, { create: true });
  await store.writeFile(REGISTRY_FILENAME, JSON.stringify(registry, null, 2));
  await store.cleanup();
  return { backend: 'indexeddb', path: REGISTRY_FILENAME };
}

export async function loadModelRegistry() {
  if (isOPFSAvailable()) {
    return readRegistryFromOpfs();
  }
  if (isIndexedDBAvailable()) {
    return readRegistryFromIdb();
  }
  throw new Error('No supported storage backend available for model registry. Supported: opfs, indexeddb.');
}

export async function saveModelRegistry(registry) {
  const normalized = normalizeRegistry(registry);
  if (isOPFSAvailable()) {
    return writeRegistryToOpfs(normalized);
  }
  if (isIndexedDBAvailable()) {
    return writeRegistryToIdb(normalized);
  }
  throw new Error('No supported storage backend available for model registry. Supported: opfs, indexeddb.');
}

export async function listRegisteredModels() {
  const registry = await loadModelRegistry();
  return registry.models || [];
}

export async function registerModel(entry) {
  if (!entry || typeof entry.modelId !== 'string' || entry.modelId.length === 0) {
    throw new Error('Model registry entry requires a modelId.');
  }

  const registry = await loadModelRegistry();
  const now = new Date().toISOString();
  const backend = entry.backend ?? getStorageBackendType() ?? 'unknown';

  const next = {
    ...entry,
    createdAt: entry.createdAt ?? now,
    backend,
  };

  const models = registry.models || [];
  const index = models.findIndex((model) => model.modelId === entry.modelId);
  if (index >= 0) {
    models[index] = next;
  } else {
    models.push(next);
  }
  registry.models = models;
  await saveModelRegistry(registry);
  return next;
}

export async function removeRegisteredModel(modelId) {
  const registry = await loadModelRegistry();
  const models = (registry.models || []).filter((model) => model.modelId !== modelId);
  registry.models = models;
  await saveModelRegistry(registry);
  return models;
}
