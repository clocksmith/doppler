

import { getRuntimeConfig } from '../config/runtime.js';
import { isIndexedDBAvailable, isOPFSAvailable } from './quota.js';
import { createIdbStore } from './backends/idb-store.js';
import { normalizeModelId as normalizeModelIdShared } from './normalize-model-id.js';

const REPORTS_DIR = 'reports';
const REPORT_MODEL_PREFIX = 'reports:';

function normalizeModelId(modelId) {
  return normalizeModelIdShared(modelId || 'unknown');
}

function formatTimestamp(value) {
  if (value instanceof Date) {
    return value.toISOString().replace(/[:]/g, '-');
  }
  if (typeof value === 'string' && value.length > 0) {
    return value.replace(/[:]/g, '-');
  }
  return new Date().toISOString().replace(/[:]/g, '-');
}

function isNodeRuntime() {
  return typeof process !== 'undefined' && !!process.versions?.node;
}

function nodeModule(specifier) {
  return `node:${specifier}`;
}

function createReportStorageError(error, backend, modelId, filename) {
  const message = error?.message || String(error);
  const wrapped = new Error(
    `Report storage ${backend} write failed for ${modelId}/${filename}: ${message}`,
    error instanceof Error ? { cause: error } : undefined
  );
  wrapped.name = error?.name || 'Error';
  if (error?.code !== undefined) {
    wrapped.code = error.code;
  }
  wrapped.details = {
    ...(error?.details && typeof error.details === 'object' ? error.details : {}),
    reportBackend: backend,
    modelId,
    filename,
  };
  return wrapped;
}

async function saveReportToOpfs(modelId, filename, payload) {
  try {
    const runtime = getRuntimeConfig();
    const root = await navigator.storage.getDirectory();
    const opfsRoot = await root.getDirectoryHandle(runtime.loading.opfsPath.opfsRootDir, { create: true });
    const reportsDir = await opfsRoot.getDirectoryHandle(REPORTS_DIR, { create: true });
    const modelDir = await reportsDir.getDirectoryHandle(modelId, { create: true });
    const fileHandle = await modelDir.getFileHandle(filename, { create: true });
    const writable = await fileHandle.createWritable();
    await writable.write(payload);
    await writable.close();
    return `reports/${modelId}/${filename}`;
  } catch (error) {
    throw createReportStorageError(error, 'opfs', modelId, filename);
  }
}

async function saveReportToIdb(modelId, filename, payload) {
  const runtime = getRuntimeConfig();
  const idbConfig = runtime.loading.storage.backend.indexeddb;
  const store = createIdbStore(idbConfig);
  const reportModelId = `${REPORT_MODEL_PREFIX}${modelId}`;
  await store.openModel(reportModelId, { create: true });
  await store.writeFile(filename, new TextEncoder().encode(payload));
  await store.cleanup();
  return `reports/${modelId}/${filename}`;
}

async function saveReportToNodeFs(modelId, filename, payload) {
  const [{ mkdir, writeFile }, { dirname, join, relative, resolve, sep }] = await Promise.all([
    import(nodeModule('fs/promises')),
    import(nodeModule('path')),
  ]);

  const rootDir = process.env.DOPPLER_REPORTS_DIR
    ? resolve(String(process.env.DOPPLER_REPORTS_DIR))
    : resolve(process.cwd(), 'reports');
  const filePath = join(rootDir, modelId, filename);
  await mkdir(dirname(filePath), { recursive: true });
  await writeFile(filePath, payload, 'utf8');

  const relativePath = relative(process.cwd(), filePath);
  const outsideCwd = relativePath.startsWith('..') || relativePath.includes(`${sep}..${sep}`);
  return outsideCwd ? filePath : relativePath.split(sep).join('/');
}

export async function saveReport(modelId, report, options = {}) {
  const normalized = normalizeModelId(modelId);
  const timestamp = formatTimestamp(options.timestamp);
  const filename = `${timestamp}.json`;
  const payload = JSON.stringify(report, null, 2);

  if (isOPFSAvailable()) {
    const path = await saveReportToOpfs(normalized, filename, payload);
    return { backend: 'opfs', path };
  }
  if (isIndexedDBAvailable()) {
    const path = await saveReportToIdb(normalized, filename, payload);
    return { backend: 'indexeddb', path };
  }
  if (isNodeRuntime()) {
    const path = await saveReportToNodeFs(normalized, filename, payload);
    return { backend: 'node-fs', path };
  }

  throw new Error('No storage backend available for reports');
}
