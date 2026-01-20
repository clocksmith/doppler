

import { getRuntimeConfig } from '../config/runtime.js';
import { isIndexedDBAvailable, isOPFSAvailable } from './quota.js';
import { createIdbStore } from './backends/idb-store.js';

const REPORTS_DIR = 'reports';
const REPORT_MODEL_PREFIX = 'reports:';

function normalizeModelId(modelId) {
  return String(modelId || 'unknown').replace(/[^a-zA-Z0-9_-]/g, '_');
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

async function saveReportToOpfs(modelId, filename, payload) {
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
}

async function saveReportToIdb(modelId, filename, payload) {
  const runtime = getRuntimeConfig();
  const idbConfig = runtime.loading.storage.backend.indexeddb;
  const store = createIdbStore(idbConfig);
  const reportModelId = `${REPORT_MODEL_PREFIX}${modelId}`;
  await store.openModel(reportModelId, { create: true });
  await store.writeFile(filename, payload);
  await store.cleanup();
  return `reports/${modelId}/${filename}`;
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

  throw new Error('No storage backend available for reports');
}
