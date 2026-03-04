import { sha256Hex } from '../utils/sha256.js';

function isNodeRuntime() {
  return typeof process !== 'undefined' && !!process.versions?.node;
}

function looksLikePath(value) {
  const normalized = String(value || '').trim();
  return normalized.includes('/') || normalized.includes('\\') || normalized.endsWith('.json');
}

function openCheckpointDB(options = {}) {
  const {
    dbName = 'doppler-training',
    storeName = 'checkpoints',
    version = 1,
  } = options;

  return new Promise((resolve, reject) => {
    const request = indexedDB.open(dbName, version);

    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(storeName)) {
        db.createObjectStore(storeName);
      }
    };

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve({ db: request.result, storeName });
  });
}

async function resolveNodeCheckpointPath(key, options = {}) {
  const [{ resolve, join, dirname }, { mkdir }] = await Promise.all([
    import('node:path'),
    import('node:fs/promises'),
  ]);
  const configuredRoot = typeof options.nodeDir === 'string' && options.nodeDir.trim()
    ? options.nodeDir.trim()
    : '.doppler-checkpoints';
  if (looksLikePath(key)) {
    const direct = resolve(String(key));
    await mkdir(dirname(direct), { recursive: true });
    return direct;
  }
  const safeKey = String(key).replace(/[^a-zA-Z0-9._-]/g, '_');
  const root = resolve(configuredRoot);
  await mkdir(root, { recursive: true });
  return join(root, `${safeKey}.json`);
}

async function readNodeCheckpointRecord(filePath) {
  const { readFile } = await import('node:fs/promises');
  try {
    const raw = await readFile(filePath, 'utf8');
    return JSON.parse(raw);
  } catch (error) {
    if (error?.code === 'ENOENT') {
      return null;
    }
    throw error;
  }
}

async function writeNodeCheckpointRecord(filePath, data) {
  const [{ writeFile, mkdir }, { dirname }] = await Promise.all([
    import('node:fs/promises'),
    import('node:path'),
  ]);
  await mkdir(dirname(filePath), { recursive: true });
  await writeFile(filePath, `${JSON.stringify(data, null, 2)}\n`, 'utf8');
}

function readCheckpointRecord(db, storeName, key) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(storeName, 'readonly');
    tx.onerror = () => reject(tx.error);
    const store = tx.objectStore(storeName);
    const request = store.get(key);
    request.onsuccess = () => resolve(request.result ?? null);
    request.onerror = () => reject(request.error);
  });
}

function stableSortObject(value) {
  if (Array.isArray(value)) {
    return value.map((entry) => stableSortObject(entry));
  }
  if (!value || typeof value !== 'object') {
    return value;
  }
  const sorted = {};
  for (const key of Object.keys(value).sort()) {
    sorted[key] = stableSortObject(value[key]);
  }
  return sorted;
}

function stableJson(value) {
  return JSON.stringify(stableSortObject(value));
}

function coalesceMetadataValue(...candidates) {
  for (const value of candidates) {
    if (value !== undefined && value !== null) {
      return value;
    }
  }
  return null;
}

function buildCheckpointHashPayload(data) {
  const metadata = data?.metadata || {};
  const lineage = metadata.lineage || {};
  return {
    payload: {
      ...data,
      metadata: undefined,
    },
    metadata: {
      configHash: metadata.configHash ?? null,
      datasetHash: metadata.datasetHash ?? null,
      tokenizerHash: metadata.tokenizerHash ?? null,
      optimizerHash: metadata.optimizerHash ?? null,
      runtimePresetId: metadata.runtimePresetId ?? null,
      kernelPathId: metadata.kernelPathId ?? null,
      environmentMetadata: metadata.environmentMetadata ?? null,
      buildProvenance: metadata.buildProvenance ?? null,
      lineage: {
        checkpointKey: lineage.checkpointKey ?? null,
        sequence: Number.isInteger(lineage.sequence) ? lineage.sequence : 0,
        previousCheckpointHash: lineage.previousCheckpointHash ?? null,
      },
    },
  };
}

export async function saveCheckpoint(key, payload, options = {}) {
  const useNodeStore = isNodeRuntime() && typeof indexedDB === 'undefined';
  const nodePath = useNodeStore ? await resolveNodeCheckpointPath(key, options) : null;
  const browserStore = useNodeStore ? null : await openCheckpointDB(options);
  const previousData = useNodeStore
    ? await readNodeCheckpointRecord(nodePath)
    : await readCheckpointRecord(browserStore.db, browserStore.storeName, key);
  const previousMetadata = previousData?.metadata || {};
  const previousLineage = previousMetadata.lineage || {};
  const previousCheckpointHash = options.priorCheckpointHash
    || previousMetadata.checkpointHash
    || previousLineage.previousCheckpointHash
    || null;
  const lineageSequence = Number.isInteger(previousLineage.sequence)
    ? previousLineage.sequence + 1
    : 1;

  const data = { ...payload };
  const payloadMetadata = data.metadata && typeof data.metadata === 'object'
    ? data.metadata
    : {};
  data.metadata = {
    ...payloadMetadata,
    timestamp: Date.now(),
    configHash: coalesceMetadataValue(options.configHash, payloadMetadata.configHash, previousMetadata.configHash),
    datasetHash: coalesceMetadataValue(options.datasetHash, payloadMetadata.datasetHash, previousMetadata.datasetHash),
    tokenizerHash: coalesceMetadataValue(options.tokenizerHash, payloadMetadata.tokenizerHash, previousMetadata.tokenizerHash),
    optimizerHash: coalesceMetadataValue(options.optimizerHash, payloadMetadata.optimizerHash, previousMetadata.optimizerHash),
    runtimePresetId: coalesceMetadataValue(options.runtimePresetId, payloadMetadata.runtimePresetId, previousMetadata.runtimePresetId),
    kernelPathId: coalesceMetadataValue(options.kernelPathId, payloadMetadata.kernelPathId, previousMetadata.kernelPathId),
    environmentMetadata: coalesceMetadataValue(
      options.environmentMetadata,
      payloadMetadata.environmentMetadata,
      previousMetadata.environmentMetadata
    ),
    buildProvenance: coalesceMetadataValue(options.buildProvenance, payloadMetadata.buildProvenance, previousMetadata.buildProvenance),
    lineage: {
      checkpointKey: key,
      sequence: lineageSequence,
      previousCheckpointHash,
    },
  };
  data.metadata.checkpointHash = sha256Hex(
    stableJson(buildCheckpointHashPayload(data))
  );

  if (useNodeStore) {
    await writeNodeCheckpointRecord(nodePath, data);
    return;
  }

  return new Promise((resolve, reject) => {
    const tx = browserStore.db.transaction(browserStore.storeName, 'readwrite');
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    const store = tx.objectStore(browserStore.storeName);
    store.put(data, key);
  });
}

export async function loadCheckpoint(key, options = {}) {
  const useNodeStore = isNodeRuntime() && typeof indexedDB === 'undefined';
  const nodePath = useNodeStore ? await resolveNodeCheckpointPath(key, options) : null;
  const data = useNodeStore
    ? await readNodeCheckpointRecord(nodePath)
    : await (async () => {
      const { db, storeName } = await openCheckpointDB(options);
      return readCheckpointRecord(db, storeName, key);
    })();

  if (!data || !data.metadata || !options.expectedMetadata) {
    return data;
  }

  const mismatches = [];
  for (const [k, v] of Object.entries(options.expectedMetadata)) {
    if (data.metadata[k] !== v) {
      mismatches.push(k);
    }
  }

  if (mismatches.length > 0) {
    if (!options.forceResume) {
      throw new Error(`Checkpoint mismatch on fields: ${mismatches.join(', ')}`);
    }
    const priorCheckpointMetadataHash = data.metadata?.checkpointHash ?? null;
    data.metadata.resumeAudits = data.metadata.resumeAudits || [];
    data.metadata.resumeAudits.push({
      timestamp: Date.now(),
      mismatchedFields: mismatches,
      source: options.forceResumeSource || 'unspecified',
      operator: options.forceResumeOperator || null,
      reason: options.forceResumeReason || 'forced resume flag provided',
      priorCheckpointMetadataHash,
    });
    await saveCheckpoint(key, data, options);
  }

  return data;
}
