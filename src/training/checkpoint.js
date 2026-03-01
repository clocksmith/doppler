import { sha256Hex } from '../utils/sha256.js';

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
  const { db, storeName } = await openCheckpointDB(options);
  const previousData = await readCheckpointRecord(db, storeName, key);
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
  data.metadata = {
    ...(data.metadata || {}),
    timestamp: Date.now(),
    configHash: options.configHash,
    datasetHash: options.datasetHash,
    tokenizerHash: options.tokenizerHash,
    optimizerHash: options.optimizerHash,
    runtimePresetId: options.runtimePresetId,
    kernelPathId: options.kernelPathId,
    environmentMetadata: options.environmentMetadata,
    buildProvenance: options.buildProvenance ?? null,
    lineage: {
      checkpointKey: key,
      sequence: lineageSequence,
      previousCheckpointHash,
    },
  };
  data.metadata.checkpointHash = sha256Hex(
    stableJson(buildCheckpointHashPayload(data))
  );

  return new Promise((resolve, reject) => {
    const tx = db.transaction(storeName, 'readwrite');
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    const store = tx.objectStore(storeName);
    store.put(data, key);
  });
}

export async function loadCheckpoint(key, options = {}) {
  const { db, storeName } = await openCheckpointDB(options);
  const data = await readCheckpointRecord(db, storeName, key);

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
    data.metadata.resumeAudits = data.metadata.resumeAudits || [];
    data.metadata.resumeAudits.push({
      timestamp: Date.now(),
      mismatchedFields: mismatches,
      reason: options.forceResumeReason || 'forced resume flag provided',
    });
    await saveCheckpoint(key, data, options);
  }

  return data;
}
