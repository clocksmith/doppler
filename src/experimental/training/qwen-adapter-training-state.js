import { readBuffer, uploadData } from '../../memory/buffer-pool.js';
import { sha256BytesHex, sha256Hex } from '../../formats/sha256.js';
import { stableSortObject } from '../../formats/stable-sort-object.js';

const ARTIFACT_TYPE = 'qwen_adapter_training_state';
const SCHEMA_VERSION = 1;

function isTensor(value) {
  return value?.buffer
    && value.dtype === 'f32'
    && Array.isArray(value.shape)
    && value.shape.length > 0;
}

function elementCount(shape) {
  return shape.reduce((product, value) => {
    if (!Number.isInteger(value) || value < 1) {
      throw new Error('Qwen adapter training state requires positive tensor dimensions.');
    }
    return product * value;
  }, 1);
}

function toBase64(bytes) {
  if (typeof Buffer !== 'undefined') {
    return Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength).toString('base64');
  }
  let binary = '';
  for (let index = 0; index < bytes.length; index += 1) {
    binary += String.fromCharCode(bytes[index]);
  }
  return btoa(binary);
}

function fromBase64(value) {
  if (typeof value !== 'string' || value.length < 1) {
    throw new Error('Qwen adapter training state tensor data is missing.');
  }
  if (typeof Buffer !== 'undefined') {
    const decoded = Buffer.from(value, 'base64');
    return new Uint8Array(decoded.buffer, decoded.byteOffset, decoded.byteLength);
  }
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

function stableJson(value) {
  return JSON.stringify(stableSortObject(value));
}

function payloadWithoutHash(payload) {
  const { payloadSha256: _payloadSha256, ...content } = payload;
  return content;
}

function normalizeEntries(entries) {
  if (!Array.isArray(entries) || entries.length < 1) {
    throw new Error('Qwen adapter training state requires parameter entries.');
  }
  const names = new Set();
  const parameters = new Set();
  return entries.map((entry) => {
    const name = String(entry?.name || '');
    if (!name || names.has(name)) {
      throw new Error(`Qwen adapter training state duplicate or missing name "${name}".`);
    }
    if (!isTensor(entry?.parameter)) {
      throw new Error(`Qwen adapter training state ${name} requires an F32 parameter tensor.`);
    }
    if (parameters.has(entry.parameter)) {
      throw new Error(`Qwen adapter training state parameter is duplicated at ${name}.`);
    }
    names.add(name);
    parameters.add(entry.parameter);
    return {
      name,
      parameter: entry.parameter,
      shape: [...entry.parameter.shape],
      byteLength: elementCount(entry.parameter.shape) * 4,
    };
  });
}

async function captureTensor(tensor, byteLength) {
  const bytes = new Uint8Array(await readBuffer(tensor.buffer, byteLength));
  return {
    dataBase64: toBase64(bytes),
    dataSha256: sha256BytesHex(bytes),
  };
}

function validateSnapshot(snapshot, byteLength, label) {
  const bytes = fromBase64(snapshot?.dataBase64);
  if (bytes.byteLength !== byteLength) {
    throw new Error(
      `Qwen adapter training state ${label} byte length mismatch: `
      + `${bytes.byteLength} != ${byteLength}.`
    );
  }
  if (snapshot?.dataSha256 !== sha256BytesHex(bytes)) {
    throw new Error(`Qwen adapter training state ${label} checksum mismatch.`);
  }
  return bytes;
}

function normalizeProgress(progress = {}) {
  const consumedRowIds = Array.isArray(progress.consumedRowIds)
    ? progress.consumedRowIds.map((value) => String(value))
    : [];
  if (consumedRowIds.some((value) => !value)) {
    throw new Error('Qwen adapter training state consumed row IDs must be non-empty.');
  }
  const microstepCount = Number(progress.microstepCount);
  const optimizerStepCount = Number(progress.optimizerStepCount);
  if (!Number.isInteger(microstepCount) || microstepCount < 0
    || !Number.isInteger(optimizerStepCount) || optimizerStepCount < 0) {
    throw new Error('Qwen adapter training state progress counters must be non-negative integers.');
  }
  const consumedPrefixSha256 = sha256Hex(JSON.stringify(consumedRowIds));
  return {
    microstepCount,
    optimizerStepCount,
    consumedRowIds,
    consumedPrefixSha256,
  };
}

export async function captureQwenAdapterTrainingState(entries, optimizer, progress = {}) {
  const normalized = normalizeEntries(entries);
  if (!(optimizer?.state instanceof Map) || !Number.isInteger(optimizer.stepCount)) {
    throw new Error('Qwen adapter training state requires an initialized optimizer.');
  }
  const normalizedProgress = normalizeProgress({
    ...progress,
    optimizerStepCount: optimizer.stepCount,
  });
  const tensors = {};
  for (const entry of normalized) {
    const state = optimizer.state.get(entry.parameter);
    if (!isTensor(state?.m) || !isTensor(state?.v)) {
      throw new Error(`Qwen adapter training state is missing AdamW slots for ${entry.name}.`);
    }
    tensors[entry.name] = {
      dtype: 'f32',
      shape: entry.shape,
      parameter: await captureTensor(entry.parameter, entry.byteLength),
      moment1: await captureTensor(state.m, entry.byteLength),
      moment2: await captureTensor(state.v, entry.byteLength),
    };
  }
  const payload = {
    artifactType: ARTIFACT_TYPE,
    schemaVersion: SCHEMA_VERSION,
    parameterNames: normalized.map((entry) => entry.name),
    progress: normalizedProgress,
    tensors,
  };
  return {
    ...payload,
    payloadSha256: sha256Hex(stableJson(payload)),
  };
}

export function validateQwenAdapterTrainingState(payload) {
  if (payload?.artifactType !== ARTIFACT_TYPE || payload?.schemaVersion !== SCHEMA_VERSION) {
    throw new Error('Qwen adapter training state artifact contract mismatch.');
  }
  const expectedHash = sha256Hex(stableJson(payloadWithoutHash(payload)));
  if (payload.payloadSha256 !== expectedHash) {
    throw new Error('Qwen adapter training state payload checksum mismatch.');
  }
  const progress = normalizeProgress(payload.progress);
  if (payload.progress?.consumedPrefixSha256 !== progress.consumedPrefixSha256) {
    throw new Error('Qwen adapter training state consumed-prefix checksum mismatch.');
  }
  if (!Array.isArray(payload.parameterNames) || payload.parameterNames.length < 1) {
    throw new Error('Qwen adapter training state parameter names are missing.');
  }
  const names = payload.parameterNames.map((value) => String(value || ''));
  if (names.some((value) => !value) || new Set(names).size !== names.length) {
    throw new Error('Qwen adapter training state parameter names must be unique and non-empty.');
  }
  if (JSON.stringify(Object.keys(payload.tensors || {})) !== JSON.stringify(names)) {
    throw new Error('Qwen adapter training state tensor keys differ from parameter names.');
  }
  return payload;
}

export async function restoreQwenAdapterTrainingState(entries, optimizer, payload) {
  const normalized = normalizeEntries(entries);
  validateQwenAdapterTrainingState(payload);
  const names = normalized.map((entry) => entry.name);
  if (JSON.stringify(names) !== JSON.stringify(payload.parameterNames)) {
    throw new Error('Qwen adapter training state parameter order differs from the live model.');
  }
  if (!(optimizer?.state instanceof Map)) {
    throw new Error('Qwen adapter training state requires an optimizer for restore.');
  }
  const restores = [];
  for (const entry of normalized) {
    const snapshot = payload.tensors?.[entry.name];
    if (snapshot?.dtype !== 'f32'
      || JSON.stringify(snapshot?.shape) !== JSON.stringify(entry.shape)) {
      throw new Error(`Qwen adapter training state tensor contract differs for ${entry.name}.`);
    }
    restores.push({
      entry,
      state: optimizer.getState(entry.parameter),
      parameter: validateSnapshot(
        snapshot.parameter,
        entry.byteLength,
        `${entry.name}.parameter`
      ),
      moment1: validateSnapshot(snapshot.moment1, entry.byteLength, `${entry.name}.moment1`),
      moment2: validateSnapshot(snapshot.moment2, entry.byteLength, `${entry.name}.moment2`),
    });
  }
  for (const restore of restores) {
    uploadData(restore.entry.parameter.buffer, restore.parameter);
    uploadData(restore.state.m.buffer, restore.moment1);
    uploadData(restore.state.v.buffer, restore.moment2);
  }
  optimizer.stepCount = payload.progress.optimizerStepCount;
  return {
    ...payload.progress,
    parameterCount: normalized.length,
    payloadSha256: payload.payloadSha256,
  };
}
