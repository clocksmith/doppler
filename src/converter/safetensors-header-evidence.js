import { sha256BytesHex } from '../utils/sha256.js';

export const SAFETENSORS_HEADER_PIN_SCHEMA_ID = 'doppler.safetensors-header-pin/v1';
export const SAFETENSORS_HEADER_EVIDENCE_SCHEMA_ID = 'doppler.safetensors-header-evidence/v1';

const MAX_HEADER_BYTES = 64 * 1024 * 1024;
const SHA256_PATTERN = /^(?:sha256:)?([0-9a-f]{64})$/;

function requireObject(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
  return value;
}

function requireString(value, label) {
  if (typeof value !== 'string' || !value.trim()) throw new Error(`${label} must be a non-empty string.`);
  return value;
}

function normalizeSha256(value, label) {
  const match = requireString(value, label).match(SHA256_PATTERN);
  if (!match) throw new Error(`${label} must be a SHA-256 digest.`);
  return match[1];
}

function requireShape(value, tensorName) {
  if (!Array.isArray(value) || value.some((dimension) => !Number.isInteger(dimension) || dimension < 0)) {
    throw new Error(`SafeTensors tensor "${tensorName}" has an invalid shape.`);
  }
  return [...value];
}

export function readSafetensorsHeaderLength(prefix) {
  if (!(prefix instanceof Uint8Array) || prefix.byteLength !== 8) {
    throw new Error('SafeTensors header prefix must contain exactly 8 bytes.');
  }
  const view = new DataView(prefix.buffer, prefix.byteOffset, prefix.byteLength);
  const length = Number(view.getBigUint64(0, true));
  if (!Number.isSafeInteger(length) || length <= 0 || length > MAX_HEADER_BYTES) {
    throw new Error(`SafeTensors header length ${String(length)} is unsupported.`);
  }
  return length;
}

export function parseSafetensorsHeaderEvidence(bytes, { sourceFile, expectedSha256 }) {
  if (!(bytes instanceof Uint8Array)) throw new Error('SafeTensors header evidence must be bytes.');
  const headerLength = readSafetensorsHeaderLength(bytes.subarray(0, 8));
  if (bytes.byteLength !== headerLength + 8) {
    throw new Error(
      `SafeTensors header evidence for "${sourceFile}" has ${bytes.byteLength} bytes; expected ${headerLength + 8}.`
    );
  }
  const observedSha256 = sha256BytesHex(bytes);
  const pinnedSha256 = normalizeSha256(expectedSha256, `header digest for "${sourceFile}"`);
  if (observedSha256 !== pinnedSha256) {
    throw new Error(
      `SafeTensors header digest mismatch for "${sourceFile}": expected ${pinnedSha256}, got ${observedSha256}.`
    );
  }
  let header;
  try {
    header = JSON.parse(new TextDecoder().decode(bytes.subarray(8)));
  } catch (error) {
    throw new Error(`SafeTensors header for "${sourceFile}" is invalid JSON: ${error.message}`);
  }
  requireObject(header, `SafeTensors header for "${sourceFile}"`);
  const tensors = {};
  for (const tensorName of Object.keys(header).filter((name) => name !== '__metadata__').sort()) {
    const descriptor = requireObject(header[tensorName], `SafeTensors tensor "${tensorName}"`);
    tensors[tensorName] = {
      dtype: requireString(descriptor.dtype, `SafeTensors tensor "${tensorName}" dtype`),
      shape: requireShape(descriptor.shape, tensorName),
      sourceFile,
    };
  }
  if (Object.keys(tensors).length === 0) {
    throw new Error(`SafeTensors header for "${sourceFile}" contains no tensors.`);
  }
  return Object.freeze({
    sourceFile,
    sourceHeaderSha256: observedSha256,
    headerLength,
    tensorCount: Object.keys(tensors).length,
    tensors,
  });
}

export async function materializeSafetensorsHeaderEvidence(pin, readRange) {
  requireObject(pin, 'SafeTensors header pin');
  if (pin.schema !== SAFETENSORS_HEADER_PIN_SCHEMA_ID) {
    throw new Error(`SafeTensors header pin requires schema "${SAFETENSORS_HEADER_PIN_SCHEMA_ID}".`);
  }
  if (typeof readRange !== 'function') throw new Error('SafeTensors header evidence requires a range reader.');
  const checkpointId = requireString(pin.checkpointId, 'checkpointId');
  const repository = requireString(pin.repository, 'repository');
  const revision = requireString(pin.revision, 'revision');
  if (!Array.isArray(pin.shards) || pin.shards.length === 0) {
    throw new Error('SafeTensors header pin requires at least one shard.');
  }

  const shardEvidence = [];
  const tensors = {};
  for (const [index, rawShard] of pin.shards.entries()) {
    const shard = requireObject(rawShard, `shards[${index}]`);
    const sourceFile = requireString(shard.sourceFile, `shards[${index}].sourceFile`);
    const expectedSha256 = normalizeSha256(shard.headerSha256, `shards[${index}].headerSha256`);
    const prefix = await readRange({ repository, revision, sourceFile, start: 0, end: 7 });
    const headerLength = readSafetensorsHeaderLength(prefix);
    const bytes = await readRange({
      repository,
      revision,
      sourceFile,
      start: 0,
      end: headerLength + 7,
    });
    const evidence = parseSafetensorsHeaderEvidence(bytes, { sourceFile, expectedSha256 });
    shardEvidence.push(evidence);
    for (const [tensorName, descriptor] of Object.entries(evidence.tensors)) {
      if (Object.hasOwn(tensors, tensorName)) {
        throw new Error(`Tensor "${tensorName}" appears in multiple SafeTensors shards.`);
      }
      tensors[tensorName] = descriptor;
    }
  }

  const [primary, ...additional] = shardEvidence;
  return Object.freeze({
    schema: SAFETENSORS_HEADER_EVIDENCE_SCHEMA_ID,
    checkpointId,
    repository,
    revision,
    sourceFile: primary.sourceFile,
    sourceHeaderSha256: primary.sourceHeaderSha256,
    additionalSourceHeaders: additional.map((item) => ({
      sourceFile: item.sourceFile,
      sourceHeaderSha256: item.sourceHeaderSha256,
      headerLength: item.headerLength,
      tensorCount: item.tensorCount,
    })),
    headerLength: primary.headerLength,
    tensorCount: Object.keys(tensors).length,
    tensors: Object.fromEntries(Object.entries(tensors).sort(([left], [right]) => left.localeCompare(right))),
  });
}
