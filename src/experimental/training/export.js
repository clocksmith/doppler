
import { readBuffer } from '../../memory/buffer-pool.js';
import { f16ToF32Array } from '../../inference/kv-cache/types.js';
import { createManifest, serializeManifest } from '../adapters/adapter-manifest.js';

function encodeBase64(bytes) {
  if (typeof btoa === 'function') {
    let binary = '';
    const view = new Uint8Array(bytes);
    for (let i = 0; i < view.length; i += 1) {
      binary += String.fromCharCode(view[i]);
    }
    return btoa(binary);
  }
  if (typeof Buffer !== 'undefined') {
    return Buffer.from(bytes).toString('base64');
  }
  throw new Error('Base64 encoding not supported in this environment');
}

async function resolveTensorData(entry) {
  const dtype = entry.dtype ?? entry.tensor?.dtype ?? 'f32';
  let shape = entry.shape ?? entry.tensor?.shape;
  if (!shape) {
    throw new Error(`Missing shape for tensor ${entry.name}`);
  }
  shape = [...shape];
  const expectedElements = shape.reduce((acc, value) => acc * Number(value), 1);
  if (!Number.isInteger(expectedElements) || expectedElements < 1) {
    throw new Error(`Invalid shape for tensor ${entry.name}`);
  }
  const bytesPerElement = dtype === 'f16' ? 2 : 4;
  const expectedBytes = expectedElements * bytesPerElement;

  const hasGPUBuffer = typeof GPUBuffer !== 'undefined';
  let data;
  if (entry.tensor instanceof Float32Array) {
    data = entry.tensor.length === expectedElements
      ? entry.tensor
      : entry.tensor.slice(0, expectedElements);
  } else if (hasGPUBuffer && entry.tensor?.buffer instanceof GPUBuffer) {
    const raw = await readBuffer(entry.tensor.buffer);
    data = dtype === 'f16'
      ? f16ToF32Array(new Uint16Array(raw.slice(0, expectedBytes)))
      : new Float32Array(raw.slice(0, expectedBytes));
  } else if (hasGPUBuffer && entry.tensor instanceof GPUBuffer) {
    const raw = await readBuffer(entry.tensor);
    data = dtype === 'f16'
      ? f16ToF32Array(new Uint16Array(raw.slice(0, expectedBytes)))
      : new Float32Array(raw.slice(0, expectedBytes));
  } else {
    throw new Error(`Unsupported tensor type for ${entry.name}`);
  }
  if (data.length !== expectedElements) {
    throw new Error(`Tensor ${entry.name} data length mismatch: expected ${expectedElements}, got ${data.length}.`);
  }

  return { dtype: 'f32', shape, data };
}

function float32ToBytes(values) {
  const bytes = new Uint8Array(values.length * 4);
  const view = new DataView(bytes.buffer);
  for (let index = 0; index < values.length; index += 1) {
    view.setFloat32(index * 4, values[index], true);
  }
  return bytes;
}

function writeUint64LE(value) {
  const bytes = new Uint8Array(8);
  const view = new DataView(bytes.buffer);
  view.setBigUint64(0, BigInt(value), true);
  return bytes;
}

async function sha256Hex(bytes) {
  if (!globalThis.crypto?.subtle) {
    throw new Error('SHA-256 hashing requires crypto.subtle.');
  }
  const hashBuffer = await globalThis.crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(hashBuffer))
    .map((value) => value.toString(16).padStart(2, '0'))
    .join('');
}

export function serializeLoRASafetensors(tensors) {
  if (!Array.isArray(tensors) || tensors.length === 0) {
    throw new Error('serializeLoRASafetensors requires tensors');
  }
  const header = {};
  const tensorBytes = [];
  let offset = 0;
  for (const tensor of tensors) {
    if (!tensor?.name) {
      throw new Error('serializeLoRASafetensors tensor name is required.');
    }
    const shape = Array.isArray(tensor.shape) ? tensor.shape.map((value) => Number(value)) : [];
    if (shape.length !== 2 || shape.some((value) => !Number.isInteger(value) || value < 1)) {
      throw new Error(`serializeLoRASafetensors tensor ${tensor.name} requires shape [rows, cols].`);
    }
    const data = tensor.data instanceof Float32Array
      ? tensor.data
      : new Float32Array(tensor.data || []);
    const expectedLength = shape[0] * shape[1];
    if (data.length !== expectedLength) {
      throw new Error(`serializeLoRASafetensors tensor ${tensor.name} shape mismatch: expected ${expectedLength}, got ${data.length}.`);
    }
    const bytes = float32ToBytes(data);
    header[tensor.name] = {
      dtype: 'F32',
      shape,
      data_offsets: [offset, offset + bytes.byteLength],
    };
    tensorBytes.push(bytes);
    offset += bytes.byteLength;
  }
  const headerBytes = new TextEncoder().encode(JSON.stringify(header));
  const sizePrefix = writeUint64LE(headerBytes.byteLength);
  const out = new Uint8Array(sizePrefix.byteLength + headerBytes.byteLength + offset);
  out.set(sizePrefix, 0);
  out.set(headerBytes, sizePrefix.byteLength);
  let writeOffset = sizePrefix.byteLength + headerBytes.byteLength;
  for (const bytes of tensorBytes) {
    out.set(bytes, writeOffset);
    writeOffset += bytes.byteLength;
  }
  return out.buffer;
}

export async function exportLoRAAdapter(options) {
  const {
    id,
    name,
    baseModel,
    rank,
    alpha,
    targetModules,
    version,
    description,
    metadata,
    tensors,
    format = 'base64',
    weightsFormat = 'json',
    weightsPath = 'adapters.safetensors',
    pretty = false,
  } = options;

  if (!Array.isArray(tensors) || tensors.length === 0) {
    throw new Error('exportLoRAAdapter requires tensors');
  }

  const manifest = createManifest({
    id,
    name,
    baseModel,
    rank,
    alpha,
    targetModules,
    version,
    description,
    metadata,
    weightsFormat,
  });

  const serialized = [];
  const resolvedTensors = [];
  let totalBytes = 0;

  for (const entry of tensors) {
    const resolved = await resolveTensorData(entry);
    const byteLength = resolved.data.byteLength;
    totalBytes += byteLength;
    const tensorSpec = {
      name: entry.name,
      shape: resolved.shape,
      dtype: resolved.dtype,
    };
    resolvedTensors.push({
      ...tensorSpec,
      data: resolved.data,
    });

    if (format === 'array') {
      tensorSpec.data = Array.from(resolved.data);
    } else {
      const slice = resolved.data.buffer.slice(
        resolved.data.byteOffset,
        resolved.data.byteOffset + resolved.data.byteLength
      );
      tensorSpec.base64 = encodeBase64(slice);
    }

    serialized.push(tensorSpec);
  }

  if (weightsFormat === 'safetensors') {
    const weights = serializeLoRASafetensors(resolvedTensors);
    const weightsBytes = new Uint8Array(weights);
    const checksum = await sha256Hex(weightsBytes);
    manifest.weightsPath = weightsPath;
    manifest.weightsSize = weightsBytes.byteLength;
    manifest.checksum = checksum;
    manifest.checksumAlgorithm = 'sha256';
    return {
      manifest,
      json: serializeManifest(manifest, pretty),
      weights,
      weightsSha256: checksum,
      weightsPath,
    };
  }

  manifest.tensors = serialized;
  manifest.weightsSize = totalBytes;

  return { manifest, json: serializeManifest(manifest, pretty) };
}
