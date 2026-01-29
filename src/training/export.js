
import { readBuffer } from '../memory/buffer-pool.js';
import { f16ToF32Array } from '../inference/kv-cache/types.js';
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

  const hasGPUBuffer = typeof GPUBuffer !== 'undefined';
  let data;
  if (entry.tensor instanceof Float32Array) {
    data = entry.tensor;
  } else if (hasGPUBuffer && entry.tensor?.buffer instanceof GPUBuffer) {
    const raw = await readBuffer(entry.tensor.buffer);
    data = dtype === 'f16'
      ? f16ToF32Array(new Uint16Array(raw))
      : new Float32Array(raw);
  } else if (hasGPUBuffer && entry.tensor instanceof GPUBuffer) {
    const raw = await readBuffer(entry.tensor);
    data = dtype === 'f16'
      ? f16ToF32Array(new Uint16Array(raw))
      : new Float32Array(raw);
  } else {
    throw new Error(`Unsupported tensor type for ${entry.name}`);
  }

  return { dtype: 'f32', shape: [...shape], data };
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
    weightsFormat: 'json',
  });

  const serialized = [];
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

  manifest.tensors = serialized;
  manifest.weightsSize = totalBytes;

  return { manifest, json: serializeManifest(manifest, pretty) };
}
