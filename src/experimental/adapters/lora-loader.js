

import { LORA_MODULE_ALIASES } from '../../inference/pipelines/text/lora.js';
import { applyAdapterManifestDefaults, validateManifest } from './adapter-manifest.js';
import { log } from '../../debug/index.js';

// ============================================================================
// Helper Functions
// ============================================================================


const parseTensorName = (name) => {
  const text = String(name || '');
  const match = text.match(/(?:^|\.)layers?\.?(\d+)\.(?:[A-Za-z0-9_]+\.)*([A-Za-z0-9_]+)\.lora_([ab])(?:\.[A-Za-z0-9_]+)?$/i);
  if (!match) return null;
  const layer = parseInt(match[1], 10);
  const rawModule = match[2].toLowerCase();
  const module = LORA_MODULE_ALIASES[rawModule];
  if (!module) return null;
  const kind = match[3].toLowerCase() === 'a' ? 'a' : 'b';
  return { layer, module, kind };
};


const decodeBase64ToFloat32 = (base64) => {
  let binary;
  if (typeof atob === 'function') {
    const decoded = atob(base64);
    binary = new Uint8Array(decoded.length);
    for (let i = 0; i < decoded.length; i++) {
      binary[i] = decoded.charCodeAt(i);
    }
  } else if (typeof Buffer !== 'undefined') {
    binary = new Uint8Array(Buffer.from(base64, 'base64'));
  } else {
    throw new Error('Base64 decode not supported in this environment');
  }
  return new Float32Array(binary.buffer.slice(binary.byteOffset, binary.byteOffset + binary.byteLength));
};


const toFloat32Array = async (tensor, options) => {
  if (tensor.data) return new Float32Array(tensor.data);
  if (tensor.base64) return decodeBase64ToFloat32(tensor.base64);
  if (tensor.opfsPath && options.readOPFS) {
    const data = await options.readOPFS(tensor.opfsPath);
    return new Float32Array(data);
  }
  if (tensor.url && options.fetchUrl) {
    const data = await options.fetchUrl(tensor.url);
    if (tensor.opfsPath && options.writeOPFS) {
      await options.writeOPFS(tensor.opfsPath, data);
    }
    return new Float32Array(data);
  }
  throw new Error(`LoRA tensor ${tensor.name} missing data`);
};


const validateShape = (tensor, data) => {
  const dtype = tensor.dtype;
  if (!dtype) {
    throw new Error(`LoRA tensor ${tensor.name} missing dtype`);
  }
  if (dtype !== 'f32') {
    throw new Error(`LoRA tensor ${tensor.name} has unsupported dtype: ${dtype}`);
  }
  const [rows, cols] = tensor.shape;
  const expected = rows * cols;
  if (data.length !== expected) {
    throw new Error(`LoRA tensor ${tensor.name} shape mismatch: expected ${expected}, got ${data.length}`);
  }
};


function asArrayBuffer(data) {
  if (data instanceof ArrayBuffer) return data;
  if (ArrayBuffer.isView(data)) {
    return data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  }
  throw new Error('Expected ArrayBuffer or typed array data');
}

async function computeSHA256(data) {
  const hashBuffer = await crypto.subtle.digest('SHA-256', asArrayBuffer(data));
  const hashArray = new Uint8Array(hashBuffer);
  return Array.from(hashArray).map(b => b.toString(16).padStart(2, '0')).join('');
}

function stripHashPrefix(value) {
  return String(value || '').replace(/^sha256:/i, '').toLowerCase();
}

function isUrl(value) {
  return /^https?:\/\//i.test(String(value || ''));
}

async function readPathBytes(sourcePath, options = {}) {
  const text = String(sourcePath || '').trim();
  if (!text) {
    throw new Error('LoRA weightsPath is required');
  }
  if (isUrl(text)) {
    if (options.fetchUrl) return options.fetchUrl(text);
    const res = await fetch(text);
    if (!res.ok) {
      throw new Error(`Failed to fetch LoRA weights: ${res.status}`);
    }
    return res.arrayBuffer();
  }
  if (options.readFile) {
    const path = options.resolvePath
      ? options.resolvePath(text)
      : (options.basePath ? `${String(options.basePath).replace(/\/$/, '')}/${text}` : text);
    return options.readFile(path);
  }
  if (options.readOPFS) {
    return options.readOPFS(text);
  }
  if (options.fetchUrl) {
    return options.fetchUrl(text);
  }
  throw new Error('Cannot load LoRA weightsPath without fetchUrl, readFile, or readOPFS');
}

async function assertChecksum(data, manifest) {
  const expected = stripHashPrefix(manifest.checksum);
  if (!expected) return;
  const algorithm = String(manifest.checksumAlgorithm || 'sha256').toLowerCase();
  if (algorithm !== 'sha256') {
    throw new Error(`Unsupported LoRA checksum algorithm: ${algorithm}`);
  }
  const computed = await computeSHA256(data);
  if (computed.toLowerCase() !== expected) {
    throw new Error(`LoRA checksum mismatch: expected ${expected}, got ${computed}`);
  }
}

async function loadExternalWeights(manifest, options = {}) {
  const format = String(manifest.weightsFormat || 'safetensors').toLowerCase();
  if (format !== 'safetensors') {
    throw new Error(`Unsupported LoRA weightsFormat for external weights: ${format}`);
  }
  const data = await readPathBytes(manifest.weightsPath, options);
  if (!options.skipVerify) {
    await assertChecksum(data, manifest);
  }
  return loadLoRAFromSafetensors(asArrayBuffer(data), manifest);
}

function assertCompleteAdapterLayers(adapter) {
  for (const [layerIndex, layer] of adapter.layers.entries()) {
    for (const [moduleName, weights] of Object.entries(layer)) {
      const hasA = weights?.a instanceof Float32Array && weights.a.length > 0;
      const hasB = weights?.b instanceof Float32Array && weights.b.length > 0;
      if (!hasA || !hasB) {
        throw new Error(
          `LoRA adapter layer ${layerIndex} module ${moduleName} is incomplete; both lora_a and lora_b tensors are required.`
        );
      }
    }
  }
}

// ============================================================================
// Core Loading Functions
// ============================================================================


export async function loadLoRAWeights(path, options = {}) {
  if (path && typeof path === 'object') {
    const manifest = applyAdapterManifestDefaults(path);
    const validation = validateManifest(manifest);
    if (!validation.valid) {
      const errors = validation.errors.map(e => `${e.field}: ${e.message}`).join('; ');
      throw new Error(`Invalid LoRA manifest: ${errors}`);
    }
    const adapter = await loadLoRAFromManifest(manifest, options);
    return {
      adapter,
      manifest,
      loadedFromCache: false,
      checksumValid: manifest.checksum && !options.skipVerify ? true : undefined,
    };
  }

  let manifestJson;
  let loadedFromCache = false;

  // Determine if path is URL or OPFS path
  const isUrl = path.startsWith('http://') || path.startsWith('https://');

  if (isUrl) {
    // Load from URL
    const res = await fetch(path);
    if (!res.ok) {
      throw new Error(`Failed to fetch LoRA manifest: ${res.status} ${res.statusText}`);
    }
    manifestJson = await res.text();
  } else if (options.readOPFS) {
    // Load from OPFS
    try {
      const buffer = await options.readOPFS(path);
      manifestJson = new TextDecoder().decode(buffer);
      loadedFromCache = true;
    } catch (e) {
      throw new Error(`Failed to read LoRA manifest from OPFS: ${e.message}`);
    }
  } else {
    throw new Error('Cannot load LoRA weights: path is not a URL and no OPFS reader provided');
  }

  // Parse and validate manifest
  let manifest;
  try {
    manifest = JSON.parse(manifestJson);
  } catch (e) {
    throw new Error(`Invalid LoRA manifest JSON: ${e.message}`);
  }

  manifest = applyAdapterManifestDefaults(manifest);
  const validation = validateManifest(manifest);
  if (!validation.valid) {
    const errors = validation.errors.map(e => `${e.field}: ${e.message}`).join('; ');
    throw new Error(`Invalid LoRA manifest: ${errors}`);
  }

  // Build base URL for relative paths
  const baseUrl = isUrl ? path.substring(0, path.lastIndexOf('/') + 1) : '';

  // Create fetch function that resolves relative URLs
  const fetchWithBase = async (url) => {
    const fullUrl = url.startsWith('http') ? url : baseUrl + url;
    if (options.fetchUrl) {
      return options.fetchUrl(fullUrl);
    }
    const res = await fetch(fullUrl);
    if (!res.ok) {
      throw new Error(`Failed to fetch: ${res.status}`);
    }
    return res.arrayBuffer();
  };

  const adapter = await loadLoRAFromManifest(
    manifest,
    { ...options, fetchUrl: fetchWithBase }
  );

  let checksumValid;
  if (manifest.checksum && !options.skipVerify && Array.isArray(manifest.tensors) && manifest.tensors.length > 0) {
    const algorithm = manifest.checksumAlgorithm;
    if (algorithm !== 'sha256') {
      throw new Error(`Unsupported LoRA checksum algorithm: ${algorithm}`);
    } else if (manifest.tensors && manifest.tensors.length > 0) {
      // For inline tensors, compute checksum over concatenated tensor data
      const tensorBuffers = [];
      for (const tensor of manifest.tensors) {
        if (tensor.base64) {
          const decoded = decodeBase64ToFloat32(tensor.base64);
          tensorBuffers.push(decoded.buffer.slice(decoded.byteOffset, decoded.byteOffset + decoded.byteLength));
        } else if (tensor.data) {
          tensorBuffers.push(new Float32Array(tensor.data).buffer);
        }
      }
      if (tensorBuffers.length > 0) {
        const totalSize = tensorBuffers.reduce((sum, buf) => sum + buf.byteLength, 0);
        const combined = new Uint8Array(totalSize);
        let offset = 0;
        for (const buf of tensorBuffers) {
          combined.set(new Uint8Array(buf), offset);
          offset += buf.byteLength;
        }
        const computedHash = await computeSHA256(combined.buffer);
        checksumValid = computedHash.toLowerCase() === stripHashPrefix(manifest.checksum);
        if (!checksumValid) {
          throw new Error(`LoRA checksum mismatch: expected ${manifest.checksum}, got ${computedHash}`);
        }
      }
    }
  }

  return {
    adapter,
    manifest,
    loadedFromCache,
    checksumValid: manifest.checksum && !options.skipVerify && checksumValid !== false ? true : checksumValid,
  };
}


export async function loadLoRAFromManifest(manifest, options = {}) {
  if (manifest?.weightsPath && !Array.isArray(manifest.tensors)) {
    return loadExternalWeights(manifest, options);
  }

  const adapter = {
    name: manifest.name,
    version: manifest.version,
    baseModel: manifest.baseModel,
    rank: manifest.rank,
    alpha: manifest.alpha,
    targetModules: manifest.targetModules,
    layers: new Map(),
  };

  const tensors = manifest.tensors;
  if (!Array.isArray(tensors)) {
    throw new Error('LoRA manifest missing tensors array');
  }
  const total = tensors.length;
  let loaded = 0;

  for (const tensor of tensors) {
    const parsed = parseTensorName(tensor.name);
    if (!parsed) {
      throw new Error(`Unrecognized LoRA tensor name: ${tensor.name}`);
    }

    let data;
    try {
      data = await toFloat32Array(tensor, options);
    } catch (error) {
      log.error('LoRA', `Failed to load tensor data for component "${tensor.name}" (layer=${parsed.layer}, module=${parsed.module}, kind=${parsed.kind}): ${error.message}`);
      throw error;
    }
    try {
      validateShape(tensor, data);
    } catch (error) {
      log.error('LoRA', `Shape validation failed for component "${tensor.name}" (layer=${parsed.layer}, module=${parsed.module}): ${error.message}`);
      throw error;
    }

    const layer = adapter.layers.get(parsed.layer) || {};
    const scale = manifest.rank > 0 ? manifest.alpha / manifest.rank : 1;

    if (!layer[parsed.module]) {
      layer[parsed.module] = {
        a: new Float32Array(0),
        b: new Float32Array(0),
        rank: manifest.rank,
        alpha: manifest.alpha,
        scale,
      };
    }

    if (parsed.kind === 'a') {
      layer[parsed.module].a = data;
    } else {
      layer[parsed.module].b = data;
    }

    adapter.layers.set(parsed.layer, layer);

    loaded++;
    if (options.onProgress) {
      options.onProgress(loaded, total);
    }
  }

  assertCompleteAdapterLayers(adapter);
  return adapter;
}


export async function loadLoRAFromUrl(url, options = {}) {
  const result = await loadLoRAWeights(url, options);
  return result.adapter;
}


export function applyDeltaWeights(baseWeight, loraA, loraB, scale) {
  // Infer dimensions from LoRA matrices
  // A is rank x in_dim, B is out_dim x rank
  // We need to compute: delta = B @ A and add to base weight

  // For efficiency, we don't actually compute the full product
  // Instead, we return the base weight with metadata indicating LoRA should be applied
  // The actual fusion happens in the inference kernel

  // This is a placeholder for direct weight fusion
  // In practice, LoRA is applied dynamically during forward pass
  log.warn('LoRA', 'Direct weight fusion not implemented - use runtime application');
  return baseWeight;
}

// ============================================================================
// Safetensors Support
// ============================================================================


export async function loadLoRAFromSafetensors(data, manifest) {
  const buffer = asArrayBuffer(data);
  // Parse safetensors header
  const view = new DataView(buffer);
  const headerSize = Number(view.getBigUint64(0, true));
  const headerJson = new TextDecoder().decode(
    new Uint8Array(buffer, 8, headerSize)
  );
  const header = JSON.parse(headerJson);

  const adapter = {
    name: manifest.name,
    version: manifest.version,
    baseModel: manifest.baseModel,
    rank: manifest.rank,
    alpha: manifest.alpha,
    targetModules: manifest.targetModules,
    layers: new Map(),
  };

  const dataOffset = 8 + headerSize;

  for (const [tensorName, tensorInfo] of Object.entries(header)) {
    if (tensorName === '__metadata__') continue;

    const parsed = parseTensorName(tensorName);
    if (!parsed) {
      throw new Error(`Unrecognized LoRA safetensors tensor name: ${tensorName}`);
    }

    const [start, end] = tensorInfo.data_offsets;
    const tensorData = new Uint8Array(buffer, dataOffset + start, end - start);

    // Convert to Float32Array based on dtype
    let floatData;
    if (tensorInfo.dtype === 'F32') {
      const aligned = tensorData.slice();
      floatData = new Float32Array(aligned.buffer, aligned.byteOffset, aligned.byteLength / 4);
    } else if (tensorInfo.dtype === 'F16' || tensorInfo.dtype === 'BF16') {
      // Convert from f16/bf16 to f32
      const aligned = tensorData.slice();
      const f16View = new Uint16Array(aligned.buffer, aligned.byteOffset, aligned.byteLength / 2);
      floatData = new Float32Array(f16View.length);
      for (let i = 0; i < f16View.length; i++) {
        if (tensorInfo.dtype === 'F16') {
          floatData[i] = float16ToFloat32(f16View[i]);
        } else {
          floatData[i] = bfloat16ToFloat32(f16View[i]);
        }
      }
    } else {
      log.warn('LoRA', `Unsupported dtype ${tensorInfo.dtype} for component "${tensorName}" (layer=${parsed.layer}, module=${parsed.module}); skipping`);
      continue;
    }

    const shape = Array.isArray(tensorInfo.shape) ? tensorInfo.shape : [];
    const expected = shape.reduce((product, value) => product * Number(value || 0), 1);
    if (shape.length !== 2 || floatData.length !== expected) {
      throw new Error(`LoRA safetensors tensor ${tensorName} shape mismatch: expected ${expected}, got ${floatData.length}`);
    }

    const layer = adapter.layers.get(parsed.layer) || {};
    const scale = manifest.rank > 0 ? manifest.alpha / manifest.rank : 1;

    if (!layer[parsed.module]) {
      layer[parsed.module] = {
        a: new Float32Array(0),
        b: new Float32Array(0),
        rank: manifest.rank,
        alpha: manifest.alpha,
        scale,
      };
    }

    if (parsed.kind === 'a') {
      layer[parsed.module].a = floatData;
    } else {
      layer[parsed.module].b = floatData;
    }

    adapter.layers.set(parsed.layer, layer);
  }

  assertCompleteAdapterLayers(adapter);
  return adapter;
}


function float16ToFloat32(h) {
  const sign = (h & 0x8000) >> 15;
  const exp = (h & 0x7C00) >> 10;
  const frac = h & 0x03FF;

  if (exp === 0) {
    if (frac === 0) return sign ? -0 : 0;
    // Subnormal
    let e = -14;
    let m = frac;
    while ((m & 0x0400) === 0) {
      m <<= 1;
      e--;
    }
    m &= 0x03FF;
    return (sign ? -1 : 1) * Math.pow(2, e) * (1 + m / 1024);
  }
  if (exp === 31) {
    return frac ? NaN : (sign ? -Infinity : Infinity);
  }

  return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + frac / 1024);
}


function bfloat16ToFloat32(bf) {
  // bfloat16 is just the upper 16 bits of float32
  const bytes = new Uint8Array(4);
  bytes[2] = bf & 0xFF;
  bytes[3] = (bf >> 8) & 0xFF;
  return new Float32Array(bytes.buffer)[0];
}
