/**
 * LoRA adapter loader.
 *
 * Supports JSON manifest with inline tensor data (array or base64).
 *
 * @module adapters/lora-loader
 */

import { LORA_MODULE_ALIASES, type LoRAAdapter, type LoRAModuleName } from '../inference/pipeline/lora.js';

export interface LoRATensorSpec {
  name: string;
  shape: [number, number];
  dtype?: 'f32';
  data?: number[];
  base64?: string;
  opfsPath?: string;
  url?: string;
}

export interface LoRAManifest {
  name: string;
  version?: string;
  baseModel?: string;
  rank: number;
  alpha: number;
  targetModules?: LoRAModuleName[];
  tensors: LoRATensorSpec[];
}

export interface LoRALoadOptions {
  readOPFS?: (path: string) => Promise<ArrayBuffer>;
  writeOPFS?: (path: string, data: ArrayBuffer) => Promise<void>;
  fetchUrl?: (url: string) => Promise<ArrayBuffer>;
}

interface ParsedTensorName {
  layer: number;
  module: LoRAModuleName;
  kind: 'a' | 'b';
}

const parseTensorName = (name: string): ParsedTensorName | null => {
  const match = name.match(/layers?\.?(\d+)\.([^\.]+)\.lora_([ab])/i);
  if (!match) return null;
  const layer = parseInt(match[1], 10);
  const rawModule = match[2].toLowerCase();
  const module = LORA_MODULE_ALIASES[rawModule];
  if (!module) return null;
  const kind = match[3].toLowerCase() === 'a' ? 'a' : 'b';
  return { layer, module, kind };
};

const decodeBase64ToFloat32 = (base64: string): Float32Array => {
  let binary: Uint8Array;
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

const toFloat32Array = async (
  tensor: LoRATensorSpec,
  options: LoRALoadOptions
): Promise<Float32Array> => {
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

const validateShape = (tensor: LoRATensorSpec, data: Float32Array): void => {
  if (tensor.dtype && tensor.dtype !== 'f32') {
    throw new Error(`LoRA tensor ${tensor.name} has unsupported dtype: ${tensor.dtype}`);
  }
  const [rows, cols] = tensor.shape;
  const expected = rows * cols;
  if (data.length !== expected) {
    throw new Error(`LoRA tensor ${tensor.name} shape mismatch: expected ${expected}, got ${data.length}`);
  }
};

export async function loadLoRAFromManifest(
  manifest: LoRAManifest,
  options: LoRALoadOptions = {}
): Promise<LoRAAdapter> {
  const adapter: LoRAAdapter = {
    name: manifest.name,
    version: manifest.version,
    baseModel: manifest.baseModel,
    rank: manifest.rank,
    alpha: manifest.alpha,
    targetModules: manifest.targetModules,
    layers: new Map(),
  };

  for (const tensor of manifest.tensors || []) {
    const parsed = parseTensorName(tensor.name);
    if (!parsed) continue;
    const data = await toFloat32Array(tensor, options);
    validateShape(tensor, data);

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
  }

  return adapter;
}

export async function loadLoRAFromUrl(
  url: string,
  options: LoRALoadOptions = {}
): Promise<LoRAAdapter> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch LoRA manifest: ${res.status}`);
  }
  const manifest = await res.json() as LoRAManifest;
  return loadLoRAFromManifest(manifest, options);
}
