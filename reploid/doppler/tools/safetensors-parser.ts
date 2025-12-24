/**
 * SafeTensors Format Parser
 * Parses HuggingFace safetensors files for tensor metadata and data.
 */

import { readFile, stat, open, readdir, type FileHandle } from 'fs/promises';
import { join, dirname } from 'path';

export const DTYPE_SIZE: Record<string, number> = {
  F64: 8,
  F32: 4,
  F16: 2,
  BF16: 2,
  I64: 8,
  I32: 4,
  I16: 2,
  I8: 1,
  U8: 1,
  BOOL: 1,
};

export const DTYPE_MAP: Record<string, string> = {
  F64: 'F64',
  F32: 'F32',
  F16: 'F16',
  BF16: 'BF16',
  I64: 'I64',
  I32: 'I32',
  I16: 'I16',
  I8: 'I8',
  U8: 'U8',
  BOOL: 'BOOL',
};

export type SafetensorsDType = keyof typeof DTYPE_SIZE;

export interface SafetensorsTensor {
  name: string;
  shape: number[];
  dtype: string;
  dtypeOriginal?: string;
  offset: number;
  size: number;
  elemSize?: number;
  byteSize?: number;
  filePath?: string;
  shardFile?: string;
  shardPath?: string;
}

export interface SafetensorsHeaderInfo {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}

export interface SafetensorsHeader {
  __metadata__?: Record<string, string>;
  [tensorName: string]: SafetensorsHeaderInfo | Record<string, string> | undefined;
}

export interface ParsedHeader {
  headerSize: number;
  dataOffset: number;
  metadata: Record<string, string>;
  tensors: SafetensorsTensor[];
}

export interface ShardInfo {
  file: string;
  path: string;
  size: number;
  tensorCount: number;
}

export interface ParsedSafetensorsFile {
  dataOffset: number;
  metadata: Record<string, string>;
  tensors: SafetensorsTensor[];
  filePath: string;
  fileSize: number;
  config?: Record<string, unknown>;
  tokenizerConfig?: Record<string, unknown>;
  tokenizerJson?: Record<string, unknown>;
}

export interface ParsedSafetensorsIndex {
  indexPath: string;
  modelDir: string;
  metadata: Record<string, unknown>;
  config: Record<string, unknown>;
  shards: ShardInfo[];
  tensors: SafetensorsTensor[];
  shardParsed: Map<string, ParsedSafetensorsFile>;
  tokenizerConfig?: Record<string, unknown>;
  tokenizerJson?: Record<string, unknown>;
}

export interface ModelFormatInfo {
  sharded: boolean;
  indexPath?: string;
  singlePath?: string;
  files?: string[];
}

export function parseSafetensorsHeader(buffer: ArrayBuffer): ParsedHeader {
  const view = new DataView(buffer);

  const headerSizeLow = view.getUint32(0, true);
  const headerSizeHigh = view.getUint32(4, true);
  const headerSize = headerSizeHigh * 0x100000000 + headerSizeLow;

  if (headerSize > 100 * 1024 * 1024) {
    throw new Error(`Header too large: ${headerSize} bytes`);
  }

  const headerBytes = new Uint8Array(buffer, 8, headerSize);
  const headerJson = new TextDecoder().decode(headerBytes);
  const header: SafetensorsHeader = JSON.parse(headerJson);

  const dataOffset = 8 + headerSize;
  const metadata = (header.__metadata__ || {}) as Record<string, string>;
  delete header.__metadata__;

  const tensors: SafetensorsTensor[] = [];
  for (const [name, info] of Object.entries(header)) {
    if (!info || typeof info !== 'object' || !('dtype' in info)) continue;
    const tensorInfo = info as SafetensorsHeaderInfo;
    const { dtype, shape, data_offsets } = tensorInfo;
    const [startOffset, endOffset] = data_offsets;

    tensors.push({
      name,
      shape,
      dtype: DTYPE_MAP[dtype] || dtype,
      dtypeOriginal: dtype,
      offset: dataOffset + startOffset,
      size: endOffset - startOffset,
      elemSize: DTYPE_SIZE[dtype] || 4,
    });
  }

  tensors.sort((a, b) => a.offset - b.offset);

  return { headerSize, dataOffset, metadata, tensors };
}

export async function parseSafetensorsFile(filePath: string): Promise<ParsedSafetensorsFile> {
  const stats = await stat(filePath);
  const fileSize = stats.size;
  const fileHandle = await open(filePath, 'r');

  try {
    const headerSizeBuffer = Buffer.alloc(8);
    await fileHandle.read(headerSizeBuffer, 0, 8, 0);

    const headerSizeLow = headerSizeBuffer.readUInt32LE(0);
    const headerSizeHigh = headerSizeBuffer.readUInt32LE(4);
    const headerSize = headerSizeHigh * 0x100000000 + headerSizeLow;

    if (headerSize > 100 * 1024 * 1024) {
      throw new Error(`Header too large: ${headerSize} bytes`);
    }

    const headerBuffer = Buffer.alloc(headerSize);
    await fileHandle.read(headerBuffer, 0, headerSize, 8);

    const headerJson = headerBuffer.toString('utf8');
    const header: SafetensorsHeader = JSON.parse(headerJson);

    const dataOffset = 8 + headerSize;
    const metadata = (header.__metadata__ || {}) as Record<string, string>;
    delete header.__metadata__;

    const tensors: SafetensorsTensor[] = [];
    for (const [name, info] of Object.entries(header)) {
      if (!info || typeof info !== 'object' || !('dtype' in info)) continue;
      const tensorInfo = info as SafetensorsHeaderInfo;
      const { dtype, shape, data_offsets } = tensorInfo;
      const [startOffset, endOffset] = data_offsets;

      tensors.push({
        name,
        dtype: DTYPE_MAP[dtype] || dtype,
        shape,
        offset: dataOffset + startOffset,
        size: endOffset - startOffset,
        byteSize: DTYPE_SIZE[dtype] || 1,
        filePath,
      });
    }

    await fileHandle.close();

    return { dataOffset, metadata, tensors, filePath, fileSize };
  } catch (e) {
    await fileHandle.close();
    throw e;
  }
}

export async function parseSafetensorsIndex(indexPath: string): Promise<ParsedSafetensorsIndex> {
  const indexBuffer = await readFile(indexPath, 'utf8');
  const index = JSON.parse(indexBuffer) as {
    metadata?: Record<string, unknown>;
    weight_map: Record<string, string>;
  };

  const { metadata = {}, weight_map } = index;
  const modelDir = dirname(indexPath);

  const shardToTensors = new Map<string, string[]>();
  for (const [tensorName, shardFile] of Object.entries(weight_map)) {
    if (!shardToTensors.has(shardFile)) {
      shardToTensors.set(shardFile, []);
    }
    shardToTensors.get(shardFile)!.push(tensorName);
  }

  const shards: ShardInfo[] = [];
  const allTensors: SafetensorsTensor[] = [];
  const shardParsed = new Map<string, ParsedSafetensorsFile>();

  for (const shardFile of shardToTensors.keys()) {
    const shardPath = join(modelDir, shardFile);
    const parsed = await parseSafetensorsFile(shardPath);
    shardParsed.set(shardFile, parsed);

    shards.push({
      file: shardFile,
      path: shardPath,
      size: parsed.fileSize,
      tensorCount: parsed.tensors.length,
    });

    for (const tensor of parsed.tensors) {
      tensor.shardFile = shardFile;
      tensor.shardPath = shardPath;
      allTensors.push(tensor);
    }
  }

  const config = extractConfigFromMetadata(metadata);

  return {
    indexPath,
    modelDir,
    metadata,
    config,
    shards,
    tensors: allTensors,
    shardParsed,
  };
}

function extractConfigFromMetadata(metadata: Record<string, unknown>): Record<string, unknown> {
  const config: Record<string, unknown> = {
    format: (metadata.format as string) || 'pt',
  };

  for (const [key, value] of Object.entries(metadata || {})) {
    if (typeof value === 'string') {
      try {
        config[key] = JSON.parse(value);
      } catch {
        config[key] = value;
      }
    } else {
      config[key] = value;
    }
  }

  return config;
}

export async function loadModelConfig(modelDir: string): Promise<Record<string, unknown> | null> {
  try {
    const configPath = join(modelDir, 'config.json');
    const configBuffer = await readFile(configPath, 'utf8');
    return JSON.parse(configBuffer);
  } catch {
    return null;
  }
}

export async function loadTokenizerConfig(modelDir: string): Promise<Record<string, unknown> | null> {
  try {
    const configPath = join(modelDir, 'tokenizer_config.json');
    const configBuffer = await readFile(configPath, 'utf8');
    return JSON.parse(configBuffer);
  } catch {
    return null;
  }
}

export async function loadTokenizerJson(modelDir: string): Promise<Record<string, unknown> | null> {
  try {
    const tokenizerPath = join(modelDir, 'tokenizer.json');
    const buffer = await readFile(tokenizerPath, 'utf8');
    return JSON.parse(buffer);
  } catch {
    return null;
  }
}

export async function detectModelFormat(modelDir: string): Promise<ModelFormatInfo> {
  const indexPath = join(modelDir, 'model.safetensors.index.json');
  const singlePath = join(modelDir, 'model.safetensors');

  try {
    await stat(indexPath);
    return { sharded: true, indexPath };
  } catch {
    // Not sharded
  }

  try {
    await stat(singlePath);
    return { sharded: false, singlePath };
  } catch {
    // No model.safetensors
  }

  const files = await readdir(modelDir);
  const safetensorFiles = files.filter(f => f.endsWith('.safetensors'));

  if (safetensorFiles.length === 1) {
    return { sharded: false, singlePath: join(modelDir, safetensorFiles[0]) };
  } else if (safetensorFiles.length > 1) {
    return { sharded: true, files: safetensorFiles.map(f => join(modelDir, f)) };
  }

  throw new Error(`No safetensors files found in ${modelDir}`);
}

export async function parseSafetensors(
  pathOrDir: string
): Promise<ParsedSafetensorsFile | ParsedSafetensorsIndex> {
  const stats = await stat(pathOrDir);

  if (stats.isDirectory()) {
    const format = await detectModelFormat(pathOrDir);
    if (format.sharded && format.indexPath) {
      const parsed = await parseSafetensorsIndex(format.indexPath);
      const modelConfig = await loadModelConfig(pathOrDir);
      if (modelConfig) {
        parsed.config = { ...parsed.config, ...modelConfig };
      }
      parsed.tokenizerConfig = (await loadTokenizerConfig(pathOrDir)) ?? undefined;
      parsed.tokenizerJson = (await loadTokenizerJson(pathOrDir)) ?? undefined;
      return parsed;
    } else if (format.singlePath) {
      const parsed = await parseSafetensorsFile(format.singlePath);
      parsed.config = (await loadModelConfig(pathOrDir)) ?? undefined;
      parsed.tokenizerConfig = (await loadTokenizerConfig(pathOrDir)) ?? undefined;
      parsed.tokenizerJson = (await loadTokenizerJson(pathOrDir)) ?? undefined;
      return parsed;
    }
  }

  if (pathOrDir.endsWith('.json')) {
    return parseSafetensorsIndex(pathOrDir);
  }

  return parseSafetensorsFile(pathOrDir);
}

export function getTensor(
  parsed: ParsedSafetensorsFile | ParsedSafetensorsIndex,
  name: string
): SafetensorsTensor | null {
  return parsed.tensors.find(t => t.name === name) || null;
}

export function getTensors(
  parsed: ParsedSafetensorsFile | ParsedSafetensorsIndex,
  pattern: RegExp
): SafetensorsTensor[] {
  return parsed.tensors.filter(t => pattern.test(t.name));
}

export async function readTensorData(
  tensor: SafetensorsTensor,
  buffer?: ArrayBuffer
): Promise<ArrayBuffer> {
  if (buffer) {
    return buffer.slice(tensor.offset, tensor.offset + tensor.size);
  }

  const filePath = tensor.shardPath || tensor.filePath;
  if (!filePath) {
    throw new Error('No file path for tensor');
  }

  const file = await open(filePath, 'r');
  try {
    const data = Buffer.alloc(tensor.size);
    await file.read(data, 0, tensor.size, tensor.offset);
    return data.buffer as ArrayBuffer;
  } finally {
    await file.close();
  }
}

export function groupTensorsByLayer(
  parsed: ParsedSafetensorsFile | ParsedSafetensorsIndex
): Map<number, SafetensorsTensor[]> {
  const layers = new Map<number, SafetensorsTensor[]>();

  for (const tensor of parsed.tensors) {
    const match = tensor.name.match(/layers?\.(\d+)\./);
    if (match) {
      const layerIdx = parseInt(match[1], 10);
      if (!layers.has(layerIdx)) {
        layers.set(layerIdx, []);
      }
      layers.get(layerIdx)!.push(tensor);
    }
  }

  return layers;
}

export function calculateTotalSize(parsed: ParsedSafetensorsFile | ParsedSafetensorsIndex): number {
  return parsed.tensors.reduce((sum, t) => sum + t.size, 0);
}
