

import { readFile, stat, open, readdir } from 'fs/promises';
import { join, dirname } from 'path';
import {
  DTYPE_SIZE,
  parseSafetensorsHeader as parseSafetensorsHeaderCore,
  parseSafetensorsIndexJsonText,
  groupTensorsByLayer as groupTensorsByLayerCore,
  calculateTotalSize as calculateTotalSizeCore,
} from './types.js';
import {
  parseConfigJsonText,
  parseTokenizerConfigJsonText,
  parseTokenizerJsonText,
} from '../tokenizer/types.js';

export { DTYPE_SIZE, DTYPE_MAP } from './types.js';

export function parseSafetensorsHeader(buffer) {
  return parseSafetensorsHeaderCore(buffer);
}

export async function parseSafetensorsFile(filePath) {
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

    const headerBuffer = Buffer.alloc(8 + headerSize);
    await fileHandle.read(headerBuffer, 0, 8 + headerSize, 0);
    const headerArrayBuffer = headerBuffer.buffer.slice(
      headerBuffer.byteOffset,
      headerBuffer.byteOffset + headerBuffer.byteLength
    );
    const parsedHeader = parseSafetensorsHeaderCore(headerArrayBuffer);

    const tensors = parsedHeader.tensors.map((tensor) => ({
      ...tensor,
      elemSize: tensor.elemSize ?? DTYPE_SIZE[tensor.dtype] ?? 1,
      byteSize: tensor.byteSize ?? tensor.elemSize ?? 1,
      filePath,
    }));

    await fileHandle.close();

    return {
      dataOffset: parsedHeader.dataOffset,
      metadata: parsedHeader.metadata,
      tensors,
      filePath,
      fileSize,
    };
  } catch (e) {
    await fileHandle.close();
    throw e;
  }
}

export async function parseSafetensorsIndex(indexPath) {
  const indexBuffer = await readFile(indexPath, 'utf8');
  const index = parseSafetensorsIndexJsonText(indexBuffer);
  const { metadata = {}, weight_map } = index;
  const modelDir = dirname(indexPath);

  const shardToTensors = new Map();
  for (const [tensorName, shardFile] of Object.entries(weight_map)) {
    if (!shardToTensors.has(shardFile)) {
      shardToTensors.set(shardFile, []);
    }
    shardToTensors.get(shardFile).push(tensorName);
  }

  const shards = [];
  const allTensors = [];
  const shardParsed = new Map();

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

function extractConfigFromMetadata(metadata) {
  const config = {
    format: metadata.format || 'pt',
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

export async function loadModelConfig(modelDir) {
  try {
    const configPath = join(modelDir, 'config.json');
    const configBuffer = await readFile(configPath, 'utf8');
    return parseConfigJsonText(configBuffer);
  } catch {
    return null;
  }
}

export async function loadTokenizerConfig(modelDir) {
  try {
    const configPath = join(modelDir, 'tokenizer_config.json');
    const configBuffer = await readFile(configPath, 'utf8');
    return parseTokenizerConfigJsonText(configBuffer);
  } catch {
    return null;
  }
}

export async function loadTokenizerJson(modelDir) {
  try {
    const tokenizerPath = join(modelDir, 'tokenizer.json');
    const buffer = await readFile(tokenizerPath, 'utf8');
    return parseTokenizerJsonText(buffer);
  } catch {
    return null;
  }
}

export async function detectModelFormat(modelDir) {
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

export async function parseSafetensors(pathOrDir) {
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

export function getTensor(parsed, name) {
  return parsed.tensors.find(t => t.name === name) || null;
}

export function getTensors(parsed, pattern) {
  return parsed.tensors.filter(t => pattern.test(t.name));
}

export async function readTensorData(tensor, buffer) {
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
    return data.buffer;
  } finally {
    await file.close();
  }
}

export function groupTensorsByLayer(parsed) {
  return groupTensorsByLayerCore(parsed);
}

export function calculateTotalSize(parsed) {
  return calculateTotalSizeCore(parsed);
}
