

import {
  parseSafetensorsHeader,
  parseSafetensorsIndexJsonText,
  groupTensorsByLayer as groupTensorsByLayerCore,
  calculateTotalSize as calculateTotalSizeCore,
} from '../formats/safetensors/types.js';
import {
  parseConfigJsonText,
  parseTokenizerJsonText,
  parseTokenizerConfigJsonText,
} from '../formats/tokenizer/types.js';
import { MAX_HEADER_SIZE, MB } from '../config/schema/index.js';
import { normalizeTensorSource } from './tensor-source-file.js';

export const DTYPE_SIZE = {
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

export const DTYPE_MAP = {
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

// ============================================================================
// Public API
// ============================================================================


async function readTextFromSource(source) {
  if (source?.file && typeof source.file.text === 'function') {
    return source.file.text();
  }
  const buffer = await source.readRange(0, source.size);
  return new TextDecoder().decode(buffer);
}

export async function parseSafetensorsFile(file) {
  const source = normalizeTensorSource(file);
  const headerSizeBuffer = await source.readRange(0, 8);
  const headerSizeView = new DataView(headerSizeBuffer);
  const headerSizeLow = headerSizeView.getUint32(0, true);
  const headerSizeHigh = headerSizeView.getUint32(4, true);
  const headerSize = headerSizeHigh * 0x100000000 + headerSizeLow;
  if (headerSize > MAX_HEADER_SIZE) {
    throw new Error(`Header too large: ${headerSize} bytes`);
  }

  const headerBuffer = await source.readRange(8, headerSize);
  const combined = new Uint8Array(8 + headerSize);
  combined.set(new Uint8Array(headerSizeBuffer), 0);
  combined.set(new Uint8Array(headerBuffer), 8);
  const parsedHeader = parseSafetensorsHeader(combined.buffer);

  const tensors = parsedHeader.tensors.map((tensor) => ({
    ...tensor,
    elemSize: tensor.elemSize ?? DTYPE_SIZE[tensor.dtype] ?? 4,
    dtypeOriginal: tensor.dtypeOriginal ?? tensor.dtype,
    file: source.file,
    source,
  }));

  return {
    headerSize: parsedHeader.headerSize,
    dataOffset: parsedHeader.dataOffset,
    metadata: parsedHeader.metadata,
    tensors,
    file: source.file,
    source,
    fileSize: source.size,
    fileName: source.name,
  };
}


export async function parseSafetensorsSharded(
  files,
  indexJson = null
) {
  const fileMap = new Map();
  for (const file of files) {
    const source = normalizeTensorSource(file);
    fileMap.set(source.name, source);
  }

  // If we have an index, use it to determine tensor locations
  let metadata = {};

  if (indexJson) {
    metadata = indexJson.metadata || {};
  }

  // Parse each safetensors file
  const shards = [];
  const allTensors = [];

  for (const file of files) {
    const source = normalizeTensorSource(file);
    if (!source.name.endsWith('.safetensors')) continue;

    const parsed = await parseSafetensorsFile(source);
    shards.push({
      file: source.name,
      size: source.size,
      tensorCount: parsed.tensors.length,
    });

    // Add shard info to tensors
    for (const tensor of parsed.tensors) {
      tensor.shardFile = source.name;
      allTensors.push(tensor);
    }
  }

  return {
    metadata,
    shards,
    tensors: allTensors,
    fileMap,
  };
}


export async function readTensorData(tensor) {
  const source = tensor.source ?? (tensor.file ? normalizeTensorSource(tensor.file) : null);
  if (!source) {
    throw new Error('No source reference for tensor');
  }
  return source.readRange(tensor.offset, tensor.size);
}


export async function* streamTensorData(
  tensor,
  chunkSize = 64 * MB
) {
  const source = tensor.source ?? (tensor.file ? normalizeTensorSource(tensor.file) : null);
  if (!source) {
    throw new Error('No source reference for tensor');
  }

  let offset = tensor.offset;
  const endOffset = tensor.offset + tensor.size;

  while (offset < endOffset) {
    const end = Math.min(offset + chunkSize, endOffset);
    const buffer = await source.readRange(offset, end - offset);
    yield new Uint8Array(buffer);
    offset = end;
  }
}


export async function parseConfigJson(configFile) {
  const source = normalizeTensorSource(configFile);
  const text = await readTextFromSource(source);
  return parseConfigJsonText(text);
}

export async function parseTokenizerConfigJson(tokenizerConfigFile) {
  const source = normalizeTensorSource(tokenizerConfigFile);
  const text = await readTextFromSource(source);
  return parseTokenizerConfigJsonText(text);
}


export async function parseTokenizerJson(tokenizerFile) {
  const source = normalizeTensorSource(tokenizerFile);
  const text = await readTextFromSource(source);
  return parseTokenizerJsonText(text);
}


export async function parseIndexJson(indexFile) {
  const source = normalizeTensorSource(indexFile);
  const text = await readTextFromSource(source);
  return parseSafetensorsIndexJsonText(text);
}


export function detectModelFormat(files) {
  // Check for index file (sharded model)
  const indexFile = files.find(f => f.name === 'model.safetensors.index.json');
  if (indexFile) {
    return {
      type: 'sharded',
      indexFile,
      safetensorsFiles: files.filter(f => f.name.endsWith('.safetensors')),
    };
  }

  // Check for single safetensors file
  const safetensorsFiles = files.filter(f => f.name.endsWith('.safetensors'));
  if (safetensorsFiles.length === 1) {
    return {
      type: 'single',
      safetensorsFile: safetensorsFiles[0],
    };
  }

  if (safetensorsFiles.length > 1) {
    return {
      type: 'sharded-no-index',
      safetensorsFiles,
    };
  }

  // Check for GGUF
  const ggufFile = files.find(f => f.name.endsWith('.gguf'));
  if (ggufFile) {
    return {
      type: 'gguf',
      ggufFile,
    };
  }

  return { type: 'unknown', files };
}


export function getAuxiliaryFiles(files) {
  return {
    config: files.find(f => f.name === 'config.json'),
    tokenizerConfig: files.find(f => f.name === 'tokenizer_config.json'),
    tokenizer: files.find(f => f.name === 'tokenizer.json'),
    tokenizerModel: files.find(f => f.name === 'tokenizer.model'),
    specialTokensMap: files.find(f => f.name === 'special_tokens_map.json'),
    generationConfig: files.find(f => f.name === 'generation_config.json'),
  };
}


export function calculateTotalSize(parsed) {
  return calculateTotalSizeCore(parsed);
}


export function groupTensorsByLayer(parsed) {
  // Cast to browser type - core tensors are a subset of browser tensors
  return groupTensorsByLayerCore(parsed);
}
