

import {
  DTYPE_SIZE,
  parseSafetensorsHeader,
  parseSafetensorsIndexJsonText,
  groupTensorsByLayer as groupTensorsByLayerCore,
  calculateTotalSize as calculateTotalSizeCore,
} from '../formats/safetensors.js';
import {
  parseConfigJsonText,
  parseTokenizerJsonText,
  parseTokenizerConfigJsonText,
} from '../formats/tokenizer.js';
import { MAX_HEADER_SIZE, MB } from '../config/schema/index.js';

export { DTYPE_SIZE, DTYPE_MAP } from '../formats/safetensors.js';

// ============================================================================
// Public API
// ============================================================================


export async function parseSafetensorsFile(file) {
  const headerSizeBuffer = await file.slice(0, 8).arrayBuffer();
  const headerSizeView = new DataView(headerSizeBuffer);
  const headerSizeLow = headerSizeView.getUint32(0, true);
  const headerSizeHigh = headerSizeView.getUint32(4, true);
  const headerSize = headerSizeHigh * 0x100000000 + headerSizeLow;
  if (headerSize > MAX_HEADER_SIZE) {
    throw new Error(`Header too large: ${headerSize} bytes`);
  }

  const headerBuffer = await file.slice(8, 8 + headerSize).arrayBuffer();
  const combined = new Uint8Array(8 + headerSize);
  combined.set(new Uint8Array(headerSizeBuffer), 0);
  combined.set(new Uint8Array(headerBuffer), 8);
  const parsedHeader = parseSafetensorsHeader(combined.buffer);

  const tensors = parsedHeader.tensors.map((tensor) => ({
    ...tensor,
    elemSize: tensor.elemSize ?? DTYPE_SIZE[tensor.dtype] ?? 4,
    dtypeOriginal: tensor.dtypeOriginal ?? tensor.dtype,
    file,
  }));

  return {
    headerSize: parsedHeader.headerSize,
    dataOffset: parsedHeader.dataOffset,
    metadata: parsedHeader.metadata,
    tensors,
    file,
    fileSize: file.size,
    fileName: file.name,
  };
}


export async function parseSafetensorsSharded(
  files,
  indexJson = null
) {
  const fileMap = new Map();
  for (const file of files) {
    fileMap.set(file.name, file);
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
    if (!file.name.endsWith('.safetensors')) continue;

    const parsed = await parseSafetensorsFile(file);
    shards.push({
      file: file.name,
      size: file.size,
      tensorCount: parsed.tensors.length,
    });

    // Add shard info to tensors
    for (const tensor of parsed.tensors) {
      tensor.shardFile = file.name;
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
  const file = tensor.file;
  if (!file) {
    throw new Error('No file reference for tensor');
  }

  const blob = file.slice(tensor.offset, tensor.offset + tensor.size);
  return blob.arrayBuffer();
}


export async function* streamTensorData(
  tensor,
  chunkSize = 64 * MB
) {
  const file = tensor.file;
  if (!file) {
    throw new Error('No file reference for tensor');
  }

  let offset = tensor.offset;
  const endOffset = tensor.offset + tensor.size;

  while (offset < endOffset) {
    const end = Math.min(offset + chunkSize, endOffset);
    const blob = file.slice(offset, end);
    const buffer = await blob.arrayBuffer();
    yield new Uint8Array(buffer);
    offset = end;
  }
}


export async function parseConfigJson(configFile) {
  const text = await configFile.text();
  return parseConfigJsonText(text);
}

export async function parseTokenizerConfigJson(tokenizerConfigFile) {
  const text = await tokenizerConfigFile.text();
  return parseTokenizerConfigJsonText(text);
}


export async function parseTokenizerJson(tokenizerFile) {
  const text = await tokenizerFile.text();
  return parseTokenizerJsonText(text);
}


export async function parseIndexJson(indexFile) {
  const text = await indexFile.text();
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
