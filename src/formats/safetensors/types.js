/**
 * Shared safetensors parsing utilities (browser + tools).
 */

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

export function parseSafetensorsIndexJsonText(text) {
  return JSON.parse(text);
}

export function parseSafetensorsHeader(buffer) {
  const view = new DataView(buffer);

  const headerSizeLow = view.getUint32(0, true);
  const headerSizeHigh = view.getUint32(4, true);
  const headerSize = headerSizeHigh * 0x100000000 + headerSizeLow;

  if (headerSize > 100 * 1024 * 1024) {
    throw new Error(`Header too large: ${headerSize} bytes`);
  }

  if (buffer.byteLength < 8 + headerSize) {
    throw new Error('Buffer does not contain full safetensors header');
  }

  const headerBytes = new Uint8Array(buffer, 8, headerSize);
  const headerJson = new TextDecoder().decode(headerBytes);
  const header = JSON.parse(headerJson);

  const dataOffset = 8 + headerSize;
  const metadata = (header.__metadata__ || {});
  delete header.__metadata__;

  const tensors = [];
  for (const [name, info] of Object.entries(header)) {
    if (!info || typeof info !== 'object' || !('dtype' in info)) continue;
    const tensorInfo = info;
    const { dtype, shape, data_offsets } = tensorInfo;
    const [startOffset, endOffset] = data_offsets;
    const elemSize = DTYPE_SIZE[dtype] || 1;

    tensors.push({
      name,
      dtype: DTYPE_MAP[dtype] || dtype,
      dtypeOriginal: dtype,
      shape,
      offset: dataOffset + startOffset,
      size: endOffset - startOffset,
      elemSize,
      byteSize: elemSize,
    });
  }

  tensors.sort((a, b) => a.offset - b.offset);

  return { headerSize, dataOffset, metadata, tensors };
}

export function groupTensorsByLayer(parsed) {
  const layers = new Map();

  for (const tensor of parsed.tensors) {
    const match = tensor.name.match(/layers?\.(\d+)\./);
    if (match) {
      const layerIdx = parseInt(match[1], 10);
      if (!layers.has(layerIdx)) {
        layers.set(layerIdx, []);
      }
      layers.get(layerIdx).push(tensor);
    }
  }

  return layers;
}

export function calculateTotalSize(parsed) {
  return parsed.tensors.reduce((sum, tensor) => sum + tensor.size, 0);
}
