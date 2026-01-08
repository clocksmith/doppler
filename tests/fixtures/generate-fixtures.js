import { writeFileSync, mkdirSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

const GGUFValueType = {
  UINT8: 0,
  INT8: 1,
  UINT16: 2,
  INT16: 3,
  UINT32: 4,
  INT32: 5,
  FLOAT32: 6,
  BOOL: 7,
  STRING: 8,
  ARRAY: 9,
  UINT64: 10,
  INT64: 11,
  FLOAT64: 12,
};

const GGMLType = {
  F32: 0,
  F16: 1,
  Q4_K: 12,
};

function generateGGUFFixture() {
  const encoder = new TextEncoder();

  const metadataKVs = [
    { key: 'general.architecture', type: GGUFValueType.STRING, value: 'llama' },
    { key: 'general.name', type: GGUFValueType.STRING, value: 'test-fixture' },
    { key: 'llama.context_length', type: GGUFValueType.UINT32, value: 512 },
    { key: 'llama.embedding_length', type: GGUFValueType.UINT32, value: 64 },
    { key: 'llama.block_count', type: GGUFValueType.UINT32, value: 2 },
    { key: 'llama.attention.head_count', type: GGUFValueType.UINT32, value: 4 },
    { key: 'llama.attention.head_count_kv', type: GGUFValueType.UINT32, value: 4 },
    { key: 'tokenizer.ggml.model', type: GGUFValueType.STRING, value: 'llama' },
    { key: 'tokenizer.ggml.bos_token_id', type: GGUFValueType.UINT32, value: 1 },
    { key: 'tokenizer.ggml.eos_token_id', type: GGUFValueType.UINT32, value: 2 },
  ];

  const tensors = [
    { name: 'token_embd.weight', shape: [100, 64], dtype: GGMLType.F16 },
    { name: 'blk.0.attn_q.weight', shape: [64, 64], dtype: GGMLType.F16 },
    { name: 'blk.0.attn_k.weight', shape: [64, 64], dtype: GGMLType.F16 },
    { name: 'blk.0.attn_v.weight', shape: [64, 64], dtype: GGMLType.F16 },
    { name: 'blk.0.ffn_gate.weight', shape: [128, 64], dtype: GGMLType.F16 },
    { name: 'blk.0.ffn_up.weight', shape: [128, 64], dtype: GGMLType.F16 },
    { name: 'blk.0.ffn_down.weight', shape: [64, 128], dtype: GGMLType.F16 },
    { name: 'blk.1.attn_q.weight', shape: [64, 64], dtype: GGMLType.F16 },
    { name: 'output.weight', shape: [100, 64], dtype: GGMLType.F16 },
  ];

  let headerSize = 0;
  headerSize += 4 + 4 + 8 + 8;

  for (const kv of metadataKVs) {
    headerSize += 8 + encoder.encode(kv.key).length;
    headerSize += 4;
    if (kv.type === GGUFValueType.STRING) {
      headerSize += 8 + encoder.encode(kv.value).length;
    } else if (kv.type === GGUFValueType.UINT32) {
      headerSize += 4;
    }
  }

  for (const tensor of tensors) {
    headerSize += 8 + encoder.encode(tensor.name).length;
    headerSize += 4;
    headerSize += tensor.shape.length * 8;
    headerSize += 4;
    headerSize += 8;
  }

  const padding = (32 - (headerSize % 32)) % 32;
  headerSize += padding;

  let tensorDataSize = 0;
  for (const tensor of tensors) {
    const numElements = tensor.shape.reduce((a, b) => a * b, 1);
    if (tensor.dtype === GGMLType.F16) {
      tensorDataSize += numElements * 2;
    } else if (tensor.dtype === GGMLType.F32) {
      tensorDataSize += numElements * 4;
    }
  }

  const totalSize = headerSize + tensorDataSize;
  const buffer = Buffer.alloc(totalSize);
  let offset = 0;

  function writeUint32(val) {
    buffer.writeUInt32LE(val, offset);
    offset += 4;
  }

  function writeUint64(val) {
    buffer.writeUInt32LE(val & 0xffffffff, offset);
    buffer.writeUInt32LE(Math.floor(val / 0x100000000), offset + 4);
    offset += 8;
  }

  function writeString(str) {
    const encoded = encoder.encode(str);
    writeUint64(encoded.length);
    encoded.forEach((b, i) => buffer[offset + i] = b);
    offset += encoded.length;
  }

  writeUint32(0x46554747);
  writeUint32(3);
  writeUint64(tensors.length);
  writeUint64(metadataKVs.length);

  for (const kv of metadataKVs) {
    writeString(kv.key);
    writeUint32(kv.type);
    if (kv.type === GGUFValueType.STRING) {
      writeString(kv.value);
    } else if (kv.type === GGUFValueType.UINT32) {
      writeUint32(kv.value);
    }
  }

  let tensorOffset = 0;
  for (const tensor of tensors) {
    writeString(tensor.name);
    writeUint32(tensor.shape.length);
    for (const dim of tensor.shape) {
      writeUint64(dim);
    }
    writeUint32(tensor.dtype);
    writeUint64(tensorOffset);

    const numElements = tensor.shape.reduce((a, b) => a * b, 1);
    if (tensor.dtype === GGMLType.F16) {
      tensorOffset += numElements * 2;
    } else if (tensor.dtype === GGMLType.F32) {
      tensorOffset += numElements * 4;
    }
  }

  const remainder = offset % 32;
  if (remainder !== 0) {
    offset += 32 - remainder;
  }

  for (let i = offset; i < totalSize; i++) {
    buffer[i] = (i * 17) % 256;
  }

  return buffer;
}

function generateSafetensorsFixture() {
  const tensors = {
    'model.embed_tokens.weight': {
      dtype: 'F16',
      shape: [100, 64],
      data_offsets: [0, 12800],
    },
    'model.layers.0.self_attn.q_proj.weight': {
      dtype: 'F32',
      shape: [64, 64],
      data_offsets: [12800, 29184],
    },
    'model.layers.0.self_attn.k_proj.weight': {
      dtype: 'F32',
      shape: [64, 64],
      data_offsets: [29184, 45568],
    },
    'model.layers.0.mlp.gate_proj.weight': {
      dtype: 'F32',
      shape: [128, 64],
      data_offsets: [45568, 78336],
    },
    'model.layers.1.self_attn.q_proj.weight': {
      dtype: 'F32',
      shape: [64, 64],
      data_offsets: [78336, 94720],
    },
    'lm_head.weight': {
      dtype: 'F16',
      shape: [100, 64],
      data_offsets: [94720, 107520],
    },
  };

  const header = {
    '__metadata__': {
      'format': 'pt',
      'model_type': 'test',
      'version': '1.0',
    },
    ...tensors,
  };

  const headerJson = JSON.stringify(header);
  const headerBytes = Buffer.from(headerJson, 'utf8');
  const headerSize = headerBytes.length;

  const dataSize = 107520;
  const totalSize = 8 + headerSize + dataSize;

  const buffer = Buffer.alloc(totalSize);

  buffer.writeUInt32LE(headerSize & 0xffffffff, 0);
  buffer.writeUInt32LE(Math.floor(headerSize / 0x100000000), 4);

  headerBytes.copy(buffer, 8);

  for (let i = 8 + headerSize; i < totalSize; i++) {
    buffer[i] = ((i - 8 - headerSize) * 23) % 256;
  }

  return buffer;
}

const ggufBuffer = generateGGUFFixture();
writeFileSync(join(__dirname, 'sample.gguf'), ggufBuffer);
console.log(`Generated sample.gguf: ${ggufBuffer.length} bytes`);

const safetensorsBuffer = generateSafetensorsFixture();
writeFileSync(join(__dirname, 'sample.safetensors'), safetensorsBuffer);
console.log(`Generated sample.safetensors: ${safetensorsBuffer.length} bytes`);
