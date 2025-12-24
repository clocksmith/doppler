#!/usr/bin/env node
/**
 * Debug script to check tensor values at specific indices.
 * Compares original SafeTensors data vs converted RDRR data.
 */

import { readFile, open } from 'fs/promises';
import { join } from 'path';

interface TensorInfo {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}

interface RDRRTensorInfo {
  shard: number;
  offset: number;
  size: number;
  shape: number[];
  dtype: string;
}

interface TensorResult {
  bf16: Uint16Array;
  dtype: string;
}

function bf16ToF32(bf16: number): number {
  const buffer = new ArrayBuffer(4);
  const u32 = new Uint32Array(buffer);
  const f32 = new Float32Array(buffer);
  u32[0] = bf16 << 16;
  return f32[0];
}

async function readSafetensorsTensor(modelDir: string, tensorName: string, indices: number[]): Promise<TensorResult | null> {
  const indexPath = join(modelDir, 'model.safetensors.index.json');
  const indexJson = JSON.parse(await readFile(indexPath, 'utf8'));

  const shardFile = indexJson.weight_map[tensorName];
  if (!shardFile) {
    console.log(`Tensor ${tensorName} not found in index`);
    return null;
  }

  const shardPath = join(modelDir, shardFile);
  console.log(`Tensor ${tensorName} is in shard: ${shardFile}`);

  const file = await open(shardPath, 'r');
  const headerSizeBuf = Buffer.alloc(8);
  await file.read(headerSizeBuf, 0, 8, 0);

  const headerSize = headerSizeBuf.readUInt32LE(0) + headerSizeBuf.readUInt32LE(4) * 0x100000000;
  console.log(`Header size: ${headerSize} bytes`);

  const headerBuf = Buffer.alloc(headerSize);
  await file.read(headerBuf, 0, headerSize, 8);
  const header = JSON.parse(headerBuf.toString('utf8')) as Record<string, TensorInfo>;

  const tensorInfo = header[tensorName];
  if (!tensorInfo) {
    console.log(`Tensor ${tensorName} not in header`);
    await file.close();
    return null;
  }

  const { dtype, shape, data_offsets } = tensorInfo;
  const [startOffset, endOffset] = data_offsets;
  const dataOffset = 8 + headerSize + startOffset;
  const tensorSize = endOffset - startOffset;

  console.log(`Tensor info: dtype=${dtype}, shape=${JSON.stringify(shape)}, size=${tensorSize} bytes`);
  console.log(`Data at file offset: ${dataOffset}`);

  const tensorBuf = Buffer.alloc(tensorSize);
  await file.read(tensorBuf, 0, tensorSize, dataOffset);
  await file.close();

  const bf16 = new Uint16Array(tensorBuf.buffer, tensorBuf.byteOffset, tensorBuf.length / 2);

  console.log(`\nValues at specified indices (BF16 -> F32):`);
  for (const idx of indices) {
    if (idx < bf16.length) {
      const bf16Val = bf16[idx];
      const f32Val = bf16ToF32(bf16Val);
      console.log(`  [${idx}]: bf16=0x${bf16Val.toString(16).padStart(4, '0')} -> f32=${f32Val.toFixed(4)}`);
    }
  }

  const startIdx = Math.max(0, indices[0] - 3);
  const endIdx = Math.min(bf16.length, indices[indices.length - 1] + 4);
  console.log(`\nFull range [${startIdx}-${endIdx}]:`);
  for (let i = startIdx; i < endIdx; i++) {
    const bf16Val = bf16[i];
    const f32Val = bf16ToF32(bf16Val);
    const marker = indices.includes(i) ? ' <-- OUTLIER INDEX' : '';
    console.log(`  [${i}]: bf16=0x${bf16Val.toString(16).padStart(4, '0')} -> f32=${f32Val.toFixed(4)}${marker}`);
  }

  return { bf16, dtype };
}

async function readRDRRTensor(rdrrDir: string, tensorName: string, indices: number[]): Promise<TensorResult | null> {
  const manifestPath = join(rdrrDir, 'manifest.json');
  const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));

  const tensorInfo = manifest.tensors[tensorName] as RDRRTensorInfo | undefined;
  if (!tensorInfo) {
    console.log(`Tensor ${tensorName} not found in RDRR manifest`);
    return null;
  }

  const { shard, offset, size, shape, dtype } = tensorInfo;
  console.log(`\nRDRR tensor info: shard=${shard}, offset=${offset}, size=${size}, dtype=${dtype}, shape=${JSON.stringify(shape)}`);

  const shardFile = `shard_${String(shard).padStart(5, '0')}.bin`;
  const shardPath = join(rdrrDir, shardFile);

  const file = await open(shardPath, 'r');
  const tensorBuf = Buffer.alloc(size);
  await file.read(tensorBuf, 0, size, offset);
  await file.close();

  const bf16 = new Uint16Array(tensorBuf.buffer, tensorBuf.byteOffset, tensorBuf.length / 2);

  console.log(`\nRDRR values at specified indices (BF16 -> F32):`);
  for (const idx of indices) {
    if (idx < bf16.length) {
      const bf16Val = bf16[idx];
      const f32Val = bf16ToF32(bf16Val);
      console.log(`  [${idx}]: bf16=0x${bf16Val.toString(16).padStart(4, '0')} -> f32=${f32Val.toFixed(4)}`);
    }
  }

  const startIdx = Math.max(0, indices[0] - 3);
  const endIdx = Math.min(bf16.length, indices[indices.length - 1] + 4);
  console.log(`\nRDRR full range [${startIdx}-${endIdx}]:`);
  for (let i = startIdx; i < endIdx; i++) {
    const bf16Val = bf16[i];
    const f32Val = bf16ToF32(bf16Val);
    const marker = indices.includes(i) ? ' <-- OUTLIER INDEX' : '';
    console.log(`  [${i}]: bf16=0x${bf16Val.toString(16).padStart(4, '0')} -> f32=${f32Val.toFixed(4)}${marker}`);
  }

  return { bf16, dtype };
}

async function checkTensorStats(rdrrDir: string, tensorName: string): Promise<void> {
  const manifestPath = join(rdrrDir, 'manifest.json');
  const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));

  const tensorInfo = manifest.tensors[tensorName] as RDRRTensorInfo | undefined;
  if (!tensorInfo) {
    console.log(`Tensor ${tensorName} not found`);
    return;
  }

  const { shard, offset, size } = tensorInfo;
  const shardFile = `shard_${String(shard).padStart(5, '0')}.bin`;
  const shardPath = join(rdrrDir, shardFile);

  const file = await open(shardPath, 'r');
  const tensorBuf = Buffer.alloc(size);
  await file.read(tensorBuf, 0, size, offset);
  await file.close();

  const bf16 = new Uint16Array(tensorBuf.buffer, tensorBuf.byteOffset, tensorBuf.length / 2);

  let maxAbs = 0, maxIdx = 0, sumSq = 0;
  for (let i = 0; i < bf16.length; i++) {
    const f32 = bf16ToF32(bf16[i]);
    if (Math.abs(f32) > maxAbs) {
      maxAbs = Math.abs(f32);
      maxIdx = i;
    }
    sumSq += f32 * f32;
  }
  const rms = Math.sqrt(sumSq / bf16.length);

  const shortName = tensorName.replace('language_model.model.layers.0.', '');
  console.log(`${shortName}: maxAbs=${maxAbs.toFixed(4)} at idx=${maxIdx}, rms=${rms.toFixed(4)}`);

  if (maxAbs > 10) {
    const startIdx = Math.max(0, maxIdx - 2);
    const endIdx = Math.min(bf16.length, maxIdx + 3);
    const vals: string[] = [];
    for (let i = startIdx; i < endIdx; i++) {
      vals.push(bf16ToF32(bf16[i]).toFixed(4));
    }
    console.log(`  values around max: [${vals.join(', ')}]`);
  }
}

async function main(): Promise<void> {
  const safetensorsDir = process.argv[2] || '/Users/xyz/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767';
  const rdrrDir = process.argv[3] || './gemma-3-4b-rdrr';

  console.log('=== All Layer 0 Norm Tensor Stats ===\n');
  const normTensors = [
    'language_model.model.layers.0.input_layernorm.weight',
    'language_model.model.layers.0.post_attention_layernorm.weight',
    'language_model.model.layers.0.pre_feedforward_layernorm.weight',
    'language_model.model.layers.0.post_feedforward_layernorm.weight',
  ];

  for (const tensor of normTensors) {
    await checkTensorStats(rdrrDir, tensor);
  }

  console.log('\n=== Layer 1 Norm Tensor Stats ===\n');
  const normTensors1 = [
    'language_model.model.layers.1.input_layernorm.weight',
    'language_model.model.layers.1.post_attention_layernorm.weight',
    'language_model.model.layers.1.pre_feedforward_layernorm.weight',
    'language_model.model.layers.1.post_feedforward_layernorm.weight',
  ];

  for (const tensor of normTensors1) {
    await checkTensorStats(rdrrDir, tensor);
  }
}

main().catch(console.error);

export { bf16ToF32, readSafetensorsTensor, readRDRRTensor, checkTensorStats };
