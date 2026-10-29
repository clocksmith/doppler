import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { hash as blake3Hash } from '../../src/storage/blake3.js';
import { dequantizeQ4KMRowWise, float16ToFloat32 } from '../../src/converter/quantizer.js';

const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const config = JSON.parse(await fs.readFile(process.argv[2], 'utf8'));
if (config.schema !== 'doppler.equal-weight-export.v1') throw new Error('Unknown export contract');
const source = path.resolve(config.sourceModelDir), output = path.resolve(config.outputDir);
const manifestBytes = await fs.readFile(path.join(source, 'manifest.json'));
if (sha(manifestBytes) !== config.sourceManifestSha256) throw new Error('Source manifest hash mismatch');
const manifest = JSON.parse(manifestBytes);
if (output === source || output.startsWith(source + path.sep)) throw new Error('Output must not modify the source model tree');
const verifiedShards = [];
for (const shard of manifest.shards) {
  const bytes = await fs.readFile(path.join(source, shard.filename));
  const digest = manifest.hashAlgorithm === 'blake3'
    ? Buffer.from(await blake3Hash(bytes)).toString('hex') : manifest.hashAlgorithm === 'sha256' ? sha(bytes) : null;
  if (bytes.length !== shard.size || !digest || digest !== shard.hash) throw new Error(`Source shard integrity failed: ${shard.filename}`);
  verifiedShards.push({ filename: shard.filename, bytes: bytes.length, manifestHash: digest, sha256: sha(bytes) });
  console.log(JSON.stringify({ verifiedShard: shard.index, total: manifest.shards.length }));
}
await fs.mkdir(output, { recursive: false });
const selected = Object.entries(manifest.tensors).filter(([name]) => name.startsWith(config.tensorPrefix));
if (selected.length === 0) throw new Error('No tensors selected');
let offset = 0;
const header = { __metadata__: { source_manifest_sha256: sha(manifestBytes), recipe: 'canonical-f32-no-requantization-v1' } };
for (const [name, tensor] of selected) {
  const size = tensor.shape.reduce((a, b) => a * b, 1) * 4;
  if (!Number.isSafeInteger(size)) throw new Error(`Unsafe tensor extent: ${name}`);
  header[name] = { dtype: 'F32', shape: tensor.shape, data_offsets: [offset, offset + size] };
  offset += size;
}
let encoded = Buffer.from(JSON.stringify(header));
encoded = Buffer.concat([encoded, Buffer.alloc((8 - encoded.length % 8) % 8, 32)]);
const length = Buffer.alloc(8); length.writeBigUInt64LE(BigInt(encoded.length));
const file = await fs.open(path.join(output, 'model.safetensors'), 'wx');
const receipt = { schema: 'doppler.equal-weight-canonical.v1', sourceManifestSha256: sha(manifestBytes),
  scope: 'Text transformer learned tensors only; no vision or speculative model; F32 reference reconstruction, no new quantization.',
  tensorPrefix: config.tensorPrefix, sourceShards: verifiedShards, tensors: [], completed: false };
const inputShards = new Map();
async function readTensor(tensor) {
  const spans = tensor.spans ?? [{ shardIndex: tensor.shard, offset: tensor.offset, size: tensor.size }];
  const pieces = [];
  for (const span of spans) {
    const shard = manifest.shards.find(s => s.index === span.shardIndex);
    if (!shard) throw new Error('Unknown source shard');
    let handle = inputShards.get(shard.index);
    if (!handle) { handle = await fs.open(path.join(source, shard.filename)); inputShards.set(shard.index, handle); }
    const bytes = Buffer.alloc(span.size);
    let n = 0;
    while (n < bytes.length) { const read = await handle.read(bytes, n, bytes.length - n, span.offset + n);
      if (!read.bytesRead) throw new Error('Truncated source tensor'); n += read.bytesRead; }
    pieces.push(bytes);
  }
  return pieces.length === 1 ? pieces[0] : Buffer.concat(pieces);
}
try {
  await fs.writeFile(path.join(output, 'canonical.json'), JSON.stringify(receipt, null, 2) + '\n');
  await file.write(length); await file.write(encoded);
  for (const [name, tensor] of selected) {
    const bytes = await readTensor(tensor);
    let values;
    if (tensor.dtype === 'F32') values = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
    else if (tensor.dtype === 'F16') {
      const packed = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
      values = new Float32Array(packed.length);
      for (let i = 0; i < packed.length; i++) values[i] = float16ToFloat32(packed[i]);
    } else if (tensor.dtype === 'Q4_K_M') {
      if (tensor.shape.length !== 2) throw new Error(`Unsupported quantized shape: ${name}`);
      values = dequantizeQ4KMRowWise(new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength), tensor.shape);
    } else throw new Error(`Unsupported source dtype ${tensor.dtype}: ${name}`);
    const expected = tensor.shape.reduce((a, b) => a * b, 1);
    if (values.length !== expected) throw new Error(`Decoded extent mismatch: ${name}`);
    for (const value of values) if (!Number.isFinite(value)) throw new Error(`Nonfinite canonical tensor: ${name}`);
    const data = Buffer.from(values.buffer, values.byteOffset, values.byteLength);
    let wrote = 0;
    while (wrote < data.length) { const result = await file.write(data, wrote, data.length - wrote);
      if (!result.bytesWritten) throw new Error('Short canonical write'); wrote += result.bytesWritten; }
    receipt.tensors.push({ name, shape: tensor.shape, dtype: 'F32', sourceDtype: tensor.dtype,
      sourceBytesSha256: sha(bytes), canonicalBytesSha256: sha(data), bytes: data.length });
    console.log(JSON.stringify({ tensor: name, bytes: data.length, completed: receipt.tensors.length, total: selected.length }));
  }
  receipt.completed = true; receipt.totalTensorBytes = offset;
} finally {
  await file.close();
  for (const handle of inputShards.values()) await handle.close();
  await fs.writeFile(path.join(output, 'canonical.json'), JSON.stringify(receipt, null, 2) + '\n');
}
await fs.copyFile(path.join(source, 'tokenizer.json'), path.join(output, 'tokenizer.json'));
