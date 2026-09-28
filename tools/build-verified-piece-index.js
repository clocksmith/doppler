import { readFile, open, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import { createStreamingHasher } from '../src/storage/shard-manager.js';

const [directory, destination, bytesArgument, manifestPath] = process.argv.slice(2);
const maxPieceBytes = Number(bytesArgument);
if (!directory || !destination || !Number.isSafeInteger(maxPieceBytes) || maxPieceBytes < 4096) {
  throw new Error('Usage: node tools/build-verified-piece-index.js MODEL_DIRECTORY OUTPUT_JSON MAX_PIECE_BYTES');
}
const sha = bytes => 'sha256:' + createHash('sha256').update(bytes).digest('hex');
const manifestBytes = await readFile(manifestPath || resolve(directory, 'manifest.json'));
const manifest = JSON.parse(manifestBytes);
const files = [];
for (const shard of manifest.shards) {
  const boundaries = new Set([0, shard.size]);
  for (const tensor of Object.values(manifest.tensors)) {
    for (const span of tensor.spans || [{ shardIndex: tensor.shard, offset: tensor.offset, size: tensor.size }]) {
      if (span.shardIndex === shard.index) { boundaries.add(span.offset); boundaries.add(span.offset + span.size); }
    }
  }
  files.push({ path: shard.filename, size: shard.size, boundaries: [...boundaries].sort((a,b) => a-b), shard });
}
// These auxiliary inputs are explicit source files, not weights inferred by name.
for (const path of [manifest.tokenizer.file]) {
  const bytes = await readFile(resolve(directory, path));
  files.push({ path, size: bytes.length, boundaries: [0, bytes.length] });
}
const index = { schema: 'doppler.verified-pieces/v1', manifestIdentity: sha(manifestBytes), maxPieceBytes, files: [] };
for (const file of files) {
  const handle = await open(resolve(directory, file.path), 'r');
  const hasher = file.shard ? await createStreamingHasher(manifest.hashAlgorithm) : null;
  const pieces = [];
  try {
    if ((await handle.stat()).size !== file.size) throw new Error('Source size mismatch: ' + file.path);
    for (let i = 1; i < file.boundaries.length; i++) {
      for (let offset = file.boundaries[i-1]; offset < file.boundaries[i]; offset += maxPieceBytes) {
        const size = Math.min(maxPieceBytes, file.boundaries[i] - offset), bytes = Buffer.alloc(size);
        const read = await handle.read(bytes, 0, size, offset);
        if (read.bytesRead !== size) throw new Error('Short source read');
        hasher?.update(bytes); pieces.push({ offset, size, identity: sha(bytes) });
      }
    }
    if (hasher && Buffer.from(await hasher.finalize()).toString('hex') !== file.shard.hash) throw new Error('Source shard hash mismatch: ' + file.path);
  } finally { await handle.close(); }
  index.files.push({ path: file.path, size: file.size, pieces });
}
const output = JSON.stringify(index, null, 2) + '\n';
await writeFile(destination, output);
console.log(JSON.stringify({ indexIdentity: sha(output), manifestIdentity: index.manifestIdentity,
  files: index.files.length, pieces: index.files.reduce((count,file) => count + file.pieces.length, 0) }));
