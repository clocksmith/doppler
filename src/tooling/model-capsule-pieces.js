import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { createVerifiedPieceReader } from '../storage/verified-piece-reader.js';

/** Verify the original index and each declared range before signing its bytes.
 * @type {import('./model-capsule-pieces.js').loadCapsulePieceIndex} */
export async function loadCapsulePieceIndex(indexPath, manifestRaw, modelDirectory) {
  if (indexPath == null) return null;
  const sourcePath = path.resolve(indexPath), bytes = await fs.readFile(sourcePath);
  const hash = 'sha256:' + createHash('sha256').update(bytes).digest('hex');
  const index = JSON.parse(bytes.toString('utf8'));
  /** @type {import('../formats/rdrr/index.js').RDRRManifest} */
  const manifest = JSON.parse(manifestRaw);
  if (manifest.hashAlgorithm !== 'sha256') throw new Error('Signed pieces require SHA-256 manifest shards.');
  const allowed = /** @type {Map<string, number | null>} */ (new Map(manifest.shards.map(shard => [shard.filename, shard.size])));
  if (manifest.tokenizer?.file) allowed.set(manifest.tokenizer.file, null);
  const reader = await createVerifiedPieceReader({ manifestBytes: new TextEncoder().encode(manifestRaw), indexBytes: bytes,
    indexIdentity: hash, async acquire(piece) {
      const handle = await fs.open(path.resolve(modelDirectory, piece.path), 'r');
      try {
        const result = new Uint8Array(piece.size);
        if ((await handle.read(result, 0, piece.size, piece.offset)).bytesRead !== piece.size) throw new Error('Short piece source read.');
        return result;
      } finally { await handle.close(); }
    } });
  try {
    for (const file of index.files) {
      if (!allowed.has(file.path) || (allowed.get(file.path) !== null && allowed.get(file.path) !== file.size)) {
        throw new Error('Piece index names an undeclared model artifact.');
      }
      if ((await fs.stat(path.resolve(modelDirectory, file.path))).size !== file.size) throw new Error('Piece source size mismatch.');
      for (const piece of file.pieces) await reader.readRange(file.path, piece.offset, piece.size);
    }
  } finally { await reader.close(); }
  return { sourcePath, hash, sizeBytes: bytes.length, path: 'artifacts/model/' + path.basename(sourcePath) };
}
