import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { mkdir, open, readFile, realpath, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildArtifactIdentity } from '../src/converter/artifact-identity.js';
import { buildIntegrityExtensions } from '../src/formats/rdrr/integrity.js';
import { parseManifest } from '../src/formats/rdrr/index.js';

const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');

// Internal migration, never an inference fallback. The caller must independently
// authenticate and pin an immutable Git-LFS inventory before invoking this tool.
export async function migrateRdrrShardHashes(options) {
  const { sourceDir, destinationDir, inventoryPath, inventoryHash, inventoryPrefix } = options;
  const source = await realpath(sourceDir);
  const destination = path.join(await realpath(path.dirname(path.resolve(destinationDir))), path.basename(destinationDir));
  if (destination === source || destination.startsWith(`${source}${path.sep}`)) {
    throw new Error('Migration requires a separate destination outside the source artifact.');
  }
  const inventoryBytes = await readFile(inventoryPath);
  if (!/^[0-9a-f]{64}$/.test(inventoryHash) || sha256(inventoryBytes) !== inventoryHash) {
    throw new Error('Independent inventory SHA-256 mismatch.');
  }
  const inventory = JSON.parse(inventoryBytes);
  if (!Array.isArray(inventory)) throw new Error('Expected pinned Git-LFS tree inventory.');
  const entries = new Map(inventory.map((entry) => [entry.path, entry]));
  const originalBytes = await readFile(path.join(source, 'manifest.json'));
  const original = JSON.parse(originalBytes);
  if (original.hashAlgorithm !== 'blake3' || original.metadata?.sourceRuntime) {
    throw new Error('Expected a legacy BLAKE3 RDRR shard artifact.');
  }
  for (const key of ['signature', 'signatures', 'pieceIndex', 'release', 'weightsRef', 'blake3Full', 'tensorsFile']) {
    if (original[key] != null) {
      throw new Error(`Migration must rebuild bound ${key} metadata explicitly.`);
    }
  }
  if (Object.keys(original.groups ?? {}).length > 0) {
    throw new Error('Migration must rebuild bound group hashes explicitly.');
  }
  const files = [];
  async function verify(filename) {
    if (path.basename(filename) !== filename) throw new Error('Artifact filename must be a basename.');
    const entry = entries.get(`${inventoryPrefix}/${filename}`);
    if (!entry || !Number.isSafeInteger(entry.size)) throw new Error(`Missing inventory entry: ${filename}`);
    const hash = createHash('sha256');
    const gitHash = createHash('sha1').update(`blob ${entry.size}\0`);
    let size = 0;
    for await (const chunk of createReadStream(path.join(source, filename))) {
      size += chunk.byteLength;
      hash.update(chunk);
      gitHash.update(chunk);
    }
    const digest = hash.digest('hex');
    if (size !== entry.size || (entry.lfs?.oid
      ? digest !== entry.lfs.oid
      : gitHash.digest('hex') !== entry.oid)) {
      throw new Error(`Independent inventory byte mismatch: ${filename}`);
    }
    const record = { filename, size, sha256: digest, inventoryLfsSha256: entry.lfs?.oid ?? null };
    files.push(record);
    return record;
  }
  await verify('manifest.json');
  const manifest = structuredClone(original);
  manifest.hashAlgorithm = 'sha256';
  for (const shard of manifest.shards) {
    const file = await verify(shard.filename);
    if (file.size !== shard.size || !file.inventoryLfsSha256) {
      throw new Error(`Shard requires independent SHA-256 inventory: ${shard.filename}`);
    }
    delete shard.blake3;
    shard.hash = file.sha256;
    shard.sha256 = file.sha256;
  }
  if (manifest.tokenizer?.file) await verify(manifest.tokenizer.file);
  const identity = { ...original.artifactIdentity };
  for (const key of ['shardSetHash', 'weightPackHash', 'weightPackId', 'weightCapsuleHash',
    'weightCapsuleId', 'manifestVariantId']) delete identity[key];
  manifest.artifactIdentity = await buildArtifactIdentity({
    explicitArtifactIdentity: identity,
    hashString: async (text) => sha256(text),
    modelId: manifest.modelId,
    modelType: manifest.modelType,
    hashAlgorithm: manifest.hashAlgorithm,
    shards: manifest.shards,
    quantizationInfo: manifest.quantizationInfo,
    inference: manifest.inference,
    manifestConfig: manifest.config,
  });
  const handles = new Map();
  try {
    if (original.integrityExtensions) {
      const rebuilt = await buildIntegrityExtensions(manifest, {
        readShardRange: async (index, offset, length) => {
          if (!handles.has(index)) {
            handles.set(index, await open(path.join(source, manifest.shards[index].filename), 'r'));
          }
          const bytes = Buffer.alloc(length);
          const { bytesRead } = await handles.get(index).read(bytes, 0, length, offset);
          if (bytesRead !== length) throw new Error('Truncated tensor while rebuilding integrity metadata.');
          return bytes;
        },
      });
      manifest.integrityExtensions = rebuilt.integrityExtensions;
    }
  } finally {
    await Promise.all([...handles.values()].map((handle) => handle.close()));
  }
  const text = `${JSON.stringify(manifest, null, 2)}\n`;
  parseManifest(text);
  const receipt = {
    contractVersion: 1,
    migration: 'legacy-shard-digests-to-standard-sha256',
    originalManifestHash: sha256(originalBytes),
    manifestHash: sha256(text),
    originalArtifactIdentity: original.artifactIdentity,
    artifactIdentity: manifest.artifactIdentity,
    independentInventoryHash: inventoryHash,
    inventoryPrefix,
    files,
    unchangedExecution: true,
    qualificationTransferred: false,
    integrityExtensionsRebuilt: original.integrityExtensions != null,
  };
  await mkdir(destination, { recursive: false });
  await writeFile(path.join(destination, 'manifest.json'), text, { flag: 'wx' });
  await writeFile(path.join(destination, 'migration-receipt.json'), `${JSON.stringify(receipt, null, 2)}\n`, { flag: 'wx' });
  return { manifest, receipt };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const optionsPath = process.argv[2];
  if (!optionsPath) throw new Error('Usage: node tools/migrate-rdrr-shard-hashes.js <options.json>');
  const { receipt } = await migrateRdrrShardHashes(JSON.parse(await readFile(optionsPath, 'utf8')));
  console.log(JSON.stringify(receipt, null, 2));
}
