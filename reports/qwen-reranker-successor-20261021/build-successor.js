import fs from 'node:fs/promises';
import { createHash } from 'node:crypto';
const [root, dir, pyHashesPath] = process.argv.slice(2);
const { computeHash } = await import(`${root}/src/storage/shards/integrity.js`);
const { hashArtifactValue } = await import(`${root}/src/converter/artifact-identity.js`);
const sha = async (s) => createHash('sha256').update(s).digest('hex');
const py = JSON.parse(await fs.readFile(pyHashesPath, 'utf8'));
const pred = await fs.readFile(`${dir}/predecessor-manifest.json`);
const m = JSON.parse(pred);
const pyByName = new Map(py.files.map((f) => [f.name, f]));
const rows = [];
for (const shard of m.shards) {
  const bytes = new Uint8Array(await fs.readFile(`${dir}/${shard.filename}`));
  const doppler = await computeHash(bytes, 'blake3');
  const python = pyByName.get(shard.filename).blake3Python;
  if (doppler !== python) throw new Error(`BLAKE3 implementations disagree for ${shard.filename}`);
  rows.push({ filename: shard.filename, legacy: shard.hash, standard: doppler });
  shard.hash = doppler;
  shard.blake3 = doppler;
}
const a = m.artifactIdentity;
a.shardSetHash = await hashArtifactValue(sha, { hashAlgorithm: m.hashAlgorithm,
  shards: m.shards.map(({ index, filename, size, hash, offset }) => ({ index, filename, size, hash, offset })) }, 'shardSet');
a.weightPackHash = await hashArtifactValue(sha, { sourceCheckpointId: a.sourceCheckpointId, sourceFormat: a.sourceFormat,
  modelType: m.modelType, modalitySet: a.modalitySet, quantizationInfo: m.quantizationInfo,
  materializationProfile: a.materializationProfile, shardSetHash: a.shardSetHash,
  sharding: { shardSizeBytes: 67108864 }, output: { textOnly: true } }, 'weightCapsule');
m.metadata = { ...m.metadata, integrityMigration: {
  schema: 'doppler.standard-blake3-successor/v1',
  predecessorManifestIdentity: `sha256:${createHash('sha256').update(pred).digest('hex')}`,
  trustedInventory: 'reports/choice-scoring/model-derivation.json (SHA-256 of all shards and tokenizer)',
  standardBlake3Implementations: ['doppler src/storage/blake3.js', 'python blake3 1.0.11'],
  weightAndTokenizerBytes: 'unchanged',
  qualification: 'required',
} };
await fs.writeFile(`${dir}/manifest.json`, `${JSON.stringify(m, null, 2)}\n`);
await fs.writeFile(`${dir}/blake3-migration.json`, `${JSON.stringify({ shards: rows, shardSetHash: a.shardSetHash, weightPackHash: a.weightPackHash }, null, 2)}\n`);
console.log(`agree on ${rows.length} shards; shardSetHash ${a.shardSetHash}; weightPackHash ${a.weightPackHash}`);
