// CPU-only selective reader control. No device, resident, or model execution.
import assert from 'node:assert/strict';
import { readFile, open } from 'node:fs/promises';
import { resolve, sep } from 'node:path';
import { createVerifiedCapsuleArtifactStore } from '../../src/client/runtime/verified-capsule-artifact-store.js';
import { verifyCapsuleV3Signature } from '../../src/config/capsule-v3.js';
const root = resolve(process.argv[2]);
const capsule = JSON.parse(await readFile(resolve(root, 'capsule-v3.json'), 'utf8'));
const signers = JSON.parse(await readFile(resolve(root, 'trusted-signers.json'), 'utf8'));
const loading = JSON.parse(await readFile(new URL('./loading.json', import.meta.url), 'utf8'));
await verifyCapsuleV3Signature(capsule, signers);
const resolveArtifact = artifact => {
  const file = resolve(root, artifact.path);
  assert(file.startsWith(root + sep));
  return file;
};
const reads = [], ranges = [];
const store = createVerifiedCapsuleArtifactStore(capsule, {
  async readArtifact(artifact) {
    assert.notEqual(artifact.role, 'weight-shard');
    reads.push(artifact.artifactId);
    return new Uint8Array(await readFile(resolveArtifact(artifact)));
  },
  async readArtifactRange(artifact, offset, length) {
    const handle = await open(resolveArtifact(artifact), 'r');
    try {
      const result = new Uint8Array(length);
      const { bytesRead } = await handle.read(result, 0, length, offset);
      assert.equal(bytesRead, length);
      ranges.push({ artifactId: artifact.artifactId, offset, length });
      return result;
    } finally { await handle.close(); }
  },
}, loading);
try {
  await store.enableResidentPieces();
  const weight = capsule.artifacts.find(row => row.role === 'weight-shard');
  const result = await store.readArtifactRange(weight, 0, 1);
  assert.equal(result.length, 1);
  console.log(JSON.stringify({ pass: true, semanticRoot: capsule.semanticRoot, reads, ranges,
    pieceVerification: store.getPieceVerification(), physicalOpening: false, publicPromotion: false }, null, 2));
} finally { await store.close(); }
