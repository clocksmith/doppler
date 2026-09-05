import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { exportProgramBundle, writeProgramBundle, verifyClosedProgramBundle } from '../../src/tooling/program-bundle.js';
import { createRerankProgram } from '../../src/tooling/program-bundle-host.js';
import { computeCanonicalSha256, hashBytesSha256 } from '../../src/formats/canonical-hash.js';
import { KERNEL_REF_CONTENT_DIGESTS } from '../../src/config/kernels/kernel-ref-digests.js';
import { createRerankReferenceFixture } from '../helpers/rerank-reference-fixture.js';
import { hashStableJson } from '../../src/tooling/program-bundle/materialize.js';

const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-rerank-bundle-'));
try {
  const fixture = createRerankReferenceFixture();
  const weights = new Uint8Array(16);
  const execution = { kernels: { embed: { kernel: 'gather.wgsl', entry: 'main',
    digest: `sha256:${KERNEL_REF_CONTENT_DIGESTS['gather.wgsl#main']}` } },
  preLayer: [['embed', 'embed']], decode: [], prefill: [], postLayer: [] };
  const manifest = {
    version: 1, modelId: fixture.modelId, modelType: 'llm', hashAlgorithm: 'sha256',
    artifactIdentity: { sourceCheckpointId: fixture.reference.source.checkpointId,
      sourceRepo: fixture.reference.source.repository, sourceRevision: fixture.reference.source.revision },
    tokenizer: { type: 'bundled', file: 'tokenizer.json' },
    shards: [{ index: 0, filename: 'shard.bin', size: weights.length, hash: hashBytesSha256(weights).slice(7), offset: 0 }],
    inference: { schema: 'doppler.execution/v1', supportsRerank: true, rerank: fixture.reference.scoringConfig, execution },
  };
  const manifestBytes = JSON.stringify(manifest);
  const report = { schema: 'doppler.rerankModelQualification.v1', passed: true,
    model: { modelId: manifest.modelId, manifestHash: hashBytesSha256(new TextEncoder().encode(manifestBytes)), artifactIdentity: manifest.artifactIdentity },
    runtime: { surface: fixture.surface, executionGraphHash: hashStableJson(execution), adapterInfo: { vendor: 'synthetic' } },
    reference: fixture.reference, referenceDigest: fixture.referenceDigest, observation: fixture.observation };
  await fs.writeFile(path.join(dir, 'manifest.json'), manifestBytes);
  await fs.writeFile(path.join(dir, 'tokenizer.json'), '{}');
  await fs.writeFile(path.join(dir, 'shard.bin'), weights);
  const reportPath = path.join(dir, 'report.json');
  await fs.writeFile(reportPath, JSON.stringify(report));
  const options = { repoRoot: process.cwd(), modelDir: dir, referenceReportPath: reportPath };
  const bundle = await exportProgramBundle(options);
  assert.equal(bundle.referenceTranscript.operation, 'rerank');
  assert.equal(bundle.host.entrypoints[0].export, 'createRerankProgram');
  assert.deepEqual(bundle.captureProfile.surfaces, [fixture.surface]);
  const written = await writeProgramBundle({ ...options, outputPath: path.join(dir, 'closed', 'program-bundle.json') });
  assert.equal((await verifyClosedProgramBundle(written.outputPath)).ok, true);
  assert.throws(() => createRerankProgram({}, bundle), /createRerankProgram/);
  const program = {};
  assert.equal(createRerankProgram({ createRerankProgram: () => program }, bundle), program);

  for (const change of [
    (value) => { value.model.manifestHash = fixture.manifestHash; },
    (value) => { value.model.modelId = 'another-model'; },
    (value) => { value.observation.outputs[0].trueLogit += 1; },
  ]) {
    const rejected = structuredClone(report);
    change(rejected);
    await fs.writeFile(reportPath, JSON.stringify(rejected));
    await assert.rejects(exportProgramBundle(options), /does not bind|source comparison failed/);
  }
} finally {
  await fs.rm(dir, { recursive: true, force: true });
}
console.log('rerank-program-bundle.test: ok (synthetic source qualification and closed artifacts)');
