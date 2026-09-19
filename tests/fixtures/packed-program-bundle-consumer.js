// Installed Forge packaging contract with synthetic data; no model qualification.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { writeProgramBundle, verifyClosedProgramBundle } from 'doppler-gpu/tooling';

const packageRoot = path.dirname(path.dirname(fileURLToPath(import.meta.resolve('doppler-gpu'))));
assert(packageRoot.endsWith(path.join('node_modules', 'doppler-gpu')));
const gatherSource = (await fs.readFile(path.join(packageRoot, 'src/gpu/kernels/gather.wgsl'), 'utf8'))
  .replace(/\r\n/g, '\n');
const directory = path.resolve('program-bundle-fixture');
await fs.mkdir(directory);
const weights = new Uint8Array(16);
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const manifest = {
  version: 1, modelId: 'installed-bundle-fixture', modelType: 'llm', hashAlgorithm: 'sha256',
  shards: [{ index: 0, filename: 'shard.bin', size: weights.length, hash: hash(weights), offset: 0 }],
  tokenizer: { type: 'bundled', file: 'tokenizer.json' },
  inference: { schema: 'doppler.execution/v1', execution: {
    kernels: { embed: { kernel: 'gather.wgsl', entry: 'main',
      digest: `sha256:${hash(`${gatherSource}\n@@entry:main`)}` } },
    preLayer: [['embed', 'embed']], decode: [], prefill: [], postLayer: [],
  } },
};
const report = { modelId: manifest.modelId, results: [{ name: 'generation', passed: true }],
  output: 'synthetic', metrics: { prompt: 'synthetic', tokensGenerated: 1,
    referenceTranscript: { tokens: { ids: [1] }, output: { tokensGenerated: 1, stopReason: 'max-tokens' } } } };
await fs.writeFile(path.join(directory, 'manifest.json'), JSON.stringify(manifest));
await fs.writeFile(path.join(directory, 'tokenizer.json'), '{}');
await fs.writeFile(path.join(directory, 'shard.bin'), weights);
await fs.writeFile(path.join(directory, 'reference.json'), JSON.stringify(report));
const written = await writeProgramBundle({ repoRoot: packageRoot, modelDir: directory,
  referenceReportPath: path.join(directory, 'reference.json'),
  outputPath: path.join(directory, 'closed/program-bundle.json') });
const closed = await verifyClosedProgramBundle(written.outputPath);
assert.equal(closed.ok, true);
const host = closed.files.find(file => file.role === 'host-source');
assert(host, 'Installed export must materialize its default host module');
const bridge = await import(pathToFileURL(host.absolutePath));
for (const name of ['createTextGenerationProgram', 'createSequenceProgram', 'createRerankProgram']) {
  const program = {};
  assert.equal(bridge[name]({ [name]: () => program }, written.bundle), program);
  assert.throws(() => bridge[name]({}, written.bundle), /required/);
}
await fs.appendFile(host.absolutePath, '\n// corrupted retained host\n');
await assert.rejects(verifyClosedProgramBundle(written.outputPath), /hash\/size mismatch/);
console.log('package program bundle smoke passed (synthetic export, host closure, corruption rejection)');
