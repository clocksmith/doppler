// CPU-only continuation of the frozen numerical fixture. No process/device launch.
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir, copyFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';

const parent = '/var/tmp/reploid-gate-three-stage-20261013';
const root = '/var/tmp/reploid-cold-compile-three-stage-20261013';
const consumer = '/var/tmp/doppler-cold-compile-package-verified-20261013/consumer/node_modules/doppler-gpu';
const archive = '/var/tmp/doppler-cold-compile-package-verified-20261013/doppler-gpu-0.6.28.tgz';
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const json = path => readFile(path, 'utf8').then(JSON.parse);
const save = (path, value) => writeFile(path, JSON.stringify(value, null, 2) + '\n');
const parentBytes = await readFile(parent + '/snapshot-binding.json');
assert.equal(hash(parentBytes), '6f69ef692a5916f1736ca356625df0bd81267a8b2e5ea6edf725e96e53220d2d');
const prior = JSON.parse(parentBytes), inventory = await json(new URL('./inventory.json', import.meta.url));
assert.equal(hash(await readFile(archive)), inventory.archiveSha256);
await mkdir(root); // Refuse to overwrite an earlier candidate.
const paths = new Set();
for (const entry of prior.files) {
  const bytes = await readFile(resolve(parent, entry.path));
  assert.equal(hash(bytes), entry.sha256, entry.path);
  if (entry.path.startsWith('vendor/')) continue;
  const target = resolve(root, entry.path); await mkdir(dirname(target), { recursive: true });
  await writeFile(target, bytes); paths.add(entry.path);
}
const runtime = root + '/vendor/doppler/0.6.28';
for (const entry of inventory.files) {
  const bytes = await readFile(resolve(consumer, entry.path));
  assert.equal(hash(bytes), entry.sha256); assert.equal(bytes.length, entry.bytes);
  const relative = 'vendor/doppler/0.6.28/' + entry.path;
  const target = resolve(root, relative); await mkdir(dirname(target), { recursive: true });
  await writeFile(target, bytes); paths.add(relative);
}
await copyFile(archive, root + '/archive.tgz');
const load = path => import(pathToFileURL(runtime + '/src/' + path));
const { assertPairedExecutionMechanismClosure } = await load('config/execution-registry-contract.js');
const graph = await json(root + '/execution.json');
// Replace the obsolete combined preparation declaration with the exact phases.
delete graph.kernels.qwen_linear_attention_prepare_paired;
graph.mechanismKernels = graph.mechanismKernels.filter(k => k !== 'qwen_linear_attention_prepare_paired');
for (const [key, shader] of [
  ['paired_beta', 'qwen_linear_attention_prepare_paired'],
  ['paired_decay_scale', 'qwen_linear_attention_scale_paired'],
  ['paired_log_decay', 'qwen_linear_attention_log_decay_paired'],
  ['paired_decay', 'qwen_linear_attention_decay_paired'],
  ...['qk', 'state', 'norm', 'gate'].map(phase => [`paired_${phase}`, `gated_delta_${phase}_paired`]),
]) {
  graph.kernels[key] = { kernel: shader + '.wgsl', entry: 'main', constants: {} };
  graph.mechanismKernels.push(key);
}
for (const kernel of Object.values(graph.kernels)) {
  // Same content domain as tools/sync-kernel-ref-digests.js, using the installed
  // shader itself; the generated developer index is intentionally not shipped.
  const shader = (await readFile(runtime + '/src/gpu/kernels/' + kernel.kernel, 'utf8')).replace(/\r\n/g, '\n');
  const digest = hash(shader + '\n@@entry:' + kernel.entry);
  assert.match(digest, /^[a-f0-9]{64}$/); kernel.digest = 'sha256:' + digest;
}
assertPairedExecutionMechanismClosure(graph);
await save(root + '/execution.json', graph);
const model = await json(root + '/manifest.json'); model.inference.execution = graph;
await save(root + '/manifest.json', model);
const manifestBytes = await readFile(root + '/manifest.json'), graphBytes = await readFile(root + '/execution.json');
const binding = await json(root + '/paired-binding.json');
Object.assign(binding, { archiveSha256: inventory.archiveSha256,
  immediateParentManifestSha256: binding.manifestSha256, manifestSha256: hash(manifestBytes), graphSha256: hash(graphBytes),
  sourceConstruction: 'Ordinary archive 9264d265a90c7c0b046adf46135965b2032d2d9f6e5e60d6a8149d10f0988ec4 from committed fca57498, including ae437c95; all bytes verified against installed consumer.',
  closureCorrection: 'Four gate phases and four recurrent phases explicitly declared; no previous signature or numerical report is reused.', qualified: false });
await save(root + '/paired-binding.json', binding);
const { bindPairedPartitionFixture } = await import(pathToFileURL(root + '/tests/fixtures/paired-partition-binding.js'));
const reference = await json(root + '/approved-reference.json');
const facts = await bindPairedPartitionFixture({ bindingBytes: await readFile(root + '/paired-binding.json'),
  graphBytes, modelBytes: manifestBytes, archiveBytes: await readFile(archive), referenceIdentity: reference.modelIdentity });
assert.equal(reference.numericalReference.steps, 55); assert.equal(reference.numericalReference.tolerance, 0.001);
const frozen = { scope: prior.scope, reploidCommit: prior.reploidCommit, dopplerCommit: 'fca57498',
  archiveSha256: inventory.archiveSha256, parentSnapshotSha256: hash(parentBytes), files: [] };
for (const path of [...paths].sort()) frozen.files.push({ path, sha256: hash(await readFile(resolve(root, path))) });
await save(root + '/snapshot-binding.json', frozen);
const config = JSON.parse((await readFile(parent + '/config-preparation.json', 'utf8')).replaceAll(parent, root));
config.model.identity = 'sha256:' + hash(manifestBytes);
for (const key of ['manifest', 'graph', 'archive', 'pairedBinding', 'delivery']) config[key].hash = 'sha256:' + hash(await readFile(config[key].path));
assert.equal(config.launchReady, false); assert.equal(config.gpuReservation.approved, false);
await save(root + '/config-preparation.json', config);
const { verifyNumericalTestDelivery } = await import(pathToFileURL(root + '/tests/fixtures/numerical-test-delivery.js'));
const delivery = await verifyNumericalTestDelivery({ root, bindingPath: config.delivery.path,
  bindingSha256: config.delivery.hash.slice(7), archivePath: archive, runtimeDirectory: runtime });
const publicRuntime = await load('partitions.js');
const plans = [2, 3, 4, 5, 6, 7].map(count => {
  const plan = publicRuntime.createOrderedLayerPartitionPlan({ modelId: model.modelId, ...model.architecture,
    splitLayers: Array.from({ length: count - 1 }, (_, i) => Math.floor(model.architecture.numLayers * (i + 1) / count)),
    activationPrecision: 'two-f32' });
  publicRuntime.validateOrderedLayerPartitionPlan(model, plan);
  return { count, hash: publicRuntime.hashLayerPartitionPlan(plan), plan };
});
await save(new URL('./model-preparation.json', import.meta.url), { root, delivery, facts, plans,
  config, gpuOpened: false, launchReady: false, next: 'Fresh host reservation, installed endpoint/provider observations and initial execution identities before physical capture' });
console.log(JSON.stringify({ root, delivery, checkedStageCounts: plans.map(p => p.count), gpuOpened: false }));
