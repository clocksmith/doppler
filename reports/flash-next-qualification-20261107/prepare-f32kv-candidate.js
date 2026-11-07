import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { widenAttentionKvToF32 } from '../../src/converter/execution-graph-transforms.js';
import { expandExecutionV1 } from '../../src/config/schema/execution-v1.schema.js';
import { buildInlineKernelPath } from '../../src/inference/pipelines/text/execution-runtime-builders.js';
import { KERNEL_REF_CONTENT_DIGESTS } from '../../src/config/kernels/kernel-ref-digests.js';

// Separate offline candidate. The original F16 reference/run/artifact stay intact.
const [baselinePath, referencePath, outputPath, observationPath, selectionPath] = process.argv.slice(2);
const hash = raw => createHash('sha256').update(raw).digest('hex');
const run = JSON.parse(await fs.readFile(baselinePath));
const manifestBytes = await fs.readFile(path.join(run.modelDir, 'manifest.json'));
assert.equal(hash(manifestBytes), run.expectedManifestSha256);
const manifest = JSON.parse(manifestBytes);
const referenceBytes = await fs.readFile(referencePath);
const reference = JSON.parse(referenceBytes);
assert.equal(reference.schema, 'doppler.qwen38-learned-forward-reference/v1');
assert.equal(reference.artifactManifestSha256, hash(manifestBytes));
assert.equal(reference.dtype, 'float32');
const before = run.candidate?.sourceExecution;
assert(before, 'Candidate must preserve the previously identified execution graph.');
const graph = widenAttentionKvToF32(before, { capabilities: { hasF16: true, hasSubgroups: true } });
assert(graph);
const changes = [];
for (const [key, entry] of Object.entries(graph.kernels)) {
  if (entry.kernel === before.kernels[key].kernel) continue;
  assert(entry.kernel.startsWith('attention_'), `Unexpected non-attention replacement: ${key}`);
  const digest = KERNEL_REF_CONTENT_DIGESTS[`${entry.kernel}#${entry.entry}`];
  assert(digest, `Missing prepared kernel identity: ${entry.kernel}`);
  entry.digest = `sha256:${digest}`;
  changes.push({ key, before: before.kernels[key].kernel, after: entry.kernel, digest: entry.digest });
}
assert(changes.some(change => change.after === 'attention_indexed.wgsl'));
if (selectionPath) {
  const selection = JSON.parse(await fs.readFile(selectionPath));
  assert.deepEqual(Object.keys(selection), ['prefillRouterKernel']);
  assert(['matmul_f32.wgsl', 'matmul_compensated_f32.wgsl'].includes(selection.prefillRouterKernel));
  const key = 'hc_accuracy_moe_router';
  const previous = graph.kernels[key];
  assert(previous, 'The identified baseline has an isolated prefill router reference.');
  assert(!JSON.stringify(graph.decode).includes(key), 'Decode must preserve its original router.');
  const digest = KERNEL_REF_CONTENT_DIGESTS[`${selection.prefillRouterKernel}#main`];
  assert(digest);
  graph.kernels[key] = { ...previous, kernel: selection.prefillRouterKernel, entry: 'main', digest: `sha256:${digest}` };
  changes.push({ key, phase: 'prefill', op: 'moe_router', before: previous.kernel,
    after: selection.prefillRouterKernel, digest: graph.kernels[key].digest });
}
const session = structuredClone(manifest.inference.session);
session.kvcache.kvDtype = 'f32';
const resolved = expandExecutionV1(graph).map(step => ({ ...step, kernelRef: { digest: step.digest } }));
const candidate = buildInlineKernelPath(resolved, session, manifest.modelId, manifest.architecture.numLayers);
assert.equal(candidate.activationDtype, 'f32');
assert.equal(candidate.kvDtype, 'f32');
const graphSha256 = hash(JSON.stringify(graph));
candidate.id += `-f32kv-${graphSha256.slice(0, 12)}`;
candidate.description = 'Separate explicit F32 KV candidate; no F16 reference replacement or promotion';
run.runtimeConfig.inference.kernelPath = candidate;
run.runtimeConfig.inference.session = { ...run.runtimeConfig.inference.session,
  kvcache: { kvDtype: 'f32' } };
run.schema = 'doppler.flash-next-learned-probe/v1';
run.referencePath = referencePath;
run.outputPath = observationPath;
run.sequences = [{ id: 'short', maxTokens: 3 }, { id: 'long', maxTokens: 1 }];
for (const sequence of run.sequences) assert(reference.sequences[sequence.id], `Missing independent reference: ${sequence.id}`);
run.captureLogitRows = true;
run.lifecycle = false;
run.captureLayers = false;
run.captureRouterLayers = [];
run.captureQsaLayers = [];
run.runtimeRevision = '934f60a9+explicit-f32kv-working-tree';
run.sourceFiles = [...new Set([...(run.sourceFiles ?? []),
  'src/gpu/kernels/qsa-indexer.js', 'src/gpu/kernels/attention_indexed.wgsl',
  'src/inference/pipelines/text/indexed-attention/attention-runner.js',
  'src/inference/pipelines/text/hyper-connection/sequence-transaction.js',
  'src/converter/execution-graph/validation.js', 'src/converter/execution-graph/kv-storage.js',
  'src/converter/execution-graph-transforms.js',
  'src/rules/inference/capability-transforms.rules.json', 'src/rules/kernels/attention.rules.json',
  'src/config/kernels/kernel-ref-digests.js', 'src/config/kernels/registry.json',
  'tests/integration/flash-next-learned-physical.test.js',
])];
delete run.referenceRouterControl;
for (const key of ['traceLayers', 'stopBeforeMoeLayer', 'captureLinearStages', 'captureAttentionKv',
  'captureAttentionInput', 'captureMoe', 'targetStages']) delete run[key];
run.candidate = { ...run.candidate, sourceExecution: graph, graphSha256, changes,
  baselineRunSha256: hash(await fs.readFile(baselinePath)), referenceSha256: hash(referenceBytes),
  originalF16ReferenceUnchanged: true, normalModelQualified: false };
run.candidate.bindings = { ...run.candidate.bindings,
  moe_router: graph.kernels.hc_accuracy_moe_router.kernel };
await fs.writeFile(outputPath, JSON.stringify(run, null, 2)+'\n');
console.log(JSON.stringify({ outputPath, graphSha256, changes }));
