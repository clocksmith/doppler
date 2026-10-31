// CPU-only workload freeze. Never opens a device or invents executable/provider qualification.
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { createOrderedLayerPartitionPlan, hashLayerPartitionPlan } from '../../src/config/layer-partition-plan.js';
import { hashPartitionGeneration } from '../../src/partitions.js';
const sha = bytes => 'sha256:' + createHash('sha256').update(bytes).digest('hex');
const previous = JSON.parse(await readFile(new URL('../paired-intermediates-20261012/signed-closure-correction/fresh-qualification/direction-0-producer-config.json', import.meta.url)));
const bound = async entry => { const bytes = await readFile(entry.path); assert.equal(sha(bytes), entry.hash); return JSON.parse(bytes); };
const manifest = await bound(previous.manifest), reference = await bound(previous.reference), graph = await bound(previous.graph);
assert.deepEqual(manifest.inference.execution, graph);
assert.equal(sha(await readFile(previous.evaluator.path)), previous.evaluator.hash);
for (const request of previous.requestContracts) await bound(request);
assert.equal(reference.numericalReference.kind, 'independent-f64');
assert.equal(reference.numericalReference.steps, 55); assert.equal(reference.numericalReference.tolerance, 0.001);
const partitionPlan = createOrderedLayerPartitionPlan({ modelId: manifest.modelId, ...manifest.architecture,
  splitLayers: [8, 16], activationDtype: 'f32', activationPrecision: 'two-f32' });
const policyPath = '/var/tmp/reploid-closure-qualification-v2-20261013/self/config/partition-policy.json';
const policyBytes = await readFile(policyPath), policy = JSON.parse(policyBytes);
assert.equal(policy.maxGpuBufferBytes, 1420000000);
const producerPath = new URL('../signed-resident-producer-20261012/produce-physical-reports.js', import.meta.url);
const freeze = {
  schema: 'doppler.ordered-physical-workload-preparation/v1', launchReady: false,
  scope: 'Frozen three-logical-stage workload, not provider/executable freeze or numerical evidence',
  model: previous.manifest, graph: previous.graph, reference: previous.reference, evaluator: previous.evaluator,
  requestContracts: previous.requestContracts, numericalReference: reference.numericalReference,
  partitionPlan, partitionPlanHash: hashLayerPartitionPlan(partitionPlan),
  generation: reference.generation, generationDigest: hashPartitionGeneration(reference.generation),
  policy: { path: policyPath, hash: sha(policyBytes) }, maxGpuBufferBytesPerExecutor: policy.maxGpuBufferBytes,
  limits: policy.limits, bufferPool: policy.bufferPool, prefillChunkLayers: policy.prefillChunkLayers,
  prefillTokenChunkSize: policy.prefillTokenChunkSize,
  deadlines: { stageMs: 300000, cleanupMs: 10000, observationMs: 2000 },
  logicalStageCount: 3, requestedPhysicalHostCount: 3,
  proposedHosts: ['Mac', 'x128 AMD', 'intel-host Intel'],
  hostScope: 'Three physical hosts require three independently observed hosts; a colocated route must declare its actual lower host count.',
  minimumRuntimeSource: '77af06b9', producerHash: sha(await readFile(producerPath)),
  pendingBeforeLaunch: [
    'Ordinary immutable N-stage runtime archive; 0fad lacks executeStage and cannot run this plan.',
    'Ordered-reference capture branch using actual executeStage and complete boundary/lifecycle observations.',
    'Root reservation and fresh executable/adapter/provider preflight on every selected host; intel-host portability is not qualified.',
    'Exact installed archive/model/graph/delivery bytes on every executor, frozen input-v2 expectations and ordered contract-v2.',
    'Per-stage initial execution identity, actual process/context identity, final-only logits and exact activation continuity.',
  ],
};
await writeFile(new URL('./three-stage-qualification-workload.json', import.meta.url), JSON.stringify(freeze, null, 2) + '\n');
console.log(JSON.stringify({ partitionPlanHash: freeze.partitionPlanHash, launchReady: false, logicalStages: 3, comparedPrefixes: 55 }));
