import assert from 'node:assert/strict';
import { createOrderedLayerPartitionPlan, createLayerPartitionPlan, hashLayerPartitionPlan } from '../../src/partitions.js';
import { computeCanonicalSha256 as hash } from '../../src/formats/canonical-hash.js';
import { assertResidentQualification } from '../../src/converter/resident-qualification.js';
import { normalizeQualificationEvidence } from '../../src/converter/forge-qualification.js';
import { assertOrderedResidentQualificationSet } from '../../src/config/ordered-resident-qualification.js';
import { createTargetPlanV2, assertQualifiedResidentPartition, validateTargetPlan } from '../../src/config/target-plan.js';
import { createInitialExecutionIdentity } from '../../src/config/initial-execution-identity.js';

// Synthetic producer inputs test structural admission only, never physical qualification.
const digest = hash('fixture');
const other = hash('other');
const manifest = { modelId: 'ordered-fixture', architecture: { numLayers: 28, hiddenSize: 4, vocabSize: 16 } };
const initialExecutionIdentity = createInitialExecutionIdentity({ executionGraphHash: digest, resolvedGraphHash: digest,
  kernelClosure: [{ moduleId: 'main', file: 'main.wgsl', entry: 'main', digest }],
  dtypeLane: { activation: 'f32', output: 'f32', kv: 'f16', math: 'f32', accumulation: 'f32' }, fusionSet: [],
  kvLayout: { layout: 'contiguous', kvDtype: 'f16' }, memoryPolicy: { kvcache: { layout: 'contiguous', kvDtype: 'f16' } },
  executionPlanDigest: digest, runtimeEngine: { schema: 'doppler.resolved-runtime-session/v1' } });
const targetBase = { targetId: 'fixture', modelId: manifest.modelId, modelIRHash: digest, executionGraphHash: digest,
  programBundleHash: digest, initialExecutionIdentity, capabilityPredicate: { requiresF16: false, requiresSubgroups: false, minBufferSize: 64 },
  dtypes: { activation: 'f32', kv: 'f16', weight: 'q4k' }, fusions: [],
  kernelClosure: [{ moduleId: 'main', digest, sourceHash: digest }],
  memoryLayout: { kvCacheLayout: 'contiguous', bufferSlots: [{ slotId: 'input', role: 'input', scope: 'transient', owner: 'runtime', usageBits: 1, size: { op: 'constant', bytes: 256 } }] },
  phases: Object.fromEntries(['prefill', 'decode'].map(phase => [phase, [{ kind: 'program-phase', phase, executionGraphHash: digest, declaredStepIds: [phase] }]])) };
for (const count of [2, 3, 4, 7]) {
  for (const activationPrecision of ['f32', 'two-f32']) {
    const partitionPlan = createOrderedLayerPartitionPlan({ modelId: manifest.modelId, numLayers: 28, hiddenSize: 4, vocabSize: 16,
      splitLayers: Array.from({ length: count - 1 }, (_, i) => Math.floor(28 * (i + 1) / count)), activationPrecision });
    const partitionPlanHash = hashLayerPartitionPlan(partitionPlan);
    const providers = Array.from({ length: count }, (_, index) => ({ adapter: `fixture-${index}`, driver: 'synthetic' }));
    const contract = { schema: 'doppler.resident-qualification-contract/v2', manifestIdentity: digest, executionGraphHash: digest,
      partitionPlan, partitionPlanHash, referenceHash: digest, evaluatorHash: other, tolerance: 0.001, comparedSteps: 2,
      stages: providers.map((provider, index) => ({ index, surface: index % 2 ? 'browser-webgpu' : 'node-webgpu', providerIdentityHash: hash(provider) })) };
    const context = { manifest, manifestIdentity: digest, executionGraphHash: digest, initialExecutionIdentity, contract };
    const reports = providers.map((providerIdentity, partitionIndex) => ({ schema: 'doppler.resident-partition-qualification/v2',
      operation: 'residentPartition', status: 'passed', modelId: manifest.modelId, manifestIdentity: digest, executionGraphHash: digest,
      initialExecutionIdentity, partitionPlan, partitionPlanHash, partitionIndex, providerIdentity, surface: contract.stages[partitionIndex].surface,
      comparison: { contractHash: hash(contract), referenceHash: digest, evaluatorHash: other, captureHash: hash(partitionIndex), tolerance: 0.001,
        complete: true, steps: [{ index: 0, maxAbsoluteError: 0 }, { index: 1, maxAbsoluteError: 0.0005 }] } }));
    const records = reports.map((report, index) => normalizeQualificationEvidence({ ...assertResidentQualification(report, context),
      evidenceHash: hash(report), sourcePath: `/synthetic/stage-${index}.json`, sizeBytes: 10 }));
    assert.doesNotThrow(() => assertOrderedResidentQualificationSet(records));
    const qualification = records.map(({ artifact, ...record }) => record);
    const target = createTargetPlanV2({ ...targetBase, qualification });
    assert.equal(validateTargetPlan(target).ok, true);
    for (const record of records) assert.doesNotThrow(() => assertQualifiedResidentPartition(target, record.surface,
      { planId: partitionPlanHash, plan: partitionPlan, model: { identity: digest }, index: record.partitionIndex }));
    assert.throws(() => assertOrderedResidentQualificationSet(records.slice(1)), /missing/);
    assert.throws(() => assertOrderedResidentQualificationSet([...records, records[0]]), /duplicate/);
    assert.equal(validateTargetPlan({ ...target, qualification: qualification.slice(1) }).ok, false);
    assert.throws(() => assertQualifiedResidentPartition(target, records[0].surface,
      { planId: partitionPlanHash, plan: partitionPlan, model: { identity: other }, index: 0 }), /no signed/);
    for (const mutate of [
      r => { r.partitionIndex = count; }, r => { r.partitionIndex = -1; }, r => { r.partitionIndex = 0.5; },
      r => { r.partitionPlan.partitions[0].layerRange[1]++; }, r => { r.partitionPlan.activationElementStrideBytes = 99; },
      r => { r.partitionPlanHash = other; }, r => { r.manifestIdentity = other; }, r => { r.modelId = 'wrong'; },
      r => { r.executionGraphHash = other; }, r => { r.providerIdentity.driver = 'other'; }, r => { delete r.providerIdentity; },
      r => { r.surface = 'other'; }, r => { r.schema = 'doppler.resident-partition-qualification/v1'; },
      r => { r.comparison.contractHash = other; }, r => { r.comparison.steps[1].maxAbsoluteError = 0.01; },
    ]) {
      const invalid = structuredClone(reports[0]); mutate(invalid);
      assert.throws(() => assertResidentQualification(invalid, context), /qualification|partition|Partition/);
    }
    for (const mutate of [
      c => { c.stages.pop(); }, c => { c.stages[1].index = 0; }, c => { c.stages.reverse(); },
      c => { c.partitionPlan.partitions[1].hasEmbedding = true; }, c => { c.partitionPlan.partitions[0].outputContract.components = 99; },
    ]) {
      const invalid = structuredClone(contract); mutate(invalid);
      assert.throws(() => assertResidentQualification(reports[0], { ...context, contract: invalid }), /qualification|partition|Partition/);
    }
    const disguised = structuredClone(records[0]); delete disguised.orderedContract;
    disguised.partitionIndex = count; disguised.sourcePath = "/synthetic/report.json"; disguised.sizeBytes = 10;
    assert.throws(() => normalizeQualificationEvidence(disguised), /plan and index/);
  }
}
// A legacy report cannot acquire ordered-stage authority through its unchanged v1 schema.
const legacy = createLayerPartitionPlan({ modelId: manifest.modelId, numLayers: 28, hiddenSize: 4, vocabSize: 16, splitLayer: 14 });
assert.notEqual(legacy.schema, 'doppler.layer-partition-contract/v3');
console.log('ordered signed qualification: synthetic 2/3/4/7-stage bindings, complete sets, model/layout/provider mutations passed');
