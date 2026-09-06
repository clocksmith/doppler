import assert from 'node:assert/strict';
import { createCapsuleAdapterExecution } from '../../src/client/runtime/capsule-adapter-execution.js';
import { createCapsuleOperationExecutor } from '../../src/client/runtime/capsule-operation-executor.js';
import { hashCapsuleObservation } from '../../src/config/capsule-operation.js';
import { sha256Hex } from '../../src/formats/sha256.js';

// Contract-only injected program: no model output or physical GPU claim.
const hash = value => `sha256:${sha256Hex(value)}`;
const bytes = new Uint8Array([1, 2, 3, 4]);
const capsule = { modelId: 'base', semanticRoot: hash('model'), envelopeDigest: hash('envelope'), artifactClosureDigest: hash('closure') };
const targetPlan = { schema: 'doppler.target-plan/v2', qualification: [{ operation: 'generate' }],
  initialExecutionIdentity: { kernelClosure: [
    { moduleId: 'matmul', file: 'matmul_f16.wgsl', entry: 'main', digest: hash('matmul') },
    { moduleId: 'scale', file: 'scale.wgsl', entry: 'main', digest: hash('scale') },
    { moduleId: 'residual', file: 'residual.wgsl', entry: 'main', digest: hash('residual') }
  ] }, adapterExecution: { schema: 'doppler.capsule-adapter-execution/v1', maxAdapters: 1,
    combination: 'single', formats: ['peft_safetensors'], operations: ['generate'], kernelModules: ['matmul', 'scale', 'residual'] } };
targetPlan.kernelClosure = targetPlan.initialExecutionIdentity.kernelClosure.map(row => ({ moduleId: row.moduleId, digest: row.digest, sourceHash: row.digest }));
const entry = { schema: 'doppler.capsule-adapter/v1', identity: hash('adapter'), baseModel: capsule, format: 'peft_safetensors',
  manifest: { id: 'adapter', baseModel: 'base', rank: 1, alpha: 1, targetModules: ['q_proj'], checksum: hash(bytes),
    checksumAlgorithm: 'sha256', weightsFormat: 'safetensors', weightsPath: 'weights.safetensors', weightsSize: 4 },
  artifact: { artifactId: 'adapter', role: 'lora-weights', path: 'weights.safetensors', hash: hash(bytes), sizeBytes: 4 } };
const request = { schema: 'doppler.capsule-operation-request/v1', operation: { name: 'generate', version: 1 },
  input: { prompt: 'public' }, options: {}, adapterSet: [entry], assignment: null,
  limits: { maxInputBytes: 10000, maxOutputBytes: 10000, deadlineAt: Date.now() + 60000 } };
let identity = null, loads = 0, unloads = 0, executions = 0, failLoad = false, failExecute = false, changeIdentity = false;
const program = {
  getActiveAdapterIdentity: () => identity,
  async loadAdapter(_manifest, control) {
    loads++;
    assert.deepEqual(control.bytes, bytes);
    identity = { schema: 'doppler.lora-execution-identity/v1', id: 'adapter', digest: hash('tensors') };
    if (failLoad) throw new Error('load failure');
  },
  async unloadAdapter() { unloads++; identity = null; }, reset() {}
};
const make = (plan = targetPlan) => createCapsuleOperationExecutor({ identity: { capsule }, assertCurrent: async () => {},
  prepareExecution: createCapsuleAdapterExecution({ program, capsule, targetPlan: plan }),
  adapters: { generate: { validate() {}, async *execute() {
    executions++;
    if (failExecute) throw new Error('model failure');
    assert.ok(identity);
    if (changeIdentity) identity = { ...identity, digest: hash('replacement tensors') };
    return { text: 'observed' };
  } } } });
const collect = async iterator => { const events = []; for await (const event of iterator) events.push(event); return events; };
const control = { adapterArtifactStore: { async readArtifact(artifact) { assert.deepEqual(artifact, entry.artifact); return bytes; } } };
const events = await collect(make()(request, control));
assert.equal(executions, 1);
assert.equal(identity, null);
assert.equal(unloads, 1);
assert.equal(events.at(-1).receipt.adapterReceipts[0].identity, entry.identity);
assert.equal(events.at(-1).receipt.adapterReceipts[0].sourceDigest, entry.artifact.hash);
assert.equal(events.at(-1).receipt.requestHash, hashCapsuleObservation(request));
await assert.rejects(collect(make({ ...targetPlan, adapterExecution: undefined })(request, control)), /explicitly declare/);
await assert.rejects(collect(make()(request, { adapterArtifactStore: { async readArtifact() { return new Uint8Array([0, 0, 0, 0]); } } })), /corruption/);
await assert.rejects(collect(make()({ ...request, adapterSet: [{ ...entry, baseModel: { ...capsule, semanticRoot: hash('other') } }] }, control)), /base model/);
await assert.rejects(collect(make()({ ...request, adapterSet: [entry, entry] }, control)), /count/);
assert.equal(loads, 1, 'rejected bytes or composition must never activate');
failLoad = true;
await assert.rejects(collect(make()(request, control)), /load failure/);
assert.equal(identity, null);
failLoad = false; failExecute = true;
await assert.rejects(collect(make()(request, control)), /model failure/);
assert.equal(identity, null);
assert.equal(unloads, 3);
failExecute = false;
const abort = new AbortController();
const cancelledStore = { async readArtifact() { abort.abort(new Error('cancelled during transfer')); return bytes; } };
await assert.rejects(collect(make()(request, { signal: abort.signal, adapterArtifactStore: cancelledStore })), /cancelled during transfer/);
assert.equal(loads, 3);
changeIdentity = true;
await assert.rejects(collect(make()(request, control)), /active adapter changed/);
assert.equal(identity, null, 'identity drift must unload before another operation');
changeIdentity = false;
const cleanupExecutor = createCapsuleOperationExecutor({ identity: { capsule }, assertCurrent: async () => {},
  prepareExecution: async () => ({ check() {}, receiptFields: {}, close() { throw new Error('unload failure'); } }),
  adapters: { generate: { validate() {}, async *execute() { throw new Error('original execution failure'); } } } });
await assert.rejects(collect(cleanupExecutor(request, control)), error => {
  assert.ok(error instanceof AggregateError);
  assert.deepEqual(error.errors.map(row => row.message), ['original execution failure', 'unload failure']);
  return true;
});
const finalCleanupFailure = createCapsuleOperationExecutor({ identity: { capsule }, assertCurrent: async () => {},
  prepareExecution: async () => ({ check() {}, receiptFields: {}, close() { throw new Error('final unload failed'); } }),
  adapters: { generate: { validate() {}, async *execute() { return { text: 'unaccepted' }; } } } });
await assert.rejects(finalCleanupFailure(request, control).next(), /final unload failed/);
console.log('capsule-adapter-execution: identity, declared kernels, corruption, base mismatch, composition, cleanup, cancellation passed');
