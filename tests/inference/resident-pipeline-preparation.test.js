// Real loader, resident opening and partition execution on a recording device
// that compiles nothing. This compares the pipelines prepared before readiness
// with the pipelines execution binds; it is not GPU arithmetic evidence.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';

globalThis.GPUBufferUsage ??= { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32,
  UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512 };
globalThis.GPUMapMode ??= { READ: 1, WRITE: 2 };
globalThis.GPUShaderStage ??= { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };
Object.defineProperty(globalThis.navigator, 'gpu', { configurable: true, value: {
  wgslLanguageFeatures: new Set(['readonly_and_readwrite_storage_textures', 'packed_4x8_integer_dot_product',
    'unrestricted_pointer_parameters', 'pointer_composite_access']),
  async requestAdapter() { return null; } } });

const { setDevice } = await import('../../src/gpu/device.js');
const { createPipeline } = await import('../../src/inference/pipelines/text.js');
const { createArtifactStorageContext } = await import('../../src/storage/artifact-storage-context.js');
const { createOrderedLayerPartitionPlan, hashLayerPartitionPlan } = await import('../../src/inference/pipelines/text/layer-partition-contract.js');
const { resolveResidentPartitionAllocation } = await import('../../src/inference/pipelines/text/resident-partition-contract.js');
const { createResidentPartitionSession } = await import('../../src/inference/pipelines/text/resident-partition.js');
const { resolveResidentPipelineProgram, prepareResidentPartitionPipelines } = await import('../../src/inference/pipelines/text/resident-pipeline-preparation.js');
const { runPipelineOperation } = await import('../../src/inference/pipelines/shader-scoped-pipeline.js');
const { createPartitionAttempt } = await import('../../src/inference/pipelines/text/partition-attempt.js');
const { executePartitionLayers } = await import('../../src/inference/pipelines/text/partition-execution.js');
const { releaseBuffer } = await import('../../src/memory/buffer-pool.js');
const { getShaderModuleIdentity, getShaderSourceIdentity } = await import('../../src/gpu/kernels/shader-cache.js');
const { canonicalizeJson } = await import('../../src/formats/canonical-hash.js');
const { resolveGenerationOptions } = await import('../../src/config/generation-contract.js');

class FakeBuffer {
  constructor(size, usage, label) { Object.assign(this, { size, usage, label, mapState: 'unmapped' }); }
  async mapAsync() { this.mapState = 'mapped'; }
  getMappedRange(offset = 0, size = this.size - offset) { return new ArrayBuffer(size); }
  unmap() { this.mapState = 'unmapped'; }
  destroy() { this.destroyed = true; }
}
function createRecordingDevice() {
  const record = { created: [], used: [], pending: [], hold: false, fail: null };
  const pipeline = descriptor => ({ label: descriptor.label, module: descriptor.compute.module,
    entryPoint: descriptor.compute.entryPoint, constants: descriptor.compute.constants ?? null, getBindGroupLayout: () => ({}) });
  const pass = { setPipeline(p) { record.used.push(p); }, setBindGroup() {}, dispatchWorkgroups() {},
    dispatchWorkgroupsIndirect() {}, end() {}, pushDebugGroup() {}, popDebugGroup() {}, insertDebugMarker() {} };
  const device = {
    label: 'recording', features: new Set(['shader-f16', 'subgroups']), lost: new Promise(() => {}),
    limits: { maxBufferSize: 2 ** 31, maxStorageBufferBindingSize: 2 ** 30, maxComputeWorkgroupsPerDimension: 65535,
      maxStorageBuffersPerShaderStage: 10, maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupSizeY: 256, maxComputeWorkgroupSizeZ: 64, maxComputeWorkgroupStorageSize: 32768,
      maxUniformBufferBindingSize: 65536, minStorageBufferOffsetAlignment: 256, minUniformBufferOffsetAlignment: 256,
      maxBindGroups: 4, maxBindingsPerBindGroup: 1000, maxDynamicStorageBuffersPerPipelineLayout: 4,
      maxDynamicUniformBuffersPerPipelineLayout: 8, maxUniformBuffersPerShaderStage: 12 },
    queue: { submit() {}, writeBuffer() {}, async onSubmittedWorkDone() {} },
    createBuffer: ({ size, usage, label }) => new FakeBuffer(size, usage, label),
    createShaderModule: ({ code, label }) => ({ label, code, async getCompilationInfo() { return { messages: [] }; } }),
    async createComputePipelineAsync(descriptor) {
      record.created.push(descriptor.label);
      if (record.fail?.(descriptor)) throw new Error(`provider rejected ${descriptor.label}`);
      if (record.hold) await new Promise(resolve => record.pending.push(resolve));
      return pipeline(descriptor);
    },
    createComputePipeline(descriptor) { record.created.push(descriptor.label); return pipeline(descriptor); },
    createBindGroupLayout: d => ({ label: d.label }), createPipelineLayout: d => ({ label: d.label }),
    createBindGroup: d => ({ label: d.label }), createQuerySet: d => ({ ...d, destroy() {} }),
    createCommandEncoder: () => ({ beginComputePass: () => pass, copyBufferToBuffer() {}, clearBuffer() {},
      resolveQuerySet() {}, writeTimestamp() {}, pushDebugGroup() {}, popDebugGroup() {}, finish: () => ({}) }),
    addEventListener() {}, removeEventListener() {}, pushErrorScope() {}, async popErrorScope() { return null; }, destroy() {},
  };
  setDevice(device, { platformConfig: null });
  return record;
}

// The frozen Qwen3.5 0.8B manifest with the deployed paired program (prefill
// chunked at 16 tokens). Layers 0, 1 (linear) and 3 (full) become a 3-layer model
// with real per-layer geometry and encodings; zero bytes stand in for weights.
const baseline = JSON.parse(gunzipSync(readFileSync(new URL('../fixtures/qwen35-frozen-reference-manifest.json.gz', import.meta.url))));
const graph = JSON.parse(readFileSync(new URL('../../artifacts/chunked-prefill-capsule-20261014/source-execution.json', import.meta.url)));
const SOURCE_LAYERS = [0, 1, 3], VOCAB = 512;
const manifest = structuredClone(baseline);
manifest.inference.execution = graph;
manifest.architecture = { ...manifest.architecture, numLayers: SOURCE_LAYERS.length, vocabSize: VOCAB };
manifest.inference.layerPattern.layerTypes = SOURCE_LAYERS.map(layer => baseline.inference.layerPattern.layerTypes[layer]);
assert.deepEqual(manifest.inference.layerPattern.layerTypes, ['linear_attention', 'linear_attention', 'full_attention']);
manifest.tensors = {};
let cursor = 0;
for (const [name, entry] of Object.entries(baseline.tensors)) {
  const match = /layers\.(\d+)\./.exec(name);
  const layer = match ? SOURCE_LAYERS.indexOf(Number(match[1])) : null;
  if (layer === -1) continue;
  const tensor = structuredClone(entry);
  delete tensor.spans;
  if (tensor.role === 'embedding' || tensor.role === 'lm_head') {
    tensor.size = VOCAB * entry.size / entry.shape[0];
    tensor.shape = [VOCAB, entry.shape[1]];
  }
  if (match) tensor.group = `layer.${layer}`;
  Object.assign(tensor, { shard: 0, offset: cursor });
  cursor += tensor.size;
  manifest.tensors[match ? name.replace(`layers.${match[1]}.`, `layers.${layer}.`) : name] = tensor;
}
manifest.shards = [{ index: 0, filename: 'shard_00000.bin', size: cursor, offset: 0, hash: '0'.repeat(64) }];
manifest.totalSize = cursor;
manifest.tokenizer = { ...manifest.tokenizer, vocabSize: VOCAB };
const tokenizerJson = JSON.stringify({ version: '1.0', model: { type: 'BPE', merges: [],
  vocab: Object.fromEntries(Array.from({ length: VOCAB }, (_, id) => [`t${id}`, id])) }, added_tokens: [] });
const storage = (target = manifest) => createArtifactStorageContext({ manifest: target,
  readRange: async (_path, _offset, length) => new ArrayBuffer(length), readText: async () => tokenizerJson,
  readBinary: async () => new TextEncoder().encode(tokenizerJson).buffer });
const withGraph = execution => ({ ...manifest, inference: { ...manifest.inference, execution } });
const runtimeConfig = { inference: { compute: { activationDtype: 'f32' }, session: {
  intermediatePrecision: 'two-f32', linearRecurrentStatePrecision: 'two-f32', prefillTokenChunkSize: 16,
  prefillChunkLayers: 1, kvcache: { maxSeqLen: 32, kvDtype: 'f16', layout: 'contiguous' },
  compute: { defaults: { activationDtype: 'f32', mathDtype: 'f32', accumDtype: 'f32', outputDtype: 'f32' } } } } };
const plan = createOrderedLayerPartitionPlan({ modelId: manifest.modelId, numLayers: 3, hiddenSize: manifest.architecture.hiddenSize,
  vocabSize: VOCAB, splitLayers: [1, 2], activationPrecision: 'two-f32' });
const model = { id: manifest.modelId, identity: 'sha256:' + 'a'.repeat(64) };
const generation = resolveGenerationOptions({ maxTokens: 2, maxSeqLen: 32, temperature: 0, topK: 1, topP: 1,
  repetitionPenalty: 1, repetitionPenaltyWindow: 0, useChatTemplate: false });
const limits = { maxTokens: 2, maxPromptTokens: 17, maxActivationBytes: 1 << 20, maxOutputCharacters: 64,
  maxAttempts: 4, maxConcurrentAttempts: 1, maxPreparationMs: 60_000 };
const allocate = (index, declared = limits, target = manifest) => resolveResidentPartitionAllocation(target, model.identity, { model, plan,
  planId: hashLayerPartitionPlan(plan), index, participantId: `stage-${index}`, limits: declared, generation });
async function load(index, declared = limits, target = manifest) {
  const pipeline = await createPipeline(target, { runtimeConfig, storage: storage(target), partition: { plan, index } });
  return { pipeline, allocation: allocate(index, declared, target) };
}
const compiled = (module, entryPoint, constants) => `${module}#${entryPoint} ${canonicalizeJson(constants ?? {})}`;
const boundIdentity = p => compiled(getShaderModuleIdentity(p.module), p.entryPoint, p.constants);
const preparedIdentity = e => compiled(getShaderSourceIdentity(e.shaderFile), e.entryPoint, e.constants);
const rolesOf = preparation => new Set(preparation.pipelines.flatMap(entry => entry.roles));
async function settle(condition = () => false) {
  for (let i = 0; i < 400 && !condition(); i++) await new Promise(resolve => setImmediate(resolve));
}

// 1. Every pipeline execution binds was created before readiness, and nothing else was, on the
// deployed graphs and on the faster candidate's graph with declared Q/K head-norm steps.
// - The chunked-prefill Capsule graph (above) and the faster candidate's (manifest 548e6559, which
//   declares rmsnorm_paired as mechanism kernel paired_qk_head_norm) have no q_norm/k_norm steps:
//   their head norms run the interim program rmsnorm/paired.
// - With q_norm and k_norm steps, the steps' program runs them: the sliced program through the
//   norm key (paired_qk_head_norm removed, so rmsnorm_paired.wgsl leaves the closure), or the
//   shipped program through the paired_qk_head_norm key.
const candidate = JSON.parse(readFileSync(new URL('../fixtures/qwen35-faster-candidate-execution.json', import.meta.url)));
const withQKNormSteps = (source, kernelKey) => {
  const stepped = structuredClone(source);
  for (const phase of ['decode', 'prefill']) {
    stepped[phase].splice(stepped[phase].findIndex(step => step[0] === 'v_proj') + 1, 0, ['q_norm', kernelKey], ['k_norm', kernelKey]);
  }
  stepped.mechanismKernels = stepped.mechanismKernels.filter(key => key !== 'paired_qk_head_norm');
  if (kernelKey !== 'paired_qk_head_norm') delete stepped.kernels.paired_qk_head_norm;
  return stepped;
};
for (const [label, execution, headNormShader] of [
  ['chunked-prefill graph without steps', graph, 'rmsnorm_paired.wgsl'],
  ['faster-candidate graph without steps', candidate, 'rmsnorm_paired.wgsl'],
  ['faster-candidate graph, steps declaring the sliced program', withQKNormSteps(candidate, 'norm'), 'rmsnorm_paired_sliced.wgsl'],
  ['faster-candidate graph, steps declaring the shipped program', withQKNormSteps(candidate, 'paired_qk_head_norm'), 'rmsnorm_paired.wgsl'],
]) {
  const target = withGraph(execution);
  const record = createRecordingDevice();
  const signal = new AbortController().signal;
  for (let index = 0; index < plan.stageCount; index++) {
    const { pipeline, allocation } = await load(index, limits, target);
    const progress = [];
    const resident = await createResidentPartitionSession(pipeline, allocation, {}, () => pipeline.unload(), signal,
      event => progress.push(event));
    try {
      const preparation = resident.getDescriptor().pipelinePreparation;
      // Observable while it runs: each pipeline's key, count and elapsed time, then one terminal record.
      assert.deepEqual(progress.slice(0, -1).map(event => [event.status, event.prepared, event.key, event.outstanding]),
        preparation.pipelines.map((entry, prepared) => ['preparing', prepared, entry.key, true]));
      assert.deepEqual({ ...progress.at(-1), elapsedMs: 0, startedAt: 0 }, { phase: 'pipelines', status: 'prepared',
        prepared: preparation.count, total: preparation.count, key: null, elapsedMs: 0, startedAt: 0,
        maxDurationMs: limits.maxPreparationMs, outstanding: false, complete: true });
      assert(progress.every((event, at) => event.total === preparation.count && event.phase === 'pipelines'
        && (at === 0 || event.elapsedMs >= progress[at - 1].elapsedMs) && event.startedAt === progress[0].startedAt));
      assert.equal(preparation.schema, 'doppler.resident-pipeline-preparation/v1');
      assert.equal(preparation.complete, true);
      assert.deepEqual(preparation.lazy, []);
      assert.equal(preparation.count, preparation.pipelines.length);
      assert(preparation.durationMs >= 0);
      const declared = new Set(Object.values(execution.kernels).map(kernel => kernel.kernel));
      for (const entry of preparation.pipelines) assert(declared.has(entry.shaderFile), `${label}: ${entry.key} is outside the declared program`);
      const createdAtReady = record.created.length, usedFrom = record.used.length;
      await runPipelineOperation(pipeline, async owner => {
        // 17 tokens prefill as chunks of 16 and 1; a 1-token prompt is a single chunk.
        for (const prompt of [17, 1]) {
          const attempt = createPartitionAttempt(owner, 32);
          try {
            for (const tokens of [prompt, 1, 1]) {
              const result = await executePartitionLayers(attempt.state, index === 0
                ? { numTokens: tokens, tokenIds: Array.from({ length: tokens }, (_, id) => id + 1) }
                : { numTokens: tokens, activationBytes: new ArrayBuffer(tokens * manifest.architecture.hiddenSize * 8) }, signal);
              if (result.logits) releaseBuffer(result.logits.logitsBuffer);
              attempt.state.currentSeqLen += tokens;
            }
          } finally { attempt.close(); }
        }
        const unsupported = resolveResidentPipelineProgram(Object.create(owner, { layerPipelinePlan: { value: { steps: [] } } }), allocation);
        assert.deepEqual(unsupported.pipelines, []);
        assert.equal(unsupported.lazy[0].scope, 'stage', 'an undeclared program shape is reported lazy, never partially prepared');
        assert.match(unsupported.lazy[0].reason, /layer pipeline plan/);
      });
      assert.equal(record.created.length, createdAtReady, `${label}: stage ${index} created pipelines after readiness`);
      const bound = new Set(record.used.slice(usedFrom).map(boundIdentity));
      const prepared = new Set(preparation.pipelines.map(preparedIdentity));
      assert.deepEqual([...bound].filter(key => !prepared.has(key)), [], `${label}: stage ${index} bound unprepared pipelines`);
      assert.deepEqual([...prepared].filter(key => !bound.has(key)), [], `${label}: stage ${index} prepared unused pipelines`);
      const roles = rolesOf(preparation);
      assert.equal(roles.has('embed'), index === 0);
      assert.equal(roles.has('logits_finalize'), index === plan.stageCount - 1);
      assert.equal(roles.has('attention'), index === 2);
      assert.equal(roles.has('linear_attention'), index < 2, 'recurrent pipelines belong only to stages with linear layers');
      // The full-attention stage prepares, and therefore binds, exactly the head-norm program the graph declares.
      for (const role of ['q_norm', 'k_norm']) {
        assert.deepEqual([...new Set(preparation.pipelines.filter(entry => entry.roles.includes(role)).map(entry => entry.shaderFile))],
          index === 2 ? [headNormShader] : [], `${label}: stage ${index} ${role} program`);
      }
    } finally { await resident.close(); }
  }
}

// 2. Scalar programs report their pipelines as lazy instead of skipping them silently.
{
  const progress = [];
  const scalar = await prepareResidentPartitionPipelines({ gpuContext: null },
    { plan: { activationPrecision: 'f32', partitions: [] }, index: 0 }, undefined, event => progress.push(event));
  assert.equal(scalar.complete, false);
  assert.equal(scalar.count, 0);
  assert.equal(scalar.lazy[0].scope, 'program');
  assert.deepEqual(progress.map(event => [event.status, event.total, event.maxDurationMs, event.complete]), [['prepared', 0, null, false]]);
}

// 3. Cancel mid-creation: reported at once, but the stage keeps ownership (the opening
// does not settle and nothing is released) until the started creation settles.
{
  const record = createRecordingDevice();
  const { pipeline, allocation } = await load(1);
  try {
    record.hold = true;
    const controller = new AbortController(), progress = [];
    let settled = false, released = false;
    const opening = createResidentPartitionSession(pipeline, allocation, {}, async () => { released = true; },
      controller.signal, event => progress.push(event));
    opening.then(() => { settled = true; }, () => { settled = true; });
    await settle(() => record.pending.length === 1);
    assert.equal(record.pending.length, 1, 'preparation started a provider creation');
    const started = record.created.length, inFlight = progress.at(-1).key;
    controller.abort(new Error('cancel preparation'));
    assert.deepEqual([progress.at(-1).status, progress.at(-1).outstanding, progress.at(-1).key], ['cancelling', true, inFlight]);
    await new Promise(resolve => setTimeout(resolve, 30)); await settle();
    assert.equal(settled, false, 'cancellation retains ownership until the started provider creation settles');
    assert.equal(released, false, 'nothing is released while compilation is outstanding');
    record.pending.shift()();
    await assert.rejects(opening, /cancel preparation/);
    assert.deepEqual([progress.at(-1).status, progress.at(-1).outstanding, progress.at(-1).key], ['cancelled', false, inFlight]);
    assert.equal(record.created.length, started, 'no pipeline creation starts after abort');
    assert.equal(released, false, 'a failed opening never reports ready or closes as a resident');
  } finally { record.hold = false; await pipeline.unload(); }
}

// 4. A failed creation fails readiness and names the pipeline.
{
  const record = createRecordingDevice();
  const { pipeline, allocation } = await load(2);
  try {
    record.fail = descriptor => descriptor.label.startsWith('attention_paired');
    const progress = [];
    await assert.rejects(createResidentPartitionSession(pipeline, allocation, {}, async () => {}, undefined,
      event => progress.push(event)), error => {
      assert.match(error.message, /Resident stage 2 could not create GPU pipeline attention\/paired/);
      assert.match(error.message, /attention_paired\.wgsl#main/);
      assert.match(error.message, /not advertised ready/);
      assert.match(error.cause.message, /provider rejected attention_paired/);
      return true;
    });
    assert.deepEqual([progress.at(-1).status, progress.at(-1).outstanding], ['failed', false]);
    assert.match(progress.at(-1).error, /attention\/paired/);
  } finally { await pipeline.unload(); }
}

// 5. The declared bound fails opening naming the pipeline in progress; the started
// creation still owns the stage until it settles. Without a declared bound, nothing starts.
{
  const record = createRecordingDevice();
  const bounded = { ...limits, maxPreparationMs: 40 };
  const { pipeline, allocation } = await load(0, bounded);
  try {
    record.hold = true;
    const progress = [];
    let settled = false;
    const opening = createResidentPartitionSession(pipeline, allocation, {}, async () => {}, undefined, event => progress.push(event));
    opening.then(() => { settled = true; }, () => { settled = true; });
    await settle(() => record.pending.length === 1);
    const started = record.created.length, inFlight = progress.at(-1).key;
    await new Promise(resolve => setTimeout(resolve, 80)); await settle();
    const exceeded = progress.at(-1);
    assert.deepEqual([exceeded.status, exceeded.outstanding, exceeded.key, exceeded.maxDurationMs], ['timed-out', true, inFlight, 40]);
    assert.match(exceeded.error, /exceeded its pipeline preparation bound \(limits\.maxPreparationMs=40\)/);
    assert(exceeded.error.includes(inFlight), 'the timeout names the pipeline in progress');
    assert.equal(settled, false, 'the bound reports failure without releasing an outstanding creation');
    record.pending.shift()();
    await assert.rejects(opening, error => error.code === 'DOPPLER_RESIDENT_PREPARATION_TIMEOUT' && error.message.includes(inFlight));
    assert.deepEqual([progress.at(-1).status, progress.at(-1).outstanding], ['timed-out', false]);
    assert.equal(record.created.length, started, 'no pipeline creation starts after the bound');
    assert.throws(() => allocate(0, { ...limits, maxPreparationMs: undefined }), /maxPreparationMs/);
    await runPipelineOperation(pipeline, owner => assert.rejects(prepareResidentPartitionPipelines(owner,
      { ...allocation, limits: { ...limits, maxPreparationMs: undefined } }), /limits\.maxPreparationMs/));
    assert.equal(record.created.length, started, 'an undeclared bound starts no creation');
  } finally { record.hold = false; await pipeline.unload(); }
}
console.log('resident pipeline preparation: prepared set equals execution-bound set per stage on the deployed graphs with and without declared Q/K head-norm steps; progress observable; cancel and bound retain ownership until settlement; failure and timeout name the pipeline');
