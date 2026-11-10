import assert from 'node:assert/strict';
import { widenAttentionKvToF32 } from '../../src/converter/execution-graph-transforms.js';
import { resolveCapabilityTransforms } from '../../src/converter/capability-transform-resolver.js';
import { loadRuntimeProfile } from '../../src/tooling/browser-harness/runtime-config.js';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';
import { resolveRuntimeConfig } from '../../src/config/runtime.js';
import { stripLegacyRuntimeIntent } from '../../src/tooling/runtime-intent.js';
import { mergeRuntimeValues } from '../../src/config/runtime-merge.js';
import { readFile } from 'node:fs/promises';
import { compileExecutionV1 } from '../../src/inference/pipelines/text/execution-v1.js';
import { getLayerSteps } from '../../src/config/kernel-path-loader.js';
import { getKernelConfig } from '../../src/gpu/kernels/kernel-configs.js';

const graph = { kernels: {
  embed: { kernel: 'gather_f16.wgsl', entry: 'main', digest: 'sha256:embed' },
  project: { kernel: 'matmul_f32.wgsl', entry: 'main', digest: 'sha256:project' },
  indexed: { kernel: 'attention_indexed_f16kv.wgsl', entry: 'main', digest: 'sha256:attention',
    precision: { activationDtype: 'f32', kvDtype: 'f16' } },
}, decode: [['attention', 'indexed']], prefill: [['attention', 'indexed']] };
const original = structuredClone(graph);
const widened = widenAttentionKvToF32(graph, {});
assert.deepEqual(graph, original, 'the artifact graph is immutable');
assert.deepEqual(widened.kernels.embed, graph.kernels.embed, 'KV widening preserves F16 embedding storage');
assert.deepEqual(widened.kernels.project, graph.kernels.project, 'KV widening preserves projection arithmetic');
assert.equal(widened.kernels.indexed.kernel, 'attention_indexed.wgsl');
assert.equal(widened.kernels.indexed.precision.kvDtype, 'f32');
assert.equal(widened.kernels.indexed.digest, null, 'new cache binding requires a new digest');
assert.deepEqual(widened.decode, graph.decode);
const caps = { hasF16: true, hasSubgroups: true };
const context = { modelId: 'qwen-3-8-flash-next-text-q4k-ehf16-af32', activationDtype: 'f32', kvDtype: 'f32' };
const chosen = resolveCapabilityTransforms(caps, {}, context);
assert.equal(chosen.kind, 'explicit-lane');
assert.deepEqual(chosen.names, ['widenAttentionKvToF32']);
assert.deepEqual(chosen.transforms[0](graph, {}), widened);
assert.notDeepEqual(resolveCapabilityTransforms(caps, {}, { ...context, kvDtype: 'f16' }).names, chosen.names);
assert.notDeepEqual(resolveCapabilityTransforms(caps, {}, { ...context, modelId: 'unrelated' }).names, chosen.names);
installNodeFileFetchShim();
const load = async id => resolveRuntimeConfig(stripLegacyRuntimeIntent((await loadRuntimeProfile(id)).runtime));
const parent = await load('profiles/flash-next-routed-qualification');
const candidate = mergeRuntimeValues(parent, await load('profiles/flash-next-routed-f32kv-qualification'));
assert.equal(candidate.inference.session.kvcache.kvDtype, 'f32');
const child = await load('profiles/flash-next-node-host-cache-qualification');
assert.deepEqual(child.inference.executionPatch, candidate.inference.executionPatch,
  'host-cache profile inherits the same preparation choices');
assert.equal(candidate.inference.kernelPath ?? null, null, 'the profile patches the graph before expansion, without a duplicated inline plan');
const recipe = JSON.parse(await readFile(new URL('../../src/config/conversion/qwen3/qwen-3-8-flash-next-text-q4k-ehf16-af32.json', import.meta.url)));
const originalGraph = structuredClone(recipe.execution);
const compile = (patch, hasSubgroups = true) => compileExecutionV1({
  manifestInference: { ...recipe.inference, schema: 'doppler.execution/v1',
    session: recipe.session, execution: recipe.execution },
  modelId: recipe.output.modelBaseId, numLayers: 48, headDim: 256, weightDtype: 'q4k',
  runtimeSession: candidate.inference.session, executionPatch: patch,
  kernelPathPolicy: candidate.inference.kernelPathPolicy,
  capabilities: { hasF16: true, hasSubgroups, maxWorkgroupStorageSize: 32768 },
  platform: { id: 'vulkan', vendor: 'amd', architecture: 'rdna-3' },
});
const baseline = compile({}).runtimeInferencePatch.kernelPath;
const prepared = compile(candidate.inference.executionPatch).runtimeInferencePatch.kernelPath;
const selected = getKernelConfig('matmul', 'f32_row_subgroup');
const roles = new Set(['moe_router', 'attn_hc_down', 'attn_hc_up', 'attn_hc_inject',
  'mlp_hc_down', 'mlp_hc_up', 'mlp_hc_inject']);
let changed = 0;
for (let layer = 0; layer < 48; layer++) {
  assert.deepEqual(getLayerSteps(baseline, layer, 'decode'), getLayerSteps(prepared, layer, 'decode'),
    `layer ${layer}: decode remains unchanged`);
  const before = getLayerSteps(baseline, layer, 'prefill');
  const after = getLayerSteps(prepared, layer, 'prefill');
  assert.equal(before.length, after.length);
  for (let index = 0; index < before.length; index++) {
    const a = before[index], b = after[index];
    assert.equal(a.op, b.op, 'operation order is preserved');
    if (roles.has(a.op)) {
      assert.equal(b.kernel, selected.shaderFile);
      assert.equal(b.entry, selected.entryPoint);
      assert.notEqual(a.kernel, b.kernel);
      assert.deepEqual({ ...b, kernel: a.kernel, entry: a.entry }, a,
        'weights, precision and all other dispatch fields remain unchanged');
      changed++;
    } else assert.deepEqual(b, a, `layer ${layer}: unrelated ${a.op} remains unchanged`);
  }
}
assert.equal(changed, 48 * roles.size, 'all layers, including PLE and QSA overrides, receive the declared seven-role patch');
assert.deepEqual(prepared.preLayer, baseline.preLayer);
assert.deepEqual(prepared.postLayer, baseline.postLayer);
assert.deepEqual(recipe.execution, originalGraph, 'preparation cannot mutate the artifact graph');
const invalidPatch = structuredClone(candidate.inference.executionPatch);
invalidPatch.set.at(-1).op = 'missing_hyper_connection_role';
assert.throws(() => compile(invalidPatch), /did not match any execution step/i,
  'a missing role cannot silently leave a partially patched preparation');
candidate.inference.session.kvcache.kvDtype = parent.inference.session.kvcache.kvDtype;
candidate.inference.executionPatch = parent.inference.executionPatch;
assert.deepEqual(candidate, parent, 'the profile changes only KV storage and the explicit prefill patch');
console.log('Flash-Next F32 KV: explicit cache policy, 336 prepared control bindings, unchanged decode/other operations, immutable graph and missing-role rejection passed (metadata only).');
