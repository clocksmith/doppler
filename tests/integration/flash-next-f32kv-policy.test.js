import assert from 'node:assert/strict';
import { widenAttentionKvToF32 } from '../../src/converter/execution-graph-transforms.js';
import { resolveCapabilityTransforms } from '../../src/converter/capability-transform-resolver.js';
import { loadRuntimeProfile } from '../../src/tooling/browser-harness/runtime-config.js';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';
import { resolveRuntimeConfig } from '../../src/config/runtime.js';
import { stripLegacyRuntimeIntent } from '../../src/tooling/runtime-intent.js';
import { mergeRuntimeValues } from '../../src/config/runtime-merge.js';

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
candidate.inference.session.kvcache.kvDtype = parent.inference.session.kvcache.kvDtype;
assert.deepEqual(candidate, parent, 'the separate profile changes only KV storage');
console.log('Flash-Next F32 KV: explicit policy, unchanged weights/projections, immutable graph, and isolated profile passed.');
