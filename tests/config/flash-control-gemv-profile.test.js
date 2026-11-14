import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import candidate from '../../src/config/runtime/profiles/flash-next-gguf-control-gemv.json' with { type: 'json' };
import policy from '../../src/config/conversion/qwen3/qwen4exp-gguf-source-policy.json' with { type: 'json' };
import { validateRuntimeProfileMetadata } from '../../src/config/schema/runtime-profile.schema.js';
import { createQwen4ExpSourceExecution } from '../../src/converter/qwen4-exp-source-execution.js';
import { applyExecutionPatch } from '../../src/inference/pipelines/text/execution-v1/validation.js';
import { getKernelConfig } from '../../src/gpu/kernels/kernel-configs.js';
import { KERNEL_REF_CONTENT_DIGESTS } from '../../src/config/kernels/kernel-ref-digests.js';

validateRuntimeProfileMetadata(candidate);
assert.deepEqual(Object.keys(candidate.runtime), ['inference']);
assert.deepEqual(Object.keys(candidate.runtime.inference), ['executionPatch']);
const patch = candidate.runtime.inference.executionPatch;
const selected = new Set(patch.set.map(step => step.op));
assert.equal(selected.size, 7);
assert.ok(patch.set.every(step => step.section === 'decode'));
const tensors = [
  { name: 'token_embd.weight', dtype: 'Q4_0' }, { name: 'per_layer_token_embd.weight', dtype: 'Q4_0' },
  ...Object.entries(policy.rootRoles).map(([, name]) => ({ name, dtype: name === 'output.weight' ? 'Q6_K' : 'BF16' })),
  ...Object.entries(policy.layerRoles).map(([role, suffix]) => ({ name: `blk.0.${suffix}`, dtype: selected.has(role) ? 'F32' : 'Q4_0' })),
];
const unique = [...new Map(tensors.map(t => [t.name, t])).values()];
const baseline = createQwen4ExpSourceExecution(unique, 1), original = structuredClone(baseline);
const changed = applyExecutionPatch(baseline, patch);
assert.deepEqual(baseline, original);
for (const section of ['preLayer', 'prefill', 'postLayer']) assert.deepEqual(changed[section], baseline[section]);
const flatten = entries => entries.flatMap(entry => Array.isArray(entry) ? [entry] : entry.steps);
const before = flatten(baseline.decode), after = flatten(changed.decode);
let replacements = 0;
for (let i = 0; i < before.length; i++) {
  if (!selected.has(before[i][0])) { assert.deepEqual(after[i], before[i]); continue; }
  assert.equal(baseline.kernels[before[i][1]].kernel, 'matmul_f32.wgsl');
  assert.equal(after[i][1], patch.addKernels[0].key);
  assert.deepEqual(after[i].slice(2), before[i].slice(2)); replacements++;
}
assert.equal(replacements, 7);
const spec = patch.addKernels[0].kernel, registry = getKernelConfig('matmul', 'gemv_f32_subgroup');
assert.equal(spec.digest, `sha256:${KERNEL_REF_CONTENT_DIGESTS[`${spec.kernel}#${spec.entry}`]}`);
assert.deepEqual(spec.constants, registry.wgslOverrides);
assert.ok(registry.requires.includes('subgroups'));
assert.match(readFileSync(new URL('../../src/gpu/kernels/matmul_gemv_f32_subgroup.wgsl', import.meta.url), 'utf8'), /requires subgroup_id;/);
console.log('Flash control GEMV profile: seven decode-only roles; prefill, packed bindings, cache and sampling unchanged.');
