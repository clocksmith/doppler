import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { expandExecutionV1 } from '../../src/config/schema/execution-v1.schema.js';
import { buildInlineKernelPath } from '../../src/inference/pipelines/text/execution-runtime-builders.js';
import { KERNEL_REF_CONTENT_DIGESTS } from '../../src/config/kernels/kernel-ref-digests.js';

// Offline candidate construction; preserves phase, geometry, stored weights,
// activation/math dtypes and every unrelated operation. No recipe promotion.
const [baselinePath, outputPath, selection] = process.argv.slice(2);
const allowedKernels = new Set(['matmul_f32_row_subgroup.wgsl', 'matmul_compensated_f32.wgsl']);
const allowedRoles = new Set(['attn_hc_down', 'attn_hc_up', 'attn_hc_inject', 'mlp_hc_down', 'mlp_hc_up', 'mlp_hc_inject', 'moe_router']);
const bindings = allowedKernels.has(selection)
  ? Object.fromEntries(['attn_hc_down', 'attn_hc_inject', 'mlp_hc_down', 'mlp_hc_inject'].map(role => [role, selection]))
  : JSON.parse(await fs.readFile(selection));
assert(Object.entries(bindings).every(([role, kernel]) => allowedRoles.has(role) && allowedKernels.has(kernel)));
const run = JSON.parse(await fs.readFile(baselinePath));
const manifestBytes = await fs.readFile(path.join(run.modelDir, 'manifest.json'));
assert.equal(createHash('sha256').update(manifestBytes).digest('hex'), run.expectedManifestSha256);
const manifest = JSON.parse(manifestBytes);
const graph = structuredClone(manifest.inference.execution);
const before = expandExecutionV1(graph);
const roundingKernels = new Set(['hyper_connection_branch.wgsl', 'hyper_connection_inject.wgsl']);
const digestUpdates = [];
for (const ref of Object.values(graph.kernels)) {
  if (!roundingKernels.has(ref.kernel)) continue;
  const digest = `sha256:${KERNEL_REF_CONTENT_DIGESTS[`${ref.kernel}#${ref.entry}`]}`;
  assert(!digest.endsWith('undefined'));
  if (digest !== ref.digest) {
    digestUpdates.push({ kernel: ref.kernel, entry: ref.entry, before: ref.digest, after: digest });
    ref.digest = digest;
  }
}
for (const [role, kernel] of Object.entries(bindings)) {
  const digest = KERNEL_REF_CONTENT_DIGESTS[`${kernel}#main`];
  assert(digest);
  graph.kernels[`hc_accuracy_${role}`] = { kernel, entry: 'main', digest: `sha256:${digest}` };
}
const roles = new Set(Object.keys(bindings));
let replacements = 0;
function replace(value) {
  if (Array.isArray(value)) {
    if (roles.has(value[0])) { value[1] = `hc_accuracy_${value[0]}`; replacements++; }
    else value.forEach(replace);
  } else if (value && typeof value === 'object') Object.values(value).forEach(replace);
}
replace(graph.prefill);
assert(replacements > 0 || digestUpdates.length > 0);
const after = expandExecutionV1(graph);
assert.equal(before.length, after.length);
for (let i = 0; i < before.length; i++) {
  const a = structuredClone(before[i]), b = structuredClone(after[i]);
  if (a.phase === 'prefill' && roles.has(a.op)) {
    assert.equal(b.kernel, bindings[a.op]);
    for (const field of ['kernel', 'entry', 'digest']) { delete a[field]; delete b[field]; }
  } else if (roundingKernels.has(a.kernel)) {
    assert.equal(b.digest, `sha256:${KERNEL_REF_CONTENT_DIGESTS[`${a.kernel}#${a.entry}`]}`);
    delete a.digest;
    delete b.digest;
  }
  assert.deepEqual(a, b, `only the declared prefill projection may change: ${i}`);
}
const resolved = after.map(step => ({ ...step, kernelRef: { digest: step.digest } }));
const candidate = buildInlineKernelPath(resolved, manifest.inference.session, manifest.modelId, manifest.architecture.numLayers);
assert(candidate && candidate.activationDtype === 'f32' && candidate.kvDtype === 'f16');
const graphSha = createHash('sha256').update(JSON.stringify(graph)).digest('hex');
candidate.id += `-hc-accuracy-${graphSha.slice(0, 12)}`;
candidate.description = 'Explicit F32 arithmetic candidate; no dtype changes or qualification claim';
run.runtimeConfig.inference = { ...run.runtimeConfig.inference, kernelPath: candidate };
run.candidate = { graphSha256: graphSha, projectionPhase: 'prefill', roles: [...roles], bindings, replacements, digestUpdates,
  builderSha256: createHash('sha256').update(await fs.readFile(new URL(import.meta.url))).digest('hex'),
  sourceExecution: graph, normalModelQualified: false };
run.positions = [0, 1017, 2003];
run.captureLinearStages = false;
run.outputPath = outputPath.replace(/-run\.json$/, '-trace.json');
await fs.writeFile(outputPath, JSON.stringify(run, null, 2) + '\n');
console.log(JSON.stringify({ outputPath, graphSha256: graphSha, replacements, bindings }));
