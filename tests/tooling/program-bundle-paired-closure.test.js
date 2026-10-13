import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { getKernelConfig } from '../../src/config/kernel-registry-contract.js';
import { readFile } from 'node:fs/promises';
import { expandExecutionV1 } from '../../src/config/schema/execution-v1.schema.js';
import { assertPairedExecutionMechanismClosure } from '../../src/config/execution-registry-contract.js';
import { buildWgslClosure } from '../../src/tooling/program-bundle/wgsl-closure.js';
import { createShaderSourceScope, runWithShaderSourceScope } from '../../src/gpu/kernels/shader-source-scope.js';
import { loadShaderSource } from '../../src/gpu/kernels/shader-cache.js';
const read = async path => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const original = await read('../../artifacts/paired-intermediates-20261012/execution.json');
const graph = await read('../../artifacts/paired-intermediates-20261012/signed-closure-correction/execution.json');
assert.throws(() => assertPairedExecutionMechanismClosure(original), /Missing pinned mechanism/);
assert.throws(() => assertPairedExecutionMechanismClosure(graph), /qwen_linear_attention_prepare\/paired_gates/);
const gate = getKernelConfig('qwen_linear_attention_prepare', 'paired_gates');
const gateSource = await readFile(new URL(`../../src/gpu/kernels/${gate.shaderFile}`, import.meta.url), 'utf8');
graph.kernels.paired_gates = { kernel: gate.shaderFile, entry: gate.entryPoint, constants: gate.wgslOverrides ?? {},
  digest: `sha256:${createHash('sha256').update(gateSource.replace(/\r\n/g, '\n') + '\n@@entry:' + gate.entryPoint).digest('hex')}` };
graph.mechanismKernels.push('paired_gates');
assert.doesNotThrow(() => assertPairedExecutionMechanismClosure(graph));
for (const declaration of Object.values(graph.kernels)) {
  const code = await readFile(new URL(`../../src/gpu/kernels/${declaration.kernel}`, import.meta.url), 'utf8');
  declaration.digest = `sha256:${createHash('sha256').update(code.replace(/\r\n/g, '\n') + '\n@@entry:' + declaration.entry).digest('hex')}`;
}
const bundle = await buildWgslClosure(graph, expandExecutionV1(graph), { repoRoot: process.cwd() });
const sources = new Map(bundle.modules.map(module => [module.file,
  bundle.packageFiles.find(file => file.path === module.sourcePath).contents]));
const scope = createShaderSourceScope(sources);
for (const filename of ['gather_f16.wgsl', 'gather_f16_vec4.wgsl', 'gather_split8_f16_vec4_f32_out.wgsl', 'logit_finalize_f32.wgsl', 'qwen_linear_attention_prepare_paired.wgsl']) {
  const source = await runWithShaderSourceScope(scope, () => loadShaderSource(filename));
  assert.equal(source, await readFile(new URL(`../../src/gpu/kernels/${filename}`, import.meta.url), 'utf8'));
}
await assert.rejects(runWithShaderSourceScope(scope, () => loadShaderSource('gather_f16_out.wgsl')), /outside the verified Capsule source closure/);
for (const key of ['gather_f16_vec4', 'gather_split8_f16_vec4_f32_out', 'logit_finalize_f32', 'paired_gates']) {
  const changed = structuredClone(graph); changed.mechanismKernels = changed.mechanismKernels.filter(id => id !== key);
  assert.throws(() => assertPairedExecutionMechanismClosure(changed), /Missing pinned mechanism/, 'Unreferenced kernel cannot authorize source');
  const changedEntry = structuredClone(graph); changedEntry.kernels[key].entry = 'not-the-selected-entry';
  assert.throws(() => assertPairedExecutionMechanismClosure(changedEntry), /Missing pinned mechanism/);
  const changedDigest = structuredClone(graph); changedDigest.kernels[key].digest = `sha256:${'0'.repeat(64)}`;
  await assert.rejects(buildWgslClosure(changedDigest, expandExecutionV1(changedDigest), { repoRoot: process.cwd() }), /kernel digest mismatch/);
}
console.log('paired program bundle closure: actual source bytes, scalar/vector/split gather, final logits and strict undeclared rejection pass');
