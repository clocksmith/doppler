import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { getKernelConfig } from '../../src/config/kernel-registry-contract.js';
import { readFile } from 'node:fs/promises';
import { expandExecutionV1 } from '../../src/config/schema/execution-v1.schema.js';
import { assertPairedExecutionMechanismClosure } from '../../src/config/execution-registry-contract.js';
import { assertPairedLinearMechanismClosure } from '../../src/inference/pipelines/text/paired-execution.js';
import { buildWgslClosure } from '../../src/tooling/program-bundle/wgsl-closure.js';
import { createShaderSourceScope, runWithShaderSourceScope } from '../../src/gpu/kernels/shader-source-scope.js';
import { loadShaderSource } from '../../src/gpu/kernels/shader-cache.js';
const read = async path => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const original = await read('../../artifacts/paired-intermediates-20261012/execution.json');
const graph = await read('../../artifacts/paired-intermediates-20261012/signed-closure-correction/execution.json');
// Pre-field manifests omit linearNormGateActivation; the closure resolves the schema default (silu).
const legacy = execution => ({ execution, normalization: {} });
const sigmoidInference = execution => ({ execution, normalization: { linearNormGateActivation: 'sigmoid' } });
assert.throws(() => assertPairedExecutionMechanismClosure(graph, []), /norm-gate mechanisms/);
assert.throws(() => assertPairedLinearMechanismClosure(legacy(original)), /Missing pinned mechanism/);
assert.throws(() => assertPairedLinearMechanismClosure(legacy(graph)), /qwen_linear_attention_prepare\/paired_beta/);
const gateVariants = ['paired_beta', 'paired_decay_scale', 'paired_log_decay', 'paired_decay'];
const recurrentVariants = ['paired_qk', 'paired_state', 'paired_norm', 'paired_gate'];
const configs = new Map([...gateVariants.map(v => [v, getKernelConfig('qwen_linear_attention_prepare', v)]), ...recurrentVariants.map(v => [v, getKernelConfig('gated_delta_recurrent', v)])]);
for (const [variant, gate] of configs) {
  const gateSource = await readFile(new URL(`../../src/gpu/kernels/${gate.shaderFile}`, import.meta.url), 'utf8');
  graph.kernels[variant] = { kernel: gate.shaderFile, entry: gate.entryPoint, constants: gate.wgslOverrides ?? {},
    digest: `sha256:${createHash('sha256').update(gateSource.replace(/\r\n/g, '\n') + '\n@@entry:' + gate.entryPoint).digest('hex')}` };
  graph.mechanismKernels.push(variant);
}
assert.doesNotThrow(() => assertPairedLinearMechanismClosure(legacy(graph)));
assert.doesNotThrow(() => assertPairedLinearMechanismClosure({ execution: graph, normalization: { linearNormGateActivation: 'silu' } }));
// The same silu closure cannot admit a sigmoid manifest: the declared silu gate
// and monolithic anchor fail the norm-gate guard before closure lookup.
assert.throws(() => assertPairedLinearMechanismClosure(sigmoidInference(graph)), /applies a silu norm gate, but the manifest declares sigmoid/);
{
  const sigmoidGraph = structuredClone(graph);
  const gate = getKernelConfig('gated_delta_recurrent', 'paired_gate_sigmoid');
  const gateSource = await readFile(new URL(`../../src/gpu/kernels/${gate.shaderFile}`, import.meta.url), 'utf8');
  for (const key of Object.keys(sigmoidGraph.kernels)) {
    const decl = sigmoidGraph.kernels[key];
    if (['gated_delta_recurrent_paired.wgsl', 'gated_delta_gate_paired.wgsl'].includes(decl.kernel)) {
      delete sigmoidGraph.kernels[key]; sigmoidGraph.mechanismKernels = sigmoidGraph.mechanismKernels.filter(id => id !== key);
    }
  }
  sigmoidGraph.kernels.paired_gate_sigmoid = { kernel: gate.shaderFile, entry: gate.entryPoint, constants: gate.wgslOverrides ?? {},
    digest: `sha256:${createHash('sha256').update(gateSource.replace(/\r\n/g, '\n') + '\n@@entry:' + gate.entryPoint).digest('hex')}` };
  sigmoidGraph.mechanismKernels.push('paired_gate_sigmoid');
  assert.doesNotThrow(() => assertPairedLinearMechanismClosure(sigmoidInference(sigmoidGraph)));
  assert.throws(() => assertPairedLinearMechanismClosure(legacy(sigmoidGraph)), /applies a sigmoid norm gate, but the manifest declares silu/);
  const missingGate = structuredClone(sigmoidGraph);
  missingGate.mechanismKernels = missingGate.mechanismKernels.filter(id => id !== 'paired_gate_sigmoid');
  assert.throws(() => assertPairedLinearMechanismClosure(sigmoidInference(missingGate)), /gated_delta_recurrent\/paired_gate_sigmoid/);
}
for (const declaration of Object.values(graph.kernels)) {
  const code = await readFile(new URL(`../../src/gpu/kernels/${declaration.kernel}`, import.meta.url), 'utf8');
  declaration.digest = `sha256:${createHash('sha256').update(code.replace(/\r\n/g, '\n') + '\n@@entry:' + declaration.entry).digest('hex')}`;
}
const bundle = await buildWgslClosure(graph, expandExecutionV1(graph), { repoRoot: process.cwd() });
const sources = new Map(bundle.modules.map(module => [module.file,
  bundle.packageFiles.find(file => file.path === module.sourcePath).contents]));
const scope = createShaderSourceScope(sources);
for (const filename of ['gather_f16.wgsl', 'gather_f16_vec4.wgsl', 'gather_split8_f16_vec4_f32_out.wgsl', 'logit_finalize_f32.wgsl', ...[...configs.values()].map(config => config.shaderFile)]) {
  const source = await runWithShaderSourceScope(scope, () => loadShaderSource(filename));
  assert.equal(source, await readFile(new URL(`../../src/gpu/kernels/${filename}`, import.meta.url), 'utf8'));
}
await assert.rejects(runWithShaderSourceScope(scope, () => loadShaderSource('gather_f16_out.wgsl')), /outside the verified Capsule source closure/);
for (const key of ['gather_f16_vec4', 'gather_split8_f16_vec4_f32_out', 'logit_finalize_f32', ...configs.keys()]) {
  const changed = structuredClone(graph); changed.mechanismKernels = changed.mechanismKernels.filter(id => id !== key);
  assert.throws(() => assertPairedLinearMechanismClosure(legacy(changed)), /Missing pinned mechanism/, 'Unreferenced kernel cannot authorize source');
  const changedEntry = structuredClone(graph); changedEntry.kernels[key].entry = 'not-the-selected-entry';
  assert.throws(() => assertPairedLinearMechanismClosure(legacy(changedEntry)), /Missing pinned mechanism/);
  const changedDigest = structuredClone(graph); changedDigest.kernels[key].digest = `sha256:${'0'.repeat(64)}`;
  await assert.rejects(buildWgslClosure(changedDigest, expandExecutionV1(changedDigest), { repoRoot: process.cwd() }), /kernel digest mismatch/);
}
console.log('paired program bundle closure: actual source bytes, scalar/vector/split gather, final logits and strict undeclared rejection pass');
