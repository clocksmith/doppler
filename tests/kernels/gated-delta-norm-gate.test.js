import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import {
  assertKernelNormGateActivation,
  requireNormGateActivation,
  resolveGatedDeltaNormGateKernel,
} from '../../src/gpu/kernels/gated-delta-norm-gate.js';
import { getKernelConfig, KERNEL_CONFIGS } from '../../src/gpu/kernels/kernel-configs.js';
import { assertLinearNormGateClosure } from '../../src/config/execution-registry-contract.js';

// Contract: the gated-delta kernel that applies the gated-RMSNorm gate is chosen
// by rule from the declared activation, silu keeps the original pinned shader
// files, and a kernel whose registry activation differs from the model fails.
// Static source checks only; GPU numerics are not established here.
const kernelSource = name => readFile(new URL(`../../src/gpu/kernels/${name}`, import.meta.url), 'utf8');

const scalar = (operation, group, inputDtype, normGateActivation) =>
  resolveGatedDeltaNormGateKernel(operation, group, 'scalarVariant', { inputDtype, normGateActivation }).shaderFile;
for (const [operation, group, base] of [
  ['gated_delta_recurrent', 'gatedDeltaRecurrent', 'gated_delta_recurrent'],
  ['gated_delta_fused_decode', 'gatedDeltaFusedDecode', 'gated_delta_fused_decode'],
]) {
  assert.equal(scalar(operation, group, 'f32', 'silu'), `${base}.wgsl`, 'silu keeps the pinned f32 shader');
  assert.equal(scalar(operation, group, 'f16', 'silu'), `${base}_f16.wgsl`, 'silu keeps the pinned f16 shader');
  assert.equal(scalar(operation, group, 'f32', 'sigmoid'), `${base}_sigmoid_gate.wgsl`);
  assert.equal(scalar(operation, group, 'f16', 'sigmoid'), `${base}_sigmoid_gate_f16.wgsl`);
}
// Compensated state retains the original compiled shader; scalar fusion must
// not change the numerical or performance path for two-component state.
for (const inputDtype of ['f32', 'f16']) {
  for (const normGateActivation of ['silu', 'sigmoid']) {
    const config = resolveGatedDeltaNormGateKernel('gated_delta_fused_decode', 'gatedDeltaFusedDecode',
      'scalarVariant', { inputDtype, normGateActivation, stateComponents: 2 });
    assert.match(config.shaderFile, /gated_delta_fused_decode_paired/);
    assert.equal(config.variantMetadata.normGateActivation, normGateActivation);
  }
}
const pairedGate = normGateActivation => resolveGatedDeltaNormGateKernel('gated_delta_recurrent', 'gatedDeltaRecurrent',
  'pairedGateVariant', { normGateActivation });
assert.equal(pairedGate('silu').variant, 'paired_gate');
assert.equal(pairedGate('silu').shaderFile, 'gated_delta_gate_paired.wgsl');
assert.equal(pairedGate('sigmoid').variant, 'paired_gate_sigmoid');

for (const invalid of [undefined, null, 'swish', 'SIGMOID']) {
  assert.throws(() => requireNormGateActivation(invalid, 'test'), /norm-gate activation/);
  assert.throws(() => pairedGate(invalid), /norm-gate activation/);
}
// Lane-mismatch guard: a kernel cannot apply a different gate than the model.
assert.throws(() => assertKernelNormGateActivation(getKernelConfig('gated_delta_recurrent', 'default'), 'sigmoid'),
  /declares norm-gate activation "silu", but the model declares "sigmoid"/);
assert.throws(() => assertKernelNormGateActivation(getKernelConfig('gated_delta_fused_decode', 'sigmoid_gate_f16'), 'silu'),
  /"sigmoid", but the model declares "silu"/);
assert.throws(() => assertKernelNormGateActivation(getKernelConfig('gated_delta_recurrent', 'paired_norm'), 'silu'),
  /declares norm-gate activation null/, 'a kernel without declared gate metadata cannot be accepted as a gate');

// Every gate-declaring kernel applies exactly its declared activation, and each
// sigmoid kernel equals its silu source apart from comments and that one call.
const gateCall = { silu: ['silu(f32(z_proj[z_index]))', 'pair_silu(z[z_index])', 'silu(z_proj[z_index])'],
  sigmoid: ['sigmoid_refined(f32(z_proj[z_index]))', 'pair_sigmoid(z[z_index])'] };
const stripComments = text => text.split('\n').filter(line => !line.trimStart().startsWith('//')).join('\n');
for (const config of Object.values(KERNEL_CONFIGS).flatMap(variants => Object.values(variants))) {
  let source = await kernelSource(config.shaderFile);
  // Split mechanisms apply the gate in their explicitly registered final pass.
  const companion = config.variantMetadata?.normGateVariant;
  if (companion) {
    const gate = getKernelConfig(config.operation, companion);
    assertKernelNormGateActivation(gate, config.variantMetadata.normGateActivation);
    source += await kernelSource(gate.shaderFile);
  }
  const applied = Object.keys(gateCall).filter(activation => gateCall[activation].some(call => source.includes(call)));
  // An undeclared gate-applying kernel would bypass the guard.
  assert.deepEqual(applied, config.variantMetadata?.normGateActivation === undefined ? []
    : [config.variantMetadata.normGateActivation], `${config.operation}/${config.variant} gate declaration`);
}
for (const [sigmoidFile, siluFile] of [
  ['gated_delta_recurrent_sigmoid_gate.wgsl', 'gated_delta_recurrent.wgsl'],
  ['gated_delta_recurrent_sigmoid_gate_f16.wgsl', 'gated_delta_recurrent_f16.wgsl'],
  ['gated_delta_fused_decode_sigmoid_gate.wgsl', 'gated_delta_fused_decode.wgsl'],
  ['gated_delta_fused_decode_sigmoid_gate_f16.wgsl', 'gated_delta_fused_decode_f16.wgsl'],
  ['gated_delta_gate_sigmoid_paired.wgsl', 'gated_delta_gate_paired.wgsl'],
]) {
  const silu = stripComments(await kernelSource(siluFile));
  const expected = silu.replace(gateCall.silu[0], gateCall.sigmoid[0]).replace(gateCall.silu[1], gateCall.sigmoid[1]);
  assert.notEqual(expected, silu);
  assert.equal(stripComments(await kernelSource(sigmoidFile)), expected, `${sigmoidFile} differs from ${siluFile} only in the gate`);
}

// Graph-level guard over a real pinned execution graph (Qwen3.8 declares the silu kernels).
const qwen38 = JSON.parse(await readFile(new URL('../../src/config/conversion/qwen3/qwen3.8-27b-text-f16-af32.json', import.meta.url), 'utf8'));
assert.doesNotThrow(() => assertLinearNormGateClosure(qwen38.execution, 'silu'));
assert.throws(() => assertLinearNormGateClosure(qwen38.execution, 'sigmoid'),
  /execution\.kernels\.gated_delta_recurrent \(gated_delta_recurrent\.wgsl\).*silu norm gate, but the manifest declares sigmoid/);
const sigmoidGraph = structuredClone(qwen38.execution);
sigmoidGraph.kernels.gated_delta_recurrent.kernel = 'gated_delta_recurrent_sigmoid_gate.wgsl';
sigmoidGraph.kernels.gated_delta_fused_decode.kernel = 'gated_delta_fused_decode_sigmoid_gate.wgsl';
assert.doesNotThrow(() => assertLinearNormGateClosure(sigmoidGraph, 'sigmoid'));
assert.throws(() => assertLinearNormGateClosure(sigmoidGraph, 'silu'), /sigmoid norm gate, but the manifest declares silu/);
assert.throws(() => assertLinearNormGateClosure(qwen38.execution, 'swish'), /Unsupported declared norm-gate activation/);
// Graphs without gate kernels (Qwen3.5-0.8B scalar graph) are unaffected by either activation.
const qwen35 = JSON.parse(await readFile(new URL('../../src/config/conversion/qwen3/qwen-3-5-0-8b-q4k-ehaf16.json', import.meta.url), 'utf8'));
for (const activation of ['silu', 'sigmoid']) assert.doesNotThrow(() => assertLinearNormGateClosure(qwen35.execution, activation));

console.log('gated-delta-norm-gate.test: rule selection, registry guard, gate-only kernel derivation and graph closure passed');
