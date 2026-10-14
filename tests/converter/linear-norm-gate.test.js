import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import {
  assertConvertedLinearNormGateActivation,
  resolveSourceLinearNormGateActivation,
} from '../../src/converter/linear-norm-gate.js';
import { resolveConversionPlan } from '../../src/converter/conversion-plan.js';
import { createConverterConfig } from '../../src/config/schema/index.js';

// Contract: source family semantics, not output_gate_type alone, decide the
// linear-attention norm gate; authored conversion values must agree with them.
const readJson = async path => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const flashNext = await readJson('../../artifacts/qwen38-flash-next-20261013/source/config.json');
const qwen38 = await readJson('../../reports/model-ir-v2/source/qwen3.8-27b-config.json');

// Pinned sources: Flash-Next (qwen4_exp, output_gate_type="sigmoid") and Qwen3.8
// (qwen3_5_text, output_gate_type="swish", hardcoded silu upstream).
assert.equal(flashNext.text_config.output_gate_type, 'sigmoid');
assert.equal(resolveSourceLinearNormGateActivation(flashNext), 'sigmoid');
assert.equal(resolveSourceLinearNormGateActivation(qwen38), 'silu');

// qwen4_exp follows `output_gate_type or hidden_act` and accepts only sigmoid/silu.
const qwen4 = textConfig => ({ model_type: 'qwen4_exp', text_config: { model_type: 'qwen4_exp_text', hidden_act: 'silu', ...textConfig } });
assert.equal(resolveSourceLinearNormGateActivation(qwen4({ output_gate_type: null })), 'silu');
assert.equal(resolveSourceLinearNormGateActivation(qwen4({ output_gate_type: '' })), 'silu');
assert.equal(resolveSourceLinearNormGateActivation(qwen4({ output_gate_type: 'silu' })), 'silu');
assert.equal(resolveSourceLinearNormGateActivation({ model_type: 'qwen4_exp_text', output_gate_type: 'sigmoid' }), 'sigmoid');
for (const unsupported of ['swish', 'gelu', 'Sigmoid']) {
  assert.throws(() => resolveSourceLinearNormGateActivation(qwen4({ output_gate_type: unsupported })), /accepts only "sigmoid" or "silu"/);
}
assert.throws(() => resolveSourceLinearNormGateActivation(qwen4({ output_gate_type: null, hidden_act: 'gelu' })), /"gelu"/);

// qwen3_5 hardcodes silu; a contradicting output_gate_type fails rather than being honored.
for (const consistent of [undefined, null, 'swish', 'silu']) {
  assert.equal(resolveSourceLinearNormGateActivation({ model_type: 'qwen3_5', text_config: { output_gate_type: consistent } }), 'silu');
}
assert.throws(() => resolveSourceLinearNormGateActivation({ model_type: 'qwen3_5', text_config: { output_gate_type: 'sigmoid' } }),
  /hardcodes silu/);
assert.equal(resolveSourceLinearNormGateActivation({ model_type: 'gemma3' }), null);

// Conversion contract.
const declared = activation => ({ normalization: activation === undefined ? {} : { linearNormGateActivation: activation } });
assert.doesNotThrow(() => assertConvertedLinearNormGateActivation(declared('sigmoid'), flashNext, 'flash-next'));
assert.throws(() => assertConvertedLinearNormGateActivation(declared(undefined), flashNext, 'flash-next'),
  /must author inference\.normalization\.linearNormGateActivation="sigmoid"/);
assert.throws(() => assertConvertedLinearNormGateActivation(declared('silu'), flashNext, 'flash-next'),
  /declares .*"silu", but the source model_type "qwen4_exp" applies sigmoid/);
assert.doesNotThrow(() => assertConvertedLinearNormGateActivation(declared(undefined), qwen38, 'qwen3.8'), 'pre-field Qwen3.5 configs stay valid');
assert.doesNotThrow(() => assertConvertedLinearNormGateActivation(declared('silu'), qwen38, 'qwen3.8'));
assert.throws(() => assertConvertedLinearNormGateActivation(declared('sigmoid'), qwen38, 'qwen3.8'), /applies silu/);
assert.throws(() => assertConvertedLinearNormGateActivation(declared('swish'), qwen38, 'qwen3.8'), /supported: silu, sigmoid/);
const unknownLinear = { model_type: 'future_hybrid', layer_types: ['linear_attention', 'full_attention'] };
assert.throws(() => assertConvertedLinearNormGateActivation(declared(undefined), unknownLinear, 'future'), /without a known norm-gate mapping/);
assert.doesNotThrow(() => assertConvertedLinearNormGateActivation(declared('sigmoid'), unknownLinear, 'future'));
assert.doesNotThrow(() => assertConvertedLinearNormGateActivation(declared(undefined), { model_type: 'gemma3' }, 'gemma'));

// The checked-in Qwen3.5 conversion config converts unchanged (no field stamped);
// the same config cannot be pointed at a sigmoid source.
const qwen35Config = await readJson('../../src/config/conversion/qwen3/qwen-3-5-0-8b-q4k-ehaf16.json');
const qwen35Raw = { model_type: 'qwen3_5', text_config: { model_type: 'qwen3_5_text' } };
const plan = resolveConversionPlan({ converterConfig: createConverterConfig(qwen35Config), rawConfig: qwen35Raw, tensors: [] });
assert.equal(plan.manifestInference.normalization.linearNormGateActivation, undefined);
assert.throws(() => resolveConversionPlan({ converterConfig: createConverterConfig(qwen35Config), rawConfig: flashNext, tensors: [] }),
  /must author inference\.normalization\.linearNormGateActivation="sigmoid"/);

console.log('linear-norm-gate.test: source semantics and conversion contract passed');
