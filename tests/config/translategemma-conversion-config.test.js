import assert from 'node:assert/strict';
import fs from 'node:fs';

import { expandExecutionV1 } from '../../src/config/schema/execution-v1.schema.js';

const converterConfig = JSON.parse(
  fs.readFileSync(
    'src/config/conversion/gemma3/translategemma-4b-it-q4k-ehf16-af32.json',
    'utf8'
  )
);
const studentConfig = JSON.parse(
  fs.readFileSync(
    'src/config/conversion/gemma3/translategemma-4b-1b-enes-q4k-ehf16-af32.json',
    'utf8'
  )
);

// === V1 config structure ===

assert.equal(converterConfig.quantization?.computePrecision, 'f32');
assert.equal(converterConfig.presets, undefined, 'v1 configs have no legacy family-indirection field');
assert.ok(converterConfig.execution?.kernels, 'v1 must have execution.kernels');
assert.ok(converterConfig.inference?.attention, 'v1 must have explicit inference.attention');
assert.ok(converterConfig.inference?.chatTemplate?.type, 'translategemma');

// === Session ===

assert.equal(converterConfig.session?.compute?.defaults?.activationDtype, 'f32');
assert.equal(converterConfig.session?.compute?.defaults?.mathDtype, 'f32');
assert.equal(converterConfig.session?.compute?.defaults?.accumDtype, 'f32');
assert.equal(converterConfig.session?.compute?.defaults?.outputDtype, 'f32');
assert.equal(converterConfig.session?.kvcache?.kvDtype, 'f16');

// === Execution graph expands correctly ===

const expanded = expandExecutionV1(converterConfig.execution);
assert.ok(expanded.length > 0, 'execution must expand to steps');
const lmHeadPrefill = expanded.find((step) => step.op === 'lm_head_prefill');
assert.equal(lmHeadPrefill?.kernel, 'matmul_f16w_f32a.wgsl');

const decodeSteps = expanded.filter((s) => s.phase === 'decode');
const prefillSteps = expanded.filter((s) => s.phase === 'prefill');
assert.ok(decodeSteps.length > 0, 'must have decode steps');
assert.ok(prefillSteps.length > 0, 'must have prefill steps');

// === TranslateGemma specific: Gemma 3 architecture ===

assert.equal(converterConfig.inference.normalization.rmsNormWeightOffset, true);
assert.equal(converterConfig.inference.normalization.rmsNormEps, 1e-6);
assert.equal(converterConfig.inference.ffn.activation, 'gelu');

// === Gamma student contract ===

assert.equal(studentConfig.modelType, 'transformer');
assert.equal(studentConfig.output.textOnly, true);
assert.equal(studentConfig.inference.chatTemplate.type, 'gemma');
assert.equal(studentConfig.inference.attention.valueNorm, false);
assert.equal(studentConfig.inference.ffn.useDoubleWideMlp, false);
assert.equal(studentConfig.inference.rope.ropeLocalPartialRotaryFactor, null);
assert.equal(studentConfig.inference.rope.ropeFrequencyBaseDim, null);
assert.equal(studentConfig.inference.rope.ropeLocalFrequencyBaseDim, null);

console.log('translategemma-conversion-config.test: ok');
