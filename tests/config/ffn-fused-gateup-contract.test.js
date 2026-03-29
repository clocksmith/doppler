import assert from 'node:assert/strict';
import { selectRuleValue } from '../../src/rules/rule-registry.js';

const BASE_FUSED_GATE_UP_CONTEXT = Object.freeze({
  hasGate: true,
  hasUp: true,
  hasDown: true,
  hasFusedWeights: false,
  inputIsSupported: true,
  hasLoRA: false,
  dtypeMatches: true,
  dtypeSupported: true,
  f16BatchSupported: true,
  batchSize: 1,
  hiddenSizeAligned32: true,
});

assert.equal(
  selectRuleValue('inference', 'ffn', 'useFusedGateUp', {
    ...BASE_FUSED_GATE_UP_CONTEXT,
    weightDtype: 'f16',
    activationDtype: 'f32',
  }),
  false,
  'Dense f16 FFN weights must not enable the widened f32 fused gate/up path'
);

assert.equal(
  selectRuleValue('inference', 'ffn', 'useFusedGateUp', {
    ...BASE_FUSED_GATE_UP_CONTEXT,
    weightDtype: 'q4k',
    activationDtype: 'f32',
  }),
  true,
  'Q4K FFN weights should retain the f32 fused gate/up path when hidden size is 32-aligned'
);

assert.equal(
  selectRuleValue('inference', 'ffn', 'useFusedGateUp', {
    ...BASE_FUSED_GATE_UP_CONTEXT,
    weightDtype: 'q4k',
    activationDtype: 'f32',
    batchSize: 4,
  }),
  true,
  'Q4K FFN fused gate/up should stay enabled for batched prefill when hidden size is 32-aligned'
);

assert.equal(
  selectRuleValue('inference', 'ffn', 'useFusedGateUp', {
    ...BASE_FUSED_GATE_UP_CONTEXT,
    weightDtype: 'f16',
    hasQ4KMaterialization: true,
    activationDtype: 'f32',
    batchSize: 4,
  }),
  true,
  'Mixed Q4K materializations should keep the widened f32 fused gate/up path enabled for batched prefill'
);

assert.equal(
  selectRuleValue('inference', 'ffn', 'useFusedGateUp', {
    ...BASE_FUSED_GATE_UP_CONTEXT,
    weightDtype: 'q4k',
    activationDtype: 'f32',
    hiddenSizeAligned32: false,
  }),
  false,
  'Q4K FFN fused gate/up must stay disabled when hidden size is not 32-aligned'
);

assert.equal(
  selectRuleValue('kernels', 'fusedFfn', 'variant', {
    isQ4K: true,
    fusedAllowed: true,
    hiddenSubblockAligned: true,
    batchSize: 4,
    weightDtype: 'q4k',
    useMultiOutput: false,
    hasF16: true,
    useF16Input: false,
  }),
  'q4k_batched',
  'Q4K fused FFN kernel should use the batched variant for prefill'
);

assert.equal(
  selectRuleValue('kernels', 'fusedFfn', 'variant', {
    isQ4K: true,
    fusedAllowed: true,
    hiddenSubblockAligned: true,
    batchSize: 1,
    weightDtype: 'q4k',
    useMultiOutput: false,
    hasF16: true,
    useF16Input: false,
  }),
  'q4k',
  'Q4K fused FFN kernel variant should stay available for 32-aligned hidden sizes'
);

assert.equal(
  selectRuleValue('kernels', 'fusedFfn', 'variant', {
    isQ4K: true,
    fusedAllowed: true,
    hiddenSubblockAligned: false,
    batchSize: 1,
    weightDtype: 'q4k',
    useMultiOutput: false,
    hasF16: true,
    useF16Input: false,
  }),
  'default',
  'Q4K fused FFN kernel variant must not apply to hidden sizes that are not 32-aligned'
);

console.log('ffn-fused-gateup-contract.test: ok');
