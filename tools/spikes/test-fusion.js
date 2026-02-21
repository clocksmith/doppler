import { selectRuleValue } from './src/rules/rule-registry.js';
console.log(selectRuleValue('inference', 'ffn', 'useFusedGateUp', {
  hasGate: true, hasUp: true, hasDown: true, hasFusedWeights: false,
  inputIsSupported: true, hasLoRA: false, dtypeMatches: true,
  dtypeSupported: true, f16BatchSupported: true, activationDtype: "f16"
}));
