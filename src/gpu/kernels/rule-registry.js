import { getRuleSet as getBaseRuleSet, selectRuleValue as selectBaseRuleValue } from '../../rules/rule-registry.js';

export function getRuleSet(group, name) {
  return getBaseRuleSet('kernels', group, name);
}

export function selectRuleValue(group, name, context) {
  return selectBaseRuleValue('kernels', group, name, context);
}
