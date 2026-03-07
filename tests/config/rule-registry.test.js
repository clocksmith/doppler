import assert from 'node:assert/strict';

import {
  getRuleSet,
  registerRuleGroup,
  selectRuleValue,
} from '../../src/rules/rule-registry.js';

assert.throws(
  () => getRuleSet('missing-domain', 'group', 'name'),
  /unknown domain/
);

assert.throws(
  () => getRuleSet('inference', 'missing-group', 'name'),
  /unknown rule group/
);

assert.throws(
  () => getRuleSet('inference', 'kernelPath', 'missing-set'),
  /unknown rule set/
);

{
  const domain = 'unit_test_rules';
  registerRuleGroup(domain, 'templates', {
    directive: [
      {
        match: { kind: 'ok' },
        value: {
          literal: 1,
          array: [
            { template: 'layer-{layer}' },
            { context: 'mode' },
          ],
          nested: {
            label: { template: '{prefix}-{suffix}' },
            selected: { context: 'selection' },
          },
        },
      },
    ],
  });

  const resolved = selectRuleValue(domain, 'templates', 'directive', {
    kind: 'ok',
    layer: 4,
    mode: 'decode',
    prefix: 'kv',
    suffix: 'cache',
    selection: 'winner',
  });

  assert.deepEqual(resolved, {
    literal: 1,
    array: ['layer-4', 'decode'],
    nested: {
      label: 'kv-cache',
      selected: 'winner',
    },
  });
}

{
  const domain = 'unit_test_rules_missing_context';
  registerRuleGroup(domain, 'templates', {
    directive: [
      {
        match: {},
        value: {
          need: { context: 'requiredKey' },
        },
      },
    ],
  });
  assert.throws(
    () => selectRuleValue(domain, 'templates', 'directive', {}),
    /missing context value/
  );
}

{
  const domain = 'unit_test_rules_missing_template';
  registerRuleGroup(domain, 'templates', {
    directive: [
      {
        match: {},
        value: {
          need: { template: '{requiredKey}' },
        },
      },
    ],
  });
  assert.throws(
    () => selectRuleValue(domain, 'templates', 'directive', {}),
    /missing template key/
  );
}

{
  const builtInRules = getRuleSet('inference', 'execution', 'decodeRecorderEnabled');
  assert.equal(Object.isFrozen(builtInRules), true);
  assert.equal(Object.isFrozen(builtInRules[0]), true);
  assert.throws(
    () => {
      builtInRules[0].value = false;
    },
    /read only|Cannot assign to read only property/i
  );
}

{
  const domain = 'unit_test_rules_immutable_registration';
  const rules = {
    directive: [
      {
        match: { kind: 'before' },
        value: 'first',
      },
      {
        match: {},
        value: 'fallback',
      },
    ],
  };

  registerRuleGroup(domain, 'templates', rules);
  rules.directive[0].value = 'mutated-after-register';

  assert.equal(
    selectRuleValue(domain, 'templates', 'directive', { kind: 'before' }),
    'first'
  );
}

console.log('rule-registry.test: ok');
