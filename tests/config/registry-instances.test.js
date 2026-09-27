import assert from 'node:assert/strict';
import { createRuleRegistry, getRuleRegistry, registerRuleGroup } from '../../src/rules/rule-registry.js';
import { createKernelRegistry, getKernelRegistry, setKernelValidator } from '../../src/gpu/kernels/kernel-configs.js';
import { KERNEL_CONFIGS } from '../../src/config/kernel-registry-contract.js';
import { resolveExecutionRegistries, assertExecutionRegistriesAccepted } from '../../src/config/execution-registry-contract.js';

const group = { choice: [{ match: {}, value: { nested: ['A'] } }] };
const a = createRuleRegistry({ extensions: [{ domain: 'test', group: 'instance', rules: group }] });
group.choice[0].value.nested[0] = 'caller mutation';
const b = createRuleRegistry({ extensions: [{ domain: 'test', group: 'instance', rules: { choice: [{ match: {}, value: 'B' }] } }] });
registerRuleGroup('test', 'instance', { choice: [{ match: {}, value: 'compatibility' }] });
assert.deepEqual(a.selectRuleValue('test', 'instance', 'choice', {}), { nested: ['A'] });
assert.equal(b.selectRuleValue('test', 'instance', 'choice', {}), 'B');
assert.equal(getRuleRegistry().selectRuleValue('test', 'instance', 'choice', {}), 'compatibility');
assert.throws(() => { a.getRuleSet('test', 'instance', 'choice')[0].value.nested.push('mutated'); }, TypeError);
assert.throws(() => a.getRuleSet('constructor', 'instance', 'choice'), /unknown domain/);
assert.throws(() => createRuleRegistry({ extensions: [{ domain: '__proto__', group: 'x', rules: {} }] }), /safe domain/);
assert.throws(() => createRuleRegistry({ extensions: [{ domain: 'x', group: 'y', rules: { invalid: new Map() } }] }), /JSON/);

const calls = [];
const variants = { gelu: { gelu: { id: 'test.validator.a/v1', validate: () => calls.push('a') } } };
const ka = createKernelRegistry({ validators: variants });
variants.gelu.gelu.validate = () => calls.push('changed');
const kb = createKernelRegistry({ validators: { gelu: { gelu: { id: 'test.validator.b/v1', validate: () => calls.push('b') } } } });
setKernelValidator('gelu', 'gelu', () => calls.push('legacy'));
ka.getKernelValidator('gelu', 'gelu')();
kb.getKernelValidator('gelu', 'gelu')();
getKernelRegistry().getKernelValidator('gelu', 'gelu')();
assert.deepEqual(calls, ['a', 'b', 'legacy']);
for (const config of [KERNEL_CONFIGS.gelu.gelu, ka.getKernelConfig('gelu', 'gelu')]) {
  assert.equal('validate' in config, false);
  for (const mutate of [() => { config.shaderFile = 'bad'; }, () => config.requires.push('bad'),
    () => { config.bindings[0].index = 9; }, () => { config.workgroupSize[0] = 1; },
    () => { config.wgslOverrides.BAD = 1; }, () => { config.uniforms.fields[0].offset = 99; }]) assert.throws(mutate, TypeError);
}
const registries = resolveExecutionRegistries({ ruleRegistry: a, kernelRegistry: ka });
assert.throws(() => assertExecutionRegistriesAccepted({}, registries), /accepted TargetPlan/);
assert.doesNotThrow(() => assertExecutionRegistriesAccepted({ initialExecutionIdentity: { runtimeEngine: { registries: registries.identity } } }, registries));
assert.throws(() => assertExecutionRegistriesAccepted({ initialExecutionIdentity: { runtimeEngine: { registries: registries.identity } } }, null), /accepted TargetPlan/);
console.log('registry-instances.test: isolated snapshots, full immutability, companion validators, plan binding passed');
