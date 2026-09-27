import assert from 'node:assert/strict';
import { createInitializedPipeline } from '../../src/inference/pipelines/factory.js';
import { createRuleRegistry, registerRuleGroup, selectRuleValue, enterRuleRegistry } from '../../src/rules/rule-registry.js';
import { createKernelRegistry, getKernelConfig, setKernelValidator, enterKernelRegistry } from '../../src/gpu/kernels/kernel-configs.js';
import { resolvePipelineRegistries } from '../../src/inference/pipelines/shader-scoped-pipeline.js';
import { log } from '../../src/debug/log.js';

const eventsA = [], eventsB = [], sequence = [];
const rules = label => createRuleRegistry({ extensions: [{ domain: 'test', group: 'scope', rules: {
  selected: [{ match: {}, value: label }],
} }] });
const kernels = label => createKernelRegistry({ extensions: { gelu: { variants: {
  gelu: { wgsl: 'gelu.wgsl', entryPoint: 'main', workgroup: [256, 1, 1], variantMetadata: { label } },
} } } });
const value = () => [selectRuleValue('test', 'scope', 'selected', {}), getKernelConfig('gelu', 'gelu').variantMetadata.label];
class Pipeline {
  async initialize() { this.prepared = value(); }
  async loadModel() {}
  async encodeSequence(signal) {
    sequence.push(`start:${value()[0]}`);
    await Promise.resolve();
    signal?.throwIfAborted();
    log.always('test', value()[0]);
    sequence.push(`end:${value()[0]}`);
    return value();
  }
  async *generate() { yield value(); await Promise.resolve(); yield value(); }
  async unload() { log.always('test', `close:${value()[0]}`); }
}
const defaults = resolvePipelineRegistries();
const restoreRules = enterRuleRegistry(rules('active-other-instance'));
const restoreKernels = enterKernelRegistry(kernels('active-other-instance'));
try {
  const independent = resolvePipelineRegistries();
  assert.equal(independent.ruleRegistry, defaults.ruleRegistry);
  assert.equal(independent.kernelRegistry, defaults.kernelRegistry);
} finally { restoreKernels(); restoreRules(); }
const a = await createInitializedPipeline(Pipeline, {}, {
  ruleRegistry: rules('A'), kernelRegistry: kernels('A'), observer: { observe: e => eventsA.push(e) },
});
const b = await createInitializedPipeline(Pipeline, {}, {
  ruleRegistry: rules('B'), kernelRegistry: kernels('B'), observer: { observe: e => eventsB.push(e) },
});
registerRuleGroup('test', 'scope', { selected: [{ match: {}, value: 'late-registration' }] });
setKernelValidator('gelu', 'gelu', () => { throw new Error('late validator'); });
assert.deepEqual(a.prepared, ['A', 'A']);
assert.deepEqual(b.prepared, ['B', 'B']);
assert.deepEqual(await Promise.all([a.encodeSequence(), b.encodeSequence()]), [['A', 'A'], ['B', 'B']]);
assert.deepEqual(sequence, ['start:A', 'end:A', 'start:B', 'end:B']);
const iterator = a.generate();
assert.deepEqual((await iterator.next()).value, ['A', 'A']);
let bRan = false;
const waiting = b.encodeSequence().then(result => { bRan = true; return result; });
await Promise.resolve();
assert.equal(bRan, false);
await iterator.return();
assert.deepEqual(await waiting, ['B', 'B']);
const control = new AbortController(); control.abort(new Error('cancelled'));
await assert.rejects(a.encodeSequence(control.signal), /cancelled/);
await Promise.all([a.unload(), a.unload()]);
assert.deepEqual(await b.encodeSequence(), ['B', 'B']);
assert.equal(eventsA.filter(e => e.message === 'close:A').length, 1);
assert.ok(eventsA.every(e => ['A', 'close:A'].includes(e.message)));
assert.ok(eventsB.every(e => e.message === 'B'));
await b.unload();
assert.equal(selectRuleValue('test', 'scope', 'selected', {}), 'late-registration');
console.log('pipeline-registry-isolation.test: construction, serialized execution, stream return, cancellation, close and observers passed (contract doubles)');
