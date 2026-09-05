import assert from 'node:assert/strict';
import { createShaderSourceScope, runWithShaderSourceScope, getScopedShaderSource,
  bindStorageShaderSourceScope, getStorageShaderSourceScope } from '../../src/gpu/kernels/shader-source-scope.js';
import { loadShaderSource, registerShaderSources, clearShaderCaches, getShaderModule } from '../../src/gpu/kernels/shader-cache.js';
import { scopePipelineShaders } from '../../src/inference/pipelines/shader-scoped-pipeline.js';

const input = new Map([['scope-test.wgsl', 'verified A']]);
const scopeA = createShaderSourceScope(input);
const scopeB = createShaderSourceScope(new Map([['scope-test.wgsl', 'verified B']]));
input.set('scope-test.wgsl', 'mutated');
const storage = {};
bindStorageShaderSourceScope(storage, scopeA);
assert.equal(getStorageShaderSourceScope(storage), scopeA);
assert.throws(() => bindStorageShaderSourceScope(storage, scopeB), /already bound/);
assert.throws(() => createShaderSourceScope(new Map([['../bad.wgsl', 'bad']])), /exact WGSL/);
await assert.rejects(runWithShaderSourceScope({}, () => {}), /Unknown/);
registerShaderSources({ 'scope-test.wgsl': 'unverified warm cache', 'missing.wgsl': 'unverified' });
assert.equal(await loadShaderSource('scope-test.wgsl'), 'unverified warm cache');
const device = { createShaderModule({ code }) { return { code, getCompilationInfo: async () => ({ messages: [] }) }; } };
assert.equal((await getShaderModule(device, 'scope-test.wgsl')).code, 'unverified warm cache');
for (const [scope, source] of [[scopeA, 'verified A'], [scopeB, 'verified B']]) {
  await runWithShaderSourceScope(scope, async () => {
    assert.equal(await loadShaderSource('scope-test.wgsl'), source);
    assert.equal((await getShaderModule(device, 'scope-test.wgsl')).code, source);
    await assert.rejects(loadShaderSource('missing.wgsl'), /outside.*closure/);
    await assert.rejects(getShaderModule(device, 'missing.wgsl'), /outside.*closure/);
  });
}
await assert.rejects(runWithShaderSourceScope(scopeA, () => { throw new Error('failed execution'); }), /failed execution/);
assert.equal(getScopedShaderSource('missing.wgsl'), null);

const observed = [];
class Pipeline {
  constructor() {
    this.generator = { async *generate() { try { yield await loadShaderSource('scope-test.wgsl'); }
      finally { observed.push(await loadShaderSource('scope-test.wgsl')); } },
    async generateTokenIds() { return loadShaderSource('scope-test.wgsl'); } };
  }
  generate() { return this.generator.generate(); }
  generateTokenIds() { return this.generator.generateTokenIds(); }
  async encodeSequence() { await Promise.resolve(); return this.nested(); }
  async nested() { return loadShaderSource('scope-test.wgsl'); }
  get loaded() { return true; }
}
const a = scopePipelineShaders(new Pipeline(), scopeA);
const b = scopePipelineShaders(new Pipeline(), scopeB);
const unbound = scopePipelineShaders(new Pipeline(), null);
assert.equal(a.loaded, true);
assert.equal(a.encodeSequence, a.encodeSequence);
assert.deepEqual(await Promise.all([a.encodeSequence(), b.encodeSequence(), unbound.encodeSequence()]),
  ['verified A', 'verified B', 'unverified warm cache']);
const stream = a.generate();
assert.equal((await stream.next()).value, 'verified A');
let otherFinished = false;
const other = b.generateTokenIds().then((value) => { otherFinished = true; return value; });
await Promise.resolve();
assert.equal(otherFinished, false);
await stream.return();
assert.equal(await other, 'verified B');
assert.deepEqual(observed, ['verified A']);
assert.equal(getScopedShaderSource('scope-test.wgsl'), null);
clearShaderCaches();
console.log('shader-source-scope.test: ok (contract doubles, not GPU execution)');
