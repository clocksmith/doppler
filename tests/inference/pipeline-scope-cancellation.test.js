import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createInitializedPipeline } from '../../src/inference/pipelines/factory.js';
import { scopePipelineShaders, runPipelineOperation } from '../../src/inference/pipelines/shader-scoped-pipeline.js';
import { createShaderSourceScope, getScopedShaderSource } from '../../src/gpu/kernels/shader-source-scope.js';

const bounded = async promise => {
  let timer;
  try {
    return await Promise.race([promise, new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error('Operation remained blocked by a paused stream')), 200);
    })]);
  } finally { clearTimeout(timer); }
};
function fixture(name, { cleanup, work } = {}) {
  const state = { calls: 0, cleaned: [], unloaded: false };
  const shader = () => getScopedShaderSource('scope.wgsl').source;
  const pipeline = scopePipelineShaders({
    isLoaded: true,
    async *generate(_prompt, options = {}) {
      try { yield shader(); await work?.promise; options.signal?.throwIfAborted(); yield shader(); }
      finally { state.cleaned.push(shader()); await cleanup?.promise; }
    },
    async generateTokenIds(_prompt, options = {}) {
      options.signal?.throwIfAborted(); state.calls++; return shader();
    },
    async unload() { assert.equal(shader(), name); state.unloaded = true; },
  }, createShaderSourceScope(new Map([['scope.wgsl', name]])));
  return { pipeline, state };
}

test('queued cancellation rejects before another model releases its paused stream, without bypassing its lease', async () => {
  const a = fixture('A'), b = fixture('B'), c = fixture('C');
  const stream = a.pipeline.generate('x');
  const controller = new AbortController(), reason = new Error('Cancel queued B');
  let later;
  try {
    assert.equal((await stream.next()).value, 'A');
    const queued = b.pipeline.generateTokenIds('x', { signal: controller.signal });
    const rejected = assert.rejects(queued, error => error === reason);
    controller.abort(reason);
    await bounded(rejected);
    assert.equal(b.state.calls, 0);
    let finished = false;
    later = c.pipeline.generateTokenIds('x').then(value => { finished = true; return value; });
    await Promise.resolve(); await Promise.resolve();
    assert.equal(finished, false, 'Cancellation must not unlock A for C');
    assert.equal(getScopedShaderSource('scope.wgsl').source, 'A');
  } finally { await stream.return(); }
  assert.equal(await later, 'C');
  assert.deepEqual(a.state.cleaned, ['A']);
});

test('aborting a paused stream closes it and releases its model context without another consumer pull', async () => {
  const a = fixture('A'), b = fixture('B'), controller = new AbortController();
  const stream = a.pipeline.generate('x', { signal: controller.signal });
  let queued;
  try {
    assert.equal((await stream.next()).value, 'A');
    queued = b.pipeline.generateTokenIds('x');
    controller.abort(new Error('Cancel paused A'));
    assert.equal(await bounded(queued), 'B');
    assert.deepEqual(a.state.cleaned, ['A']);
  } finally { await stream.return(); await queued; }
});

test('unload closes an abandoned paused iterator before draining owned resources', async () => {
  const a = fixture('A'), b = fixture('B'), stream = a.pipeline.generate('x');
  let closing;
  try {
    await stream.next();
    closing = a.pipeline.unload();
    await bounded(closing);
    assert.equal(a.state.unloaded, true);
    assert.deepEqual(a.state.cleaned, ['A']);
    assert.equal(await b.pipeline.generateTokenIds('x'), 'B');
  } finally { await stream.return(); await closing; }
});

test('cancellation retains the lease until active work and iterator cleanup settle', async () => {
  const work = Promise.withResolvers(), cleanup = Promise.withResolvers();
  const a = fixture('A', { work, cleanup }), b = fixture('B'), controller = new AbortController();
  const stream = a.pipeline.generate('x', { signal: controller.signal });
  let step, queued;
  try {
    await stream.next();
    step = assert.rejects(stream.next(), /Cancel active A/);
    queued = b.pipeline.generateTokenIds('x');
    controller.abort(new Error('Cancel active A'));
    await Promise.resolve(); assert.equal(b.state.calls, 0);
    work.resolve();
    await Promise.resolve(); await Promise.resolve();
    assert.equal(b.state.calls, 0, 'Cleanup still owns A’s shader scope');
    cleanup.resolve();
    await step; assert.equal(await queued, 'B');
    assert.deepEqual(a.state.cleaned, ['A']);
  } finally { work.resolve(); cleanup.resolve(); await stream.return(); await step; await queued; }
});


test('surviving waiters remain FIFO across a cancelled middle waiter', async () => {
  const a = fixture('A'), order = [], controller = new AbortController();
  const stream = a.pipeline.generate('x');
  await stream.next();
  const b = fixture('B').pipeline.generateTokenIds('x').then(value => order.push(value));
  const cancelled = fixture('cancelled').pipeline.generateTokenIds('x', { signal: controller.signal });
  const rejected = assert.rejects(cancelled, /middle/);
  const c = fixture('C').pipeline.generateTokenIds('x').then(value => order.push(value));
  controller.abort(new Error('middle'));
  await bounded(rejected);
  await stream.return();
  await Promise.all([b, c]);
  assert.deepEqual(order, ['B', 'C']);
});

test('shutdown reports background cleanup failure and still unloads under its context', async () => {
  const a = fixture('A', { cleanup: { get promise() { return Promise.reject(new Error('cleanup failed')); } } });
  const stream = a.pipeline.generate('x');
  await stream.next();
  await assert.rejects(a.pipeline.unload(), /cleanup failed/);
  assert.equal(a.state.unloaded, true);
  assert.equal(await fixture('B').pipeline.generateTokenIds('x'), 'B');
  await assert.rejects(stream.return(), /cleanup failed/);
});

test('factory loading observes cancellation while waiting for another model scope', async () => {
  const a = fixture('A'), stream = a.pipeline.generate('x'), controller = new AbortController();
  let initialized = false;
  class Pipeline {
    async initialize() { initialized = true; }
    async loadModel() {}
  }
  await stream.next();
  try {
    const loading = createInitializedPipeline(Pipeline, {}, { signal: controller.signal });
    const rejected = assert.rejects(loading, /cancel load/);
    controller.abort(new Error('cancel load'));
    await bounded(rejected);
    assert.equal(initialized, false);
    assert.equal(getScopedShaderSource('scope.wgsl').source, 'A');
  } finally { await stream.return(); }
});

test('factory preserves construction and cleanup failures', async () => {
  class Pipeline {
    async initialize() { throw new Error('initialize failed'); }
    async loadModel() {}
    async unload() { throw new Error('construction cleanup failed'); }
  }
  await assert.rejects(createInitializedPipeline(Pipeline, {}), error => error instanceof AggregateError
    && error.errors.map(cause => cause.message).join(';') === 'initialize failed;construction cleanup failed');
});


test('shutdown preserves the original unload failure for every caller', async () => {
  const failure = new Error('original unload failure');
  const pipeline = scopePipelineShaders({ async unload() { throw failure; } }, null);
  await assert.rejects(pipeline.unload(), error => error === failure);
  await assert.rejects(pipeline.unload(), error => error === failure);
});

test('cleanup that yields retains context until all of its work settles', async () => {
  const cleanup = Promise.withResolvers(), controller = new AbortController(), seen = [];
  const pipeline = scopePipelineShaders({
    async *generate() {
      try { yield 'first'; }
      finally {
        yield 'cleanup yield';
        await cleanup.promise;
        seen.push(getScopedShaderSource('scope.wgsl').source);
      }
    },
  }, createShaderSourceScope(new Map([['scope.wgsl', 'A']])));
  const stream = pipeline.generate('x', { signal: controller.signal });
  await stream.next();
  const b = fixture('B');
  const queued = b.pipeline.generateTokenIds('x');
  controller.abort();
  await Promise.resolve(); await Promise.resolve();
  assert.equal(b.state.calls, 0);
  cleanup.resolve();
  assert.equal(await bounded(queued), 'B');
  assert.deepEqual(seen, ['A']);
  await stream.return();
});


test('host evidence operations forward cancellation to lease acquisition', async () => {
  const a = fixture('A'), stream = a.pipeline.generate('x'), b = fixture('B'), controller = new AbortController();
  await stream.next();
  try {
    const operation = runPipelineOperation(b.pipeline, owner => owner.generateTokenIds('x'), controller.signal);
    const rejected = assert.rejects(operation, /cancel evidence/);
    controller.abort(new Error('cancel evidence'));
    await bounded(rejected);
    assert.equal(b.state.calls, 0);
  } finally { await stream.return(); }
});
